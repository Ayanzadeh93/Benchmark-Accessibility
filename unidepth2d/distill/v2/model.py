"""CoDeQ model family and the legacy EfficientViT rungs (docs/V2_PLAN.md sections 3 and 7).

Contract
--------
``build_v2_model(cfg: dict) -> nn.Module``

``cfg`` keys (unknown keys raise ``ValueError`` so a typo cannot silently change a ladder run):

* ``variant`` (required): ``"P" | "N" | "S" | "legacy"``.
* ``decoder_mode``: ``"s4" | "s2" | "ps"`` (closed decoder, plan 3.3) or ``"fpn"`` (the legacy
  FPN decoder of ``distill/student.py`` on the plain encoder, needed by rung R4 so R4 changes
  the encoder only). Default ``"ps"``. Must be omitted / ``"fpn"`` for ``variant="legacy"``.
* ``pretrained`` (default ``True``): ImageNet weights for the encoder, read offline from the local
  HF cache when present (else timm's hub loader). ``weights_file``: optional local timm checkpoint
  path (CoDeQ only). The source and SHA-256 are recorded in ``model.encoder_weights``.
* ``scale_head`` (default ``False``, run T2-13 only): global log-scale head on f4,
  ``log_depth = rel + s``. Adds one global pooling node to the exported graph (measured with
  torch 2.11: ``GlobalAveragePool`` from the TorchScript exporter, ``ReduceMean`` from dynamo).
* ``fpn_channels`` (default 96): decoder width of the ``"fpn"`` mode (legacy ``decoder_channels``).
* ``depth_min_m`` / ``depth_max_m`` (default 0.2 / 30): set ``log_min`` / ``log_max``.
* Legacy rungs R0-R3 (``variant="legacy"``): ``legacy_rung`` in ``LEGACY_RUNGS`` fills defaults
  for ``legacy_backbone`` (``efficientvit_b1.r224_in1k``), ``legacy_decoder_channels`` (96),
  ``legacy_use_intrinsics`` (FiLM), ``legacy_ray_map`` and ``legacy_default_k``
  (``(0.8, 1.08, 0.5, 0.5)``; ``None`` restores the raw legacy K=None behaviour that skips FiLM).

Every returned model:

* ``forward(x, intrinsics=None)``: ``x`` float ``(B,3,H,W)`` ImageNet-normalised RGB ->
  ``log_depth`` ``(B,1,H,W)`` at input resolution, natural-log metres, unclamped. CoDeQ ignores
  ``intrinsics`` (RGB only, no ray map / FiLM). The legacy variant uses them, and substitutes
  ``legacy_default_k`` when they are ``None`` (camera mode "default-k").
* ``log_min`` / ``log_max`` attributes (ln 0.2, ln 30); exp and clamp stay outside the graph
  (``infer_metres`` applies them).
* ``variant``, ``decoder_mode``, ``v2_config`` (resolved cfg dict) and ``allowed_onnx_ops``
  (frozenset for the closed decoder, ``None`` for the legacy/fpn graphs) attributes.
* ``encoder_weights``: ``{"kind", "path", "sha256"}`` of the pretrained encoder weights
  (``kind`` ``"none"`` when ``pretrained=False``; ``"timm-pretrained"`` for the legacy rungs).

CoDeQ input height and width must be multiples of 32 (checked eagerly, skipped while tracing).

Closed-decoder graph (plan 3.3), with ``CBR`` = Conv(bias off)+BN+ReLU, ``DS`` = depthwise CBR 3x3
then pointwise CBR 1x1, ``up2`` = bilinear x2, half-pixel, constant scale:

* ``L_i = CBR(f_i, dc, 1)`` for i = 1..4; ``y4 = DS(L4)``; ``y_i = DS(L_i + up2(y_{i+1}))``.
* ``s4``: ``log_depth = up4(Conv1x1(DS(y1)))`` (one bilinear x4 Resize).
* ``s2``/``ps``: ``y0 = DS(CBR(f0, d2, 1) + up2(CBR(y1, d2, 1)))``; ``c = Conv1x1(DS(y0))`` with
  bias ln(sqrt(0.2*30)) = 0.896; ``s2`` outputs ``up2(c)``; ``ps`` outputs
  ``up2(c) + PixelShuffle(2)(Conv3x3(y0 -> 4))`` with that conv zero-initialised (so ``ps`` equals
  ``s2`` at init). The s4 head bias is initialised to the same 0.896.

Measured at 480x640 (``count_params`` / ``count_macs``, random init): P 1,169,005 params / 1.479
GMAC, N 4,319,797 / 4.805, S 7,017,781 / 5.609 (mode ps), matching plan table 3.1.

Env: conda ``depth``. ONNX checks need ``depth_cmp``. Tests: tests/test_v2_model.py.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from distill.student import DepthStudent, StudentConfig, _ConvBNAct
from distill.v2.encoder import (
    CODEQ_N_DELETE,
    Add,
    ConvBNAct,
    PlainMobileNetV4,
    build_plain_encoder,
)

__all__ = [
    "V2_VARIANTS",
    "LEGACY_RUNGS",
    "DECODER_MODES",
    "ALLOWED_ONNX_OPS",
    "build_v2_model",
    "count_params",
    "count_macs",
    "CoDeQ",
    "ClosedDecoder",
    "PlainFPNStudent",
    "LegacyStudent",
]

#: Family table (plan section 3.1). ``plan_*`` are the numbers measured by the design
#: prototype at 480x640 in decoder mode "ps"; tests assert the rebuilt models stay within 2%.
V2_VARIANTS: Dict[str, dict] = {
    "P": dict(timm_name="mobilenetv4_conv_small.e2400_r224_in1k", delete_blocks=(), dc=48, d2=24,
              plan_encoder_M=1.137, plan_decoder_k=32.1, plan_total_M=1.169, plan_gmac=1.48,
              feature_channels=(32, 32, 64, 96, 128)),
    "N": dict(timm_name="mobilenetv4_conv_medium.e500_r256_in1k", delete_blocks=CODEQ_N_DELETE, dc=64, d2=32,
              plan_encoder_M=4.257, plan_decoder_k=62.3, plan_total_M=4.320, plan_gmac=4.80,
              feature_channels=(32, 48, 80, 160, 256)),
    "S": dict(timm_name="mobilenetv4_conv_medium.e500_r256_in1k", delete_blocks=(), dc=64, d2=32,
              plan_encoder_M=6.955, plan_decoder_k=62.3, plan_total_M=7.018, plan_gmac=5.61,
              feature_channels=(32, 48, 80, 160, 256)),
}

DECODER_MODES: Tuple[str, ...] = ("s4", "s2", "ps", "fpn")

#: ONNX operator allowlist of the closed decoder (plan 3.3 contract). The scale head (T2-13
#: onward, own config only) additionally needs one global pooling op: the plan names
#: AveragePool; torch 2.11 emits GlobalAveragePool (TorchScript exporter) or ReduceMean (dynamo).
ALLOWED_ONNX_OPS = frozenset({"Conv", "Relu", "Add", "Resize", "DepthToSpace"})
SCALE_HEAD_EXTRA_OPS = frozenset({"GlobalAveragePool", "ReduceMean"})

_DEFAULT_K = (0.8, 1.08, 0.5, 0.5)

#: Legacy ladder rungs (plan 7.1 with amendment A7: R1 is split into R1a/R1b; A22: R3r dropped).
#: R0..R2 = efficientvit_b1 with the S16 camera inputs (ray map + FiLM, verified in the
#: outputs/checkpoints/S16_prox/best.pt cfg: use_intrinsics=True, ray_map=True, decoder 96);
#: they differ only in the loss. R3 removes ray map and FiLM (RGB only). R4 onward are CoDeQ
#: models (R4: variant N, decoder_mode "fpn"; R4d onward: closed decoder).
_LEGACY_BASE = dict(legacy_backbone="efficientvit_b1.r224_in1k", legacy_decoder_channels=96,
                    legacy_default_k=_DEFAULT_K)
LEGACY_RUNGS: Dict[str, dict] = {
    "R0": dict(_LEGACY_BASE, legacy_use_intrinsics=True, legacy_ray_map=True),
    "R1a": dict(_LEGACY_BASE, legacy_use_intrinsics=True, legacy_ray_map=True),
    "R1b": dict(_LEGACY_BASE, legacy_use_intrinsics=True, legacy_ray_map=True),
    "R2": dict(_LEGACY_BASE, legacy_use_intrinsics=True, legacy_ray_map=True),
    "R3": dict(_LEGACY_BASE, legacy_use_intrinsics=False, legacy_ray_map=False),
}

_COMMON_KEYS = {"variant", "decoder_mode", "pretrained", "depth_min_m", "depth_max_m"}
_CODEQ_KEYS = _COMMON_KEYS | {"weights_file", "scale_head", "fpn_channels"}
_LEGACY_KEYS = _COMMON_KEYS | {"legacy_rung", "legacy_backbone", "legacy_decoder_channels",
                               "legacy_use_intrinsics", "legacy_ray_map", "legacy_default_k"}


# --------------------------------------------------------------------------------------
# building blocks (plan 3.3 definitions)
# --------------------------------------------------------------------------------------

def cbr(cin: int, cout: int, k: int = 1, groups: int = 1) -> ConvBNAct:
    """``CBR(a,b,k)`` = Conv(k, bias off) + BatchNorm + ReLU."""
    return ConvBNAct(nn.Conv2d(cin, cout, k, 1, k // 2, groups=groups, bias=False),
                     nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


def ds(cin: int, cout: int) -> nn.Sequential:
    """``DS(a,b)`` = depthwise ``CBR(a,a,3,groups=a)`` then pointwise ``CBR(a,b,1)``."""
    return nn.Sequential(cbr(cin, cin, 3, groups=cin), cbr(cin, cout, 1))


def up(scale: int) -> nn.Upsample:
    """Bilinear resize with a constant scale, half-pixel (align_corners=False)."""
    return nn.Upsample(scale_factor=float(scale), mode="bilinear", align_corners=False)


class ClosedDecoder(nn.Module):
    """The op-closed decoder of plan 3.3 (Conv/BN/ReLU/Add/Resize/PixelShuffle only).

    ``mode="s4"``: predict from y1 (stride 4) and upsample x4 bilinearly (one Resize).
    ``mode="s2"``: output ``up2(c)`` only. ``mode="ps"``: ``up2(c) + PixelShuffle(2)(Conv3x3(y0))``
    with the residual conv zero-initialised, so at init ``ps`` equals ``s2``."""

    def __init__(self, enc_channels: Sequence[int], dc: int, d2: int, mode: str, head_bias: float):
        super().__init__()
        if mode not in ("s4", "s2", "ps"):
            raise ValueError(f"closed decoder mode must be s4/s2/ps, got {mode!r}")
        if len(enc_channels) != 5:
            raise ValueError("closed decoder needs five encoder features")
        self.mode = mode
        c = list(enc_channels)
        self.lat = nn.ModuleList([cbr(ci, dc, 1) for ci in c[1:]])          # L1..L4
        self.fuse = nn.ModuleList([ds(dc, dc) for _ in c[1:]])              # fuse[i] -> y_{i+1}
        self.up = nn.ModuleList([up(2) for _ in range(3)])
        self.add = nn.ModuleList([Add() for _ in range(3)])
        if mode == "s4":
            self.head4 = nn.Sequential(ds(dc, dc), nn.Conv2d(dc, 1, 1))
            self.up4 = up(4)
            head = self.head4[-1]
        else:
            self.lat0 = cbr(c[0], d2, 1)
            self.proj1 = cbr(dc, d2, 1)
            self.up0 = up(2)
            self.add0 = Add()
            self.fuse0 = ds(d2, d2)
            self.coarse = nn.Sequential(ds(d2, d2), nn.Conv2d(d2, 1, 1))
            self.up_c = up(2)
            head = self.coarse[-1]
            if mode == "ps":
                self.res_conv = nn.Conv2d(d2, 4, 3, padding=1)
                self.shuffle = nn.PixelShuffle(2)
                self.add_out = Add()
                nn.init.zeros_(self.res_conv.weight)
                nn.init.zeros_(self.res_conv.bias)
        nn.init.constant_(head.bias, float(head_bias))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        lat = [l(f) for l, f in zip(self.lat, feats[1:])]
        y = self.fuse[3](lat[3])                                            # y4
        for i in (2, 1, 0):                                                 # y3, y2, y1
            y = self.fuse[i](self.add[i](lat[i], self.up[i](y)))
        if self.mode == "s4":
            return self.up4(self.head4(y))
        y0 = self.fuse0(self.add0(self.lat0(feats[0]), self.up0(self.proj1(y))))
        c = self.up_c(self.coarse(y0))
        if self.mode == "s2":
            return c
        return self.add_out(c, self.shuffle(self.res_conv(y0)))


class GlobalScaleHead(nn.Module):
    """T2-13 global log-scale head: GAP(f4) -> 1x1 conv -> ReLU -> 1x1 conv -> (B,1,1,1).

    The plan does not specify its shape; this is the smallest op-closed form. The last conv has
    zero weights and bias ``head_bias`` so the model starts at the same output as without it."""

    def __init__(self, cin: int, head_bias: float, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, hidden, 1)
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, float(head_bias))

    def forward(self, f4: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.pool(f4))))


def _tracing() -> bool:
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return True
    try:
        if torch.onnx.is_in_onnx_export():
            return True
    except Exception:  # pragma: no cover - API differences across torch versions
        pass
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # pragma: no cover
        return False


class CoDeQ(nn.Module):
    """Plain MobileNetV4 encoder + closed decoder (+ optional global scale head)."""

    def __init__(self, encoder: PlainMobileNetV4, variant: str, mode: str, dc: int, d2: int,
                 depth_min_m: float = 0.2, depth_max_m: float = 30.0, scale_head: bool = False):
        super().__init__()
        self.encoder = encoder
        self.variant = variant
        self.decoder_mode = mode
        self.log_min = math.log(depth_min_m)
        self.log_max = math.log(depth_max_m)
        head_bias = 0.5 * (self.log_min + self.log_max)                     # ln sqrt(0.2*30) = 0.896
        self.has_scale_head = bool(scale_head)
        self.decoder = ClosedDecoder(encoder.out_channels, dc, d2, mode,
                                     head_bias=0.0 if scale_head else head_bias)
        self.scale_head = GlobalScaleHead(encoder.out_channels[-1], head_bias) if scale_head else None
        self.scale_add = Add() if scale_head else None
        self.allowed_onnx_ops = ALLOWED_ONNX_OPS | (SCALE_HEAD_EXTRA_OPS if scale_head else frozenset())
        self.v2_config: dict = {}

    def _check_input(self, x: torch.Tensor) -> None:
        if _tracing():
            return
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W) input, got {tuple(x.shape)}")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"CoDeQ input H and W must be multiples of 32, got {tuple(x.shape[-2:])}")

    def forward_parts(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """``{"log_depth", "rel_log", "scale"}``; ``scale`` is (B,1,1,1) or None without a scale head."""
        self._check_input(x)
        feats = self.encoder(x)
        rel = self.decoder(feats)
        if self.scale_head is None:
            return {"log_depth": rel, "rel_log": rel, "scale": None}
        s = self.scale_head(feats[-1])
        return {"log_depth": self.scale_add(rel, s), "rel_log": rel, "scale": s}

    def forward(self, x: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        # intrinsics are accepted for interface parity with the legacy students and ignored.
        return self.forward_parts(x)["log_depth"]

    @torch.no_grad()
    def infer_metres(self, x: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.exp(self.forward(x).clamp(self.log_min, self.log_max))


class _FeatureSlice(nn.Module):
    """Returns ``encoder(x)[start:]`` (the legacy FPN consumes strides 4..32 only)."""

    def __init__(self, body: PlainMobileNetV4, start: int = 1):
        super().__init__()
        self.body = body
        self.start = int(start)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.body(x)[self.start:]


class PlainFPNStudent(DepthStudent):
    """Rung R4: the legacy FPN decoder (``distill/student.py`` forward, reused unchanged) on the
    plain MobileNetV4 encoder, RGB only.

    The FPN consumes f1..f4 (strides 4..32) and predicts at stride 4, exactly like the
    EfficientViT students (whose four features are at strides 4..32), so R3 -> R4 changes only
    the encoder. Parameter names follow DepthStudent (laterals/smooths/head)."""

    def __init__(self, encoder: PlainMobileNetV4, variant: str, decoder_channels: int = 96,
                 depth_min_m: float = 0.2, depth_max_m: float = 30.0):
        nn.Module.__init__(self)  # DepthStudent.__init__ would build a timm encoder; set fields directly
        self.cfg = StudentConfig(backbone=encoder.timm_name or "plain_mobilenetv4", pretrained_backbone=False,
                                 decoder_channels=decoder_channels, depth_min_m=depth_min_m,
                                 depth_max_m=depth_max_m)
        self.encoder = _FeatureSlice(encoder, start=1)
        chs = list(encoder.out_channels[1:])
        dc = int(decoder_channels)
        self.laterals = nn.ModuleList([nn.Conv2d(c, dc, 1) for c in chs])
        self.smooths = nn.ModuleList([_ConvBNAct(dc, dc, 3) for _ in chs])
        self.bins = 0
        self.head = nn.Sequential(_ConvBNAct(dc, dc, 3), nn.Conv2d(dc, 1, 1))
        mid_log = math.log(max(1e-3, (depth_min_m * depth_max_m) ** 0.5))
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[0] = mid_log
        self.uncertainty_head = False
        self.ray_map = False
        self.affine_head = False
        self.use_intrinsics = False
        self.log_min = math.log(depth_min_m)
        self.log_max = math.log(depth_max_m)
        self.variant = variant
        self.decoder_mode = "fpn"
        self.allowed_onnx_ops = None
        self.v2_config = {}

    def forward(self, x: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, None)


class LegacyStudent(DepthStudent):
    """Rungs R0-R3: ``distill/student.py`` DepthStudent (unchanged), state-dict compatible with
    the legacy checkpoints, plus the default-K rule: when ``intrinsics`` is None and the model has
    FiLM or a ray map, ``default_k`` is used so FiLM stays on (plan section 2, reviewer item 1)."""

    def __init__(self, scfg: StudentConfig, default_k: Optional[Sequence[float]] = _DEFAULT_K):
        super().__init__(scfg)
        self.default_k = tuple(float(v) for v in default_k) if default_k is not None else None
        self.variant = "legacy"
        self.decoder_mode = "fpn"
        self.allowed_onnx_ops = None
        self.v2_config = {}

    def forward(self, x: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        if intrinsics is None and self.default_k is not None and (self.use_intrinsics or self.ray_map):
            intrinsics = torch.tensor(self.default_k, dtype=x.dtype, device=x.device).view(1, 4).expand(x.shape[0], 4)
        out = super().forward(x, intrinsics)
        return out if out.shape[1] == 1 else out[:, :1]


# --------------------------------------------------------------------------------------
# public builders / helpers
# --------------------------------------------------------------------------------------

def _resolve(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise TypeError("build_v2_model expects a dict")
    if "variant" not in cfg:
        raise ValueError("cfg['variant'] is required (P, N, S or legacy)")
    variant = str(cfg["variant"])
    if variant == "legacy":
        unknown = set(cfg) - _LEGACY_KEYS
        if unknown:
            raise ValueError(f"unknown model cfg keys for variant legacy: {sorted(unknown)}")
        mode = cfg.get("decoder_mode", "fpn")
        if mode != "fpn":
            raise ValueError("variant 'legacy' has the legacy FPN decoder only (decoder_mode must be fpn)")
        out = dict(_LEGACY_BASE, legacy_use_intrinsics=False, legacy_ray_map=False)
        rung = cfg.get("legacy_rung")
        if rung is not None:
            if rung not in LEGACY_RUNGS:
                raise ValueError(f"unknown legacy_rung {rung!r}; choose from {sorted(LEGACY_RUNGS)}")
            out.update(LEGACY_RUNGS[rung])
        elif "legacy_use_intrinsics" not in cfg or "legacy_ray_map" not in cfg:
            raise ValueError("variant 'legacy' needs legacy_rung or both legacy_use_intrinsics and legacy_ray_map")
        out.update({k: v for k, v in cfg.items()})
        out["decoder_mode"] = "fpn"
        out.setdefault("pretrained", True)
        out.setdefault("depth_min_m", 0.2)
        out.setdefault("depth_max_m", 30.0)
        if out["legacy_default_k"] is not None:
            k = tuple(float(v) for v in out["legacy_default_k"])
            if len(k) != 4:
                raise ValueError("legacy_default_k must have four values (fx/W, fy/H, cx/W, cy/H)")
            out["legacy_default_k"] = k
        _check_range(out)
        return out
    if variant not in V2_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(V2_VARIANTS) + ['legacy']}")
    unknown = set(cfg) - _CODEQ_KEYS
    if unknown:
        raise ValueError(f"unknown model cfg keys for variant {variant}: {sorted(unknown)}")
    out = dict(decoder_mode="ps", pretrained=True, weights_file=None, scale_head=False, fpn_channels=96,
               depth_min_m=0.2, depth_max_m=30.0)
    out.update(cfg)
    if out["decoder_mode"] not in DECODER_MODES:
        raise ValueError(f"decoder_mode must be one of {DECODER_MODES}, got {out['decoder_mode']!r}")
    if out["scale_head"] and out["decoder_mode"] == "fpn":
        raise ValueError("scale_head is defined for the closed decoder only")
    _check_range(out)
    return out


def _check_range(out: dict) -> None:
    lo, hi = float(out["depth_min_m"]), float(out["depth_max_m"])
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < depth_min_m < depth_max_m, got {lo}, {hi}")


def build_v2_model(cfg: dict) -> nn.Module:
    """Build a CoDeQ variant (P/N/S, closed or fpn decoder) or a legacy EfficientViT rung."""
    r = _resolve(cfg)
    variant = r["variant"]
    if variant == "legacy":
        scfg = StudentConfig(backbone=r["legacy_backbone"], pretrained_backbone=bool(r["pretrained"]),
                             decoder_channels=int(r["legacy_decoder_channels"]),
                             depth_min_m=float(r["depth_min_m"]), depth_max_m=float(r["depth_max_m"]),
                             use_intrinsics=bool(r["legacy_use_intrinsics"]), ray_map=bool(r["legacy_ray_map"]))
        model: nn.Module = LegacyStudent(scfg, default_k=r["legacy_default_k"])
        model.encoder_weights = {"kind": "timm-pretrained" if r["pretrained"] else "none",
                                 "path": None, "sha256": None}
    else:
        spec = V2_VARIANTS[variant]
        enc = build_plain_encoder(spec["timm_name"], delete_blocks=spec["delete_blocks"], drop_last_stage=True,
                                  pretrained=bool(r["pretrained"]), weights_file=r.get("weights_file"))
        if r["decoder_mode"] == "fpn":
            model = PlainFPNStudent(enc, variant, decoder_channels=int(r["fpn_channels"]),
                                    depth_min_m=float(r["depth_min_m"]), depth_max_m=float(r["depth_max_m"]))
        else:
            model = CoDeQ(enc, variant, r["decoder_mode"], spec["dc"], spec["d2"],
                          depth_min_m=float(r["depth_min_m"]), depth_max_m=float(r["depth_max_m"]),
                          scale_head=bool(r["scale_head"]))
        model.encoder_weights = dict(enc.weights_source)
    model.v2_config = dict(r)
    return model


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (buffers such as BN running stats excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def count_macs(model: nn.Module, input_hw: Tuple[int, int] = (480, 640), in_chans: int = 3) -> int:
    """Conv multiply-accumulates for one image (the plan's E2 method: conv layers only).

    Runs one CPU forward in eval mode with hooks; the model's train/eval state is restored."""
    total = [0]
    hooks = []

    def hook(mod, _inp, out):
        kh, kw = mod.kernel_size
        total[0] += kh * kw * (mod.in_channels // mod.groups) * out.shape[1] * out.shape[2] * out.shape[3]

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        p = next(model.parameters())
        model(torch.zeros(1, in_chans, *input_hw, dtype=p.dtype, device=p.device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return int(total[0])
