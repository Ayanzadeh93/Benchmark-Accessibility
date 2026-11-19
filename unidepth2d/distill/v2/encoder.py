"""Plain MobileNetV4-Conv encoder for the CoDeQ family (docs/V2_PLAN.md section 3.2).

Contract
--------
``build_plain_encoder(timm_name, delete_blocks=(), drop_last_stage=True, pretrained=False,
weights_file=None) -> PlainMobileNetV4``

* Builds ``timm.create_model(timm_name, features_only=True, out_indices=(0,1,2,3,4))`` and
  rebuilds it from plain modules only: ``nn.Conv2d``, ``nn.BatchNorm2d``, ``nn.ReLU``,
  ``nn.Identity`` and an explicit :class:`Add` for every residual skip (so a quantization
  observer can attach to each Add). timm's ``BatchNormAct2d`` becomes ``BatchNorm2d`` plus
  the same activation (ReLU or Identity). Any other timm block feature (SE, anti-aliasing,
  layer scale, attention, non-ReLU activation) raises ``TypeError`` instead of being
  silently dropped.
* ``delete_blocks`` (CoDeQ-N: ``(2, 3, 5, 8, 9)``) deletes those indices of ``blocks[3]``.
  Each deleted block must be a stride-1, in==out channel block with a residual skip
  (otherwise ``ValueError``), so deleting it keeps every tensor shape.
* ``drop_last_stage=True`` (all CoDeQ variants) drops ``blocks[4]`` (the 1x1 conv to 960
  channels). The stride-32 feature then is the output of ``blocks[3]``.
  ``drop_last_stage=False`` keeps ``blocks[4]`` and returns exactly timm's five features
  (used by the parity test).
* ``forward(x)`` returns a list of five NCHW features ``[f0..f4]`` at strides
  2/4/8/16/32 (``.reductions``) with channels ``.out_channels``: f0 = stem output,
  f1..f3 = outputs of ``blocks[0..2]``, f4 = output of ``blocks[3]`` (or ``blocks[4]``
  when the last stage is kept).
* Weights are copied from the timm model, so the plain encoder is numerically identical
  to timm (eval mode, measured max abs diff 0.0 on random and pretrained weights).
* ``encoder.timm_key_map`` maps every plain ``state_dict`` key to its timm key (including
  the original block index after deletion). :func:`load_timm_state_dict` uses it to load a
  timm state dict (features model or full classification checkpoint) strictly.
* Pretrained weights: ``weights_file`` loads a local timm checkpoint with no network access.
  With ``pretrained=True`` and no ``weights_file``, :func:`find_cached_timm_weights` first looks
  for the checkpoint in the local Hugging Face cache (``HF_HUB_CACHE``, ``HF_HOME/hub`` or
  ``~/.cache/huggingface/hub``) and loads it offline; only when it is absent does timm's hub
  loader run. ``encoder.weights_source`` records ``{"kind": "file"|"hf-cache"|"timm-hub"|"none",
  "path", "sha256"}`` for provenance.

Env: conda ``depth`` (torch + timm 1.0.27). Tests: tests/test_v2_model.py.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "Add",
    "ConvBNAct",
    "PlainBlock",
    "PlainMobileNetV4",
    "build_plain_encoder",
    "build_timm_features",
    "convert_timm_mobilenetv4",
    "find_cached_timm_weights",
    "load_timm_state_dict",
    "CODEQ_N_DELETE",
]

#: blocks[3] indices removed for CoDeQ-N (plan section 3.2 step 2).
CODEQ_N_DELETE: Tuple[int, ...] = (2, 3, 5, 8, 9)

_BN_KEYS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


class Add(nn.Module):
    """Explicit residual add, a module so observers / fake-quant can attach to it."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> (ReLU | Identity). Plain modules only."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, act: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PlainBlock(nn.Module):
    """A sequence of ConvBNAct layers with an optional explicit residual Add."""

    def __init__(self, body: Sequence[ConvBNAct], skip: bool):
        super().__init__()
        self.body = nn.Sequential(*body)
        self.has_skip = bool(skip)
        self.add = Add() if skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.add is not None:
            y = self.add(y, x)
        return y


class PlainMobileNetV4(nn.Module):
    """Stem + stages of PlainBlocks; returns five features fine -> coarse."""

    def __init__(self, stem: ConvBNAct, stages: Sequence[Sequence[PlainBlock]],
                 out_channels: List[int], timm_key_map: Dict[str, str],
                 timm_name: str = "", deleted: Tuple[int, ...] = (), dropped_last_stage: bool = True):
        super().__init__()
        self.stem = stem
        self.stages = nn.ModuleList(nn.Sequential(*st) for st in stages)
        self.out_channels = list(out_channels)
        self.reductions = [2, 4, 8, 16, 32]
        self.timm_key_map = dict(timm_key_map)
        self.timm_name = timm_name
        self.deleted_blocks = tuple(deleted)
        self.dropped_last_stage = bool(dropped_last_stage)
        self.weights_source: Dict[str, Optional[str]] = {"kind": "none", "path": None, "sha256": None}
        if len(self.stages) not in (4, 5):
            raise ValueError(f"expected 4 or 5 stages, got {len(self.stages)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        feats = [x]
        for st in self.stages:
            x = st(x)
            feats.append(x)
        # 4 stages -> [stem, b0, b1, b2, b3]; 5 stages (timm parity) -> [stem, b0, b1, b2, b4]
        return feats[:4] + [feats[-1]]


# --------------------------------------------------------------------------------------
# conversion helpers
# --------------------------------------------------------------------------------------

def _plain_act(act: nn.Module, where: str) -> nn.Module:
    if isinstance(act, nn.ReLU):
        return nn.ReLU(inplace=False)
    if isinstance(act, nn.Identity):
        return nn.Identity()
    raise TypeError(f"{where}: unsupported activation {type(act).__name__} (plain encoder is ReLU-only)")


def _require_identity(mod: Optional[nn.Module], where: str) -> None:
    if mod is not None and not isinstance(mod, nn.Identity):
        raise TypeError(f"{where}: unsupported non-identity module {type(mod).__name__}")


def _plain_conv(conv: nn.Module, where: str) -> nn.Conv2d:
    if type(conv) is not nn.Conv2d:
        raise TypeError(f"{where}: unsupported conv type {type(conv).__name__}")
    if conv.padding_mode != "zeros" or isinstance(conv.padding, str):
        raise TypeError(f"{where}: unsupported padding {conv.padding!r}/{conv.padding_mode}")
    return copy.deepcopy(conv)


def _plain_bn(bn: nn.Module, where: str) -> Tuple[nn.BatchNorm2d, nn.Module]:
    """timm BatchNormAct2d / BatchNorm2d -> (nn.BatchNorm2d with copied state, activation)."""
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"{where}: unsupported norm {type(bn).__name__}")
    _require_identity(getattr(bn, "drop", None), where + ".drop")
    act = _plain_act(getattr(bn, "act", nn.Identity()), where + ".act")
    out = nn.BatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum,
                         affine=bn.affine, track_running_stats=bn.track_running_stats)
    sd = {k: v for k, v in bn.state_dict().items() if k.split(".")[-1] in _BN_KEYS}
    out.load_state_dict(sd, strict=True)
    return out, act


class _Converter:
    """Builds plain layers and records plain-key -> timm-key for each parameter/buffer."""

    def __init__(self):
        self.key_map: Dict[str, str] = {}

    def cna(self, conv: nn.Module, bn: nn.Module, plain_prefix: str, conv_key: str, bn_key: str,
            extra_act: Optional[nn.Module] = None) -> ConvBNAct:
        c = _plain_conv(conv, conv_key)
        b, act = _plain_bn(bn, bn_key)
        if extra_act is not None and not isinstance(extra_act, nn.Identity):
            if not isinstance(act, nn.Identity):
                raise TypeError(f"{bn_key}: activation given twice")
            act = _plain_act(extra_act, bn_key + "(act1)")
        layer = ConvBNAct(c, b, act)
        for k in c.state_dict():
            self.key_map[f"{plain_prefix}.conv.{k}"] = f"{conv_key}.{k}"
        for k in b.state_dict():
            self.key_map[f"{plain_prefix}.bn.{k}"] = f"{bn_key}.{k}"
        return layer

    def block(self, blk: nn.Module, plain_prefix: str, timm_prefix: str) -> PlainBlock:
        name = type(blk).__name__
        drop_path = getattr(blk, "drop_path", None)
        if drop_path is not None and not isinstance(drop_path, nn.Identity):
            raise TypeError(f"{timm_prefix}: drop_path must be Identity (build with drop_path_rate=0)")
        body: List[ConvBNAct] = []
        if name == "UniversalInvertedResidual":
            _require_identity(getattr(blk, "se", None), timm_prefix + ".se")
            _require_identity(getattr(blk, "layer_scale", None), timm_prefix + ".layer_scale")
            for part in ("dw_start", "pw_exp", "dw_mid", "pw_proj", "dw_end"):
                m = getattr(blk, part)
                if isinstance(m, nn.Identity):
                    continue
                _require_identity(getattr(m, "aa", None), f"{timm_prefix}.{part}.aa")
                i = len(body)
                body.append(self.cna(m.conv, m.bn, f"{plain_prefix}.body.{i}",
                                     f"{timm_prefix}.{part}.conv", f"{timm_prefix}.{part}.bn"))
        elif name == "EdgeResidual":
            _require_identity(getattr(blk, "aa", None), timm_prefix + ".aa")
            _require_identity(getattr(blk, "se", None), timm_prefix + ".se")
            body.append(self.cna(blk.conv_exp, blk.bn1, f"{plain_prefix}.body.0",
                                 f"{timm_prefix}.conv_exp", f"{timm_prefix}.bn1"))
            body.append(self.cna(blk.conv_pwl, blk.bn2, f"{plain_prefix}.body.1",
                                 f"{timm_prefix}.conv_pwl", f"{timm_prefix}.bn2"))
        elif name == "ConvBnAct":
            _require_identity(getattr(blk, "aa", None), timm_prefix + ".aa")
            body.append(self.cna(blk.conv, blk.bn1, f"{plain_prefix}.body.0",
                                 f"{timm_prefix}.conv", f"{timm_prefix}.bn1"))
        else:
            raise TypeError(f"{timm_prefix}: unsupported timm block {name}")
        return PlainBlock(body, bool(getattr(blk, "has_skip", False)))


def _block_is_deletable(blk: nn.Module) -> Tuple[bool, str]:
    if not bool(getattr(blk, "has_skip", False)):
        return False, "has no residual skip"
    convs = [m for m in blk.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return False, "has no conv"
    if any(tuple(c.stride) != (1, 1) for c in convs):
        return False, "is not stride 1"
    if convs[0].in_channels != convs[-1].out_channels:
        return False, "changes the channel count"
    return True, ""


def convert_timm_mobilenetv4(timm_features: nn.Module, delete_blocks: Iterable[int] = (),
                             drop_last_stage: bool = True, timm_name: str = "") -> PlainMobileNetV4:
    """Convert a timm ``MobileNetV3Features`` (MobileNetV4-Conv) into :class:`PlainMobileNetV4`.

    The timm module is not modified. Deleted indices refer to ``blocks[3]``."""
    t = timm_features
    for attr in ("conv_stem", "bn1", "act1", "blocks"):
        if not hasattr(t, attr):
            raise TypeError(f"not a timm MobileNetV3Features model (missing {attr})")
    if len(t.blocks) != 5:
        raise TypeError(f"expected 5 timm stages, got {len(t.blocks)}")
    delete = sorted(set(int(i) for i in delete_blocks))
    n3 = len(t.blocks[3])
    for i in delete:
        if not 0 <= i < n3:
            raise ValueError(f"delete index {i} out of range for blocks[3] (len {n3})")
        ok, why = _block_is_deletable(t.blocks[3][i])
        if not ok:
            raise ValueError(f"blocks[3][{i}] cannot be deleted: it {why}")

    cv = _Converter()
    stem = cv.cna(t.conv_stem, t.bn1, "stem", "conv_stem", "bn1", extra_act=t.act1)
    n_stages = 4 if drop_last_stage else 5
    stages: List[List[PlainBlock]] = []
    for s in range(n_stages):
        out_blocks: List[PlainBlock] = []
        for j, blk in enumerate(t.blocks[s]):
            if s == 3 and j in delete:
                continue
            out_blocks.append(cv.block(blk, f"stages.{s}.{len(out_blocks)}", f"blocks.{s}.{j}"))
        stages.append(out_blocks)

    def _out_ch(blocks: List[PlainBlock]) -> int:
        return blocks[-1].body[-1].conv.out_channels

    out_ch = [stem.conv.out_channels] + [_out_ch(st) for st in stages[:3]] + [_out_ch(stages[-1])]
    enc = PlainMobileNetV4(stem, stages, out_ch, cv.key_map, timm_name=timm_name,
                           deleted=tuple(delete), dropped_last_stage=drop_last_stage)
    enc.train(t.training)
    missing = set(enc.state_dict().keys()) - set(cv.key_map)
    if missing:  # defensive: every plain tensor must trace back to a timm tensor
        raise RuntimeError(f"key map incomplete: {sorted(missing)[:5]}")
    return enc


def _hf_hub_dirs() -> List[str]:
    out = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        out.append(os.path.join(os.environ["HF_HOME"], "hub"))
    out.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    seen, uniq = set(), []
    for d in out:
        k = os.path.normcase(os.path.abspath(d))
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return uniq


def find_cached_timm_weights(timm_name: str) -> Optional[str]:
    """Path of ``timm/<timm_name>`` ``model.safetensors`` (or ``pytorch_model.bin``) in the local
    Hugging Face cache, or ``None``. Prefers the snapshot named by ``refs/main``; otherwise the
    most recently modified snapshot. Never touches the network."""
    for hub in _hf_hub_dirs():
        repo = os.path.join(hub, f"models--timm--{timm_name}")
        if not os.path.isdir(repo):
            continue
        snaps = []
        ref = os.path.join(repo, "refs", "main")
        if os.path.isfile(ref):
            with open(ref, "r", encoding="utf-8") as fh:
                snaps.append(os.path.join(repo, "snapshots", fh.read().strip()))
        others = sorted(glob.glob(os.path.join(repo, "snapshots", "*")), key=os.path.getmtime, reverse=True)
        snaps += [s for s in others if s not in snaps]
        for s in snaps:
            for fname in ("model.safetensors", "pytorch_model.bin"):
                p = os.path.join(s, fname)
                if os.path.isfile(p):
                    return p
    return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_timm_features(timm_name: str, pretrained: bool = False,
                        weights_file: Optional[str] = None) -> nn.Module:
    """timm MobileNetV4 features model with all five features (out_indices 0..4).

    ``weights_file`` loads a local timm checkpoint (e.g. the HF-cache model.safetensors) without
    any network access; otherwise ``pretrained=True`` uses timm's hub loader."""
    import timm

    kwargs = dict(features_only=True, out_indices=(0, 1, 2, 3, 4))
    if weights_file:
        if not os.path.isfile(str(weights_file)):
            raise FileNotFoundError(f"weights_file not found: {weights_file}")
        kwargs["pretrained_cfg_overlay"] = dict(file=str(weights_file))
        pretrained = True
    return timm.create_model(timm_name, pretrained=pretrained, **kwargs)


def build_plain_encoder(timm_name: str, delete_blocks: Iterable[int] = (), drop_last_stage: bool = True,
                        pretrained: bool = False, weights_file: Optional[str] = None,
                        prefer_local_cache: bool = True) -> PlainMobileNetV4:
    """Build the timm model (optionally pretrained) and convert it to the plain encoder.

    With ``pretrained=True`` and no ``weights_file`` the local HF-cache checkpoint is used when
    ``prefer_local_cache`` (offline, reproducible); see the module docstring."""
    kind, path = "none", None
    if weights_file:
        kind, path = "file", str(weights_file)
    elif pretrained:
        path = find_cached_timm_weights(timm_name) if prefer_local_cache else None
        kind = "hf-cache" if path else "timm-hub"
    t = build_timm_features(timm_name, pretrained=pretrained, weights_file=path)
    enc = convert_timm_mobilenetv4(t, delete_blocks=delete_blocks, drop_last_stage=drop_last_stage,
                                   timm_name=timm_name)
    enc.weights_source = {"kind": kind, "path": os.path.abspath(path) if path else None,
                          "sha256": _sha256(path) if path else None}
    return enc


# classifier-head keys of the full timm MobileNetV4 checkpoints (verified on the cached
# conv_small / conv_medium model.safetensors: top-level keys are conv_stem, bn1, blocks,
# conv_head, norm_head, classifier).
_IGNORABLE_TIMM_PREFIXES = ("conv_head.", "norm_head.", "classifier.")


def load_timm_state_dict(encoder: PlainMobileNetV4, timm_sd: Dict[str, torch.Tensor],
                         strict: bool = True) -> Dict[str, List[str]]:
    """Load a timm state dict (features model or full classifier checkpoint) via ``timm_key_map``.

    Strict mode raises ``KeyError`` when a plain key has no timm source, or when a timm key is
    neither mapped nor expected to be unused. Expected-unused keys are: deleted ``blocks.3.*``
    entries, ``blocks.4.*`` when the last stage was dropped, and classifier-head keys.
    Returns ``{"missing": [...], "unused": [...]}`` (plain keys / timm keys)."""
    kmap = encoder.timm_key_map
    target = encoder.state_dict()
    missing = [pk for pk in target if kmap.get(pk) not in timm_sd]
    used = set(kmap.values())
    expected_unused_prefixes = list(_IGNORABLE_TIMM_PREFIXES)
    expected_unused_prefixes += [f"blocks.3.{i}." for i in encoder.deleted_blocks]
    if encoder.dropped_last_stage:
        expected_unused_prefixes.append("blocks.4.")
    unused = [k for k in timm_sd if k not in used]
    unexpected = [k for k in unused if not any(k.startswith(p) for p in expected_unused_prefixes)]
    if strict and (missing or unexpected):
        raise KeyError(f"strict timm load failed: missing={missing[:5]} unexpected={unexpected[:5]}")
    new_sd = {pk: timm_sd[tk] for pk, tk in kmap.items() if tk in timm_sd}
    for pk, v in new_sd.items():
        if tuple(v.shape) != tuple(target[pk].shape):
            raise ValueError(f"shape mismatch for {pk} <- {kmap[pk]}: {tuple(v.shape)} vs {tuple(target[pk].shape)}")
    encoder.load_state_dict(new_sd, strict=strict)
    return {"missing": missing, "unused": unused}
