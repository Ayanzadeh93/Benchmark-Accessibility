"""V2 training objective (docs/V2_PLAN.md section 4, amendment A7).

Contract
--------
``V2Loss(cfg: dict)``; ``loss(pred_log, batch, aux=None) -> (loss: Tensor, terms: dict[str, float])``

* ``pred_log``: (B,1,H,W) predicted natural-log depth (any float dtype; the loss runs in fp32
  with autocast disabled).
* ``batch`` keys: ``"target_log"`` (B,1,H,W), ``"valid"`` (B,1,H,W) bool, ``"is_gt"`` (B,) bool
  (required only by row-dependent Tier-2 terms), ``"dmax"`` (B,) float metres (optional; when
  present the mask also enforces ``d <= dmax``). Optional ``"region"`` (B,1,H,W) bool marks the
  pasted CutMix region (T2-06 only). Optional ``"pixel_weight"`` (B,1,H,W) float replaces the
  computed range factor ``1 + alpha exp(-d/3)`` (it must already contain it); it exists so the
  legacy dataset's precomputed ``weight`` channel can be replayed bit-exactly (R0 regression).
  All tensors are moved to ``pred_log.device``.
* ``aux`` (T2-14 only): ``{"rel_log": (B,1,H,W), "scale": (B,1,1,1) or (B,)}`` from
  ``CoDeQ.forward_parts``.

Mask: ``M = valid & isfinite(y) & d >= depth_min_m (0.2) & d <= dmax`` (plan notation). Invalid
target pixels are replaced by 0 before any arithmetic, so NaN/inf labels never leak in.

Presets (``cfg["preset"]``, default ``"base"``; any other key overrides one field; unknown keys
raise ``ValueError``; :meth:`V2Loss.resolved_config` returns the resolved dict so the ladder can
compare loss weights, as amendment A7 requires):

* ``"R0"``: the legacy S16 objective, bit-equal to ``distill.losses.DistillLoss`` for fp32 inputs:
  ``silog_loss`` (weights squared, lambda .85) + .5 prox + .5 grad + .3 eg_ssi, every term on the
  range-weighted ``eff_mask = M * (1 + exp(-d/3))`` (legacy ``range`` tag, alpha 1).
* ``"R1a"``: corrected ``L_SI`` (weights applied once, lambda .45, range weight); prox, grad and
  eg_ssi keep ``eff_mask`` (A7).
* ``"R1b"``: R1a with the range weight removed from prox and grad (binary M). eg_ssi keeps
  ``eff_mask`` because A7 names only prox and grad.
* ``"R2"`` / ``"base"``: ``L = 1.0 L_SI + 0.5 L_prox + 0.5 L_grad``, L_SI range-weighted, prox and
  grad on binary M, no eg_ssi.

``L_SI = mean_i sqrt(max(0, sum(w e^2)/sum(w) - lam (sum(w e)/sum(w))^2) + 1e-6)``,
``w = M (1 + alpha exp(-d/3))``. For a pure global offset ``e = c`` it equals
``sqrt((1-lam) c^2 + 1e-6)``: the global-scale coefficient is exactly ``1 - lam`` (.55), for any
weights. The legacy ``silog_loss`` instead gives ``sum(w^2)/sum(w) - lam``.

Tier-2 options (each off by default, each its own run):

* T2-03 ``grad_pseudo_weight=2.0``: pseudo-labelled rows (``~is_gt``) weigh 2.0 in L_grad, GT rows
  keep ``grad_weight``; the batch-pooled denominator of the legacy gradient loss is kept.
* T2-04 ``trim_frac=0.10``: on pseudo-labelled rows drop the 10% of M pixels with the largest
  ``|e - median(e)|`` (detached) from every term.
* T2-06 ``cutmix_si_lambda=1.0``: rows with a ``region`` map get L_SI computed per region (mean
  over non-empty regions) at that lambda.
* T2-07 ``si_lambda=0.85``; T2-08 ``range_alpha=0.0`` (w = M): plain overrides.
* T2-14 ``decoupled_scale=True``: L_SI is replaced by ``L_SI(lam=1)`` on ``aux["rel_log"]`` plus
  ``0.5 * Huber(s - stopgrad(median_M(y - y_rel)), delta=.1)``; prox and grad stay on ``pred_log``.
* T2-05 (photometric perturbation) is data-side and lives in ``distill/v2/data.py``.

QAT: :func:`qat_consistency_loss` gives ``1.0 mean|p_q - p_f| + 10 (mean(p_q - p_f))^2``
(per-image bias, then averaged; the same definition as ``distill.v2.qat.QATLoss`` per_image).

Ladder support (A7): :func:`resolve_loss_config` returns the fully resolved dict for a cfg and
:func:`loss_config_diff` lists the resolved fields that differ between two cfgs, so a ladder
runner can check one-factor changes on resolved weights rather than on declared names.

Precision: the loss always runs in fp32 with autocast disabled. Under CUDA AMP the legacy
trainer evaluated ``eg_ssi_loss``'s Sobel conv inside autocast (fp16), so an R0 run here is the
same objective but not bit-identical to the old AMP numerics; on fp32 inputs R0 is bit-equal to
``DistillLoss`` (tests/test_v2_losses.py).

Env: conda ``depth`` (torch only). Tests: tests/test_v2_losses.py (CPU, offline).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from distill.losses import (
    eg_ssi_loss,
    gradient_matching_loss,
    proximity_weighted_silog,
    silog_loss,
)

__all__ = [
    "V2Loss",
    "LOSS_PRESETS",
    "range_factor",
    "si_per_row",
    "si_loss",
    "si_loss_regions",
    "gradient_matching_rows",
    "trim_mask",
    "decoupled_scale_terms",
    "qat_consistency_loss",
    "legacy_r0_distill_cfg",
    "resolve_loss_config",
    "loss_config_diff",
]

_BASE: Dict[str, object] = dict(
    si_form="corrected",        # "corrected" (weights once) | "legacy" (distill.losses.silog_loss)
    si_weight=1.0,
    si_lambda=0.45,
    range_alpha=1.0,            # w = M (1 + alpha exp(-d / range_scale_m)); 0 -> w = M (T2-08)
    range_scale_m=3.0,
    prox_weight=0.5,
    prox_lambda=0.85,
    prox_band_m=2.0,
    prox_gain=2.0,
    prox_soft_m=0.5,
    prox_mask="binary",         # "binary" (M) | "eff" (range-weighted M, legacy)
    grad_weight=0.5,
    grad_scales=4,
    grad_mask="binary",
    eg_ssi_weight=0.0,
    eg_ssi_mask="eff",
    depth_min_m=0.2,
    enforce_range=True,         # M also requires d >= depth_min_m and d <= dmax (when given)
    eps=1e-6,
    # Tier 2 (off by default)
    grad_pseudo_weight=None,    # T2-03: 2.0
    trim_frac=0.0,              # T2-04: 0.10
    cutmix_si_lambda=None,      # T2-06: 1.0
    decoupled_scale=False,      # T2-14
    decoupled_lambda=1.0,
    decoupled_huber_delta=0.1,
    decoupled_weight=0.5,
)

LOSS_PRESETS: Dict[str, Dict[str, object]] = {
    "R0": dict(_BASE, si_form="legacy", si_lambda=0.85, prox_mask="eff", grad_mask="eff",
               eg_ssi_weight=0.3, eg_ssi_mask="eff"),
    "R1a": dict(_BASE, prox_mask="eff", grad_mask="eff", eg_ssi_weight=0.3, eg_ssi_mask="eff"),
    "R1b": dict(_BASE, prox_mask="binary", grad_mask="binary", eg_ssi_weight=0.3, eg_ssi_mask="eff"),
    "R2": dict(_BASE),
    "base": dict(_BASE),
}

_TIER2_KEYS = ("grad_pseudo_weight", "trim_frac", "cutmix_si_lambda", "decoupled_scale")


def legacy_r0_distill_cfg() -> dict:
    """The S16 ``DistillLoss`` config (read from outputs/checkpoints/S16_prox/best.pt cfg:
    silog 1.0 / lambda .85, grad .5 x4 scales, eg_ssi .3, prox .5, teacher confidence off,
    range_weight_alpha 1.0 applied by the legacy dataset as the per-pixel ``weight``)."""
    return {
        "distill": {
            "loss": {"silog_weight": 1.0, "silog_lambda": 0.85, "grad_weight": 0.5, "grad_scales": 4,
                     "ssi_weight": 0.0, "eg_ssi_weight": 0.3, "prox_silog_weight": 0.5},
            "use_teacher_confidence": False,
        },
        "student": {"depth_min_m": 0.2, "depth_max_m": 30.0},
    }


# --------------------------------------------------------------------------------------
# functional pieces
# --------------------------------------------------------------------------------------

def range_factor(target_log: torch.Tensor, alpha: float = 1.0, scale_m: float = 3.0) -> torch.Tensor:
    """``1 + alpha * exp(-clip(d, 0, 60) / scale_m)`` with ``d = exp(target_log)`` (legacy range tag,
    distill/dataset.py). Multiply by the mask to get the per-pixel weight."""
    d = torch.exp(target_log.clamp(max=math.log(60.0)))
    return 1.0 + float(alpha) * torch.exp(-d / float(scale_m))


def si_per_row(pred_log: torch.Tensor, target_log: torch.Tensor, weight: torch.Tensor,
               lam: float = 0.45, eps: float = 1e-6) -> torch.Tensor:
    """Per-image corrected scale-invariant log loss, shape (B,). Weights are applied once."""
    w = weight.flatten(1)
    e = (pred_log - target_log).flatten(1)
    sw = w.sum(1).clamp(min=1e-12)
    m1 = (w * e).sum(1) / sw
    m2 = (w * e * e).sum(1) / sw
    return torch.sqrt((m2 - float(lam) * m1 * m1).clamp(min=0.0) + float(eps))


def si_loss(pred_log: torch.Tensor, target_log: torch.Tensor, weight: torch.Tensor,
            lam: float = 0.45, eps: float = 1e-6) -> torch.Tensor:
    """Corrected L_SI: mean over images of :func:`si_per_row`."""
    return si_per_row(pred_log, target_log, weight, lam, eps).mean()


def si_loss_regions(pred_log: torch.Tensor, target_log: torch.Tensor, weight: torch.Tensor,
                    region: torch.Tensor, lam: float = 1.0, eps: float = 1e-6,
                    lam_plain: float = 0.45) -> torch.Tensor:
    """T2-06 CutMix L_SI: rows whose ``region`` map is non-empty get the mean of the per-region
    losses (regions with zero weight skipped) at ``lam``; other rows use ``lam_plain``."""
    reg = region.bool()
    has_cut = reg.flatten(1).any(1)
    plain = si_per_row(pred_log, target_log, weight, lam_plain, eps)
    parts, counts = [], []
    for r in (reg, ~reg):
        wr = weight * r.to(weight.dtype)
        parts.append(si_per_row(pred_log, target_log, wr, lam, eps))
        counts.append((wr.flatten(1).sum(1) > 0).to(weight.dtype))
    n = (counts[0] + counts[1]).clamp(min=1.0)
    cut = (parts[0] * counts[0] + parts[1] * counts[1]) / n
    return torch.where(has_cut, cut, plain).mean()


def gradient_matching_rows(pred_log: torch.Tensor, target_log: torch.Tensor, weight: torch.Tensor,
                           scales: int = 4, row_mult: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``distill.losses.gradient_matching_loss`` (batch-pooled ratio per scale) with an optional
    per-row multiplier on the numerator only. ``row_mult=None`` or all ones reproduces the legacy
    value (up to float summation order)."""
    diff = pred_log - target_log
    B = diff.shape[0]
    rm = torch.ones(B, dtype=diff.dtype, device=diff.device) if row_mult is None else row_mult.to(diff.dtype)
    total = diff.new_zeros(())
    d, w = diff, weight
    for s in range(scales):
        gx = (d[..., :, 1:] - d[..., :, :-1]).abs()
        wx = w[..., :, 1:] * w[..., :, :-1]
        gy = (d[..., 1:, :] - d[..., :-1, :]).abs()
        wy = w[..., 1:, :] * w[..., :-1, :]
        num_b = (gx * wx).flatten(1).sum(1) + (gy * wy).flatten(1).sum(1)
        den = wx.sum() + wy.sum()
        total = total + (rm * num_b).sum() / den.clamp(min=1.0)
        if s < scales - 1:
            d = d[..., ::2, ::2]
            w = w[..., ::2, ::2]
    return total / scales


@torch.no_grad()
def trim_mask(pred_log: torch.Tensor, target_log: torch.Tensor, mask: torch.Tensor,
              rows: torch.Tensor, frac: float = 0.10) -> torch.Tensor:
    """T2-04: for each selected row drop ``floor(frac * n_valid)`` mask pixels with the largest
    ``|e - median(e)|`` (``e = pred - target``, detached). Returns a new bool mask."""
    out = mask.clone().bool()
    if frac <= 0:
        return out
    e = (pred_log - target_log).detach()
    for b in range(mask.shape[0]):
        if not bool(rows[b]):
            continue
        m = out[b].flatten()
        n = int(m.sum())
        k = int(math.floor(float(frac) * n))
        if k <= 0:
            continue
        eb = e[b].flatten()
        med = eb[m].median()
        r = (eb - med).abs().masked_fill(~m, float("-inf"))
        idx = torch.topk(r, k, largest=True, sorted=False).indices
        m = m.clone()
        m[idx] = False
        out[b] = m.view_as(out[b])
    return out


def decoupled_scale_terms(rel_log: torch.Tensor, scale: torch.Tensor, target_log: torch.Tensor,
                          weight: torch.Tensor, mask: torch.Tensor, lam: float = 1.0,
                          delta: float = 0.1, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """T2-14: ``(L_SI(rel_log; lam), Huber(s - stopgrad(median_M(y - rel_log)), delta))``.
    The Huber term is averaged over rows that have at least one mask pixel."""
    si_rel = si_loss(rel_log, target_log, weight, lam, eps)
    B = rel_log.shape[0]
    s = scale.reshape(B)
    m = mask.bool()
    with torch.no_grad():
        resid = (target_log - rel_log).detach()
        med = torch.zeros(B, dtype=s.dtype, device=s.device)
        has = torch.zeros(B, dtype=torch.bool, device=s.device)
        for b in range(B):
            mb = m[b]
            if bool(mb.any()):
                med[b] = resid[b][mb].median()
                has[b] = True
    hub = F.huber_loss(s, med, delta=float(delta), reduction="none")
    hub = (hub * has.to(hub.dtype)).sum() / has.sum().clamp(min=1).to(hub.dtype)
    return si_rel, hub


def qat_consistency_loss(p_q: torch.Tensor, p_f: torch.Tensor, mask: Optional[torch.Tensor] = None,
                         l1_weight: float = 1.0, bias_weight: float = 10.0) -> torch.Tensor:
    """Plan 4 QAT term: ``l1_weight * mean|p_q - p_f| + bias_weight * mean_i(mean(p_q - p_f))^2``.
    ``p_f`` (frozen float twin) is detached; the bias term is per image, then averaged."""
    d = (p_q.float() - p_f.detach().float())
    w = torch.ones_like(d) if mask is None else mask.to(d.dtype)
    sw = w.flatten(1).sum(1).clamp(min=1e-12)
    l1 = (d.abs() * w).flatten(1).sum(1) / sw
    bias = (d * w).flatten(1).sum(1) / sw
    return float(l1_weight) * l1.mean() + float(bias_weight) * (bias * bias).mean()


# --------------------------------------------------------------------------------------
# V2Loss
# --------------------------------------------------------------------------------------

def resolve_loss_config(cfg: Optional[dict] = None) -> dict:
    """Fully resolved V2Loss config (preset defaults + overrides), validated.

    Raises ``ValueError`` on an unknown preset, an unknown key or an invalid combination."""
    cfg = dict(cfg or {})
    preset = cfg.pop("preset", "base")
    if preset not in LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {preset!r}; choose from {sorted(LOSS_PRESETS)}")
    unknown = set(cfg) - set(_BASE)
    if unknown:
        raise ValueError(f"unknown V2Loss cfg keys: {sorted(unknown)}")
    r = dict(LOSS_PRESETS[preset])
    r.update(cfg)
    r["preset"] = preset
    for k in ("prox_mask", "grad_mask", "eg_ssi_mask"):
        if r[k] not in ("binary", "eff"):
            raise ValueError(f"{k} must be 'binary' or 'eff', got {r[k]!r}")
    if r["si_form"] not in ("corrected", "legacy"):
        raise ValueError(f"si_form must be 'corrected' or 'legacy', got {r['si_form']!r}")
    for k in ("si_weight", "prox_weight", "grad_weight", "eg_ssi_weight", "range_alpha"):
        if float(r[k]) < 0:
            raise ValueError(f"{k} must be >= 0, got {r[k]}")
    if int(r["grad_scales"]) < 1:
        raise ValueError("grad_scales must be >= 1")
    tier2_on = (r["grad_pseudo_weight"] is not None or float(r["trim_frac"]) > 0
                or r["cutmix_si_lambda"] is not None or bool(r["decoupled_scale"]))
    if r["si_form"] == "legacy" and tier2_on:
        raise ValueError("Tier-2 options are not defined for the legacy (R0) objective")
    if r["grad_pseudo_weight"] is not None and float(r["grad_weight"]) <= 0:
        raise ValueError("grad_pseudo_weight needs grad_weight > 0")
    if r["cutmix_si_lambda"] is not None and bool(r["decoupled_scale"]):
        raise ValueError("cutmix_si_lambda and decoupled_scale are separate runs; enable one")
    if not 0.0 <= float(r["trim_frac"]) < 1.0:
        raise ValueError("trim_frac must be in [0, 1)")
    return r


def loss_config_diff(cfg_a: Optional[dict], cfg_b: Optional[dict]) -> Dict[str, Tuple[object, object]]:
    """``{field: (value_a, value_b)}`` for every resolved field that differs (``preset`` ignored)."""
    a, b = resolve_loss_config(cfg_a), resolve_loss_config(cfg_b)
    return {k: (a[k], b[k]) for k in sorted(set(a) | set(b)) if k != "preset" and a.get(k) != b.get(k)}


class V2Loss:
    """Configurable V2 objective; see the module docstring for presets and terms."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = resolve_loss_config(cfg)

    def resolved_config(self) -> dict:
        return dict(self.cfg)

    # mask ---------------------------------------------------------------------------
    def _mask(self, target_log: torch.Tensor, batch: dict) -> torch.Tensor:
        valid = torch.as_tensor(batch["valid"], device=target_log.device).bool()
        if valid.shape != target_log.shape:
            raise ValueError(f"valid shape {tuple(valid.shape)} != target {tuple(target_log.shape)}")
        m = valid & torch.isfinite(target_log)
        if self.cfg["enforce_range"]:
            tol = 1e-6
            safe = torch.where(m, target_log, torch.zeros_like(target_log))
            m = m & (safe >= math.log(float(self.cfg["depth_min_m"])) - tol)
            dmax = batch.get("dmax")
            if dmax is not None:
                dmax_t = torch.as_tensor(dmax, dtype=torch.float32, device=target_log.device).reshape(-1, 1, 1, 1)
                m = m & (safe <= torch.log(dmax_t) + tol)
        return m

    def __call__(self, pred_log: torch.Tensor, batch: dict,
                 aux: Optional[dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.autocast(device_type=pred_log.device.type, enabled=False):
            return self._compute(pred_log, batch, aux)

    def _compute(self, pred_log: torch.Tensor, batch: dict, aux: Optional[dict]):
        c = self.cfg
        pred = pred_log.float()
        if pred.ndim != 4 or pred.shape[1] != 1:
            raise ValueError(f"pred_log must be (B,1,H,W), got {tuple(pred.shape)}")
        target_raw = batch["target_log"].to(pred.device).float()
        if target_raw.shape != pred.shape:
            raise ValueError(f"target_log shape {tuple(target_raw.shape)} != pred {tuple(pred.shape)}")
        M = self._mask(target_raw, batch)
        is_gt = batch.get("is_gt")
        needs_rows = c["grad_pseudo_weight"] is not None or float(c["trim_frac"]) > 0
        if needs_rows:
            if is_gt is None:
                raise KeyError("batch['is_gt'] is required by grad_pseudo_weight / trim_frac")
            is_gt = torch.as_tensor(is_gt, device=pred.device).bool().reshape(-1)
        if float(c["trim_frac"]) > 0:
            target0 = torch.where(M, target_raw, torch.zeros_like(target_raw))
            M = trim_mask(pred, target0, M, ~is_gt, float(c["trim_frac"]))
        target = torch.where(M, target_raw, torch.zeros_like(target_raw))
        binary = M.float()
        eff = binary * range_factor(target, float(c["range_alpha"]), float(c["range_scale_m"])) \
            if float(c["range_alpha"]) != 0.0 else binary
        masks = {"binary": binary, "eff": eff}

        terms: Dict[str, torch.Tensor] = {}
        loss = pred.new_zeros(())
        eps = float(c["eps"])

        # 1) scale-invariant term ----------------------------------------------------------
        if float(c["si_weight"]) > 0:
            if c["si_form"] == "legacy":
                s = silog_loss(pred, target, eff, float(c["si_lambda"]), None, False)
                terms["si"] = s
                loss = loss + float(c["si_weight"]) * s
            elif c["decoupled_scale"]:
                if aux is None or aux.get("rel_log") is None or aux.get("scale") is None:
                    raise KeyError("decoupled_scale needs aux={'rel_log', 'scale'} (CoDeQ.forward_parts)")
                si_rel, hub = decoupled_scale_terms(aux["rel_log"].float(), aux["scale"].float(), target, eff, M,
                                                    lam=float(c["decoupled_lambda"]),
                                                    delta=float(c["decoupled_huber_delta"]), eps=eps)
                terms["si_rel"] = si_rel
                terms["scale_huber"] = hub
                loss = loss + float(c["si_weight"]) * si_rel
                loss = loss + float(c["decoupled_weight"]) * hub
            elif c["cutmix_si_lambda"] is not None and batch.get("region") is not None:
                region = batch["region"].to(pred.device)
                s = si_loss_regions(pred, target, eff, region, lam=float(c["cutmix_si_lambda"]), eps=eps,
                                    lam_plain=float(c["si_lambda"]))
                terms["si"] = s
                loss = loss + float(c["si_weight"]) * s
            else:
                s = si_loss(pred, target, eff, float(c["si_lambda"]), eps)
                terms["si"] = s
                loss = loss + float(c["si_weight"]) * s

        # 2) proximity-weighted SILog (existing function, weights applied once) -------------
        if float(c["prox_weight"]) > 0:
            ps = proximity_weighted_silog(pred, target, masks[c["prox_mask"]], float(c["prox_lambda"]),
                                          band_m=float(c["prox_band_m"]), gain=float(c["prox_gain"]),
                                          soft_m=float(c["prox_soft_m"]))
            terms["prox"] = ps
            loss = loss + float(c["prox_weight"]) * ps

        # 3) multi-scale gradient matching ---------------------------------------------------
        if float(c["grad_weight"]) > 0:
            gmask = masks[c["grad_mask"]]
            if c["grad_pseudo_weight"] is not None:
                rel = float(c["grad_pseudo_weight"]) / float(c["grad_weight"])
                row_mult = torch.where(is_gt, torch.ones_like(is_gt, dtype=pred.dtype),
                                       torch.full_like(is_gt, rel, dtype=pred.dtype))
                g = gradient_matching_rows(pred, target, gmask, int(c["grad_scales"]), row_mult)
            else:
                g = gradient_matching_loss(pred, target, gmask, int(c["grad_scales"]), None, False)
            terms["grad"] = g
            loss = loss + float(c["grad_weight"]) * g

        # 4) legacy edge-guided SSI (R0, R1a, R1b only) --------------------------------------
        if float(c["eg_ssi_weight"]) > 0:
            e = eg_ssi_loss(pred, target, masks[c["eg_ssi_mask"]])
            terms["eg_ssi"] = e
            loss = loss + float(c["eg_ssi_weight"]) * e

        out = {k: float(v.detach()) for k, v in terms.items()}
        out["total"] = float(loss.detach())
        out["valid_frac"] = float(binary.mean())
        return loss, out
