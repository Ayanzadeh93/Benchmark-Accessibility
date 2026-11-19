"""Metric-preserving distillation losses.

Primary: **SILog** on log-depth (keeps absolute scale at lambda<1) — this is what makes the student
reproduce real metres, not just relative structure. Auxiliary: **multi-scale gradient matching** for
sharp edges. Optional: a **scale-shift-invariant (SSI)** structure term (Distill-Any-Depth style),
OFF by default because it discards absolute scale — enable only to experiment with structure quality.

All terms respect a validity mask and (optionally) per-pixel teacher confidence.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _weighted(
    mask: torch.Tensor, conf: Optional[torch.Tensor], use_conf: bool, conf_floor: float = 0.2
) -> torch.Tensor:
    """Per-pixel loss weight. When using confidence, FLOOR it so even zero-confidence valid pixels
    keep ``conf_floor`` weight — otherwise a degenerate (near-all-zero) confidence map starves the
    student of gradient and collapses it to a constant (the bug this guards against)."""
    w = mask.float()
    if use_conf and conf is not None:
        c = conf.clamp(0.0, 1.0)
        w = w * (conf_floor + (1.0 - conf_floor) * c)
    return w


def silog_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.85,
    conf: Optional[torch.Tensor] = None,
    use_conf: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-Invariant Log loss (Eigen). lam=0.85 keeps most absolute scale (metric)."""
    w = _weighted(mask, conf, use_conf)
    diff = (pred_log - target_log) * w  # zero where invalid
    denom = w.flatten(1).sum(1).clamp(min=1.0)
    mu = diff.flatten(1).sum(1) / denom
    mu2 = (diff * diff).flatten(1).sum(1) / denom
    var_term = (mu2 - lam * mu * mu).clamp(min=0.0)
    return torch.sqrt(var_term + eps).mean()


def gradient_matching_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    scales: int = 4,
    conf: Optional[torch.Tensor] = None,
    use_conf: bool = True,
) -> torch.Tensor:
    """Multi-scale L1 on the gradient of (pred_log - target_log). Sharpens edges."""
    w0 = _weighted(mask, conf, use_conf)
    diff = pred_log - target_log
    total = diff.new_zeros(())
    d, w = diff, w0
    for s in range(scales):
        # gradient in x and y, valid only where both neighbours are weighted
        gx = (d[..., :, 1:] - d[..., :, :-1]).abs()
        wx = w[..., :, 1:] * w[..., :, :-1]
        gy = (d[..., 1:, :] - d[..., :-1, :]).abs()
        wy = w[..., 1:, :] * w[..., :-1, :]
        num = (gx * wx).sum() + (gy * wy).sum()
        den = wx.sum() + wy.sum()
        total = total + num / den.clamp(min=1.0)
        if s < scales - 1:
            d = d[..., ::2, ::2]
            w = w[..., ::2, ::2]
    return total / scales


def _ssi_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample scale-shift normalize using median + mean-abs-deviation over valid pixels."""
    out = torch.zeros_like(x)
    B = x.shape[0]
    for b in range(B):
        m = mask[b].bool()
        if m.sum() < 10:
            continue
        v = x[b][m]
        med = v.median()
        scale = (v - med).abs().mean().clamp(min=1e-6)
        out[b] = (x[b] - med) / scale
    return out


def ssi_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scale-shift-invariant L1 on log-depth (structure only; discards absolute scale)."""
    pn = _ssi_normalize(pred_log, mask)
    tn = _ssi_normalize(target_log, mask)
    w = mask.float()
    diff = (pn - tn).abs() * w
    return diff.sum() / w.sum().clamp(min=1.0)


_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def eg_ssi_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    edge_floor: float = 0.2,
) -> torch.Tensor:
    """Edge-guided scale-shift-invariant L1 (aux term, after UniDepth v2's L_EG-SSI).

    Structure-only supervision concentrated on depth discontinuities of the TARGET: an edge map
    (Sobel magnitude of target log-depth, per-image max-normalized) reweights an SSI-normalized L1.
    Scale-free by construction — keep it auxiliary next to the metric SILog anchor.
    """
    with torch.no_grad():
        sx = _SOBEL_X.to(target_log.device, target_log.dtype)
        sy = _SOBEL_Y.to(target_log.device, target_log.dtype)
        t = torch.nan_to_num(target_log, nan=0.0)
        gx = F.conv2d(t, sx, padding=1)
        gy = F.conv2d(t, sy, padding=1)
        edge = (gx.abs() + gy.abs())
        emax = edge.flatten(1).max(dim=1).values.clamp(min=1e-6).view(-1, 1, 1, 1)
        edge_w = edge_floor + (1.0 - edge_floor) * (edge / emax).clamp(0, 1)
    pn = _ssi_normalize(pred_log, mask)
    tn = _ssi_normalize(target_log, mask)
    w = mask.float() * edge_w
    return ((pn - tn).abs() * w).sum() / w.sum().clamp(min=1.0)


def laplace_nll_loss(
    pred_log_mu: torch.Tensor,
    pred_logsigma: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """beta-NLL (Seitzer et al.) with a Laplace likelihood on log-depth.

    Plain NLL scales the mu-gradient by 1/sigma, so the model inflates sigma and stops improving
    already-accurate regions (variance-dominated gradient shrinkage). beta-NLL multiplies the
    per-pixel NLL by a DETACHED sigma^(2*beta) factor (beta=0.5), restoring gradient flow while
    keeping calibrated uncertainty. beta=0 reduces to plain NLL.
    """
    ls = pred_logsigma.clamp(-3.0, 2.0)
    w = mask.float()
    nll = (pred_log_mu - target_log).abs() * torch.exp(-ls) + ls
    if beta > 0:
        nll = nll * torch.exp(2.0 * beta * ls).detach()
    return (nll * w).sum() / w.sum().clamp(min=1.0)


def bins_ce_loss(
    logits: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    centres_log: torch.Tensor,
) -> torch.Tensor:
    """Soft cross-entropy for the hybrid bin head: target = two-hot linear interpolation of the
    teacher's log-depth between adjacent bin centres. Classification over depth modes gives sharp
    boundaries where regression averages across discontinuities."""
    B = centres_log.numel()
    c0, c1 = float(centres_log[0]), float(centres_log[-1])
    step = (c1 - c0) / (B - 1)
    t = ((target_log - c0) / step).clamp(0, B - 1 - 1e-4)      # fractional bin index
    lo = t.floor().long()
    frac = (t - lo.float())
    logp = torch.log_softmax(logits.float(), dim=1)
    lp_lo = torch.gather(logp, 1, lo)
    lp_hi = torch.gather(logp, 1, (lo + 1).clamp(max=B - 1))
    ce = -((1 - frac) * lp_lo + frac * lp_hi)
    w = mask.float()
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def normal_consistency_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Surface-normal agreement between student and teacher depth, using the camera intrinsics.

    Back-project each depth map to points P(u,v) = d * K^-1 [u,v,1]^T, take finite-difference
    tangents, n = normalize(dP/du x dP/dv), loss = 1 - <n_s, n_t> on interior valid pixels.
    Enforces planarity on indoor surfaces (walls/floors) where photometric cues are weak. fp32.
    """
    B, _, H, W = pred_log.shape
    fx, fy, cx, cy = (intrinsics[:, i].float().view(B, 1, 1) for i in range(4))
    u = ((torch.arange(W, device=pred_log.device).float() + 0.5) / W).view(1, 1, W)
    v = ((torch.arange(H, device=pred_log.device).float() + 0.5) / H).view(1, H, 1)
    x_dir = (u - cx) / fx.clamp(min=1e-4)
    y_dir = (v - cy) / fy.clamp(min=1e-4)

    def normals(log_d):
        d = torch.exp(log_d.float().squeeze(1).clamp(-1.7, 3.5))
        P = torch.stack([d * x_dir.expand_as(d), d * y_dir.expand_as(d), d], dim=1)  # (B,3,H,W)
        du = P[:, :, :, 1:] - P[:, :, :, :-1]
        dv = P[:, :, 1:, :] - P[:, :, :-1, :]
        du = du[:, :, :-1, :]
        dv = dv[:, :, :, :-1]
        n = torch.cross(du, dv, dim=1)
        return n / n.norm(dim=1, keepdim=True).clamp(min=1e-6)

    n_s = normals(pred_log)
    n_t = normals(target_log)
    m = (mask[:, 0, :-1, :-1] & mask[:, 0, :-1, 1:] & mask[:, 0, 1:, :-1]).float()
    cos = (n_s * n_t).sum(dim=1)
    return ((1.0 - cos) * m).sum() / m.sum().clamp(min=1.0)


def object_distance_loss(
    pred_log: torch.Tensor,
    boxes: torch.Tensor,
    dists: torch.Tensor,
    counts: torch.Tensor,
    inset: float = 0.20,
    log_min: float = -1.61,
    log_max: float = 3.41,
) -> torch.Tensor:
    """Differentiable per-object distance consistency (our Task-Weighted Distillation term).

    boxes: (B, K, 4) normalized xyxy on the prediction grid; dists: (B, K) teacher metres;
    counts: (B,) valid boxes per sample. For each box, grid_sample the paper's 5 points
    (centre + 4 corners inset 20%), soft-median (mean of middle 3 of the 5 sorted values),
    L1 in log space vs the teacher's cached distance. fp32 for AMP safety.
    """
    B, K, _ = boxes.shape
    pred = pred_log.float()
    total = pred.new_zeros(())
    n = 0
    for b in range(B):
        c = int(counts[b])
        if c == 0:
            continue
        bx = boxes[b, :c].float()                      # (c,4) x1,y1,x2,y2 in [0,1]
        w_ = bx[:, 2] - bx[:, 0]
        h_ = bx[:, 3] - bx[:, 1]
        cx = (bx[:, 0] + bx[:, 2]) * 0.5
        cy = (bx[:, 1] + bx[:, 3]) * 0.5
        ix, iy = inset * w_, inset * h_
        xs = torch.stack([cx, bx[:, 0] + ix, bx[:, 2] - ix, bx[:, 0] + ix, bx[:, 2] - ix], dim=1)
        ys = torch.stack([cy, bx[:, 1] + iy, bx[:, 1] + iy, bx[:, 3] - iy, bx[:, 3] - iy], dim=1)
        grid = torch.stack([xs, ys], dim=-1) * 2.0 - 1.0          # (c,5,2) in [-1,1]
        samp = F.grid_sample(pred[b:b + 1], grid.view(1, c, 5, 2),
                             mode="bilinear", align_corners=False)  # (1,1,c,5)
        vals = samp[0, 0].clamp(log_min, log_max)                  # (c,5) log-depth
        sorted_v, _ = torch.sort(vals, dim=1)
        soft_med = sorted_v[:, 1:4].mean(dim=1)                    # mean of middle 3
        t_log = torch.log(dists[b, :c].float().clamp(min=1e-3))
        total = total + (soft_med - t_log).abs().sum()
        n += c
    return total / max(1, n)


def proximity_weighted_silog(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.85,
    band_m: float = 2.0,
    gain: float = 2.0,
    soft_m: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """SILog with the safety-critical NEAR band up-weighted (Phase-4 novelty).

    A plain SILog weights every valid pixel equally, so abundant mid/far background sets the
    scale and near-field error — the band that dominates assistive/navigation decisions ("is it
    within 2 m?") — is under-penalized. Each valid pixel here gets a smooth up-weight
    ``w = 1 + gain·sigmoid((band_m − d)/soft_m)`` that saturates to ``1+gain`` well inside
    ``band_m`` and decays to 1 well beyond it (d = exp(target_log), detached). This is a proper
    weighted-variance SILog (mu, mu2 both use the SAME weights), so it stays scale-invariant and
    metric at ``lam≈0.85``. Use as an ADDITIVE term next to the global SILog anchor.
    """
    with torch.no_grad():
        d = torch.exp(target_log.float().clamp(max=4.0))
        prox = torch.sigmoid((band_m - d) / max(1e-3, soft_m))
        w = mask.float() * (1.0 + gain * prox)
    base = (pred_log - target_log)
    sw = w.flatten(1).sum(1).clamp(min=1.0)
    mu = (w * base).flatten(1).sum(1) / sw
    mu2 = (w * base * base).flatten(1).sum(1) / sw
    var_term = (mu2 - lam * mu * mu).clamp(min=0.0)
    return torch.sqrt(var_term + eps).mean()


def anchor_consistency_loss(
    pred_log: torch.Tensor,
    boxes: torch.Tensor,
    sizes_m: torch.Tensor,
    counts: torch.Tensor,
    intrinsics: torch.Tensor,
    inset: float = 0.20,
    log_min: float = -1.61,
    log_max: float = 3.41,
) -> torch.Tensor:
    """Known-size object anchor AS a training signal (A1-as-loss, Phase-4 novelty).

    For each detected object whose class has a tight physical-height prior (``sizes_m>0``), the
    pinhole model yields a LABEL-FREE metric distance from geometry alone:

        d_geo = fy_norm · H_real / h_norm      (h_norm = normalized box height; fy_norm = fy/H)

    We sample the student's predicted per-object log-distance (soft-median of the paper's 5
    points, exactly as ``object_distance_loss``) and pull it toward ``log(d_geo)`` in L1. Unlike
    ``object_distance_loss`` (which distills the TEACHER's cached distance), this supervises
    absolute metric SCALE from real-world size priors — the differentiable form of the
    training-free size anchor (prototypes/anchor_lock.py A1). Objects with ``sizes_m==0``
    (unknown class) are ignored, so the term self-gates and never fires on un-annotated boxes.

    boxes: (B,K,4) normalized xyxy on the prediction grid; sizes_m: (B,K) metres (0 = skip);
    counts: (B,) valid boxes/sample; intrinsics: (B,4) normalized (fx/W, fy/H, cx/W, cy/H). fp32.
    """
    B, K, _ = boxes.shape
    pred = pred_log.float()
    fy = intrinsics[:, 1].float().clamp(min=1e-3)      # normalized fy = fy_px / H
    total = pred.new_zeros(())
    n = 0
    for b in range(B):
        c = int(counts[b])
        if c == 0:
            continue
        bx = boxes[b, :c].float()
        sz = sizes_m[b, :c].float()
        h_norm = (bx[:, 3] - bx[:, 1]).clamp(min=1e-3)
        valid = (sz > 0) & (h_norm > 0.02)
        if not bool(valid.any()):
            continue
        d_geo = (fy[b] * sz / h_norm).clamp(min=1e-3)
        w_ = bx[:, 2] - bx[:, 0]
        h_ = bx[:, 3] - bx[:, 1]
        cx = (bx[:, 0] + bx[:, 2]) * 0.5
        cy = (bx[:, 1] + bx[:, 3]) * 0.5
        ix, iy = inset * w_, inset * h_
        xs = torch.stack([cx, bx[:, 0] + ix, bx[:, 2] - ix, bx[:, 0] + ix, bx[:, 2] - ix], dim=1)
        ys = torch.stack([cy, bx[:, 1] + iy, bx[:, 1] + iy, bx[:, 3] - iy, bx[:, 3] - iy], dim=1)
        grid = torch.stack([xs, ys], dim=-1) * 2.0 - 1.0
        samp = F.grid_sample(pred[b:b + 1], grid.view(1, c, 5, 2),
                             mode="bilinear", align_corners=False)
        vals = samp[0, 0].clamp(log_min, log_max)
        sorted_v, _ = torch.sort(vals, dim=1)
        soft_med = sorted_v[:, 1:4].mean(dim=1)
        diff = (soft_med - torch.log(d_geo)).abs()[valid]
        total = total + diff.sum()
        n += int(valid.sum())
    return total / max(1, n)


def scale_bias_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
    is_gt: torch.Tensor,
    rel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-image SCALE supervision on real-GT rows only (S19 'scale' tag).

    Diagnosis (benchmark/scale_gap.py, 2026-07-20): the dominant zero-shot failure of
    students is a PER-IMAGE log-scale offset (IBims: d1 .71 raw -> .90 with one scalar
    per image; the offset is scene-driven, K-uncorrelated, std ~0.18). SILog with
    lambda=0.85 penalizes exactly this per-image mean at only (1-lambda)=0.15 relative
    weight — the objective structurally under-trains the one latent that dominates OOD
    transfer. This term makes it first-class: the squared per-image weighted-mean
    log-error, on rows whose target is REAL sensor GT (is_gt=1). Pseudo-label rows are
    excluded so the teacher's own metric drift (+2.1..2.6% family-wide) is never
    distilled into the scale latent.
    """
    w = mask.float()
    denom = w.flatten(1).sum(1).clamp(min=1.0)
    mu = ((pred_log - target_log) * w).flatten(1).sum(1) / denom     # per-image bias
    if is_gt is None:                      # 'scale2': every row, target = GT or teacher
        if rel_weight is not None:         # 'rel': down-weight teacher-scale-unreliable images
            rw = rel_weight.float().clamp(min=0.0)
            return (mu * mu * rw).sum() / rw.sum().clamp(min=1e-6)
        return (mu * mu).mean()
    g = is_gt.float().clamp(0, 1)
    if float(g.sum()) < 1:
        return pred_log.new_zeros(())
    return (mu * mu * g).sum() / g.sum()


def cam_eq_loss(
    pred_log_crop: torch.Tensor,
    pred_log_full_cropped: torch.Tensor,
    border: int = 8,
    beta: float = 0.1,
) -> torch.Tensor:
    """SCED camera-equivariance consistency (S19 'sced' tag).

    A crop-and-resize is a VIRTUAL CAMERA: intrinsics change (f' = f/s, principal
    point shifts) while the metric depth of every corresponding pixel is unchanged.
    A camera-blind network cannot satisfy this (zoomed-in content looks nearer), so
    pulling the crop-view prediction toward the (detached) full-view prediction at
    corresponding pixels forces the FiLM/ray conditioning pathway to become causal
    and regularizes the per-image scale latent across cameras. Contrast N5/ICD
    (S15, null result): horizontal flips leave K unchanged, so the scale law was
    never exercised — the transformation class is what matters.

    Both inputs are (B,1,H,W) log-depth on the SAME grid (full-view prediction
    already cropped+resized by the caller, detached). A border margin absorbs
    resampling edge effects. Huber in log space.
    """
    if border > 0:
        pred_log_crop = pred_log_crop[..., border:-border, border:-border]
        pred_log_full_cropped = pred_log_full_cropped[..., border:-border, border:-border]
    return F.smooth_l1_loss(pred_log_crop, pred_log_full_cropped, beta=beta)


def icd_consistency_loss(
    pred_log: torch.Tensor,
    pred_log_ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """N5 — Invariance-Consistency Distillation.

    UniDepthV2's training contribution is a *geometric-invariance* property: two augmented views of
    the same scene must agree after camera-conditioning. We distill that PROPERTY (not just depth
    values) into the student: ``pred_log`` is the student's log-depth on the original view and
    ``pred_log_ref`` is the student's prediction on a geometric transform (here a horizontal flip)
    mapped back to the original frame — computed detached (mean-teacher style, memory-cheap). The
    student is pulled toward flip-consistency, transferring V2's cross-view robustness to a small net.
    """
    w = mask.float()
    return ((pred_log - pred_log_ref).abs() * w).sum() / w.sum().clamp(min=1.0)


class DistillLoss(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["distill"]["loss"]
        self.silog_w = float(lc.get("silog_weight", 1.0))
        self.silog_lambda = float(lc.get("silog_lambda", 0.85))
        self.grad_w = float(lc.get("grad_weight", 0.5))
        self.grad_scales = int(lc.get("grad_scales", 4))
        self.ssi_w = float(lc.get("ssi_weight", 0.0))
        self.eg_ssi_w = float(lc.get("eg_ssi_weight", 0.0))          # S4-edge
        self.nll_w = float(lc.get("nll_weight", 0.0))                # S6-unc
        self.nll_beta = float(lc.get("nll_beta", 0.5))               # beta-NLL (0 = plain NLL)
        self.obj_dist_w = float(lc.get("obj_dist_weight", 0.0))      # S7-task
        self.bins_w = float(lc.get("bins_weight", 0.0))              # hybrid bin head CE
        self.norm_w = float(lc.get("normal_weight", 0.0))            # normal consistency
        self.prox_silog_w = float(lc.get("prox_silog_weight", 0.0))  # proximity-weighted SILog (safety band)
        self.prox_band_m = float(lc.get("prox_band_m", 2.0))
        self.prox_gain = float(lc.get("prox_gain", 2.0))
        self.anchor_w = float(lc.get("anchor_weight", 0.0))          # known-size anchor-consistency (A1-as-loss)
        self.scale_bias_w = float(lc.get("scale_bias_weight", 0.0))  # S19: per-image scale on GT rows
        # S21 'scale2': per-image scale toward the row target on ALL rows (own rows =
        # the TEACHER's per-image scale). Rationale (view_var + transplant probes,
        # 2026-07-20): the scale error is a view-stable per-scene commitment; the
        # teacher's scale is 2x more stable OOD (b-std .10 vs .18), and transplanting
        # V2-L's per-image scale onto S16's structure scores IBims d1 .8425 > official
        # V2-S .8253 — the ceiling this loss chases at training time.
        self.scale_bias_all = bool(lc.get("scale_bias_all", False))
        # P-series 'rel': weight the scale loss by teacher scale-reliability, derived from the
        # N-view consensus spread sigma (build_consensus_scale.py): w_i = 1/(1+(sigma_i/tau)^2).
        self.scale_rel = bool(lc.get("scale_rel", False))
        self.scale_rel_tau = float(lc.get("scale_rel_tau", 0.05))
        self.use_conf = bool(cfg["distill"].get("use_teacher_confidence", True))
        self.nll_active = True   # train.py toggles False for warmup epochs
        s = cfg["student"]
        import math
        self._log_min = math.log(float(s.get("depth_min_m", 0.2)))
        self._log_max = math.log(float(s.get("depth_max_m", 30.0)))

    def forward(self, pred_log, target_log, mask, conf=None, weight=None,
                pred_logsigma=None, boxes=None, box_dists=None, box_counts=None,
                bin_logits=None, bin_centres=None, intrinsics=None, box_sizes=None,
                is_gt=None, scale_sigma=None):
        """weight: precomposed per-pixel weight from the dataset (conf-rank x task x agreement).
        When given, it REPLACES the legacy conf path (pass it via mask multiplication)."""
        terms = {}
        loss = pred_log.new_zeros(())
        if weight is not None:
            eff_mask = mask.float() * weight        # silog/grad treat mask as float weights
            conf_arg, use_conf = None, False
        else:
            eff_mask, conf_arg, use_conf = mask, conf, self.use_conf

        if self.silog_w > 0:
            s = silog_loss(pred_log, target_log, eff_mask, self.silog_lambda, conf_arg, use_conf)
            terms["silog"] = s.detach()
            loss = loss + self.silog_w * s
        if self.prox_silog_w > 0:
            ps = proximity_weighted_silog(pred_log, target_log, eff_mask, self.silog_lambda,
                                          band_m=self.prox_band_m, gain=self.prox_gain)
            terms["prox_silog"] = ps.detach()
            loss = loss + self.prox_silog_w * ps
        if self.scale_bias_w > 0 and (is_gt is not None or self.scale_bias_all):
            rw = None
            if self.scale_rel and self.scale_bias_all and scale_sigma is not None:
                s = scale_sigma.float()
                rw = 1.0 / (1.0 + (s / self.scale_rel_tau) ** 2)
                # GT rows have sigma=0 -> weight 1 (their scale target is exact by definition)
            sb = scale_bias_loss(pred_log, target_log, mask,
                                 None if self.scale_bias_all else is_gt, rel_weight=rw)
            terms["scale_bias"] = sb.detach()
            loss = loss + self.scale_bias_w * sb
        if self.grad_w > 0:
            g = gradient_matching_loss(pred_log, target_log, eff_mask, self.grad_scales, conf_arg, use_conf)
            terms["grad"] = g.detach()
            loss = loss + self.grad_w * g
        # Auxiliary terms receive the SAME composite per-pixel weight as the metric anchor.
        # Previously they used the raw `mask`, so a recipe's own weighting (e.g. S6's rank
        # confidence) never reached the term it targeted — a half-wired ablation (audit finding).
        # Structure (ssi/eg_ssi) and uncertainty (nll) are per-pixel, so eff_mask applies cleanly.
        # bins-CE and the box-level object term keep `mask` (distinct semantics: soft-CE / per-box).
        if self.ssi_w > 0:
            si = ssi_loss(pred_log, target_log, eff_mask)
            terms["ssi"] = si.detach()
            loss = loss + self.ssi_w * si
        if self.eg_ssi_w > 0:
            e = eg_ssi_loss(pred_log, target_log, eff_mask)
            terms["eg_ssi"] = e.detach()
            loss = loss + self.eg_ssi_w * e
        if self.nll_w > 0 and pred_logsigma is not None and self.nll_active:
            n = laplace_nll_loss(pred_log, pred_logsigma, target_log, eff_mask, beta=self.nll_beta)
            terms["nll"] = n.detach()
            loss = loss + self.nll_w * n
        if self.obj_dist_w > 0 and boxes is not None and box_counts is not None and int(box_counts.sum()) > 0:
            o = object_distance_loss(pred_log, boxes, box_dists, box_counts,
                                     log_min=self._log_min, log_max=self._log_max)
            terms["obj"] = o.detach()
            loss = loss + self.obj_dist_w * o
        if (self.anchor_w > 0 and boxes is not None and box_sizes is not None
                and box_counts is not None and intrinsics is not None
                and int(box_counts.sum()) > 0 and float(box_sizes.sum()) > 0):
            ac = anchor_consistency_loss(pred_log, boxes, box_sizes, box_counts, intrinsics,
                                         log_min=self._log_min, log_max=self._log_max)
            terms["anchor"] = ac.detach()
            loss = loss + self.anchor_w * ac
        if self.bins_w > 0 and bin_logits is not None and bin_centres is not None:
            b = bins_ce_loss(bin_logits, target_log, mask, bin_centres)
            terms["bins"] = b.detach()
            loss = loss + self.bins_w * b
        if self.norm_w > 0 and intrinsics is not None:
            nm = normal_consistency_loss(pred_log, target_log, mask, intrinsics)
            terms["norm"] = nm.detach()
            loss = loss + self.norm_w * nm
        terms["total"] = loss.detach()
        return loss, terms
