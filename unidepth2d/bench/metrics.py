"""UniDepthV2's EXACT depth metrics, ported from the official
`unidepth/utils/evaluation_depth.py` (read this session), with the chamfer path swapped for our
validated pure-PyTorch `bench.chamfer_torch`. Semantics are preserved 1:1:

  * `delta(exp)` = mean( max(gt/pred,pred/gt) < 1.25**exp ).
  * `arel` (raw) = mean(|gt-pred|/gt); the paper's headline δ1/AbsRel/τ are ALSO reported after
    `ssi` (least-squares scale+shift align of pred to gt) and `si` (median-ratio align).
  * `silog` = 100 * std(log pred - log gt).
  * `d_auc` = trapz over exponents in [0.01,5] of delta, /5.
  * 2D metrics operate on the masked 1-D vectors, exactly like `eval_depth`.
  * `f1_3d` (F_A) = AUC over chamfer thresholds on the back-projected point clouds.

`eval_depth_2d(gt, pred, mask, max_depth)` mirrors the official `eval_depth` loop and returns a
dict of scalars (already reduced over the masked pixels) for ONE sample; the caller averages over
the dataset. No scale/median alignment is applied to the RAW metrics — alignment only feeds the
explicit `_ssi`/`_si` variants, faithful to the paper.
"""

from __future__ import annotations

import torch

from bench.chamfer_torch import f1_score as _f1_cloud


# ── rescalings (official) ────────────────────────────────────────────────────────
def ssi(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Align pred to gt by least-squares scale+shift (scale-shift-invariant)."""
    stab = 1e-9 * torch.eye(2, device=gt.device, dtype=pred.dtype)
    pred1 = torch.stack([pred.detach(), torch.ones_like(pred).detach()], dim=1)  # (N,2)
    scale_shift = torch.inverse(pred1.T @ pred1 + stab) @ (pred1.T @ gt.unsqueeze(1))
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return pred * scale + shift


def si(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Align pred to gt by median ratio (scale-invariant)."""
    return pred * torch.median(gt) / torch.median(pred)


# ── scalar metrics on masked 1-D vectors ─────────────────────────────────────────
def delta(gt, pred, exponent):
    inlier = torch.maximum(gt / pred, pred / gt)
    return (inlier < 1.25 ** exponent).float().mean()


def tau(gt, pred, perc=0.03):
    inlier = torch.maximum(gt / pred, pred / gt)
    return (inlier < (1.0 + perc)).float().mean()


def arel(gt, pred):
    return (torch.abs(gt - pred) / gt).mean()


def silog(gt, pred):
    return 100 * torch.std(torch.log(pred) - torch.log(gt))


def d_auc(gt, pred):
    # Vectorized: compute the inlier ratio ONCE, compare against all 100 thresholds at once
    # (the loop version recomputed max(gt/pred,pred/gt) 100x per image — the eval hotspot).
    exps = torch.linspace(0.01, 5.0, steps=100, device=gt.device)
    inlier = torch.maximum(gt / pred, pred / gt)                     # (N,)
    thr = 1.25 ** exps                                               # (100,)
    ds = (inlier[None, :] < thr[:, None]).float().mean(dim=1)        # (100,)
    return torch.trapz(ds, exps) / 5.0


_SCALARS = {
    "d1": lambda g, p: delta(g, p, 1.0),
    "d2": lambda g, p: delta(g, p, 2.0),
    "d3": lambda g, p: delta(g, p, 3.0),
    "rmse": lambda g, p: torch.sqrt(((g - p) ** 2).mean()),
    "rmselog": lambda g, p: torch.sqrt(((torch.log(g) - torch.log(p)) ** 2).mean()),
    "arel": arel,
    "sqrel": lambda g, p: (((g - p) ** 2) / g).mean(),
    "log10": lambda g, p: torch.abs(torch.log10(p) - torch.log10(g)).mean(),
    "silog": silog,
    "d_auc": d_auc,
    "tau": tau,
}
# Metrics the paper additionally reports under ssi / si rescaling.
_RESCALED = ("d1", "arel", "tau")


def eval_depth_2d(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, max_depth=None) -> dict:
    """gt/pred/mask: (H,W) tensors (pred already interpolated to gt size). Returns {name: scalar}."""
    if max_depth is not None:
        mask = mask & (gt <= max_depth)
    g, p = gt[mask], pred[mask]
    out = {}
    if g.numel() < 10:
        return out
    for name, fn in _SCALARS.items():
        out[name] = float(fn(g, p))
        if name in _RESCALED:
            out[f"{name}_ssi"] = float(fn(g, ssi(g, p)))
            out[f"{name}_si"] = float(fn(g, si(g, p)))
    return out


# ── 3D metric F_A (chamfer-AUC F1) ───────────────────────────────────────────────
def backproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """depth (H,W) metres, K (3,3) -> points (3,H,W) in camera space."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                          torch.arange(W, device=depth.device, dtype=depth.dtype), indexing="ij")
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return torch.stack([x, y, depth], dim=0)


# Official eval_3d downsamples the cloud before chamfer ("rescale to avoid OOM"). Chamfer is
# O(N^2), so we cap N to keep F_A fast. 20k points give a stable chamfer-F1 estimate and, crucially,
# F_A is computed with the SAME cap AND the SAME pixel subset for every model: the subset is drawn
# from the GT-valid mask with a fixed-seed generator, so it depends only on the image, not on the
# model or the run. The absolute F_A can differ slightly from the paper's ~76.8k-point / Pytorch3D
# value. (Set higher for a closer-to-paper absolute at O(N^2) cost.)
_F1_MAX_POINTS = 20000


def f1_3d(gt_pts: torch.Tensor, pred_pts: torch.Tensor, mask: torch.Tensor,
          max_depth: float, n_thresh: int = 20, max_points: int = _F1_MAX_POINTS,
          seed: int = 0) -> float:
    """F_A: AUC of chamfer-F1 over thresholds up to (max_depth/20)^2 (squared, matching the
    official squared-distance comparison). gt_pts/pred_pts: (3,H,W); mask: (H,W).

    When more than ``max_points`` pixels are valid, a subset is drawn with a CPU generator seeded
    by ``seed``: identical for every model scored on the same mask, and identical across runs."""
    m = mask.reshape(-1)
    idx = torch.nonzero(m, as_tuple=False).squeeze(1)
    if idx.numel() < 10:
        return float("nan")
    if idx.numel() > max_points:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        sel = torch.randperm(idx.numel(), generator=gen)[:max_points].to(idx.device)
        idx = idx[sel]
    g = gt_pts.reshape(3, -1)[:, idx].T.contiguous()      # (N,3)
    p = pred_pts.reshape(3, -1)[:, idx].T.contiguous()
    # thresholds are compared against SQUARED distances; paper uses F-score up to 1/20 max depth.
    tmax = (max_depth / 20.0) ** 2
    th = torch.linspace(tmax / n_thresh, tmax, n_thresh, device=g.device, dtype=g.dtype)
    return float(_f1_cloud(p, g, th))


def reduce_dataset(rows: list[dict]) -> dict:
    """Average a list of per-sample metric dicts (missing keys skipped per-metric)."""
    keys = set().union(*[set(r) for r in rows]) if rows else set()
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out
