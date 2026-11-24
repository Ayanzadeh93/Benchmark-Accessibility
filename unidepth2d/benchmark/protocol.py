"""Paper-grade evaluation protocol, mirroring UniDepthV2's official implementation.

Depth metrics reproduce `UniDepth-main/unidepth/utils/evaluation_depth.py` exactly
(per-image, GT-side masking only, predictions NEVER clipped, unweighted mean over images
— their `get_evaluation` uses `.mean()` across accumulated per-image values).
3D F-score thresholds reproduce `base_dataset.accumulate_metrics_3d`:
exp(linspace(log(min_depth), log(max_depth/20), 100)).

Additions beyond the official file (each documented in docs/PROTOCOL_NOTES.md):
  * AUSE / nAUSE w.r.t. delta1 + Spearman rho (paper Tab. VII semantics).
  * rho_A: AUC of mean camera-ray angular error up to 15 deg (paper Sec. IV-A).
  * Scale-invariant boundary F1 (DepthPro-style ratio boundaries; dense-GT sets only).

Everything is float64-safe numpy on top of float32 tensors; no GPU needed.
Self-test: `python -m benchmark.protocol` cross-checks shared metrics against the official
torch implementation on random tensors (requires torch; 3D part skipped if the official
chamfer import fails).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

try:  # scipy is available in both project envs; guard anyway
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# numpy 2.x renamed trapz -> trapezoid; official torch code uses torch.trapz (same rule)
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ──────────────────────────────────────────────────────────────────────────────
# Alignment helpers (official `ssi` / `si`)
# ──────────────────────────────────────────────────────────────────────────────

def ssi_align(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Least-squares scale+shift of PRED onto GT in depth space (official `ssi`).

    Solves min_{s,t} ||s*pred + t - gt||^2 with a 1e-9 Tikhonov diagonal, matching
    the official closed form. Inputs are 1-D valid-pixel vectors.
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)          # (N,2)
    ata = A.T @ A + 1e-9 * np.eye(2)
    st = np.linalg.inv(ata) @ (A.T @ g[:, None])
    s, t = float(st[0, 0]), float(st[1, 0])
    return (p * s + t).astype(np.float64)


def _median_t(x: np.ndarray) -> float:
    """torch.median semantics: lower of the two middle elements (no averaging)."""
    xs = np.sort(x.ravel())
    return float(xs[(xs.size - 1) // 2])


def si_align(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Median-ratio scaling of PRED onto GT (official `si`, torch-median semantics)."""
    med_p = _median_t(pred)
    if med_p == 0:
        return pred.astype(np.float64)
    return pred.astype(np.float64) * (_median_t(gt) / med_p)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image depth metrics (official DICT_METRICS + aligned variants)
# ──────────────────────────────────────────────────────────────────────────────

def _delta(gt, pred, exponent):
    inlier = np.maximum(gt / pred, pred / gt)
    return float((inlier < 1.25 ** exponent).mean())


def _tau(gt, pred, perc=0.03):
    inlier = np.maximum(gt / pred, pred / gt)
    return float((inlier < 1.0 + perc).mean())


def _d_auc(gt, pred):
    exponents = np.linspace(0.01, 5.0, 100)
    deltas = [_delta(gt, pred, e) for e in exponents]
    return float(_trapz(np.asarray(deltas), exponents) / 5.0)


def per_image_depth_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    max_depth: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """All official per-image depth metrics on one image.

    gt/pred: (H,W) metres. valid: (H,W) bool GT validity (sensor mask, crops, min-depth).
    max_depth masks GT only (mirrors official `mask & (gt <= max_depth)`).
    Predictions are NOT clipped. Returns None if no valid pixels.
    """
    m = valid.astype(bool)
    if max_depth is not None:
        m = m & (gt <= max_depth)
    if m.sum() == 0:
        return None
    g = gt[m].astype(np.float64)
    p = pred[m].astype(np.float64)
    # guard: official code implicitly assumes pred > 0 (log/ratios); record clamp count
    n_nonpos = int((p <= 0).sum())
    p = np.clip(p, 1e-6, None)

    out = {
        "d1": _delta(g, p, 1.0),
        "d2": _delta(g, p, 2.0),
        "d3": _delta(g, p, 3.0),
        "rmse": float(np.sqrt(((g - p) ** 2).mean())),
        "mae": float(np.abs(g - p).mean()),
        "rmselog": float(np.sqrt(((np.log(g) - np.log(p)) ** 2).mean())),
        "arel": float((np.abs(g - p) / g).mean()),
        "sqrel": float((((g - p) ** 2) / g).mean()),
        "log10": float(np.abs(np.log10(p) - np.log10(g)).mean()),
        # torch.std default = Bessel-corrected (ddof=1); torch.median = lower-middle
        "silog": float(100.0 * np.std(np.log(p) - np.log(g), ddof=1)),
        "medianlog": float(100.0 * abs(_median_t(np.log(p) - np.log(g)))),
        "d_auc": _d_auc(g, p),
        "tau": _tau(g, p, 0.03),
        "n_nonpos_pred": float(n_nonpos),
    }
    # official loop adds ssi/si-aligned variants for tau, d1, arel
    for name, fn in (("tau", _tau), ("d1", lambda a, b: _delta(a, b, 1.0)),
                     ("arel", lambda a, b: float((np.abs(a - b) / a).mean()))):
        for align_name, align in (("ssi", ssi_align), ("si", si_align)):
            pa = np.clip(align(g, p), 1e-6, None)
            out[f"{name}_{align_name}"] = float(fn(g, pa))
    return out


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Unweighted mean over images (official `get_evaluation` semantics).

    nan-mean: per-image NaNs (e.g., nAUSE on an image whose oracle equals the
    random curve) are excluded rather than poisoning the dataset mean.
    """
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows if k in r], dtype=np.float64)
        with np.errstate(all="ignore"):
            out[k] = round(float(np.nanmean(vals)), 5) if np.isfinite(vals).any() else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3D metrics: chamfer + F1-AUC (F_A)
# ──────────────────────────────────────────────────────────────────────────────

def fa_thresholds(min_depth: float, max_depth: float, steps: int = 100) -> np.ndarray:
    """Official F-score thresholds: log-spaced min_depth .. max_depth/20."""
    return np.exp(np.linspace(math.log(min_depth), math.log(max_depth / 20.0), steps))


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Depth (H,W) + pinhole K (3x3) -> points (3, H*W). Pixel centers at +0.5."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5,
                       np.arange(h, dtype=np.float64) + 0.5)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.astype(np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)


def _downsample_for_3d(arr: np.ndarray, mask: np.ndarray):
    """Official OOM guard: rescale so valid count ~<= 240*320, nearest-exact."""
    ratio = min(1.0, (240 * 320 / max(1, int(mask.sum()))) ** 0.5)
    if ratio >= 1.0:
        return arr, mask
    h, w = mask.shape
    hn, wn = int(h * ratio), int(w * ratio)
    yi = (np.linspace(0, h - 1, hn)).round().astype(int)
    xi = (np.linspace(0, w - 1, wn)).round().astype(int)
    return arr[..., yi[:, None], xi[None, :]], mask[yi[:, None], xi[None, :]]


def pointcloud_f1_chamfer(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    thresholds: np.ndarray,
    squared_compat: bool = True,
) -> Dict[str, float]:
    """Bidirectional NN distances (exact, cKDTree) -> chamfer + F1-AUC.

    gt_points/pred_points: (3, N) / (3, M). Mirrors official `f1_score`:
    precision from pred->gt distances, recall from gt->pred, F1 per threshold,
    AUC = trapz(F1)/len(thresholds)  (unit-spaced trapz over the threshold INDEX,
    exactly as the official code does).

    squared_compat (BUG-COMPAT, default True): the official knn_points returns
    SQUARED distances (pytorch3d convention) and `f1_score`/`auc` compare them
    directly against metre-valued thresholds — the effective threshold is sqrt(t).
    We replicate that to match published F_A. chamfer always uses true distances
    (their chamfer_dist applies sqrt explicitly).
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy required for point-cloud metrics")
    gp = np.ascontiguousarray(gt_points.T)
    pp = np.ascontiguousarray(pred_points.T)
    d_pred2gt, _ = cKDTree(gp).query(pp, k=1, workers=-1)   # dist1: pred -> gt
    d_gt2pred, _ = cKDTree(pp).query(gp, k=1, workers=-1)   # dist2: gt -> pred
    dp = d_pred2gt ** 2 if squared_compat else d_pred2gt
    dg = d_gt2pred ** 2 if squared_compat else d_gt2pred
    precisions = np.array([(dp < t).mean() for t in thresholds])
    recalls = np.array([(dg < t).mean() for t in thresholds])
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * precisions * recalls / (precisions + recalls)
    f1 = np.nan_to_num(f1, nan=0.0)
    return {
        "F1": float(_trapz(f1) / len(thresholds)),
        "chamfer": float((d_pred2gt.mean() + d_gt2pred.mean()) / 2.0),
    }


def eval_3d_image(
    gt_depth: np.ndarray,
    pred_points: np.ndarray,       # (3,H,W) — model's own points (pred camera x depth)
    valid: np.ndarray,
    gt_K: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> Optional[Dict[str, float]]:
    """Official eval_3d for one image: GT points from GT camera, pred points as given.

    NOTE: mirroring the official pipeline, the 3D mask does NOT apply the max_depth
    cut (their `prepare_points_eval` passes the raw depth_mask); max_depth only
    enters via the F-score threshold range.
    """
    m = valid & (gt_depth > 0)
    if m.sum() == 0:
        return None
    gt_pts_img = backproject(gt_depth, gt_K).reshape(3, *gt_depth.shape)
    stacked = np.concatenate([gt_pts_img, pred_points], axis=0)  # (6,H,W)
    stacked_ds, m_ds = _downsample_for_3d(stacked, m)
    if m_ds.sum() == 0:
        return None
    gt_pts = stacked_ds[:3][:, m_ds]
    pr_pts = stacked_ds[3:][:, m_ds]
    return pointcloud_f1_chamfer(gt_pts, pr_pts, fa_thresholds(min_depth, max_depth))


# ──────────────────────────────────────────────────────────────────────────────
# Camera rho_A: AUC of mean ray angular error up to 15 degrees
# ──────────────────────────────────────────────────────────────────────────────

def rays_from_K(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """Unit rays (3,H,W) for pinhole K at pixel centers."""
    pts = backproject(np.ones((h, w)), K).reshape(3, h, w)
    return pts / np.linalg.norm(pts, axis=0, keepdims=True).clip(1e-9)


def mean_ray_angular_error_deg(pred_rays: np.ndarray, gt_rays: np.ndarray) -> float:
    """Mean per-pixel angle (deg) between two unit-ray fields (3,H,W)."""
    cos = np.clip((pred_rays * gt_rays).sum(axis=0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def rho_a(per_image_mean_err_deg: Sequence[float], max_deg: float = 15.0,
          steps: int = 100) -> float:
    """Dataset-level AUC of recall(threshold) for per-image mean angular error.

    recall(t) = fraction of images with mean ray error < t, t in (0, max_deg];
    AUC normalized to [0,1] (x100 to compare with the paper's rho_A).
    Interpretation documented in PROTOCOL_NOTES; validated against published rows
    at the reproduction gate.
    """
    errs = np.asarray(per_image_mean_err_deg, dtype=np.float64)
    ts = np.linspace(max_deg / steps, max_deg, steps)
    recall = np.array([(errs < t).mean() for t in ts])
    return float(_trapz(recall, ts) / max_deg)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty: AUSE / nAUSE w.r.t. delta1 + Spearman rho (paper Tab. VII)
# ──────────────────────────────────────────────────────────────────────────────

def ause_delta1(
    gt: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    valid: np.ndarray,
    max_depth: Optional[float] = None,
    fractions: Optional[np.ndarray] = None,
    max_pixels: int = 200_000,
) -> Optional[Dict[str, float]]:
    """Sparsification analysis on one image.

    Rank pixels by predicted uncertainty (higher = more uncertain), progressively
    remove the most-uncertain fraction f, recompute delta1 on the rest; oracle ranks
    by true per-pixel error max(gt/pred, pred/gt). AUSE = integral of
    (delta1_oracle - delta1_uncertainty) over f; nAUSE = AUSE / AUSE_random where the
    random curve is flat at the base delta1 (0 = oracle, 1 = random, per the paper).
    Also returns Spearman rho between uncertainty and true error.
    """
    m = valid.astype(bool)
    if max_depth is not None:
        m = m & (gt <= max_depth)
    if m.sum() < 100:
        return None
    g = gt[m].astype(np.float64)
    p = np.clip(pred[m].astype(np.float64), 1e-6, None)
    u = uncertainty[m].astype(np.float64)
    if g.size > max_pixels:  # subsample deterministically for tractability
        idx = np.linspace(0, g.size - 1, max_pixels).astype(int)
        g, p, u = g[idx], p[idx], u[idx]
    err = np.maximum(g / p, p / g)
    if fractions is None:
        fractions = np.linspace(0.0, 0.99, 34)

    order_u = np.argsort(-u)     # most uncertain first
    order_e = np.argsort(-err)   # largest true error first
    n = g.size
    inlier = (err < 1.25).astype(np.float64)

    def curve(order):
        inlier_sorted = inlier[order]
        # removing top-k of `order` keeps the suffix; suffix means via reverse cumsum
        csum = np.cumsum(inlier_sorted[::-1])[::-1]  # csum[k] = sum(inlier_sorted[k:])
        vals = []
        for f in fractions:
            k = min(n - 1, int(round(f * n)))
            vals.append(csum[k] / (n - k))
        return np.asarray(vals)

    c_unc = curve(order_u)
    c_or = curve(order_e)
    base_d1 = float(inlier.mean())
    ause = float(_trapz(c_or - c_unc, fractions))
    ause_rand = float(_trapz(c_or - base_d1, fractions))
    rho = float(spearmanr(u, err).statistic) if _HAVE_SCIPY else float("nan")
    return {
        "ause": ause,
        "nause": ause / ause_rand if ause_rand > 1e-9 else float("nan"),
        "spearman": rho,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scale-invariant boundary F1 (DepthPro-style ratio boundaries; dense GT only)
# ──────────────────────────────────────────────────────────────────────────────

def boundary_f1_si(
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    t_list: Sequence[float] = (5, 10, 15, 20, 25),
) -> Optional[float]:
    """Occluding-boundary F1 averaged over ratio thresholds t (%).

    A directed neighbor pair (near, far) is a boundary at threshold t iff
    d_far / d_near > 1 + t/100. Precision/recall over all valid neighbor pairs
    (horizontal + vertical, both directions), F1 per t, mean over t_list.
    Scale-invariant by construction (ratios). Our implementation of DepthPro's
    protocol — flagged as such in PROTOCOL_NOTES; use on dense-GT sets only
    (NYU filled, IBims-1, Sintel).
    """
    v = valid.astype(bool)

    def pairs(d):
        return (
            (d[:, 1:], d[:, :-1]),   # right vs left
            (d[:, :-1], d[:, 1:]),
            (d[1:, :], d[:-1, :]),   # down vs up
            (d[:-1, :], d[1:, :]),
        )

    def masks():
        return (
            v[:, 1:] & v[:, :-1],
            v[:, :-1] & v[:, 1:],
            v[1:, :] & v[:-1, :],
            v[:-1, :] & v[1:, :],
        )

    f1s = []
    for t in t_list:
        thr = 1.0 + t / 100.0
        tp = fp = fn = 0
        for (ga, gb), (pa, pb), m in zip(pairs(gt), pairs(pred), masks()):
            with np.errstate(divide="ignore", invalid="ignore"):
                g_ratio = np.where(gb > 0, ga / np.where(gb <= 0, 1.0, gb), 0.0)
                p_ratio = np.where(pb > 0, pa / np.where(pb <= 0, 1.0, pb), 0.0)
            g_edge = (g_ratio > thr) & m
            p_edge = (p_ratio > thr) & m
            tp += int((g_edge & p_edge).sum())
            fp += int((~g_edge & p_edge).sum())
            fn += int((g_edge & ~p_edge).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec > 0:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else None


# ──────────────────────────────────────────────────────────────────────────────
# Self-test vs the official torch implementation
# ──────────────────────────────────────────────────────────────────────────────

def _self_test():  # pragma: no cover
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "UniDepth-main"))
    import torch

    rng = np.random.default_rng(0)
    h, w = 60, 80
    gt = rng.uniform(0.4, 9.5, (h, w)).astype(np.float32)
    pred = (gt * rng.uniform(0.8, 1.25, (h, w))).astype(np.float32)
    valid = rng.uniform(size=(h, w)) > 0.15

    ours = per_image_depth_metrics(gt, pred, valid, max_depth=8.0)

    from unidepth.utils.evaluation_depth import eval_depth  # official
    t_gt = torch.from_numpy(gt)[None, None]
    t_pr = torch.from_numpy(pred)[None, None]
    t_ma = torch.from_numpy(valid)[None, None]
    theirs = {k: float(v.mean()) for k, v in
              eval_depth(t_gt, t_pr, t_ma, max_depth=8.0).items()}

    keymap = ["d1", "d2", "d3", "rmse", "rmselog", "arel", "sqrel", "log10",
              "silog", "medianlog", "d_auc", "tau",
              "tau_ssi", "tau_si", "d1_ssi", "d1_si", "arel_ssi", "arel_si"]
    max_diff = 0.0
    for k in keymap:
        if k not in theirs:
            print(f"  [skip] official missing {k}")
            continue
        d = abs(ours[k] - theirs[k])
        max_diff = max(max_diff, d)
        flag = "OK " if d < 5e-4 else "FAIL"
        print(f"  [{flag}] {k:12s} ours={ours[k]:.6f} official={theirs[k]:.6f} diff={d:.2e}")
    print(f"max diff: {max_diff:.3e}")
    assert max_diff < 5e-4, "protocol mismatch vs official implementation"

    # 3D: compare our cKDTree chamfer/F1 against official (needs their chamfer import)
    try:
        from unidepth.utils.evaluation_depth import eval_3d
        K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
        pred_pts = backproject(pred.astype(np.float64), K).reshape(3, h, w)
        ours3d = eval_3d_image(gt, pred_pts, valid, K, 0.005, 8.0)
        gt_pts_t = torch.from_numpy(backproject(gt.astype(np.float64), K).reshape(3, h, w))[None].float()
        pr_pts_t = torch.from_numpy(pred_pts)[None].float()
        thr = torch.from_numpy(fa_thresholds(0.005, 8.0)).float()
        theirs3d = {k: float(v.mean()) for k, v in
                    eval_3d(gt_pts_t, pr_pts_t, torch.from_numpy(valid)[None, None],
                            thresholds=thr).items()}
        for k in ("F1", "chamfer"):
            d = abs(ours3d[k] - theirs3d[k])
            flag = "OK " if d < 2e-3 else "FAIL"
            print(f"  [{flag}] 3D {k:8s} ours={ours3d[k]:.6f} official={theirs3d[k]:.6f} diff={d:.2e}")
    except Exception as e:
        print(f"  [skip] official 3D comparison unavailable: {type(e).__name__}: {e}")

    print("self-test done")


if __name__ == "__main__":
    _self_test()
