"""V2 metric fixes and additions, defined NEXT TO the legacy definitions in benchmark/protocol.py.

Contract
--------
Nothing in benchmark/protocol.py changes. Every legacy number stays reproducible, and each
corrected metric here has a new name so a result file can never mix the two definitions.
All functions are numpy-only (CPU), deterministic, and take per-image arrays in metres:
gt/pred (H,W) float, valid (H,W) bool, exactly like protocol.per_image_depth_metrics.

What is fixed or added, and why (V2_PLAN sections 5.1, 6.1, 9.1; harness audit pitfalls):

1. ``boundary_f1_si_v2`` - corrected scale-invariant boundary F1.
   The legacy ``protocol.boundary_f1_si`` *skips* a threshold when the prediction has no
   edge (tp+fp == 0), when the GT has no edge (tp+fn == 0), or when precision+recall == 0
   (edges in the wrong place), and returns None when every threshold was skipped; run_bench
   then drops that image from the dataset mean. A blurry model is therefore scored only on
   the thresholds and images it happens to fire on, which biases it upward (audit: a correct
   edge that fires only at t=5 % scores 1.0 instead of 0.2).
   V2 rule: every threshold the legacy function would skip scores 0 (this matches DepthPro's
   reference ``boundary_f1``, where r+p == 0 gives 0). An image is "non-contributing" only
   when its GT has no boundary pair at ANY threshold (nothing to detect); such images are
   excluded and COUNTED by ``aggregate_boundary_f1`` (n_contributing, n_no_gt_boundary).
   Option ``both_empty="exclude"`` drops thresholds where both GT and prediction are empty
   instead of scoring them 0; it is NOT the V2 default and must be named in any table using it.

2. ``per_image_log_shift`` / ``dataset_scale_summary`` / ``d1_dataset_scale`` - plan 9.1
   "mean bias, b_std" and "delta1 after one per-dataset scale". The legacy tool
   (benchmark/scale_gap.py) estimates d1 after a global shift on a 2,000-pixel random
   subsample per image; here it is computed on every valid pixel. Sign convention is kept
   from scale_gap.py: b = mean(ln gt - ln pred), i.e. the log correction that would fix the
   image (b > 0: the model predicts too near). b_std uses ddof=0, as scale_gap.py does, so
   past b_std values (S16 .180, V2-S .174 on IBims) stay comparable. The per-dataset scale
   is fitted on the evaluated set itself, so it is an ALIGNED (oracle) column, never raw.

3. ``pair_rows_by_id`` - compare_paired.compare_rows pairs per-image rows by list ORDER and
   only checks the count, so two runs that dropped different images pair the wrong images
   silently. This joins on the image id and raises on duplicates, missing ids or mismatched
   id sets (unless intersection is requested explicitly, in which case the drop is reported).

4. ``paired_bootstrap_clustered`` - plan 5.1: D_SUN statistics use a paired bootstrap over
   near-duplicate CLUSTERS (connected components at pHash Hamming <= 8), not over images.
   Image-level resampling of near-duplicates understates the CI. With clusters=None this
   delegates to compare_paired.paired_bootstrap (identical numbers).

5. ``si_boundary_f1_depthpro`` - a reference re-implementation of DepthPro's published
   ``SI_boundary_F1`` (apple/ml-depth-pro, src/depth_pro/eval/boundary_metrics.py; structure
   checked against the upstream source on 2026-11-19): inverse depth (depth clipped at 1e-6),
   10 thresholds linspace(1.05, 1.25) weighted by t / sum(t), precision and recall averaged
   over the four directional edge maps (each with a max(count, 1) denominator) and F1 = 0
   when p + r == 0. The audit (reviewer 7) found the in-house boundary F1 reading 1.30-1.53x
   above published UniDepthV2 Table IV values. The in-house metric differs from DepthPro in
   three ways: 5 unweighted thresholds, counts POOLED over the four directions (DepthPro
   averages per direction, so a direction with no GT edge contributes 0 to recall), and the
   skipped-threshold rule fixed in item 1. This function exists so the size of each
   difference can be measured; it does NOT establish which definition UniDepthV2 used.
   DepthPro has no validity mask; ``valid=None`` reproduces it, a mask restricts every
   directional edge map to pairs whose two pixels are valid (an extension, named as such).

Tests: tests/test_protocol_v2.py (env: depth or depth_cmp; numpy/scipy only, CPU, offline).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BF_T_LIST = (5, 10, 15, 20, 25)
BF_KEY = "bf1_si_v2"            # per-image row key for the corrected boundary F1
LN_125 = math.log(1.25)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Corrected boundary F1
# ──────────────────────────────────────────────────────────────────────────────

def _pair_views(d: np.ndarray):
    """Directed 4-neighbour pairs (a, b) in the same order as protocol.boundary_f1_si."""
    return (
        (d[:, 1:], d[:, :-1]),   # right vs left
        (d[:, :-1], d[:, 1:]),
        (d[1:, :], d[:-1, :]),   # down vs up
        (d[:-1, :], d[1:, :]),
    )


def _valid_pairs(v: np.ndarray):
    return (
        v[:, 1:] & v[:, :-1],
        v[:, :-1] & v[:, 1:],
        v[1:, :] & v[:-1, :],
        v[:-1, :] & v[1:, :],
    )


def boundary_counts(
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    t_list: Sequence[float] = BF_T_LIST,
) -> List[Tuple[int, int, int]]:
    """(tp, fp, fn) per ratio threshold, pooled over the four directed neighbour sets.

    Counting is identical to the legacy protocol.boundary_f1_si: pair (a, b) is an edge at
    t iff a / b > 1 + t/100 (ratio 0 where b <= 0), restricted to pairs whose two pixels are
    both valid. Only the scoring of degenerate thresholds differs between legacy and V2.
    """
    v = valid.astype(bool)
    out = []
    for t in t_list:
        thr = 1.0 + t / 100.0
        tp = fp = fn = 0
        for (ga, gb), (pa, pb), m in zip(_pair_views(gt), _pair_views(pred), _valid_pairs(v)):
            with np.errstate(divide="ignore", invalid="ignore"):
                g_ratio = np.where(gb > 0, ga / np.where(gb <= 0, 1.0, gb), 0.0)
                p_ratio = np.where(pb > 0, pa / np.where(pb <= 0, 1.0, pb), 0.0)
            g_edge = (g_ratio > thr) & m
            p_edge = (p_ratio > thr) & m
            tp += int((g_edge & p_edge).sum())
            fp += int((~g_edge & p_edge).sum())
            fn += int((g_edge & ~p_edge).sum())
        out.append((tp, fp, fn))
    return out


def _f1_zero(tp: int, fp: int, fn: int) -> float:
    """F1 with every undefined ratio scored as 0 (DepthPro convention)."""
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def boundary_f1_si_v2(
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    t_list: Sequence[float] = BF_T_LIST,
    both_empty: str = "zero",
    return_detail: bool = False,
):
    """Corrected occluding-boundary F1, mean over ratio thresholds t (%).

    Returns None (non-contributing image) when the image has no valid neighbour pair or its
    GT has no boundary pair at any threshold. Otherwise returns the mean over thresholds of
    F1_t, where every threshold the legacy function would skip scores 0:
      * prediction has no edge at t        -> recall 0      -> F1_t = 0
      * edges exist but none coincide       -> F1_t = 0
      * GT has no edge at t, prediction has -> precision 0   -> F1_t = 0
      * neither has an edge at t            -> 0 with both_empty="zero" (default, DepthPro),
                                               dropped from the mean with both_empty="exclude"
    With return_detail=True a dict is returned instead:
      {"bf1", "f1_per_t", "n_t", "n_t_zero_scored", "n_t_both_empty", "counts"}.
    """
    if both_empty not in ("zero", "exclude"):
        raise ValueError(f"both_empty must be 'zero' or 'exclude', got {both_empty!r}")
    v = valid.astype(bool)
    if not any(m.any() for m in _valid_pairs(v)):
        return None
    counts = boundary_counts(gt, pred, v, t_list)
    if sum(tp + fn for tp, _, fn in counts) == 0:
        return None                                    # nothing to detect: non-contributing
    f1s, n_zero, n_both = [], 0, 0
    for tp, fp, fn in counts:
        if tp + fp == 0 and tp + fn == 0:
            n_both += 1
            if both_empty == "exclude":
                continue
        f1s.append(_f1_zero(tp, fp, fn))
        n_zero += int(tp == 0)        # exactly the thresholds the legacy function skips
    bf1 = float(np.mean(f1s)) if f1s else 0.0
    if not return_detail:
        return bf1
    return {"bf1": bf1, "f1_per_t": f1s, "n_t": len(f1s), "n_t_zero_scored": n_zero,
            "n_t_both_empty": n_both, "counts": counts}


DEPTHPRO_T_MIN, DEPTHPRO_T_MAX, DEPTHPRO_N = 1.05, 1.25, 10


def _fgbg_depthpro(d: np.ndarray, t: float):
    """DepthPro ``fgbg_depth``: (left, top, right, bottom) neighbour maps with ratio > t."""
    with np.errstate(divide="ignore", invalid="ignore"):
        right = (d[:, 1:] / d[:, :-1]) > t
        left = (d[:, :-1] / d[:, 1:]) > t
        bottom = (d[1:, :] / d[:-1, :]) > t
        top = (d[:-1, :] / d[1:, :]) > t
    return left, top, right, bottom


def boundary_f1_depthpro_at(pred: np.ndarray, gt: np.ndarray, t: float,
                            valid: Optional[np.ndarray] = None) -> float:
    """DepthPro ``boundary_f1(invert_depth(pred), invert_depth(gt), t)`` at one ratio t (> 1)."""
    ip = 1.0 / np.maximum(np.asarray(pred, dtype=np.float64), 1e-6)
    ig = 1.0 / np.maximum(np.asarray(gt, dtype=np.float64), 1e-6)
    P, G = _fgbg_depthpro(ip, t), _fgbg_depthpro(ig, t)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        mh, mv = v[:, :-1] & v[:, 1:], v[:-1, :] & v[1:, :]
        masks = (mh, mv, mh, mv)                     # left, top, right, bottom
        P = tuple(p & m for p, m in zip(P, masks))
        G = tuple(g & m for g, m in zip(G, masks))
    r = 0.25 * sum(np.count_nonzero(p & g) / max(np.count_nonzero(g), 1) for p, g in zip(P, G))
    p = 0.25 * sum(np.count_nonzero(p & g) / max(np.count_nonzero(p), 1) for p, g in zip(P, G))
    return 0.0 if r + p == 0 else float(2.0 * r * p / (r + p))


def si_boundary_f1_depthpro(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                            t_min: float = DEPTHPRO_T_MIN, t_max: float = DEPTHPRO_T_MAX,
                            n: int = DEPTHPRO_N) -> float:
    """DepthPro ``SI_boundary_F1(pred, gt)``: threshold-weighted F1 over linspace(t_min, t_max, n).

    Argument order follows DepthPro (prediction first). Never returns None (DepthPro has no
    non-contributing case); with ``valid`` given, see the module docstring (item 5).
    """
    ts = np.linspace(t_min, t_max, n)
    w = ts / ts.sum()
    return float(sum(wi * boundary_f1_depthpro_at(pred, gt, float(ti), valid) for ti, wi in zip(ts, w)))


def aggregate_boundary_f1(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    """Dataset boundary F1 = unweighted mean over CONTRIBUTING images, with the counts.

    values: one entry per evaluated image (None = non-contributing, e.g. no GT boundary).
    Returns {"bf1", "n_images", "n_contributing", "n_no_gt_boundary"}; bf1 is None when no
    image contributes. Always report n_contributing next to bf1.
    """
    vals = list(values)
    got = [float(v) for v in vals if v is not None]
    return {
        "bf1": float(np.mean(got)) if got else None,
        "n_images": len(vals),
        "n_contributing": len(got),
        "n_no_gt_boundary": len(vals) - len(got),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mean bias, b_std, delta1 after one per-dataset scale
# ──────────────────────────────────────────────────────────────────────────────

def _valid_log_ratio(gt, pred, valid, max_depth=None) -> Optional[np.ndarray]:
    """r = ln gt - ln pred on valid pixels (pred clipped to 1e-6, as protocol.py does)."""
    m = valid.astype(bool) & np.isfinite(gt) & (gt > 0)
    if max_depth is not None:
        m = m & (gt <= max_depth)
    if not m.any():
        return None
    g = gt[m].astype(np.float64)
    p = np.clip(pred[m].astype(np.float64), 1e-6, None)
    return np.log(g) - np.log(p)


def per_image_log_shift(gt, pred, valid, max_depth=None) -> Optional[float]:
    """b = mean(ln gt - ln pred) over valid pixels (scale_gap.py 'b_shift' convention)."""
    r = _valid_log_ratio(gt, pred, valid, max_depth)
    return None if r is None else float(r.mean())


def dataset_scale_summary(b_values: Sequence[float]) -> Dict[str, float]:
    """b_mean and b_std (ddof=0, as scale_gap.py) over images; b > 0 = predicted too near."""
    b = np.asarray([x for x in b_values if x is not None], dtype=np.float64)
    if b.size == 0:
        return {"b_mean": float("nan"), "b_std": float("nan"), "n": 0}
    return {"b_mean": float(b.mean()), "b_std": float(b.std()), "n": int(b.size)}


class LogRatioStore:
    """Keeps r = ln gt - ln pred per image (float32) so a dataset-level shift can be applied
    after the pass. ``max_pixels`` (None = exact, the default) caps memory with a
    deterministic stride subsample; the cap is recorded in the summary."""

    def __init__(self, max_pixels: Optional[int] = None):
        self.max_pixels = max_pixels
        self.ids: List[str] = []
        self.r: List[np.ndarray] = []
        self.n_full: List[int] = []

    def add(self, sample_id: str, gt, pred, valid, max_depth=None) -> Optional[float]:
        r = _valid_log_ratio(gt, pred, valid, max_depth)
        if r is None:
            return None
        self.n_full.append(int(r.size))
        if self.max_pixels is not None and r.size > self.max_pixels:
            r = r[np.linspace(0, r.size - 1, self.max_pixels).astype(np.int64)]
        self.ids.append(str(sample_id))
        self.r.append(r.astype(np.float32))
        return float(r.mean())


def d1_dataset_scale(log_ratios: Sequence[np.ndarray], b: Optional[float] = None,
                     ratio: float = 1.25) -> Dict[str, float]:
    """delta1 after ONE dataset-level log shift applied to every image.

    log_ratios: per-image arrays of r = ln gt - ln pred (valid pixels). b defaults to the
    mean of per-image means (scale_gap.py's b_global). A pixel is an inlier iff
    |r - b| < ln(ratio), i.e. max(g/p', p'/g) < ratio with p' = p * e^b. Returns the
    unweighted image mean "d1_dataset_scale", the shift used, and per-image raw d1 for
    reference ("d1_raw").
    """
    rs = [np.asarray(r, dtype=np.float64) for r in log_ratios if r is not None and np.size(r)]
    if not rs:
        return {"d1_dataset_scale": float("nan"), "b_global": float("nan"), "n": 0,
                "d1_raw": float("nan")}
    if b is None:
        b = float(np.mean([r.mean() for r in rs]))
    lr = math.log(ratio)
    d1_g = [float((np.abs(r - b) < lr).mean()) for r in rs]
    d1_r = [float((np.abs(r) < lr).mean()) for r in rs]
    return {"d1_dataset_scale": float(np.mean(d1_g)), "b_global": float(b), "n": len(rs),
            "d1_raw": float(np.mean(d1_r))}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Join per-image rows on image id (not list order)
# ──────────────────────────────────────────────────────────────────────────────

def pair_rows_by_id(rows_a: Sequence[dict], rows_b: Sequence[dict], id_key: str = "id",
                    allow_intersection: bool = False):
    """Align two per-image row lists on ``id_key``.

    Raises ValueError when a row lacks the id, an id repeats within one list, or the id
    sets differ (unless allow_intersection=True). Returns (ids, aligned_a, aligned_b, info)
    with info = {"n_a", "n_b", "n_paired", "only_a", "only_b"} (only_* are id lists).
    Order follows rows_a.
    """
    def index(rows, name):
        out = {}
        for i, r in enumerate(rows):
            if id_key not in r or r[id_key] is None:
                raise ValueError(f"{name}[{i}] has no '{id_key}'; per-image rows need image ids")
            k = str(r[id_key])
            if k in out:
                raise ValueError(f"duplicate id {k!r} in {name}")
            out[k] = r
        return out

    ia, ib = index(rows_a, "rows_a"), index(rows_b, "rows_b")
    only_a = [k for k in ia if k not in ib]
    only_b = [k for k in ib if k not in ia]
    if (only_a or only_b) and not allow_intersection:
        raise ValueError(f"id sets differ: {len(only_a)} only in A (e.g. {only_a[:3]}), "
                         f"{len(only_b)} only in B (e.g. {only_b[:3]})")
    ids = [k for k in ia if k in ib]
    info = {"n_a": len(ia), "n_b": len(ib), "n_paired": len(ids),
            "only_a": only_a, "only_b": only_b}
    return ids, [ia[k] for k in ids], [ib[k] for k in ids], info


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cluster-aware paired bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def paired_bootstrap_clustered(a: Sequence[float], b: Sequence[float],
                               clusters: Optional[Sequence] = None, n_boot: int = 10000,
                               seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Paired percentile bootstrap of mean(b - a), resampling whole clusters.

    Same output keys as compare_paired.paired_bootstrap (n, mean_a, mean_b, diff, ci_lo,
    ci_hi, significant) plus n_clusters and "resampling". The statistic of one replicate is
    the image-weighted mean of (b - a) over all images in the drawn clusters. NaN pairs are
    dropped. clusters=None delegates to compare_paired.paired_bootstrap.
    """
    from benchmark.compare_paired import paired_bootstrap

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs differ in length: {a.shape} vs {b.shape}")
    if clusters is None:
        out = paired_bootstrap(a, b, n_boot=n_boot, seed=seed, alpha=alpha)
        out.update({"n_clusters": out["n"], "resampling": "image"})
        return out
    c = np.asarray([str(x) for x in clusters])
    if c.shape != a.shape:
        raise ValueError(f"clusters length {c.shape} != values length {a.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b, c = a[keep], b[keep], c[keep]
    if a.size == 0:
        raise ValueError("no finite pairs to compare")
    d = b - a
    labels, inv = np.unique(c, return_inverse=True)
    k = labels.size
    sums = np.bincount(inv, weights=d, minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(a.size), "n_clusters": int(k), "resampling": "cluster",
        "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
        "ci_lo": float(lo), "ci_hi": float(hi), "significant": bool(lo > 0 or hi < 0),
    }
