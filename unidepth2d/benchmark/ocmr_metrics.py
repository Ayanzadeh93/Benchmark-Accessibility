"""OCMR — Object-Centric Metric Ranging: metric definitions.

Inputs are per-object arrays over an evaluation set:
  gt_r    ground-truth visible-surface range (m) — median GT depth in the object mask
  pred_r  model estimate of the same quantity (protocol-dependent sampling)
  unc     optional per-object predicted uncertainty (any monotone scale)

Metric family (each also reported per range-band and per-class elsewhere):
  objAbsRel   mean |pred-gt|/gt over objects
  objD1/D2    fraction with max(pred/gt, gt/pred) < 1.25 (resp 1.25^2)
  MAE / MAE@5 mean |pred-gt| (all / gt<=5m)
  RMSE        sqrt(mean (pred-gt)^2)
  bias_log    mean(log pred - log gt)  (signed scale drift: <0 = under-ranging)
  prox_F1@t   precision/recall/F1 of the safety decision "object within t metres"
              (pred<t vs gt<t), default t=1.0 and 2.0
  AUSE@obj    sparsification AUSE of objAbsRel under predicted-uncertainty ranking
              (oracle = ranking by true |rel error|), plus Spearman rho
Design notes: visible-surface range (not 3D centroid) is what a monocular ranger
estimates and what proximity safety needs; stratification bands follow indoor
assistive ranges. Novel-metric citations: docs/RELATED_WORK_DOSSIER.md Area 6.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_trapz = getattr(np, "trapezoid", None) or np.trapz

RANGE_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))          # indoor default
RANGE_BANDS_OUTDOOR = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0))


def box5_sample(depth: np.ndarray, bbox_xyxy, inset: float = 0.20, min_valid: int = 3):
    """THE box5 sampling recipe — single source of truth for every OCMR track.

    Median + std of the valid samples at the 5 canonical points (bbox centre + 4 corners
    inset 20%), requiring >= min_valid of 5 valid (finite, >0) — the deployed >=3-of-5
    rule. Returns (median, std, n_valid); (nan, nan, n) when under min_valid.
    Comparability contract: SUN-annotation, KITTI, and anchoring evaluations must all call
    this function so no track can drift to a different sampling/validity rule.
    """
    h, w = depth.shape
    x0, y0, x1, y1 = bbox_xyxy
    bw, bh = x1 - x0, y1 - y0
    pts = [((x0 + x1) / 2, (y0 + y1) / 2),
           (x0 + inset * bw, y0 + inset * bh), (x1 - inset * bw, y0 + inset * bh),
           (x0 + inset * bw, y1 - inset * bh), (x1 - inset * bw, y1 - inset * bh)]
    vals = []
    for x, y in pts:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        v = float(depth[yi, xi])
        if np.isfinite(v) and v > 0:
            vals.append(v)
    if len(vals) < min_valid:
        return float("nan"), float("nan"), len(vals)
    return float(np.median(vals)), float(np.std(vals)), len(vals)


def core(gt_r: np.ndarray, pred_r: np.ndarray,
         bands: Sequence = RANGE_BANDS) -> Dict[str, float]:
    gt = np.asarray(gt_r, dtype=np.float64)
    pr = np.clip(np.asarray(pred_r, dtype=np.float64), 1e-6, None)
    ratio = np.maximum(gt / pr, pr / gt)
    near5 = gt <= 5.0
    out = {
        "n": int(gt.size),
        "objAbsRel": float((np.abs(pr - gt) / gt).mean()),
        "objD1": float((ratio < 1.25).mean()),
        "objD2": float((ratio < 1.25 ** 2).mean()),
        "MAE": float(np.abs(pr - gt).mean()),
        "MAE@5m": float(np.abs(pr[near5] - gt[near5]).mean()) if near5.any() else float("nan"),
        "RMSE": float(np.sqrt(((pr - gt) ** 2).mean())),
        "bias_log": float((np.log(pr) - np.log(gt)).mean()),
    }
    for lo, hi in bands:
        m = (gt >= lo) & (gt < hi)
        out[f"objAbsRel@{lo:g}-{hi:g}m"] = (
            float((np.abs(pr[m] - gt[m]) / gt[m]).mean()) if m.any() else float("nan"))
        out[f"n@{lo:g}-{hi:g}m"] = int(m.sum())
    return out


def proximity(gt_r: np.ndarray, pred_r: np.ndarray,
              thresholds: Sequence[float] = (1.0, 2.0)) -> Dict[str, float]:
    gt = np.asarray(gt_r, dtype=np.float64)
    pr = np.asarray(pred_r, dtype=np.float64)
    out = {}
    for t in thresholds:
        gt_pos, pr_pos = gt < t, pr < t
        tp = int((gt_pos & pr_pos).sum())
        fp = int((~gt_pos & pr_pos).sum())
        fn = int((gt_pos & ~pr_pos).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan"))
        out[f"proxP@{t:g}m"], out[f"proxR@{t:g}m"], out[f"proxF1@{t:g}m"] = prec, rec, f1
        out[f"n_pos@{t:g}m"] = int(gt_pos.sum())
    return out


def ause_objects(gt_r: np.ndarray, pred_r: np.ndarray, unc: np.ndarray,
                 fractions: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Object-level sparsification: remove most-uncertain objects, track objAbsRel."""
    gt = np.asarray(gt_r, dtype=np.float64)
    pr = np.clip(np.asarray(pred_r, dtype=np.float64), 1e-6, None)
    u = np.asarray(unc, dtype=np.float64)
    err = np.abs(pr - gt) / gt
    n = gt.size
    if n < 20:
        return {"AUSE@obj": float("nan"), "nAUSE@obj": float("nan"), "spearman@obj": float("nan")}
    if fractions is None:
        fractions = np.linspace(0.0, 0.95, 20)

    def curve(order):
        e_sorted = err[order]
        csum = np.cumsum(e_sorted[::-1])[::-1]
        vals = []
        for f in fractions:
            k = min(n - 1, int(round(f * n)))
            vals.append(csum[k] / (n - k))
        return np.asarray(vals)

    c_unc = curve(np.argsort(-u))
    c_or = curve(np.argsort(-err))
    base = float(err.mean())
    ause = float(_trapz(c_unc - c_or, fractions))
    ause_rand = float(_trapz(np.full_like(c_or, base) - c_or, fractions))
    rho = float(spearmanr(u, err).statistic) if _HAVE_SCIPY else float("nan")
    return {
        "AUSE@obj": ause,
        "nAUSE@obj": ause / ause_rand if ause_rand > 1e-12 else float("nan"),
        "spearman@obj": rho,
    }


def conformal_log_intervals(
    gt_cal: np.ndarray, pred_cal: np.ndarray,
    gt_test: np.ndarray, pred_test: np.ndarray,
    alpha: float = 0.1,
    unc_cal: Optional[np.ndarray] = None,
    unc_test: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Split-conformal per-object range intervals in LOG space (Phase-4 calibration companion).

    Calibrate a log-residual quantile on a held-out CALIBRATION split, then form test intervals
    with target coverage 1-alpha. Distribution-free and finite-sample valid under exchangeability
    — the object-level complement to AUSE@obj (calibration vs discrimination). Two variants:

      plain      nonconformity s = |log pred - log gt|; interval [pred·e^-q, pred·e^+q] (constant
                 log half-width q). rel_width = e^q - e^-q.
      normalized (when unc_* given) s = |log pred - log gt| / (unc + eps); half-width scales with
                 predicted uncertainty -> ADAPTIVE intervals (tight where confident).

    Returns empirical test coverage (should be ~1-alpha), log half-width q, and mean relative
    width. A well-calibrated model reaches target coverage at small width.
    """
    def _q(residuals: np.ndarray) -> float:
        n = residuals.size
        if n < 2:
            return float("nan")
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)   # finite-sample conformal correction
        return float(np.quantile(residuals, level, method="higher"))

    lg_c = np.log(np.clip(gt_cal, 1e-6, None)); lp_c = np.log(np.clip(pred_cal, 1e-6, None))
    lg_t = np.log(np.clip(gt_test, 1e-6, None)); lp_t = np.log(np.clip(pred_test, 1e-6, None))
    q = _q(np.abs(lp_c - lg_c))
    cov = float((np.abs(lp_t - lg_t) <= q).mean()) if q == q else float("nan")
    out = {
        "alpha": alpha, "target_coverage": 1.0 - alpha,
        "coverage": cov, "q_log": q,
        "rel_width": float(np.exp(q) - np.exp(-q)) if q == q else float("nan"),
        "n_cal": int(gt_cal.size), "n_test": int(gt_test.size),
    }
    if unc_cal is not None and unc_test is not None:
        eps = 1e-6
        s = np.abs(lp_c - lg_c) / (np.asarray(unc_cal, dtype=np.float64) + eps)
        qn = _q(s)
        hw = qn * (np.asarray(unc_test, dtype=np.float64) + eps)          # per-object log half-width
        out["coverage_norm"] = float((np.abs(lp_t - lg_t) <= hw).mean()) if qn == qn else float("nan")
        out["rel_width_norm"] = float(np.mean(np.exp(hw) - np.exp(-hw))) if qn == qn else float("nan")
    return out


def per_class(gt_r, pred_r, classes, min_count: int = 25) -> Dict[str, Dict[str, float]]:
    gt = np.asarray(gt_r, dtype=np.float64)
    pr = np.asarray(pred_r, dtype=np.float64)
    cls = np.asarray(classes)
    out = {}
    for c in sorted(set(cls.tolist())):
        m = cls == c
        if m.sum() < min_count:
            continue
        r = np.maximum(gt[m] / pr[m], pr[m] / gt[m])
        out[c] = {
            "n": int(m.sum()),
            "objAbsRel": float((np.abs(pr[m] - gt[m]) / gt[m]).mean()),
            "objD1": float((r < 1.25).mean()),
            "MAE": float(np.abs(pr[m] - gt[m]).mean()),
        }
    return out


def summarize(gt_r, pred_r, unc=None, classes=None,
              bands: Sequence = RANGE_BANDS,
              prox_thresholds: Sequence[float] = (1.0, 2.0)) -> Dict:
    res = {}
    res.update(core(gt_r, pred_r, bands=bands))
    res.update(proximity(gt_r, pred_r, thresholds=prox_thresholds))
    if unc is not None and np.isfinite(np.asarray(unc, dtype=np.float64)).all():
        res.update(ause_objects(gt_r, pred_r, unc))
    if classes is not None:
        res["per_class"] = per_class(gt_r, pred_r, classes)
    return res
