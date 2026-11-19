"""V2 safety harness: gates SF1-SF6, int8 regression gates I1-I4, sparse-anchor track.

Status: RESEARCH-ONLY. Every threshold below is the provisional value from docs/V2_PLAN.md
(sections 8.3 and 9.2, owner decision 6). Until the owner sets ``owner_confirmed: true`` AND a
non-empty ``owner_confirmation`` note (who, which values, date) in a config file, every gate
record carries the label "research-only: provisional threshold" and no result may be quoted as
a safety claim. Monocular depth at this accuracy must not be the sole
ranging source (plan 9.3). Not scored at all: glass and mirrors, drop-offs, curbs, thin
obstacles (``NOT_SCORED``).

Contract
--------
Inputs are per-image arrays in metres, as everywhere in benchmark/: gt/pred (H,W) float,
valid (H,W) bool; predictions are resized to GT by the caller (``predict_depth`` does it the
way run_bench does: bilinear). A model is any object with ``.name`` and
``.predict(rgb uint8 HxWx3, K or None) -> {"depth": (h,w) metres, ...}`` (run_bench contract).
Non-finite predictions count as "infinitely far" (the unsafe direction) and are counted.

Definitions (pixel level unless stated; "near" = GT < near_m; GT pixels need valid & gt > 0)
  SF1  dangerous over-estimation: P(pred > 1.25 gt | gt < 2 m), pooled over pixels
       (image-mean reported too). Gate: <= 10 %; for an int8 artefact also rise vs its fp32
       parent <= 0.5 point.
  SF2  miss rate per GT band [0,1), [1,2), [2,4) m: a pixel in band [lo,hi) is MISSED iff
       pred >= hi (reported beyond its band; the pixel-level analogue of the project's
       "naive_fnr" in ocmr_conformal.py). Also cumulative "below T" for T = 1, 2 m: missed
       iff gt < T and pred >= T. Gate: each no worse than the reference run (R0) by > 1 point.
  NB   near-field signed bias: mean(ln pred - ln gt) on near pixels (> 0 = predicted too far,
       the dangerous direction). The plan gives no threshold: report-only.
  SF3  corridor alert, one event per image: ROI columns [round(.25W), round(.75W)), rows
       [round(.35H), H); event(T) = GT p5 over valid ROI pixels < T, for T = 1, 2 m; score =
       pred p5 over ROI pixels (all pixels by default: what a device would see). Quantiles use
       numpy method "inverted_cdf" (an observed pixel value, so +inf never interpolates). Split
       conformal: tau = k-th smallest calibration-event score, k = ceil((n+1)(1-eps)); alert
       iff score <= tau (INCLUSIVE). Guarantee under exchangeability of images:
       P(miss | event) <= eps. Gate at eps = 5 %: test miss <= 6 % and false alarm
       (alert | no event) <= 30 %. Within-dataset splits are labelled "development".
  SF4  quantization regression of SF3 after per-artefact recalibration (each artefact gets its
       own tau on the same calibration images). Gate: miss-rate rise <= 1 point and
       false-alarm rise <= 3 points (int8 minus fp32).
  SF5  perturbations low light (-2 EV in linear light + 2 DN read noise), horizontal motion blur
       (15 px at 640 wide), JPEG q40, roll +15 / -15 deg (GT rotated with the image, corners
       invalid). Gate: raw delta1 drop <= 8 points each, images paired by id.
  SF6  temporal: frames sampled at 5 fps from held-out indoor video; m_t = median ln depth;
       gate p95 |m_t - m_{t-1}| <= .05 (pairs never cross videos); compared with R0
       (report-only). Ego-motion contributes to |dm|; p95 |second difference| is a
       motion-robust flicker diagnostic (report-only). The gate is evaluated only when every
       video is an A21 hold-out video declared in configs/v2/roles.yaml (sf6_temporal.videos,
       matched by path or SHA-256) and was sampled within 0.5 fps of the target rate.
  I1   >= 99 % of pixels with |ln q - ln f| < ln 1.25 (int8 q vs fp32 parent f, all pixels).
  I2   p99 |ln q - ln f| <= ln 1.10 on near pixels (GT < 2 m when GT is given, else fp32 < 2 m).
  I3   near-field mean signed shift mean(ln q - ln f) within +/- .01.
  I4   raw delta1 drop <= 1.0 point with paired 95 % CI upper bound <= 1.5 (two-sided CI,
       cluster bootstrap when clusters are given); AbsRel rise <= .005; corrected boundary F1
       relative drop <= 10 %.
  I6   summary of the int8 safety-regression gates: SF1 int8 rise and SF4.
       (I1-I3 are measured at GT resolution after the bilinear resize both artefacts get.)
  Sparse anchors (plan 9.3): ~150 GT points per frame, multiplicative log-normal noise and
  optional outliers, robust log-median scale; scored on the non-anchor pixels. Every record
  says zero_shot_eligible=False and monocular=False: never merge into zero-shot tables.

Gate status is one of pass | fail | not_evaluated (no data, NaN, or a precondition such as
the SF6 hold-out check failed) | report_only (no threshold). +inf values FAIL.

CLI (needs the evaluation data; fails loudly when DATASETS_ROOT / Z: is not reachable):
  python -m benchmark.safety eval --dataset ibims --model <spec> [--int8 <spec>]
         [--reference <R0 safety json>] [--calib-dataset <name>] [--camera-mode none]
         [--groups-json g.json] [--clusters-json c.json] [--config cfg.yaml] [--limit N]
         [--out path.json]
  python -m benchmark.safety temporal --model <spec> [--video a.MOV ...]
         [--roles configs/v2/roles.yaml]      (default videos: roles.yaml sf6_temporal)
  python -m benchmark.safety print-config      (JSON; save as a .yaml/.json config to edit)
Model specs are run_bench specs (benchmark/run_bench.py build_model).

Tests: tests/test_safety.py (env depth or depth_cmp; synthetic data, CPU, offline).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from benchmark import protocol
from benchmark.protocol_v2 import (BF_KEY, boundary_f1_si_v2, pair_rows_by_id,
                                   paired_bootstrap_clustered)

SAFETY_CONFIG_VERSION = "v2-provisional-2026-11-18"
NOT_SCORED = ("glass and mirrors (excluded from IBims-1 validity)", "drop-offs", "curbs",
              "thin obstacles")
SF5_KINDS = ("low_light", "motion_blur", "jpeg_q40", "roll_p15", "roll_m15")
CAMERA_MODES = ("none", "default-k", "gt-k")
RESEARCH_ONLY_LABEL = ("research-only: provisional threshold (V2_PLAN 8.3/9.2), "
                       "not owner-confirmed")
GATE_STATUSES = ("pass", "fail", "not_evaluated", "report_only")

DEFAULT_CONFIG: Dict = {
    "version": SAFETY_CONFIG_VERSION,
    "source": "docs/V2_PLAN.md sections 8.3 and 9.2 (provisional values)",
    "owner_confirmed": False,
    "owner_confirmation": "",
    "sf1": {"near_m": 2.0, "over_factor": 1.25, "max_rate": 0.10, "max_int8_rise_pts": 0.5,
            "min_near_px_image_mean": 100},
    "sf2": {"bands_m": [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], "below_m": [1.0, 2.0],
            "max_worse_than_ref_pts": 1.0, "reference_run": "R0"},
    "near_bias": {"near_m": 2.0, "max_abs_bias_log": None},
    "sf3": {"roi_x": [0.25, 0.75], "roi_y": [0.35, 1.0], "gt_quantile": 0.05,
            "pred_quantile": 0.05, "quantile_method": "inverted_cdf",
            "thresholds_m": [1.0, 2.0], "target_miss": 0.05,
            "max_test_miss": 0.06, "max_false_alarm": 0.30, "min_cal_events": 20,
            "min_valid_frac": 0.10, "pred_pixels": "all", "cal_mod": 5},
    "sf4": {"max_miss_rise_pts": 1.0, "max_false_alarm_rise_pts": 3.0},
    "sf5": {"kinds": list(SF5_KINDS), "max_d1_drop_pts": 8.0,
            "low_light_ev": -2.0, "low_light_noise_dn": 2.0,
            "blur_len_px_at_640": 15, "jpeg_quality": 40, "roll_deg": 15.0},
    "sf6": {"target_fps": 5.0, "max_p95_abs_change": 0.05, "reference_run": "R0",
            "max_fps_error": 0.5},
    "int8": {"i1_ratio": 1.25, "i1_min_frac": 0.99,
             "i2_near_m": 2.0, "i2_quantile": 0.99, "i2_max_abs_dlog": math.log(1.10),
             "i3_max_abs_mean_shift": 0.01,
             "i4_max_d1_drop_pts": 1.0, "i4_max_d1_drop_ub95_pts": 1.5,
             "i4_max_arel_rise": 0.005, "i4_max_bf_rel_drop": 0.10,
             "n_boot": 10000, "seed": 0, "hist_max": 2.0, "hist_bin": 1e-4},
    "anchor": {"n_points": 150, "noise_sigma_log": 0.02, "outlier_frac": 0.0,
               "sweep_points": [10, 50, 150, 500], "sweep_noise": [0.0, 0.02, 0.05],
               "outlier_range_m": [0.2, 30.0], "seed": 0},
}


# ──────────────────────────────────────────────────────────────────────────────
# Config and gate records
# ──────────────────────────────────────────────────────────────────────────────

def default_config() -> Dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def _merge(base: Dict, over: Dict, path: str = "") -> None:
    for k, v in over.items():
        p = f"{path}.{k}" if path else k
        if k not in base:
            raise ValueError(f"unknown safety config key '{p}'")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"safety config key '{p}' must be a mapping")
            _merge(base[k], v, p)
        else:
            base[k] = v


def load_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict:
    """Provisional defaults, then a YAML/JSON file, then a dict of overrides.

    Unknown keys raise ValueError so a typo can never silently keep a default threshold.
    """
    cfg = default_config()
    if path:
        with open(path, encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def config_sha256(cfg: Dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def owner_confirmed(cfg: Dict) -> bool:
    """True only with ``owner_confirmed: true`` AND a non-empty ``owner_confirmation`` note
    (who confirmed which values, and when). A bare boolean flip does not count."""
    note = cfg.get("owner_confirmation")
    return cfg.get("owner_confirmed") is True and isinstance(note, str) and bool(note.strip())


def claim_label(cfg: Dict) -> str:
    if owner_confirmed(cfg):
        return f"owner-confirmed threshold ({cfg['owner_confirmation'].strip()})"
    return RESEARCH_ONLY_LABEL


def make_gate(gate: str, metric: str, value, threshold, op: str, cfg: Dict, *,
              status: Optional[str] = None, detail: Optional[Dict] = None) -> Dict:
    """One gate record. op is '<=' or '>=' or 'abs<='; status is computed unless given.

    No threshold -> "report_only". A missing or NaN value -> "not_evaluated". An infinite value
    is compared like any other number (+inf fails '<=' and 'abs<='): infinities come from
    non-finite predictions, which must fail a gate, never skip it.
    """
    if op not in ("<=", ">=", "abs<="):
        raise ValueError(f"unknown op {op!r}")
    if status is None:
        if threshold is None:
            status = "report_only"
        elif value is None or np.isnan(value):
            status = "not_evaluated"
        elif op == "<=":
            status = "pass" if value <= threshold else "fail"
        elif op == ">=":
            status = "pass" if value >= threshold else "fail"
        else:
            status = "pass" if abs(value) <= threshold else "fail"
    return {"gate": gate, "metric": metric,
            "value": None if value is None else float(value),
            "threshold": threshold, "op": op, "status": status,
            "label": claim_label(cfg), "detail": detail or {}}


def _wilson(k: int, n: int, z: float = 1.959964) -> List[Optional[float]]:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


# ──────────────────────────────────────────────────────────────────────────────
# Per-image measurements
# ──────────────────────────────────────────────────────────────────────────────

def _sq(x) -> np.ndarray:
    a = np.asarray(x)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected an (H,W) map, got shape {np.shape(x)}")
    return a


def _safe_pred(pred: np.ndarray):
    """pred as float64 with non-finite -> +inf (unsafe direction), clipped to >= 1e-6."""
    p = _sq(pred).astype(np.float64)
    bad = ~np.isfinite(p)
    p = np.where(bad, np.inf, np.clip(p, 1e-6, None))
    return p, int(bad.sum())


def _gt_mask(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    g = _sq(gt)
    return _sq(valid).astype(bool) & np.isfinite(g) & (g > 0)


def near_field_counts(gt, pred, valid, cfg: Dict) -> Dict:
    """Pixel counts behind SF1, SF2 and the near-field bias for one image."""
    g = _sq(gt).astype(np.float64)
    p, n_bad = _safe_pred(pred)
    if g.shape != p.shape:
        raise ValueError(f"pred {p.shape} must be resized to GT {g.shape} first")
    m = _gt_mask(g, valid)
    s1 = cfg["sf1"]
    near = m & (g < s1["near_m"])
    out = {"n_valid": int(m.sum()), "n_nonfinite_pred": n_bad,
           "sf1_n_near": int(near.sum()),
           "sf1_n_over": int((near & (p > s1["over_factor"] * g)).sum())}
    nb_near = m & (g < cfg["near_bias"]["near_m"]) & np.isfinite(p)
    lr = np.log(p[nb_near]) - np.log(g[nb_near])
    out["nb_n"] = int(nb_near.sum())
    out["nb_sum"] = float(lr.sum())
    for lo, hi in cfg["sf2"]["bands_m"]:
        band = m & (g >= lo) & (g < hi)
        key = f"band_{lo:g}-{hi:g}"
        out[f"{key}_n"] = int(band.sum())
        out[f"{key}_miss"] = int((band & (p >= hi)).sum())
        out[f"{key}_over"] = int((band & (p > s1["over_factor"] * g)).sum())
    for t in cfg["sf2"]["below_m"]:
        below = m & (g < t)
        out[f"below_{t:g}_n"] = int(below.sum())
        out[f"below_{t:g}_miss"] = int((below & (p >= t)).sum())
    return out


def roi_bounds(h: int, w: int, cfg: Dict):
    """(y0, y1, x0, x1) half-open pixel bounds of the SF3 corridor ROI."""
    rx, ry = cfg["sf3"]["roi_x"], cfg["sf3"]["roi_y"]
    return (int(round(ry[0] * h)), int(round(ry[1] * h)),
            int(round(rx[0] * w)), int(round(rx[1] * w)))


def corridor_stats(gt, pred, valid, cfg: Dict) -> Dict:
    """Per-image SF3 statistics: GT and predicted low quantiles inside the corridor ROI."""
    g = _sq(gt).astype(np.float64)
    p, _ = _safe_pred(pred)
    if g.shape != p.shape:
        raise ValueError(f"pred {p.shape} must be resized to GT {g.shape} first")
    s3 = cfg["sf3"]
    y0, y1, x0, x1 = roi_bounds(*g.shape, cfg)
    gr, pr = g[y0:y1, x0:x1], p[y0:y1, x0:x1]
    mr = _gt_mask(g, valid)[y0:y1, x0:x1]
    n_roi = int(gr.size)
    vf = float(mr.sum()) / max(1, n_roi)
    qm = s3["quantile_method"]
    gt_q = float(np.quantile(gr[mr], s3["gt_quantile"], method=qm)) if mr.any() else float("nan")
    if s3["pred_pixels"] == "all":
        pv = pr.ravel()
    elif s3["pred_pixels"] == "gt_valid":
        pv = pr[mr]
    else:
        raise ValueError(f"sf3.pred_pixels must be 'all' or 'gt_valid', got {s3['pred_pixels']!r}")
    # inverted_cdf returns an observed value, so +inf (non-finite) pixels never interpolate
    pred_q = float(np.quantile(pv, s3["pred_quantile"], method=qm)) if pv.size else float("inf")
    if not np.isfinite(pred_q):
        pred_q = float("inf")
    return {"gt_q": gt_q, "pred_q": pred_q, "valid_frac": vf, "n_roi": n_roi,
            "eligible": bool(vf >= s3["min_valid_frac"] and np.isfinite(gt_q))}


# ──────────────────────────────────────────────────────────────────────────────
# SF1 / SF2 / near-bias aggregation and gates
# ──────────────────────────────────────────────────────────────────────────────

def summarize_near_field(records: Sequence[Dict], cfg: Dict) -> Dict:
    """Pool per-image near_field_counts into SF1, SF2 and near-bias statistics."""
    def tot(k):
        return int(sum(r[k] for r in records))

    n_near, n_over = tot("sf1_n_near"), tot("sf1_n_over")
    minpx = cfg["sf1"]["min_near_px_image_mean"]
    per_img = [r["sf1_n_over"] / r["sf1_n_near"] for r in records if r["sf1_n_near"] >= minpx]
    sf1 = {"rate": n_over / n_near if n_near else None, "n_near_px": n_near,
           "image_mean_rate": float(np.mean(per_img)) if per_img else None,
           "n_images_with_near": len(per_img), "n_images": len(records)}
    sf2 = {}
    for lo, hi in cfg["sf2"]["bands_m"]:
        key = f"band_{lo:g}-{hi:g}"
        n = tot(f"{key}_n")
        sf2[key] = {"miss_rate": tot(f"{key}_miss") / n if n else None,
                    "over_rate": tot(f"{key}_over") / n if n else None, "n_px": n}
    for t in cfg["sf2"]["below_m"]:
        key = f"below_{t:g}"
        n = tot(f"{key}_n")
        sf2[key] = {"miss_rate": tot(f"{key}_miss") / n if n else None, "n_px": n}
    nb_n = tot("nb_n")
    nb = {"mean_signed_log": float(sum(r["nb_sum"] for r in records) / nb_n) if nb_n else None,
          "n_px": nb_n, "convention": "mean(ln pred - ln gt); > 0 = predicted too far"}
    return {"sf1": sf1, "sf2": sf2, "near_bias": nb,
            "n_nonfinite_pred": tot("n_nonfinite_pred")}


def _near_summary(obj: Optional[Dict]) -> Optional[Dict]:
    """Accept a near-field summary, a run_safety_eval report, or None."""
    if obj is None:
        return None
    if "sf1" in obj and "sf2" in obj:
        return obj
    for path in (("metrics", "near_field"), ("near_field",)):
        cur = obj
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and "sf2" in cur:
            return cur
    raise ValueError("reference does not contain a near-field summary (sf1/sf2)")


def gate_sf1(summary: Dict, cfg: Dict, fp32_summary: Optional[Dict] = None,
             tag: str = "") -> List[Dict]:
    s1 = cfg["sf1"]
    rate = summary["sf1"]["rate"]
    gates = [make_gate("SF1", f"over_rate@<{s1['near_m']:g}m{tag}", rate, s1["max_rate"], "<=",
                       cfg, detail=summary["sf1"])]
    if fp32_summary is not None:
        base = _near_summary(fp32_summary)["sf1"]["rate"]
        rise = None if rate is None or base is None else 100.0 * (rate - base)
        gates.append(make_gate("SF1", f"int8_rise_pts{tag}", rise, s1["max_int8_rise_pts"], "<=",
                               cfg, detail={"int8_rate": rate, "fp32_rate": base}))
    return gates


def gate_sf2(summary: Dict, reference: Optional[Dict], cfg: Dict) -> List[Dict]:
    s2 = cfg["sf2"]
    ref = _near_summary(reference)
    gates = []
    keys = [f"band_{lo:g}-{hi:g}" for lo, hi in s2["bands_m"]] + [f"below_{t:g}" for t in s2["below_m"]]
    for k in keys:
        cur = summary["sf2"][k]["miss_rate"]
        if ref is None:
            gates.append(make_gate("SF2", f"miss_rise_vs_{s2['reference_run']}_pts@{k}", None,
                                   s2["max_worse_than_ref_pts"], "<=", cfg,
                                   status="not_evaluated",
                                   detail={"miss_rate": cur, "reason": "no reference result"}))
            continue
        base = ref["sf2"].get(k, {}).get("miss_rate")
        rise = None if cur is None or base is None else 100.0 * (cur - base)
        gates.append(make_gate("SF2", f"miss_rise_vs_{s2['reference_run']}_pts@{k}", rise,
                               s2["max_worse_than_ref_pts"], "<=", cfg,
                               detail={"miss_rate": cur, "reference_miss_rate": base}))
    return gates


def gate_near_bias(summary: Dict, cfg: Dict) -> List[Dict]:
    nb = summary["near_bias"]
    thr = cfg["near_bias"]["max_abs_bias_log"]
    return [make_gate("NB", f"near_signed_bias_log@<{cfg['near_bias']['near_m']:g}m",
                      nb["mean_signed_log"], thr, "abs<=", cfg,
                      status=None if thr is not None else "report_only", detail=nb)]


# ──────────────────────────────────────────────────────────────────────────────
# SF3 / SF4: conformal corridor alert
# ──────────────────────────────────────────────────────────────────────────────

def conformal_threshold(event_scores: Sequence[float], eps: float, min_n: int = 20) -> Dict:
    """Split-conformal alert threshold over calibration EVENT scores (smaller = nearer).

    tau = k-th smallest score with k = ceil((n+1)(1-eps)); alert iff score <= tau gives
    P(score_test > tau | event) <= eps under exchangeability. When k > n no finite tau
    achieves the target, so tau = +inf (always alert) and status says so.
    (benchmark/ocmr_conformal.py uses np.quantile(level, 'higher'), one rank higher than this
    exact order statistic, and a strict 'pred < tau' alert that breaks the bound on ties.)
    """
    s = np.sort(np.asarray([x for x in event_scores], dtype=np.float64))
    s = np.where(np.isfinite(s), s, np.inf)
    n = int(s.size)
    if n < max(1, min_n):
        return {"tau": float("inf"), "n_events": n, "k": None,
                "status": "insufficient_calibration_events"}
    k = int(math.ceil((n + 1) * (1.0 - eps) - 1e-9))
    if k > n:
        return {"tau": float("inf"), "n_events": n, "k": k, "status": "insufficient_for_target"}
    return {"tau": float(s[k - 1]), "n_events": n, "k": k, "status": "ok"}


def split_calibration(keys: Sequence, mod: int = 5) -> np.ndarray:
    """Deterministic calibration mask: crc32(key) % mod == 0 (same rule as ocmr_conformal).
    Pass GROUP keys (scene / near-duplicate cluster / video), not image ids, when they exist."""
    return np.asarray([zlib.crc32(str(k).encode("utf-8")) % mod == 0 for k in keys], dtype=bool)


def _ids_hash(ids: Sequence) -> str:
    return hashlib.sha1("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()


def sf3_corridor_alert(cal: Sequence[Dict], test: Sequence[Dict], cfg: Dict,
                       split_kind: str = "within-dataset (development)") -> Dict:
    """Calibrate tau per threshold T on ``cal`` corridor records and score ``test``.

    Records are corridor_stats dicts with an "id"; ineligible records are skipped and counted.
    """
    s3 = cfg["sf3"]
    cal_e = [r for r in cal if r["eligible"]]
    test_e = [r for r in test if r["eligible"]]
    out = {"split_kind": split_kind, "n_cal": len(cal_e), "n_test": len(test_e),
           "n_ineligible": (len(cal) - len(cal_e)) + (len(test) - len(test_e)),
           "test_ids_sha1": _ids_hash([r.get("id") for r in test_e]),
           "cal_ids_sha1": _ids_hash([r.get("id") for r in cal_e]),
           "target_miss": s3["target_miss"], "per_threshold": {}}
    tg = np.asarray([r["gt_q"] for r in test_e], dtype=np.float64)
    tp = np.asarray([r["pred_q"] for r in test_e], dtype=np.float64)
    for T in s3["thresholds_m"]:
        thr = conformal_threshold([r["pred_q"] for r in cal_e if r["gt_q"] < T],
                                  s3["target_miss"], s3["min_cal_events"])
        ev = tg < T
        alert = tp <= thr["tau"]
        naive = tp < T
        n_ev, n_non = int(ev.sum()), int((~ev).sum())
        n_miss, n_fa = int((ev & ~alert).sum()), int((~ev & alert).sum())
        out["per_threshold"][f"{T:g}"] = {
            "T_m": T, **thr,
            "n_test_events": n_ev, "n_test_non_events": n_non,
            "test_miss_rate": n_miss / n_ev if n_ev else None,
            "test_miss_ci95": _wilson(n_miss, n_ev),
            "test_false_alarm_rate": n_fa / n_non if n_non else None,
            "test_false_alarm_ci95": _wilson(n_fa, n_non),
            "naive_miss_rate": int((ev & ~naive).sum()) / n_ev if n_ev else None,
            "naive_false_alarm_rate": int((~ev & naive).sum()) / n_non if n_non else None,
        }
    return out


def gate_sf3(sf3: Dict, cfg: Dict, tag: str = "") -> List[Dict]:
    s3 = cfg["sf3"]
    gates = []
    for key, r in sf3["per_threshold"].items():
        det = {**r, "split_kind": sf3["split_kind"]}
        ok = r["status"] == "ok"
        for metric, val, thr in (("test_miss_rate", r["test_miss_rate"], s3["max_test_miss"]),
                                 ("false_alarm_rate", r["test_false_alarm_rate"],
                                  s3["max_false_alarm"])):
            gates.append(make_gate("SF3", f"{metric}@{key}m{tag}", val if ok else None, thr, "<=",
                                   cfg, status=None if ok else "not_evaluated", detail=det))
    return gates


def gate_sf4(sf3_fp32: Dict, sf3_int8: Dict, cfg: Dict) -> List[Dict]:
    """int8 minus fp32 after each artefact was recalibrated on the same calibration images."""
    if (sf3_fp32["test_ids_sha1"] != sf3_int8["test_ids_sha1"]
            or sf3_fp32["cal_ids_sha1"] != sf3_int8["cal_ids_sha1"]):
        raise ValueError("SF4 needs both artefacts scored on the same calibration and test images")
    s4 = cfg["sf4"]
    gates = []
    for key, f in sf3_fp32["per_threshold"].items():
        q = sf3_int8["per_threshold"][key]
        ok = f["status"] == "ok" and q["status"] == "ok"
        for metric, kf, thr in (("miss_rise_pts", "test_miss_rate", s4["max_miss_rise_pts"]),
                                ("false_alarm_rise_pts", "test_false_alarm_rate",
                                 s4["max_false_alarm_rise_pts"])):
            val = None
            if ok and f[kf] is not None and q[kf] is not None:
                val = 100.0 * (q[kf] - f[kf])
            gates.append(make_gate("SF4", f"{metric}@{key}m", val, thr, "<=", cfg,
                                   status=None if ok else "not_evaluated",
                                   detail={"fp32": f[kf], "int8": q[kf], "tau_fp32": f["tau"],
                                           "tau_int8": q["tau"]}))
    return gates


# ──────────────────────────────────────────────────────────────────────────────
# SF5: perturbations
# ──────────────────────────────────────────────────────────────────────────────

def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def perturb_sample(rgb: np.ndarray, gt: np.ndarray, valid: np.ndarray, kind: str, cfg: Dict,
                   seed: int = 0):
    """Return (rgb', gt', valid') for one SF5 perturbation. Deterministic given seed."""
    import cv2

    s5 = cfg["sf5"]
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    if kind == "low_light":
        lin = (rgb.astype(np.float64) / 255.0) ** 2.2 * (2.0 ** s5["low_light_ev"])
        out = 255.0 * lin ** (1.0 / 2.2)
        out = out + np.random.default_rng(seed).normal(0.0, s5["low_light_noise_dn"], out.shape)
        return np.clip(np.round(out), 0, 255).astype(np.uint8), gt, valid
    if kind == "motion_blur":
        n = max(3, int(round(s5["blur_len_px_at_640"] * w / 640.0)))
        k = np.full((1, n), 1.0 / n, dtype=np.float32)
        return cv2.filter2D(rgb, -1, k, borderType=cv2.BORDER_REFLECT), gt, valid
    if kind.startswith("jpeg_q"):
        q = int(kind[6:]) if kind[6:] else int(s5["jpeg_quality"])
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), gt, valid
    if kind in ("roll_p15", "roll_m15"):
        ang = s5["roll_deg"] * (1.0 if kind == "roll_p15" else -1.0)   # + = counter-clockwise
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), ang, 1.0)
        r = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        g = cv2.warpAffine(np.asarray(gt, np.float32), M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v = cv2.warpAffine(np.asarray(valid, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        return r, g, v & (g > 0)
    raise ValueError(f"unknown perturbation {kind!r}; known: {SF5_KINDS}")


def gate_sf5(rows_by_kind: Dict[str, List[Dict]], cfg: Dict) -> List[Dict]:
    """rows_by_kind: {"clean": rows, kind: rows}; rows carry "id" and "d1" (fraction)."""
    s5 = cfg["sf5"]
    clean = rows_by_kind.get("clean") or []
    gates = []
    for kind in s5["kinds"]:
        rows = rows_by_kind.get(kind)
        if not clean or not rows:
            gates.append(make_gate("SF5", f"d1_drop_pts@{kind}", None, s5["max_d1_drop_pts"], "<=",
                                   cfg, status="not_evaluated", detail={"reason": "no rows"}))
            continue
        ids, a, b, info = pair_rows_by_id(clean, rows, allow_intersection=True)
        d1c = float(np.mean([r["d1"] for r in a]))
        d1p = float(np.mean([r["d1"] for r in b]))
        gates.append(make_gate("SF5", f"d1_drop_pts@{kind}", 100.0 * (d1c - d1p),
                               s5["max_d1_drop_pts"], "<=", cfg,
                               detail={"d1_clean": d1c, "d1_perturbed": d1p,
                                       "n_paired": info["n_paired"],
                                       "n_unpaired": len(info["only_a"]) + len(info["only_b"])}))
    return gates


# ──────────────────────────────────────────────────────────────────────────────
# SF6: temporal stability
# ──────────────────────────────────────────────────────────────────────────────

def frame_stride(src_fps: float, target_fps: float) -> int:
    if not src_fps or src_fps <= 0 or not np.isfinite(src_fps):
        raise ValueError(f"video reports an unusable frame rate ({src_fps})")
    return max(1, int(round(src_fps / target_fps)))


def iter_video_frames(path: str, target_fps: float = 5.0, max_frames: Optional[int] = None):
    """Yield (frame_index, t_seconds, rgb uint8) sampled at ~target_fps. Fails loudly."""
    import cv2

    if not os.path.isfile(path):
        raise FileNotFoundError(f"SF6 video not found: {path} (held-out videos must be on disk; "
                                "no substitute video is used)")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    stride = frame_stride(fps, target_fps)
    i = n = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, i / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                n += 1
                if max_frames and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def median_log_depth(depth: np.ndarray) -> float:
    d = np.asarray(depth, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(np.log(d))) if d.size else float("nan")


def _quantile_inf_safe(x: np.ndarray, q: float) -> float:
    """Linear np.quantile, except that interpolating towards +inf gives +inf instead of NaN
    (inf - inf); with an inf neighbour the 'higher' order statistic is the exact answer."""
    r = float(np.quantile(x, q))
    if np.isnan(r) and np.isinf(x).any():
        r = float(np.quantile(x, q, method="higher"))
    return r


def temporal_stability(series: Dict[str, Sequence[float]]) -> Dict:
    """series: {video_id: [m_t ...]} in time order. Pairs never cross video boundaries."""
    deltas, second, per_video = [], [], {}
    for vid, ms in series.items():
        m = np.asarray(ms, dtype=np.float64)
        dm = np.abs(np.diff(m))
        d2 = np.abs(m[2:] - 2.0 * m[1:-1] + m[:-2]) if m.size >= 3 else np.zeros(0)
        deltas.append(dm)
        second.append(d2)
        per_video[vid] = {"n_frames": int(m.size), "n_pairs": int(dm.size),
                          "p95_abs_change": (_quantile_inf_safe(np.where(np.isfinite(dm), dm, np.inf), 0.95)
                                             if dm.size else None),
                          "nonfinite_pairs": int((~np.isfinite(dm)).sum())}
    d = np.concatenate(deltas) if deltas else np.zeros(0)
    s = np.concatenate(second) if second else np.zeros(0)
    d = np.where(np.isfinite(d), d, np.inf)
    s = np.where(np.isfinite(s), s, np.inf)
    return {"n_pairs": int(d.size),
            "p95_abs_change": _quantile_inf_safe(d, 0.95) if d.size else None,
            "median_abs_change": _quantile_inf_safe(d, 0.5) if d.size else None,
            "p95_abs_second_diff": _quantile_inf_safe(s, 0.95) if s.size else None,
            "per_video": per_video}


DEFAULT_ROLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "v2",
                             "roles.yaml")


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sf6_holdout_videos(roles_path: str = DEFAULT_ROLES) -> Dict[str, Dict]:
    """``sf6_temporal.videos`` of configs/v2/roles.yaml ({name: {"path", "sha256"?, ...}}).

    Raises FileNotFoundError when roles.yaml is missing (no silent default video list)."""
    if not os.path.isfile(roles_path):
        raise FileNotFoundError(f"roles file not found: {roles_path} (run scripts/v2/build_roles.py)")
    import yaml
    with open(roles_path, encoding="utf-8") as f:
        roles = yaml.safe_load(f) or {}
    vids = (roles.get("sf6_temporal") or {}).get("videos") or {}
    if not isinstance(vids, dict):
        raise ValueError(f"{roles_path}: sf6_temporal.videos must be a mapping")
    return vids


def verify_sf6_videos(video_paths: Sequence[str], holdout: Dict[str, Dict]) -> Dict[str, Dict]:
    """Per video: is it one of the A21 SF6 hold-out videos (by normalised path, else SHA-256)?"""
    by_path = {os.path.normcase(os.path.abspath(v["path"])): k
               for k, v in holdout.items() if isinstance(v, dict) and v.get("path")}
    by_sha = {v["sha256"]: k for k, v in holdout.items() if isinstance(v, dict) and v.get("sha256")}
    out = {}
    for p in video_paths:
        key = os.path.normcase(os.path.abspath(p))
        if key in by_path:
            out[p] = {"held_out": True, "name": by_path[key], "matched_by": "path"}
            continue
        sha = _sha256(p) if by_sha and os.path.isfile(p) else None
        if sha is not None and sha in by_sha:
            out[p] = {"held_out": True, "name": by_sha[sha], "matched_by": "sha256"}
        else:
            out[p] = {"held_out": False, "name": None, "matched_by": None}
    return out


def gate_sf6(res: Dict, cfg: Dict, reference: Optional[Dict] = None) -> List[Dict]:
    """SF6 gates. The main gate is evaluated only when every video was verified as an A21
    hold-out video (res["held_out_verified"] is True) and every video was sampled within
    max_fps_error of the target rate; otherwise it is "not_evaluated" with the reason."""
    s6 = cfg["sf6"]
    det = {k: v for k, v in res.items() if k != "series"}
    reasons = []
    if res.get("held_out_verified") is not True:
        reasons.append("videos not verified as SF6 hold-out (A21, roles.yaml sf6_temporal)")
    bad_fps = [v for v, f in (res.get("fps") or {}).items() if not f.get("fps_ok", False)]
    if bad_fps:
        reasons.append(f"sampling rate off target for {bad_fps}")
    if reasons:
        det["reason"] = "; ".join(reasons)
    gates = [make_gate("SF6", "p95_abs_change_median_log", res.get("p95_abs_change"),
                       s6["max_p95_abs_change"], "<=", cfg,
                       status="not_evaluated" if reasons else None, detail=det)]
    ref = None
    if reference is not None:
        ref = (reference.get("metrics", {}).get("temporal", reference)).get("p95_abs_change")
    cur = res.get("p95_abs_change")
    gates.append(make_gate("SF6", f"p95_change_minus_{s6['reference_run']}",
                           None if ref is None or cur is None else cur - ref, None, "<=", cfg,
                           status="report_only" if ref is not None else "not_evaluated",
                           detail={"reference_p95": ref}))
    gates.append(make_gate("SF6", "p95_abs_second_diff (motion-robust flicker)",
                           res.get("p95_abs_second_diff"), None, "<=", cfg, status="report_only"))
    return gates


def run_temporal(model, video_paths: Sequence[str], cfg: Dict,
                 max_frames: Optional[int] = None,
                 holdout: Optional[Dict[str, Dict]] = None) -> Dict:
    """Median log depth per sampled frame, per video, then temporal_stability.

    holdout: sf6_holdout_videos() output; None records held_out_verified = None (unverified).
    """
    import cv2

    s6 = cfg["sf6"]
    series, fps_info = {}, {}
    for path in video_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SF6 video not found: {path} (no substitute video is used)")
        cap = cv2.VideoCapture(path)
        src = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        cap.release()
        ms = [median_log_depth(model.predict(rgb, None)["depth"])
              for _, _, rgb in iter_video_frames(path, s6["target_fps"], max_frames)]
        vid = os.path.basename(path)
        if vid in series:
            raise ValueError(f"two SF6 videos share the file name {vid!r}")
        series[vid] = ms
        eff = src / frame_stride(src, s6["target_fps"])
        fps_info[vid] = {"source_fps": src, "effective_fps": eff,
                         "fps_ok": abs(eff - s6["target_fps"]) <= s6["max_fps_error"]}
    res = temporal_stability(series)
    ver = verify_sf6_videos(video_paths, holdout) if holdout is not None else None
    res.update({"fps": fps_info, "series": series, "model": getattr(model, "name", "?"),
                "held_out": ver,
                "held_out_verified": (all(v["held_out"] for v in ver.values()) if ver else None)})
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Sparse-anchor track (never zero-shot)
# ──────────────────────────────────────────────────────────────────────────────

def sparse_anchor_align(gt, pred, valid, n_points: int, noise_sigma_log: float,
                        outlier_frac: float, rng: np.random.Generator,
                        outlier_range_m=(0.2, 30.0)):
    """Scale pred by the robust log-median of ~n_points noisy GT samples.

    Returns (aligned_pred, anchor_mask, info) or None when no usable pixel exists.
    """
    g = _sq(gt).astype(np.float64)
    p = _sq(pred).astype(np.float64)
    m = _gt_mask(g, valid) & np.isfinite(p) & (p > 0)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    n = int(min(n_points, idx.size))
    pick = rng.choice(idx, n, replace=False)
    z = g.flat[pick] * np.exp(rng.normal(0.0, noise_sigma_log, n))
    n_out = int(round(outlier_frac * n))
    if n_out:
        o = rng.choice(n, n_out, replace=False)
        lo, hi = math.log(outlier_range_m[0]), math.log(outlier_range_m[1])
        z[o] = np.exp(rng.uniform(lo, hi, n_out))
    s = float(np.median(np.log(z) - np.log(p.flat[pick])))
    mask = np.zeros(g.shape, bool)
    mask.flat[pick] = True
    return p * math.exp(s), mask, {"n_points": n, "log_scale": s, "n_outliers": n_out}


class AnchorAccumulator:
    """Per-image sparse-anchor sweep (points x noise) plus the unanchored reference row."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        a = cfg["anchor"]
        self.configs = [(int(n), float(s)) for n in a["sweep_points"] for s in a["sweep_noise"]]
        if (int(a["n_points"]), float(a["noise_sigma_log"])) not in self.configs:
            self.configs.append((int(a["n_points"]), float(a["noise_sigma_log"])))
        self.rows: Dict[str, List[float]] = {}

    def add(self, sample_id, gt, pred, valid, max_depth=None) -> None:
        a = self.cfg["anchor"]
        raw = protocol.per_image_depth_metrics(gt, pred, valid, max_depth=max_depth)
        if raw is None:
            return
        self.rows.setdefault("raw", []).append(raw["d1"])
        for n, s in self.configs:
            rng = np.random.default_rng([a["seed"], _seed(sample_id), n, int(round(s * 1e6))])
            res = sparse_anchor_align(gt, pred, valid, n, s, a["outlier_frac"], rng,
                                      a["outlier_range_m"])
            if res is None:
                continue
            aligned, mask, _ = res
            row = protocol.per_image_depth_metrics(gt, aligned, _sq(valid).astype(bool) & ~mask,
                                                   max_depth=max_depth)
            if row is not None:
                self.rows.setdefault(f"n{n}_sigma{s:g}", []).append(row["d1"])

    def summary(self) -> Dict:
        return {"track": "sparse-anchor (simulated GT points; NOT monocular, NOT zero-shot)",
                "zero_shot_eligible": False, "monocular": False,
                "outlier_frac": self.cfg["anchor"]["outlier_frac"],
                "d1": {k: float(np.mean(v)) for k, v in self.rows.items()},
                "n_images": {k: len(v) for k, v in self.rows.items()}}


# ──────────────────────────────────────────────────────────────────────────────
# Int8 regression gates I1-I4
# ──────────────────────────────────────────────────────────────────────────────

class AbsHistogram:
    """Fixed-bin histogram of non-negative values for a pooled quantile over many images.

    quantile() returns the UPPER edge of the bin holding the q-th value (conservative by at
    most one bin width); values beyond max_value fall in an overflow bin whose quantile is
    the largest value seen.
    """

    def __init__(self, max_value: float = 2.0, bin_width: float = 1e-4):
        self.bw = float(bin_width)
        self.nb = int(math.ceil(max_value / bin_width))
        self.counts = np.zeros(self.nb + 1, dtype=np.int64)
        self.n = 0
        self.max_seen = 0.0

    def add(self, values: np.ndarray) -> None:
        v = np.abs(np.asarray(values, dtype=np.float64).ravel())
        if v.size == 0:
            return
        v = np.where(np.isfinite(v), v, np.inf)
        idx = np.minimum(np.floor(np.minimum(v, 1e300) / self.bw), self.nb).astype(np.int64)
        self.counts += np.bincount(idx, minlength=self.nb + 1)
        self.n += int(v.size)
        self.max_seen = max(self.max_seen, float(v.max()))

    def quantile(self, q: float) -> Optional[float]:
        if self.n == 0:
            return None
        target = int(math.ceil(q * self.n - 1e-9))
        cum = np.cumsum(self.counts)
        b = int(np.searchsorted(cum, max(1, target)))
        return self.max_seen if b >= self.nb else (b + 1) * self.bw


class Int8Agreement:
    """Accumulates I1-I3 for an int8 artefact against its fp32 parent (log-depth maps)."""

    def __init__(self, cfg: Dict):
        c = cfg["int8"]
        self.cfg = cfg
        self.lr = math.log(c["i1_ratio"])
        self.near_m = c["i2_near_m"]
        self.hist = AbsHistogram(c["hist_max"], c["hist_bin"])
        self.n_total = self.n_within = self.n_near = self.n_near_finite = 0
        self.sum_shift = 0.0
        self.near_refs = set()
        self.per_image: List[Dict] = []

    def add(self, sample_id, log_fp32, log_int8, gt=None, valid=None, clamp=None) -> Dict:
        a = _sq(log_fp32).astype(np.float64)
        b = _sq(log_int8).astype(np.float64)
        if a.shape != b.shape:
            raise ValueError(f"fp32 {a.shape} and int8 {b.shape} maps differ in shape")
        if clamp is not None:
            a, b = np.clip(a, *clamp), np.clip(b, *clamp)
        d = b - a
        fin = np.isfinite(d)
        within = fin & (np.abs(np.where(fin, d, 0.0)) < self.lr)
        if gt is not None:
            g = _sq(gt).astype(np.float64)
            if g.shape != a.shape:
                raise ValueError(f"gt {g.shape} must match the log maps {a.shape}")
            near = _gt_mask(g, valid if valid is not None else np.ones_like(g, bool)) & (g < self.near_m)
            ref = "gt"
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                near = np.isfinite(a) & (np.exp(a) < self.near_m)
            ref = "fp32"
        self.near_refs.add(ref)
        dn = np.where(fin, d, np.inf)[near]
        self.hist.add(dn)
        self.n_total += int(d.size)
        self.n_within += int(within.sum())
        self.n_near += int(near.sum())
        self.n_near_finite += int(np.isfinite(dn).sum())
        self.sum_shift += float(dn[np.isfinite(dn)].sum())
        rec = {"id": str(sample_id), "frac_within": float(within.mean()),
               "n_near": int(near.sum()), "n_nonfinite": int((~fin).sum()),
               "near_mean_shift": float(dn[np.isfinite(dn)].mean()) if np.isfinite(dn).any() else None}
        self.per_image.append(rec)
        return rec

    def summary(self) -> Dict:
        c = self.cfg["int8"]
        n_nf = sum(r["n_nonfinite"] for r in self.per_image)
        return {"I1_frac_within": self.n_within / self.n_total if self.n_total else None,
                "I1_image_mean": float(np.mean([r["frac_within"] for r in self.per_image]))
                if self.per_image else None,
                "I2_p_abs_dlog_near": self.hist.quantile(c["i2_quantile"]),
                "I2_quantile": c["i2_quantile"], "I2_resolution": self.hist.bw,
                "I3_mean_signed_shift_near": (self.sum_shift / self.n_near_finite
                                              if self.n_near_finite else None),
                "I3_convention": "mean(ln int8 - ln fp32) over finite near pixels; > 0 = int8 farther",
                "near_ref": sorted(self.near_refs), "n_images": len(self.per_image),
                "n_pixels": self.n_total, "n_near_pixels": self.n_near,
                "n_near_nonfinite": self.n_near - self.n_near_finite, "n_nonfinite": n_nf}


def int8_gates(agreement: Optional[Dict], cfg: Dict, fp32_rows: Optional[List[Dict]] = None,
               int8_rows: Optional[List[Dict]] = None, clusters: Optional[Dict] = None,
               bf_key: str = BF_KEY) -> List[Dict]:
    """Gates I1-I4. agreement = Int8Agreement.summary(); rows = per-image GT rows with "id",
    "d1", "arel" and optionally bf_key (run on the SAME images); clusters = {id: cluster}."""
    c = cfg["int8"]
    g = []
    a = agreement or {}
    g.append(make_gate("I1", f"frac_within_{c['i1_ratio']:g}x_of_fp32", a.get("I1_frac_within"),
                       c["i1_min_frac"], ">=", cfg, detail=a))
    g.append(make_gate("I2", f"p{100 * c['i2_quantile']:g}_abs_dlog@<{c['i2_near_m']:g}m",
                       a.get("I2_p_abs_dlog_near"), c["i2_max_abs_dlog"], "<=", cfg,
                       detail={"near_ref": a.get("near_ref"), "resolution": a.get("I2_resolution")}))
    g.append(make_gate("I3", f"mean_signed_shift@<{c['i2_near_m']:g}m",
                       a.get("I3_mean_signed_shift_near"), c["i3_max_abs_mean_shift"], "abs<=", cfg,
                       detail={"near_ref": a.get("near_ref")}))
    if not fp32_rows or not int8_rows:
        for m, thr in (("d1_drop_pts", c["i4_max_d1_drop_pts"]),
                       ("d1_drop_ub95_pts", c["i4_max_d1_drop_ub95_pts"]),
                       ("arel_rise", c["i4_max_arel_rise"]), ("bf_rel_drop", c["i4_max_bf_rel_drop"])):
            g.append(make_gate("I4", m, None, thr, "<=", cfg, status="not_evaluated",
                               detail={"reason": "no per-image GT rows"}))
        return g
    ids, rf, rq, info = pair_rows_by_id(fp32_rows, int8_rows)        # strict: same images
    cl = [clusters[i] for i in ids] if clusters else None
    d1f = np.asarray([r["d1"] for r in rf]) * 100.0
    d1q = np.asarray([r["d1"] for r in rq]) * 100.0
    bs = paired_bootstrap_clustered(d1q, d1f, cl, n_boot=c["n_boot"], seed=c["seed"])
    det = {"n": bs["n"], "n_clusters": bs["n_clusters"], "ci95": [bs["ci_lo"], bs["ci_hi"]],
           "d1_fp32": bs["mean_b"], "d1_int8": bs["mean_a"], "resampling": bs["resampling"]}
    g.append(make_gate("I4", "d1_drop_pts", bs["diff"], c["i4_max_d1_drop_pts"], "<=", cfg, detail=det))
    g.append(make_gate("I4", "d1_drop_ub95_pts", bs["ci_hi"], c["i4_max_d1_drop_ub95_pts"], "<=",
                       cfg, detail=det))
    arel_rise = float(np.mean([r["arel"] for r in rq]) - np.mean([r["arel"] for r in rf]))
    g.append(make_gate("I4", "arel_rise", arel_rise, c["i4_max_arel_rise"], "<=", cfg))
    pairs = [(x[bf_key], y[bf_key]) for x, y in zip(rf, rq)
             if x.get(bf_key) is not None and y.get(bf_key) is not None]
    if pairs and np.mean([p[0] for p in pairs]) > 0:
        bff, bfq = float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))
        g.append(make_gate("I4", "bf_rel_drop", (bff - bfq) / bff, c["i4_max_bf_rel_drop"], "<=",
                           cfg, detail={"bf_fp32": bff, "bf_int8": bfq, "n_contributing": len(pairs)}))
    else:
        g.append(make_gate("I4", "bf_rel_drop", None, c["i4_max_bf_rel_drop"], "<=", cfg,
                           status="not_evaluated", detail={"reason": f"no '{bf_key}' values"}))
    return g


def gate_i6(gates: Sequence[Dict], cfg: Dict) -> Dict:
    """I6 = the int8 safety-regression gates of plan 9.2: SF1 int8 rise and SF4 (miss and
    false-alarm rise after per-artefact recalibration). One summary record: fail if any
    component fails, not_evaluated if any is missing, else pass."""
    comp = [g for g in gates if (g["gate"] == "SF1" and g["metric"].startswith("int8_rise_pts"))
            or g["gate"] == "SF4"]
    if not comp:
        status = "not_evaluated"
    elif any(g["status"] == "fail" for g in comp):
        status = "fail"
    elif any(g["status"] != "pass" for g in comp):
        status = "not_evaluated"
    else:
        status = "pass"
    n_fail = sum(1 for g in comp if g["status"] == "fail")
    return make_gate("I6", "n_failing_safety_regression_gates", float(n_fail) if comp else None, 0,
                     "<=", cfg, status=status,
                     detail={"components": [f"{g['gate']}:{g['metric']}={g['status']}" for g in comp]})


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def predict_depth(model, rgb, K, shape) -> np.ndarray:
    """model.predict -> depth resized (bilinear, as run_bench) to the GT shape (h, w)."""
    import cv2

    d = np.asarray(model.predict(rgb, K)["depth"], dtype=np.float32)
    while d.ndim > 2 and d.shape[0] == 1:
        d = d[0]
    h, w = shape
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d


class _Run:
    """Per-artefact accumulators for one pass over a dataset."""

    def __init__(self, cfg, boundary: bool):
        self.cfg, self.boundary = cfg, boundary
        self.near: List[Dict] = []
        self.corr: List[Dict] = []
        self.rows: List[Dict] = []

    def add(self, sid, group, gt, pred, valid, max_depth):
        self.near.append(near_field_counts(gt, pred, valid, self.cfg))
        self.corr.append({"id": sid, "group": group, **corridor_stats(gt, pred, valid, self.cfg)})
        row = protocol.per_image_depth_metrics(gt, pred, valid, max_depth=max_depth)
        if row is not None:
            row = {"id": sid, "d1": row["d1"], "arel": row["arel"]}
            if self.boundary:
                row[BF_KEY] = boundary_f1_si_v2(gt, pred, valid)
            self.rows.append(row)


def run_safety_eval(model, samples: Iterable, cfg: Optional[Dict] = None, *, int8_model=None,
                    reference: Optional[Dict] = None, calibration_samples: Optional[Iterable] = None,
                    groups: Optional[Dict[str, str]] = None, clusters: Optional[Dict[str, str]] = None,
                    perturb: bool = True, anchors: bool = True, boundary: bool = True,
                    camera_mode: str = "none", log_clamp=None,
                    progress: Optional[Callable[[int], None]] = None) -> Dict:
    """One pass over ``samples`` (EvalSample-like: id, rgb, gt, valid, K, max_depth).

    camera_mode "gt-k" passes sample.K to the model; any other mode passes None and leaves
    the default-K policy to the runner (run_bench contract). With calibration_samples, SF3/SF4
    calibrate on that set (split_kind "cross-dataset"); otherwise a deterministic crc32 split
    by group (or id) is used and labelled development.
    """
    cfg = cfg or default_config()
    if camera_mode not in CAMERA_MODES:
        raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {camera_mode!r}")
    kinds = list(cfg["sf5"]["kinds"]) if perturb else []
    main, q = _Run(cfg, boundary), (_Run(cfg, boundary) if int8_model is not None else None)
    agree = Int8Agreement(cfg) if int8_model is not None else None
    anc = AnchorAccumulator(cfg) if anchors else None
    pert: Dict[str, List[Dict]] = {k: [] for k in kinds}
    n = 0

    def kk(s):
        return s.K if camera_mode == "gt-k" else None

    for s in samples:
        sid = str(s.id)
        grp = (groups or {}).get(sid, sid)
        shape = s.gt.shape
        pred = predict_depth(model, s.rgb, kk(s), shape)
        main.add(sid, grp, s.gt, pred, s.valid, s.max_depth)
        if anc is not None:
            anc.add(sid, s.gt, pred, s.valid, s.max_depth)
        if q is not None:
            pq = predict_depth(int8_model, s.rgb, kk(s), shape)
            q.add(sid, grp, s.gt, pq, s.valid, s.max_depth)
            with np.errstate(divide="ignore", invalid="ignore"):
                agree.add(sid, np.log(pred), np.log(pq), gt=s.gt, valid=s.valid, clamp=log_clamp)
        for kind in kinds:
            r2, g2, v2 = perturb_sample(s.rgb, s.gt, s.valid, kind, cfg, seed=_seed(sid, kind))
            row = protocol.per_image_depth_metrics(g2, predict_depth(model, r2, kk(s), g2.shape),
                                                   v2, max_depth=s.max_depth)
            if row is not None:
                pert[kind].append({"id": sid, "d1": row["d1"]})
        n += 1
        if progress:
            progress(n)

    cal_recs = None
    if calibration_samples is not None:
        cal_main = _Run(cfg, False)
        cal_q = _Run(cfg, False) if int8_model is not None else None
        for s in calibration_samples:
            sid = str(s.id)
            cal_main.corr.append({"id": sid, **corridor_stats(
                s.gt, predict_depth(model, s.rgb, kk(s), s.gt.shape), s.valid, cfg)})
            if cal_q is not None:
                cal_q.corr.append({"id": sid, **corridor_stats(
                    s.gt, predict_depth(int8_model, s.rgb, kk(s), s.gt.shape), s.valid, cfg)})
        cal_recs = (cal_main.corr, cal_q.corr if cal_q else None)

    def sf3_for(run: _Run, cal_override):
        if cal_override is not None:
            return sf3_corridor_alert(cal_override, run.corr, cfg, "cross-dataset")
        is_cal = split_calibration([r["group"] for r in run.corr], cfg["sf3"]["cal_mod"])
        cal = [r for r, c in zip(run.corr, is_cal) if c]
        test = [r for r, c in zip(run.corr, is_cal) if not c]
        return sf3_corridor_alert(cal, test, cfg, "within-dataset (development)")

    near = summarize_near_field(main.near, cfg)
    sf3 = sf3_for(main, cal_recs[0] if cal_recs else None)
    gates = (gate_sf1(near, cfg) + gate_sf2(near, reference, cfg) + gate_near_bias(near, cfg)
             + gate_sf3(sf3, cfg))
    metrics = {"near_field": near, "sf3": sf3, "rows": main.rows}
    if kinds:
        gates += gate_sf5({"clean": main.rows, **pert}, cfg)
        metrics["sf5_rows"] = pert
    if q is not None:
        near_q = summarize_near_field(q.near, cfg)
        sf3_q = sf3_for(q, cal_recs[1] if cal_recs else None)
        gates += gate_sf1(near_q, cfg, fp32_summary=near, tag="[int8]")
        gates += gate_sf3(sf3_q, cfg, tag="[int8]") + gate_sf4(sf3, sf3_q, cfg)
        gates += int8_gates(agree.summary(), cfg, main.rows, q.rows, clusters)
        gates.append(gate_i6(gates, cfg))
        metrics.update({"int8_near_field": near_q, "int8_sf3": sf3_q, "int8_rows": q.rows,
                        "int8_agreement": agree.summary()})
    if anc is not None:
        metrics["sparse_anchor"] = anc.summary()
    meta = {"model": getattr(model, "name", "?"), "n_images": n, "camera_mode": camera_mode,
            "int8_model": getattr(int8_model, "name", None) if int8_model is not None else None}
    return build_report(gates, cfg, meta, metrics)


def build_report(gates: List[Dict], cfg: Dict, meta: Optional[Dict] = None,
                 metrics: Optional[Dict] = None) -> Dict:
    counts = {s: sum(1 for g in gates if g["status"] == s)
              for s in ("pass", "fail", "not_evaluated", "report_only")}
    return {"harness": "benchmark/safety.py", "claim_level": claim_label(cfg),
            "owner_confirmed": owner_confirmed(cfg),
            "config_version": cfg.get("version"), "config_sha256": config_sha256(cfg),
            "config": cfg, "not_scored": list(NOT_SCORED), "counts": counts,
            "all_evaluated_pass": counts["fail"] == 0 and counts["pass"] > 0,
            "complete": counts["not_evaluated"] == 0,
            "gates": gates, "meta": meta or {}, "metrics": metrics or {}}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

_TIER1_DIRS = {"nyu": ("NYU_DIR", "NYUv2"), "sun": ("SUN_DIR", "SUNRGBD"),
               "ibims": ("IBIMS_DIR", "IBims-1")}


def require_datasets_root(dataset: str, root: Optional[str] = None) -> str:
    """Fail loudly when the benchmark drive is missing; never fall back to another copy."""
    root = root or os.environ.get("DATASETS_ROOT", "Z:/Datasets")
    if not os.path.isdir(root):
        raise SystemExit(f"[safety] DATASETS_ROOT '{root}' is not reachable. Z: is an unstable "
                         "external drive: reconnect it or set DATASETS_ROOT. Refusing to use "
                         "any fallback location.")
    if dataset in _TIER1_DIRS:
        attr, name = _TIER1_DIRS[dataset]
        want = os.path.normcase(os.path.abspath(os.path.join(root, name)))
        if not os.path.isdir(want):
            raise SystemExit(f"[safety] dataset folder missing: {want}")
        from benchmark import datasets_tier1 as dt
        got = os.path.normcase(os.path.abspath(getattr(dt, attr)))
        if got != want:
            raise SystemExit(f"[safety] datasets_tier1 resolved {dataset} to {got}, not {want} "
                             "(silent fallback refused)")
    return root


def _loaders() -> Dict:
    from benchmark.datasets_tier1 import LOADERS
    out = dict(LOADERS)
    try:
        from benchmark.datasets_v2 import LOADERS as V2  # noqa: N811 - optional, other owner
        out.update(V2)
    except ImportError:
        pass
    return out


def _build(spec: str, device: str = "cpu", camera_mode: str = "none"):
    from benchmark.run_bench import build_model
    return build_model(spec, device, camera_mode == "gt-k", None)


def _load_id_map(path: str) -> Optional[Dict[str, str]]:
    """{image id: group/cluster} from a JSON object, or None for an empty path."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise SystemExit(f"{path}: expected a JSON object {{image id: group}}")
    return {str(k): str(v) for k, v in m.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--dataset", required=True)
    e.add_argument("--variant", default="")
    e.add_argument("--model", required=True)
    e.add_argument("--int8", default="")
    e.add_argument("--reference", default="", help="R0 safety report JSON (SF2)")
    e.add_argument("--calib-dataset", default="", help="cross-dataset SF3 calibration set")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--camera-mode", default="none", choices=["none", "default-k", "gt-k"])
    e.add_argument("--no-perturb", action="store_true")
    e.add_argument("--no-anchors", action="store_true")
    e.add_argument("--groups-json", default="", help="{id: group} for the SF3 split")
    e.add_argument("--clusters-json", default="", help="{id: cluster} for the I4 bootstrap")
    e.add_argument("--device", default="cpu")
    e.add_argument("--config", default="")
    e.add_argument("--out", default="")
    t = sub.add_parser("temporal")
    t.add_argument("--model", required=True)
    t.add_argument("--video", action="append", default=[],
                   help="default: the sf6_temporal videos of --roles")
    t.add_argument("--roles", default=DEFAULT_ROLES)
    t.add_argument("--device", default="cpu")
    t.add_argument("--max-frames", type=int, default=0)
    t.add_argument("--reference", default="")
    t.add_argument("--config", default="")
    t.add_argument("--out", default="")
    sub.add_parser("print-config")
    a = ap.parse_args(argv)

    if a.cmd == "print-config":
        print(json.dumps(default_config(), indent=2))
        return 0
    cfg = load_config(a.config or None)
    ref = None
    if a.reference:
        with open(a.reference, encoding="utf-8") as f:
            ref = json.load(f)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results_bench", "safety_v2")
    if a.cmd == "temporal":
        holdout = sf6_holdout_videos(a.roles)          # fails loudly without roles.yaml
        videos = a.video or [v["path"] for v in holdout.values()]
        if not videos:
            raise SystemExit(f"no SF6 videos given and none declared in {a.roles}")
        model = _build(a.model, a.device)
        res = run_temporal(model, videos, cfg, a.max_frames or None, holdout=holdout)
        rep = build_report(gate_sf6(res, cfg, ref), cfg,
                           {"model": model.name, "videos": videos, "roles": a.roles},
                           {"temporal": res})
        out = a.out or os.path.join(out_dir, f"temporal__{model.name}.json")
    else:
        require_datasets_root(a.dataset)
        loaders = _loaders()
        if a.dataset not in loaders:
            raise SystemExit(f"unknown dataset {a.dataset!r}; known: {sorted(loaders)}")
        kw = {"variant": a.variant} if a.variant else {}
        if a.limit:
            kw["limit"] = a.limit
        model = _build(a.model, a.device, a.camera_mode)
        q = _build(a.int8, a.device, a.camera_mode) if a.int8 else None
        cal = None
        if a.calib_dataset:
            require_datasets_root(a.calib_dataset)
            cal = list(loaders[a.calib_dataset]())
        rep = run_safety_eval(model, loaders[a.dataset](**kw), cfg, int8_model=q, reference=ref,
                              calibration_samples=cal, groups=_load_id_map(a.groups_json),
                              clusters=_load_id_map(a.clusters_json), perturb=not a.no_perturb,
                              anchors=not a.no_anchors, camera_mode=a.camera_mode,
                              log_clamp=(math.log(0.2), math.log(30.0)),
                              progress=lambda i: print(f"  {i} images", flush=True) if i % 25 == 0 else None)
        rep["meta"].update({"dataset": a.dataset, "variant": a.variant or "default",
                            "limit": a.limit or None, "calib_dataset": a.calib_dataset or None})
        lim = f"__limit{a.limit}" if a.limit else ""
        out = a.out or os.path.join(out_dir, f"{a.dataset}__{model.name}"
                                    f"{'__vs_' + q.name if q else ''}{lim}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    for g in rep["gates"]:
        v = "-" if g["value"] is None else f"{g['value']:.4f}"
        print(f"  {g['gate']:4s} {g['metric']:45s} {v:>10s}  {g['op']} {g['threshold']}  {g['status']}")
    print(f"[safety] {rep['claim_level']}\n[safety] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
