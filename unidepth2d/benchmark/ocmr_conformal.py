"""Wire split-conformal per-object range intervals + risk-controlled proximity
alerts into the OCMR results (paper claim C4 / Tab. 6).

  python -m benchmark.ocmr_conformal                    # all perobj CSVs found
  python -m benchmark.ocmr_conformal --alpha 0.1 --fnr 0.05 --protocol box5

Consumes results_ocmr/<model>__perobj.csv (written by ocmr_eval_ann; 57,677 SUN
objects: image, class, gt_r, gt_p10, gt_spread, area_px, box5, box5_std,
maskmed, center, unc_mask). Two outputs per model:

1) Split-conformal log-range intervals (benchmark/ocmr_metrics.py::
   conformal_log_intervals — previously defined but never called anywhere):
   plain q on |log pred − log gt|, and the uncertainty-normalized variant using
   the per-object relative box5 spread (box5_std / box5) where available.

2) Risk-controlled proximity alert (safety framing): alert "object within T m"
   iff pred < tau, with tau chosen on the calibration split as the finite-sample
   (1−eps)-quantile of pred over TRUE positives (gt < T). Under exchangeability
   within the positive stratum this bounds the miss rate: P(no alert | gt < T)
   <= eps. We report achieved test FNR and the FPR paid for the guarantee.
   GT convention: gt_r (median visible surface; the scored quantity) — gt_p10
   (near-surface) reported as a sensitivity row.

Calibration/test split is BY IMAGE (objects within an image are correlated),
deterministic (crc32(image) % 5 == 0 -> cal, ~20%), and IDENTICAL across models
so rows are comparable side-by-side.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import zlib

import numpy as np

from benchmark.ocmr_metrics import conformal_log_intervals

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_ocmr")


def _load(csv_path, protocol):
    img, gt, gtp10, pred, unc = [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = float(r[protocol])
            g = float(r["gt_r"])
            if not (np.isfinite(p) and p > 0 and np.isfinite(g) and g > 0):
                continue
            img.append(r["image"])
            gt.append(g)
            gtp10.append(float(r["gt_p10"]))
            pred.append(p)
            if protocol == "box5":
                s = float(r["box5_std"])
                u = s / p if np.isfinite(s) and p > 0 else np.nan
            elif protocol == "maskmed":
                u = float(r["unc_mask"])
            else:
                u = np.nan
            unc.append(u)
    return (np.asarray(img), np.asarray(gt), np.asarray(gtp10),
            np.asarray(pred), np.asarray(unc))


def _cal_mask(images: np.ndarray) -> np.ndarray:
    return np.asarray([zlib.crc32(s.encode()) % 5 == 0 for s in images])


def _fnr_controlled_alert(gt_cal, pred_cal, gt_test, pred_test, thr, eps):
    """tau = finite-sample (1-eps) upper quantile of pred over cal TRUE positives."""
    pos_cal = pred_cal[gt_cal < thr]
    n = pos_cal.size
    if n < 20:
        return None
    level = min(1.0, np.ceil((n + 1) * (1.0 - eps)) / n)
    tau = float(np.quantile(pos_cal, level, method="higher"))
    tp = gt_test < thr
    alert = pred_test < tau
    fn = float((tp & ~alert).sum())
    fp = float((~tp & alert).sum())
    return {
        "thr_m": thr, "eps": eps, "tau_m": round(tau, 4), "n_cal_pos": int(n),
        "test_fnr": round(fn / max(1, tp.sum()), 4),
        "test_fpr": round(fp / max(1, (~tp).sum()), 4),
        "test_precision": round(float((tp & alert).sum()) / max(1.0, alert.sum()), 4),
        "naive_fnr": round(float((tp & ~(pred_test < thr)).sum()) / max(1, tp.sum()), 4),
    }


def run_model(csv_path, protocols, alpha, fnr_eps, prox_thresholds):
    model = os.path.basename(csv_path).replace("__perobj.csv", "")
    out = {"model": model, "alpha": alpha, "split": "crc32(image)%5==0 -> cal"}
    for proto in protocols:
        img, gt, gtp10, pred, unc = _load(csv_path, proto)
        cal = _cal_mask(img)
        res = {"n_cal": int(cal.sum()), "n_test": int((~cal).sum())}
        # normalized intervals need a non-degenerate normalizer: near-zero spreads
        # explode s = err/(unc+eps). Standard remedy: FLOOR the normalizer (Romano
        # et al.-style locally adaptive conformal uses sigma+floor); floor = 25th pct
        # of the calibration spreads (>=0.05 relative), applied to cal AND test.
        u_ok = np.isfinite(unc).all() and np.unique(unc).size > 10
        if u_ok:
            floor = max(0.05, float(np.nanquantile(unc[cal], 0.25)))
            unc_f = np.clip(unc, floor, None)
            res["unc_floor"] = round(floor, 4)
        res["intervals"] = conformal_log_intervals(
            gt[cal], pred[cal], gt[~cal], pred[~cal], alpha=alpha,
            unc_cal=unc_f[cal] if u_ok else None,
            unc_test=unc_f[~cal] if u_ok else None)
        res["prox_alert"] = {}
        for thr in prox_thresholds:
            for eps in fnr_eps:
                r = _fnr_controlled_alert(gt[cal], pred[cal], gt[~cal], pred[~cal],
                                          thr, eps)
                if r:
                    res["prox_alert"][f"t{thr}_eps{eps}"] = r
        # sensitivity: nearest-surface GT convention (safety-relevant)
        r10 = _fnr_controlled_alert(gtp10[cal], pred[cal], gtp10[~cal], pred[~cal],
                                    prox_thresholds[0], fnr_eps[0])
        if r10:
            res["prox_alert_p10gt"] = r10
        out[proto] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1, help="1-alpha interval coverage")
    ap.add_argument("--fnr", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--protocol", nargs="+", default=["box5", "maskmed"])
    ap.add_argument("--model", default="", help="substring filter on perobj CSVs")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*__perobj.csv")))
    paths = [p for p in paths if "kitti" not in os.path.basename(p)]
    if args.model:
        paths = [p for p in paths if args.model in os.path.basename(p)]
    if not paths:
        raise SystemExit("no perobj CSVs matched")

    all_out = []
    for p in paths:
        out = run_model(p, args.protocol, args.alpha, args.fnr, args.thresholds)
        all_out.append(out)
        o = os.path.join(RESULTS_DIR, f"conformal__{out['model']}.json")
        with open(o, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        iv = out[args.protocol[0]]["intervals"]
        pa = out[args.protocol[0]]["prox_alert"].get(
            f"t{args.thresholds[0]}_eps{args.fnr[0]}", {})
        print(f"[{out['model']:26s}] {args.protocol[0]}: cover "
              f"{iv['coverage']:.3f}/{iv['target_coverage']:.2f} "
              f"relW {iv['rel_width']:.3f}"
              + (f" | normW {iv['rel_width_norm']:.3f}" if "rel_width_norm" in iv else "")
              + (f" | FNR@{args.thresholds[0]}m {pa.get('test_fnr')}<= {args.fnr[0]}"
                 f" (naive {pa.get('naive_fnr')}, FPR {pa.get('test_fpr')})" if pa else ""))

    summary_path = os.path.join(RESULTS_DIR, "conformal_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
