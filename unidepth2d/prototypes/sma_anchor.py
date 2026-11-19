"""Zero-shot / training-free metric anchoring — first evidence, targeting the prox<1m gap.

Consumes the per-object records dumped by benchmark/ocmr_eval.py --dump-csv (columns:
img, model, cls, gt, pred, box_h_frac, box_w_frac, img_w, img_h) and evaluates three things:

  (0) DIAGNOSTIC — is the student's object-range error a systematic, distance-dependent bias?
      We bin objects by true distance and report the median predicted/GT ratio per bin. A ratio
      that drifts from 1.0 (esp. in the <1 m bin) is a correctable systematic bias, not noise.

  (A) AFFINE RECALIBRATION (post-hoc, no backprop, transfers zero-shot across domains):
      fit log(gt) ~= a*log(pred)+b by robust (Huber) regression on a CALIBRATION set (NYU objects),
      apply the SAME (a,b) to a disjoint TEST set (SUN objects). This is the depth analogue of
      temperature scaling. If the student has a systematic scale/near-field bias, this removes it
      and improves obj-AbsRel + proximity-F1 with zero test-time supervision.

  (B) SEMANTIC SIZE ANCHOR (pure zero-shot, no labels at all):
      for anchor-class objects of known real-world height, the pinhole model gives an independent
      distance d_geo = fy_norm * H_real / box_h_frac. Per image, the robust ratio median(d_geo/pred)
      is a scale correction applied to all objects in that image. Falls back to identity when an
      image has no anchor (so it can never hurt a no-anchor image).

Reports obj-AbsRel, MAE, delta1@obj, prox<1m F1, prox<2m F1 BEFORE vs AFTER each method, per model.
No network training; the teacher is included as a reference (it should barely move — it is already
well-calibrated, which is itself a validity check on the method).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

# Known real-world heights (m) for reliable-size COCO classes YOLO detects indoors. Deliberately a
# small, well-constrained set; noisy/variable classes (bed, plant, bag) are excluded.
ANCHOR_SIZES = {
    0: 1.70,   # person (standing height; robustified per-image + Huber, occlusion tolerated)
    56: 0.90,  # chair
    57: 0.85,  # couch (seat-back height)
    60: 0.75,  # dining table
    62: 0.55,  # tv
    63: 0.02,  # laptop (skip — too small/variable) -> excluded below
    39: 0.25,  # bottle
    75: 0.30,  # vase
}
ANCHOR_EXCLUDE = {63}
# Normalized vertical focal length fy/H per dataset (Kinect constants). NYU fy=519.47 @480 -> 1.082;
# SUN-RGBD Kinect fy ~525 @480 -> ~1.094 (cameras vary; used only for the size-anchor pinhole cue).
FY_NORM = {"nyu": 519.47 / 480.0, "sun": 1.094}


def _load(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"img": int(r["img"]), "model": r["model"], "cls": int(r["cls"]),
                         "gt": float(r["gt"]), "pred": float(r["pred"]),
                         "bh": float(r["box_h_frac"])})
    return rows


def _metrics(gt, pr):
    gt = np.asarray(gt, float); pr = np.clip(np.asarray(pr, float), 1e-3, None)
    thr = np.maximum(gt / pr, pr / gt)
    out = {"obj_AbsRel": float(np.mean(np.abs(pr - gt) / gt)),
           "obj_MAE_m": float(np.mean(np.abs(pr - gt))),
           "delta1_obj": float(np.mean(thr < 1.25)), "n": int(len(gt))}
    for X in (1.0, 2.0):
        gpos, ppos = gt < X, pr < X
        tp = int(np.sum(gpos & ppos)); fp = int(np.sum(~gpos & ppos)); fn = int(np.sum(gpos & ~ppos))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"prox{int(X)}m_F1"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _fmt(m):
    return (f"AbsRel {m['obj_AbsRel']:.4f}  MAE {m['obj_MAE_m']:.3f}  d1 {m['delta1_obj']:.3f}  "
            f"prox1m_F1 {m['prox1m_F1']:.3f}  prox2m_F1 {m['prox2m_F1']:.3f}  (n={m['n']})")


def diagnostic(rows, model):
    r = [x for x in rows if x["model"] == model]
    gt = np.array([x["gt"] for x in r]); pr = np.array([x["pred"] for x in r])
    print(f"\n[diagnostic] {model}: bias = median(pred/gt) per true-distance bin")
    edges = [0, 1, 2, 3, 5, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gt >= lo) & (gt < hi)
        if m.sum() < 5:
            continue
        ratio = np.median(pr[m] / gt[m])
        print(f"   gt in [{lo},{hi}) m: n={int(m.sum()):5d}  median(pred/gt)={ratio:.3f}  "
              f"{'OVER' if ratio > 1.03 else 'UNDER' if ratio < 0.97 else 'ok'}")


def fit_affine_log(rows_cal, model, iters=20):
    """Robust (IRLS/Huber) fit of log(gt) ~= a*log(pred)+b on the calibration model rows."""
    r = [x for x in rows_cal if x["model"] == model]
    lp = np.log(np.clip([x["pred"] for x in r], 1e-3, None))
    lg = np.log(np.clip([x["gt"] for x in r], 1e-3, None))
    A = np.stack([lp, np.ones_like(lp)], 1)
    w = np.ones_like(lp)
    ab = np.array([1.0, 0.0])
    for _ in range(iters):
        Aw = A * w[:, None]
        ab = np.linalg.lstsq(Aw, lg * w, rcond=None)[0]
        res = np.abs(A @ ab - lg)
        s = 1.4826 * np.median(res) + 1e-6
        w = np.clip(s / np.maximum(res, 1e-6), 0.0, 1.0)   # Huber-ish
    return float(ab[0]), float(ab[1])


def apply_affine(pred, a, b):
    return np.exp(a * np.log(np.clip(pred, 1e-3, None)) + b)


def size_anchor(rows_test, model, fy_norm):
    """Per-image robust size-anchor scale; identity for images with no reliable anchor."""
    by_img = defaultdict(list)
    for x in rows_test:
        if x["model"] == model:
            by_img[x["img"]].append(x)
    gt_all, raw_all, cor_all = [], [], []
    n_anchored = 0
    for img, objs in by_img.items():
        ratios = []
        for x in objs:
            c = x["cls"]
            if c in ANCHOR_SIZES and c not in ANCHOR_EXCLUDE and x["bh"] > 0.02:
                d_geo = fy_norm * ANCHOR_SIZES[c] / x["bh"]
                if 0.2 < d_geo < 15.0 and x["pred"] > 1e-3:
                    ratios.append(d_geo / x["pred"])
        scale = float(np.median(ratios)) if ratios else 1.0
        scale = min(2.0, max(0.5, scale))   # guard against a wild single anchor
        if ratios:
            n_anchored += 1
        for x in objs:
            gt_all.append(x["gt"]); raw_all.append(x["pred"]); cor_all.append(x["pred"] * scale)
    return np.array(gt_all), np.array(raw_all), np.array(cor_all), n_anchored, len(by_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True, help="calibration CSV (e.g. ocmr_obj_nyu.csv)")
    ap.add_argument("--cal-name", default="nyu")
    ap.add_argument("--test", required=True, help="test CSV (e.g. ocmr_obj_sun.csv)")
    ap.add_argument("--test-name", default="sun")
    ap.add_argument("--out", default="outputs/checkpoints/sma_results.json")
    args = ap.parse_args()

    cal = _load(args.cal)
    test = _load(args.test)
    models = sorted({x["model"] for x in test})
    fy = FY_NORM.get(args.test_name, 1.09)
    report = {"calibrate_on": args.cal_name, "test_on": args.test_name, "fy_norm": round(fy, 4), "models": {}}

    for model in models:
        print("=" * 92)
        print(f"MODEL: {model}   (calibrate on {args.cal_name}, evaluate on {args.test_name})")
        diagnostic(cal, model)

        tr = [x for x in test if x["model"] == model]
        gt = np.array([x["gt"] for x in tr]); pr = np.array([x["pred"] for x in tr])
        base = _metrics(gt, pr)
        print(f"\n  baseline           : {_fmt(base)}")

        # (A) affine recalibration fit on calibration set, applied zero-shot to test set
        a, b = fit_affine_log(cal, model)
        m_aff = _metrics(gt, apply_affine(pr, a, b))
        print(f"  + affine recal a={a:.3f} b={b:+.3f} : {_fmt(m_aff)}")

        # (B) per-image semantic size anchor (pure zero-shot on the test set itself)
        gA, rA, cA, n_anch, n_img = size_anchor(test, model, fy)
        m_size = _metrics(gA, cA)
        print(f"  + size anchor ({n_anch}/{n_img} imgs anchored) : {_fmt(m_size)}")

        report["models"][model] = {
            "baseline": base,
            "affine_recal": {"a": round(a, 4), "b": round(b, 4), **m_aff},
            "size_anchor": {"imgs_anchored": n_anch, "imgs": n_img, **m_size},
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
