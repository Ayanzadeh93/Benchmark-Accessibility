"""OCMR — Object-Centric Metric Ranging evaluation against REAL sensor ground truth.

⚠️ SUPERSEDED FOR PAPER NUMBERS (comparability): this track runs on the LEGACY HF streams
(vikhyatk NYU / wyrx SUN first-N), which is a DIFFERENT split from the official
NYU-filtered alltest used by every other OCMR protocol. For a split-comparable
detector-in-the-loop row use `benchmark.ocmr_eval_ann --det-yolo <weights>` (det_box5).
This file remains for the legacy transfer-evidence numbers only.

The existing benchmark/gt_eval.py evaluates DENSE pixel depth vs sensor GT (NYU/SUN). The existing
per-object numbers (eval_all.json, own_eval/*) are vs the TEACHER (pseudo-GT), not real GT. This
script fills the gap the whole project is about — per-OBJECT metric range accuracy vs real GT —
and does it with the EXACT recipe the deployed AidGPT system uses at run time:

  1. Detect objects with YOLO (one detector run per image; identical boxes for every depth model,
     so the model comparison is fair regardless of detector error).
  2. For each box sample the paper's 5 points (centre + 4 corners inset 20%) on both the SENSOR-GT
     depth and each model's predicted depth; take the median of the valid samples (>=3 of 5).
  3. A GT object is kept only if it has >=3 valid GT samples AND their std <= uncertain_std (the
     teacher's own "uncertain" gate) — so we never score against a hole-riddled GT box.
  4. Pool over all (image, object) and report object-level metrics vs real GT:
       obj_AbsRel, obj_MAE_m, obj_MAE_5m (within 5 m), delta1@obj (max(g/p,p/g)<1.25),
       and proximity F1 at <1 m and <2 m (the BLV "is it close?" decision).

Datasets + models are reused from gt_eval.py (NYU-654 validation, SUN test), so protocol (splits,
decode, caps, no scale alignment) is identical to the dense eval.

  conda run -n depth python -m benchmark.ocmr_eval --config config.yaml --dataset nyu \
      --model teacher --model student:outputs/checkpoints/S10_ultra_L/best.pt --limit 200
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from common.io_utils import load_config
from benchmark.gt_eval import (DATASETS, MIN_D, _decode_sun_depth, build_model, eigen_crop_mask)

INSET = 0.20                 # 4 corner samples inset 20% from each edge (paper recipe)
UNCERTAIN_STD_M = 1.0        # GT box rejected if the 5 GT samples' std exceeds this
PROX_THRESHOLDS = (1.0, 2.0)  # metres; "is the object within X m?" decision points


def _five_points(x1, y1, x2, y2, W, H):
    """Return the 5 (px,py) sample coords: centre + 4 corners inset 20%, clamped to the image."""
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    ix, iy = (x2 - x1) * INSET, (y2 - y1) * INSET
    pts = [(cx, cy), (x1 + ix, y1 + iy), (x2 - ix, y1 + iy), (x1 + ix, y2 - iy), (x2 - ix, y2 - iy)]
    return [(int(min(max(px, 0), W - 1)), int(min(max(py, 0), H - 1))) for px, py in pts]


def _sample_median(depth, pts, lo, hi):
    """Median of valid depth samples at pts. Returns (median or None, std, n_valid)."""
    vals = [float(depth[py, px]) for px, py in pts]
    good = [v for v in vals if np.isfinite(v) and lo < v < hi]
    if len(good) < 3:
        return None, None, len(good)
    return float(np.median(good)), float(np.std(good)), len(good)


def _detect(yolo, rgb, conf):
    """Run YOLO, return list of (x1,y1,x2,y2,cls_id) pixel boxes."""
    res = yolo.predict(rgb[:, :, ::-1], conf=conf, verbose=False)[0]  # YOLO wants BGR-ish; RGB->BGR
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xy = res.boxes.xyxy.cpu().numpy()
    cl = res.boxes.cls.cpu().numpy()
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(c)) for b, c in zip(xy, cl)]


def object_metrics(rows):
    """rows: list of dict(gt, pred). Pooled object-level metrics vs real GT."""
    gt = np.array([r["gt"] for r in rows], dtype=np.float64)
    pr = np.clip(np.array([r["pred"] for r in rows], dtype=np.float64), 1e-3, None)
    absrel = float(np.mean(np.abs(pr - gt) / gt))
    mae = float(np.mean(np.abs(pr - gt)))
    m5 = gt <= 5.0
    mae5 = float(np.mean(np.abs(pr[m5] - gt[m5]))) if m5.any() else float("nan")
    thr = np.maximum(gt / pr, pr / gt)
    d1 = float(np.mean(thr < 1.25))
    out = {"obj_AbsRel": round(absrel, 4), "obj_MAE_m": round(mae, 4),
           "obj_MAE_5m": round(mae5, 4), "delta1_obj": round(d1, 4), "n_obj": len(rows)}
    for X in PROX_THRESHOLDS:
        gpos, ppos = gt < X, pr < X
        tp = int(np.sum(gpos & ppos)); fp = int(np.sum(~gpos & ppos)); fn = int(np.sum(gpos & ~ppos))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[f"prox{int(X)}m_F1"] = round(f1, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--model", action="append", required=True, help="teacher | vits | student:<ckpt>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--yolo", default="C:/Tim/nature/yolo11n.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/checkpoints/ocmr_results.json")
    ap.add_argument("--dump-csv", default="", help="also write per-object records here for analysis")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = DATASETS[args.dataset]
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    models = [build_model(m, cfg, device) for m in args.model]

    from datasets import load_dataset
    kw = dict(split=spec["split"], streaming=True)
    ds = load_dataset(spec["repo"], spec["config"], **kw) if spec["config"] else load_dataset(spec["repo"], **kw)
    n_target = args.limit or spec["n_default"]
    print(f"[OCMR {args.dataset}] up to {n_target} imgs; models {[m.name for m in models]}; yolo {os.path.basename(args.yolo)}")

    acc = {m.name: [] for m in models}
    dump_rows = []
    n_done = n_obj = 0
    for i, s in enumerate(ds):
        if n_done >= n_target:
            break
        rgb = np.array(s["image"].convert("RGB") if hasattr(s["image"], "convert") else s["image"])
        raw = np.array(s[spec["depth_key"]])
        gt = _decode_sun_depth(raw.astype(np.uint16)) if args.dataset == "sun" else raw.astype(np.float32)
        H, W = gt.shape[:2]
        gt_ok = eigen_crop_mask(H, W) if spec["eigen_crop"] else np.ones((H, W), bool)
        gtm = np.where(gt_ok, gt, 0.0)
        boxes = _detect(yolo, rgb, args.yolo_conf)
        if not boxes:
            n_done += 1
            continue
        preds = {m.name: (lambda p: cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
                          if p.shape[:2] != (H, W) else p)(m.predict(rgb)) for m in models}
        for (x1, y1, x2, y2, cls_id) in boxes:
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            pts = _five_points(x1, y1, x2, y2, W, H)
            g_med, g_std, g_n = _sample_median(gtm, pts, MIN_D, spec["max_d"])
            if g_med is None or g_std is None or g_std > UNCERTAIN_STD_M:
                continue                                   # unreliable GT object — skip for all models
            for name, pmap in preds.items():
                p_med, _, _ = _sample_median(pmap, pts, 1e-3, 1e3)
                if p_med is not None:
                    acc[name].append({"gt": g_med, "pred": p_med})
                    if args.dump_csv:
                        dump_rows.append({"img": n_done, "model": name, "cls": cls_id,
                                          "gt": round(g_med, 4), "pred": round(p_med, 4),
                                          "box_h_frac": round((y2 - y1) / H, 4),
                                          "box_w_frac": round((x2 - x1) / W, 4),
                                          "img_w": W, "img_h": H})
            n_obj += 1
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{n_target} imgs, {n_obj} objects")

    if args.dump_csv and dump_rows:
        import csv as _csv
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)
        with open(args.dump_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(dump_rows[0].keys()))
            w.writeheader(); w.writerows(dump_rows)
        print(f"dumped {len(dump_rows)} per-object records -> {args.dump_csv}")

    result = {"dataset": args.dataset, "task": "OCMR (object range vs real sensor GT)",
              "num_images": n_done, "yolo": os.path.basename(args.yolo),
              "models": {name: object_metrics(rows) for name, rows in acc.items() if rows}}
    print(json.dumps(result, indent=2))
    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allr = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    allr.append(result)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(allr, f, indent=2)
    print("appended ->", out)


if __name__ == "__main__":
    main()
