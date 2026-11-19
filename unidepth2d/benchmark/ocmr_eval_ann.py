"""OCMR-SUN (annotation-GT track): per-object metric range on the OFFICIAL SUN split.

Complementary to benchmark/ocmr_eval.py (detector-in-the-loop track, YOLO boxes +
sensor-GT 5-point sampling on the legacy HF stream): here ground truth comes from SUN
RGB-D 2D instance polygons on the NYU-filtered official alltest split (cache built by
`python -m benchmark.ocmr_sun build`), and models come from benchmark.run_bench
(official v2-s/b/l, v1-l, student:<ckpt>).

Sampling protocols per object:
  box5      deployed recipe: median of centre + 4 corners inset 20% of the tight bbox
  maskmed   median of predicted depth inside the GT polygon (mask oracle)
  center    bbox centre pixel (naive)
  det_box5  OPTIONAL (--det-yolo <weights>): detector-in-the-loop on the SAME official
            split — YOLO boxes (identical for every depth model), GT = box5-median of the
            RAW sensor depth at those boxes (std<=1m gate), pred = box5-median of model
            depth. Supersedes the legacy benchmark/ocmr_eval.py HF-stream track, which is
            NOT split-comparable with the annotation protocols (comparability contract §9).

All protocols use the single shared sampler ocmr_metrics.box5_sample (>=3-of-5 validity).
Per-object uncertainty: box5 sample std; mean error-map value inside the mask for
models with an uncertainty output (UniDepthV2 confidence head is error-like).

  python -m benchmark.ocmr_eval_ann --model v2-l --res-level 9 [--limit 200]
  python -m benchmark.ocmr_eval_ann --model v2-l --res-level 9 --det-yolo C:/Tim/nature/yolo11n.pt

Outputs: results_ocmr/<model>__perobj.csv + <model>__summary.json
        (+ <model>__det_perobj.csv when --det-yolo is set).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

from benchmark import ocmr_metrics
from benchmark.datasets_tier1 import SUN_DIR, _sun_decode_depth
from benchmark.ocmr_sun import CACHE
from benchmark.run_bench import build_model

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results_ocmr")
DET_GT_STD_MAX_M = 1.0     # detector-track GT gate: reject boxes whose 5 sensor samples spread >1m


def box5_sample(depth: np.ndarray, bbox, inset: float = 0.20):
    """Deployed 5-point recipe — delegates to the shared single-source implementation
    (ocmr_metrics.box5_sample, >=3-of-5 validity) so every OCMR track samples identically."""
    med, std, _ = ocmr_metrics.box5_sample(depth, bbox, inset=inset)
    return med, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="v2-s|v2-b|v2-l|v1-l|student:<ckpt>")
    ap.add_argument("--res-level", type=int, default=None)
    ap.add_argument("--gt-cam", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-yolo", default="", help="YOLO weights -> add same-split det_box5 protocol")
    ap.add_argument("--det-conf", type=float, default=0.25)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, device, args.gt_cam, args.res_level)
    os.makedirs(RESULTS, exist_ok=True)
    yolo = None
    if args.det_yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.det_yolo)

    sun_base = os.path.join(SUN_DIR, "SUNRGBD")
    rows = []
    det_rows = []
    n_img = 0
    for line in open(CACHE, encoding="utf-8"):
        rec = json.loads(line)
        img_path = os.path.join(sun_base, rec["image"])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rec["h"], rec["w"]
        K = np.asarray(rec["K"], dtype=np.float64).reshape(3, 3)
        out = model.predict(rgb, K)
        pred = out["depth"]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        conf = out.get("conf")
        if conf is not None and conf.shape != (h, w):
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

        # det_box5: same split, same images, same shared sampler — detector boxes are
        # identical for every depth model, so the model comparison stays fair.
        if yolo is not None:
            dep_dir = os.path.join(sun_base, rec["rel"], "depth")
            try:
                dep_path = os.path.join(dep_dir, os.listdir(dep_dir)[0])
                png = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
            except (OSError, IndexError):
                png = None
            if png is not None:
                gt_depth = _sun_decode_depth(png.astype(np.uint16))
                res = yolo.predict(np.ascontiguousarray(rgb[:, :, ::-1]),
                                   conf=args.det_conf, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    names = res.names
                    for bxyxy, cid in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy()):
                        bb = [float(v) for v in bxyxy]
                        if bb[2] - bb[0] < 4 or bb[3] - bb[1] < 4:
                            continue
                        g5, gstd, _ = ocmr_metrics.box5_sample(gt_depth, bb)
                        if not np.isfinite(g5) or gstd > DET_GT_STD_MAX_M:
                            continue                     # unreliable sensor GT at this box
                        p5, _, _ = ocmr_metrics.box5_sample(pred, bb)
                        if not np.isfinite(p5):
                            continue
                        det_rows.append({"image": rec["rel"],
                                         "class": str(names.get(int(cid), int(cid))),
                                         "gt_r": round(g5, 4), "pred": round(p5, 4)})

        for obj in rec["objects"]:
            bbox = obj["bbox_xyxy"]
            r_box5, box5_std = box5_sample(pred, bbox)
            pts = np.stack([np.clip(np.asarray(obj["poly_x"], dtype=np.float64), 0, w - 1),
                            np.clip(np.asarray(obj["poly_y"], dtype=np.float64), 0, h - 1)], 1)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.round().astype(np.int32)], 1)
            mvals = pred[mask > 0]
            r_mask = float(np.median(mvals)) if mvals.size else float("nan")
            cx = int(np.clip(round((bbox[0] + bbox[2]) / 2), 0, w - 1))
            cy = int(np.clip(round((bbox[1] + bbox[3]) / 2), 0, h - 1))
            r_center = float(pred[cy, cx])
            unc_mask = (float(conf[mask > 0].mean())
                        if conf is not None and (mask > 0).any() else float("nan"))
            rows.append({
                "image": rec["rel"], "class": obj["class"],
                "gt_r": obj["r_vis_med"], "gt_p10": obj["r_vis_p10"],
                "gt_spread": obj["spread_iqr"], "area_px": obj["area_px"],
                "box5": r_box5, "box5_std": box5_std,
                "maskmed": r_mask, "center": r_center,
                "unc_mask": unc_mask,
            })
        n_img += 1
        if n_img % 200 == 0:
            print(f"  {n_img} images / {len(rows)} objects", flush=True)
        if args.limit and n_img >= args.limit:
            break

    name = model.name
    csv_path = os.path.join(RESULTS, f"{name}__perobj.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    gt = np.array([r["gt_r"] for r in rows])
    classes = np.array([r["class"] for r in rows])
    summary = {"model": name, "gt": "SUN alltest (NYU-filtered) polygons x raw depth",
               "n_images": n_img, "n_objects": len(rows), "protocols": {}}
    for proto in ("box5", "maskmed", "center"):
        pr = np.array([r[proto] for r in rows])
        ok = np.isfinite(pr) & (pr > 0)
        unc = None
        if proto == "box5":
            u = np.array([r["box5_std"] for r in rows])[ok]
            unc = u if np.isfinite(u).all() else None
        elif proto == "maskmed":
            u = np.array([r["unc_mask"] for r in rows])[ok]
            unc = u if np.isfinite(u).all() else None
        summary["protocols"][proto] = ocmr_metrics.summarize(
            gt[ok], pr[ok], unc=unc, classes=classes[ok])
    if det_rows:
        gt_d = np.array([r["gt_r"] for r in det_rows])
        pr_d = np.array([r["pred"] for r in det_rows])
        cls_d = np.array([r["class"] for r in det_rows])
        summary["protocols"]["det_box5"] = ocmr_metrics.summarize(gt_d, pr_d, classes=cls_d)
        summary["det"] = {"yolo": os.path.basename(args.det_yolo), "conf": args.det_conf,
                          "gt": "sensor raw depth box5 (std<=1m gate)",
                          "n_objects": len(det_rows)}
        det_csv = os.path.join(RESULTS, f"{name}__det_perobj.csv")
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
            wr.writeheader()
            wr.writerows(det_rows)
        print(f"det_box5: {len(det_rows)} objects -> {det_csv}")
    js_path = os.path.join(RESULTS, f"{name}__summary.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== OCMR-ann {name}: {len(rows)} objects / {n_img} images ===")
    for proto, s in summary["protocols"].items():
        print(f"[{proto:8s}] objAbsRel={s['objAbsRel']:.4f} objD1={s['objD1']:.4f} "
              f"MAE={s['MAE']:.3f} MAE@5m={s['MAE@5m']:.3f} "
              f"proxF1@1m={s.get('proxF1@1m', float('nan'))}")
    print("wrote", csv_path, "and", js_path)


if __name__ == "__main__":
    main()
