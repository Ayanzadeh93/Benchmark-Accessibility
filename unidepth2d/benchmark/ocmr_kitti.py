"""OCMR-KITTI: outdoor per-object metric ranging from KITTI 3D-object ground truth.

The Z:\\Datasets\\KITTI drop is the KITTI *object-detection benchmark* layout
(training/{image_2,label_2,calib}, 7,481 frames) — no velodyne/depth maps, so a dense
Eigen-split depth eval is NOT possible from it. What it does provide is exactly what
OCMR needs: per-object 3D boxes (class, 2D bbox, dimensions h/w/l, location X/Y/Z in
rectified-camera coords, yaw) → metric range GT for ~40k objects across the full range
2–80 m. This extends OCMR beyond indoor SUN to the outdoor/driving slice, per the
locked plan decision ("extend to KITTI 3D boxes iff outdoor tier approved").

GT range conventions (all cached; eval picks one, default z_front):
  z_front   Z − ( l/2·|cos ry| + w/2·|sin ry| )  — depth of the box's NEAREST face along
            the optical axis. Matches both the OCMR visible-surface convention and what a
            depth map physically measures inside the box (plane-parallel z, not ray range).
  z_center  Z — box-center longitudinal depth (the convention of most per-object distance
            prior art, e.g. DisNet/MOM); reported as a protocol ablation.
  r_euclid  sqrt(X² + (Y−h/2)² + Z²) — Euclidean distance to the mid-height center.

Eval gates (KITTI difficulty conventions): 2D box height ≥ 25 px, truncation ≤ 0.5,
occlusion ≤ 2, GT range in [2, 80] m; per-object records keep class/difficulty/occlusion
so results can be sliced (e.g. pedestrians-only = the curated pedestrian_subset slice).

  python -m benchmark.ocmr_kitti build   [--root Z:/Datasets/KITTI] [--limit N]
  python -m benchmark.ocmr_kitti eval    --model v2-l --res-level 9 [--classes pedestrian]
                                         [--gt-conv z_front|z_center|euclid] [--limit N]

Outputs: data_bench/ocmr_kitti_gt.jsonl (cache) and
results_ocmr/kitti__<model>__{perobj.csv,summary.json} (same schema as OCMR-SUN, with
outdoor range bands 0-5-10-20-40-80 m and proximity thresholds 5/10 m).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

from benchmark import ocmr_metrics
from benchmark.datasets_tier1 import DATA_ROOT, KITTI_ROOT

CACHE = os.path.join(DATA_ROOT, "ocmr_kitti_gt.jsonl")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results_ocmr")
DEFAULT_ROOT = KITTI_ROOT

KEEP_CLASSES = {"car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram"}
PROX_OUTDOOR = (5.0, 10.0)          # braking/attention-relevant "within X m" decisions
MIN_BOX_H_PX = 25
MAX_TRUNC = 0.5
MAX_OCC = 2
RANGE_MIN, RANGE_MAX = 2.0, 80.0


def _difficulty(h_px: float, occ: int, trunc: float) -> str:
    if h_px >= 40 and occ == 0 and trunc <= 0.15:
        return "easy"
    if h_px >= 25 and occ <= 1 and trunc <= 0.30:
        return "moderate"
    if h_px >= 25 and occ <= 2 and trunc <= 0.50:
        return "hard"
    return "unknown"


def _read_K(calib_path: str) -> np.ndarray:
    """K of the left color camera (cam2) = P2[:, :3] after rectification."""
    with open(calib_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = np.array([float(v) for v in line.split()[1:]], dtype=np.float64)
                return vals.reshape(3, 4)[:, :3]
    raise ValueError(f"no P2 in {calib_path}")


def build(root: str = DEFAULT_ROOT, limit: int = 0):
    lab_dir = os.path.join(root, "training", "label_2")
    cal_dir = os.path.join(root, "training", "calib")
    img_dir = os.path.join(root, "training", "image_2")
    frames = sorted(os.path.splitext(f)[0] for f in os.listdir(lab_dir) if f.endswith(".txt"))
    n_img = n_obj = 0
    skipped = {"no_files": 0, "no_objects": 0}
    with open(CACHE, "w", encoding="utf-8") as out:
        for fr in frames:
            lab_p = os.path.join(lab_dir, fr + ".txt")
            cal_p = os.path.join(cal_dir, fr + ".txt")
            img_p = os.path.join(img_dir, fr + ".png")
            if not (os.path.exists(cal_p) and os.path.exists(img_p)):
                skipped["no_files"] += 1
                continue
            img = cv2.imread(img_p, cv2.IMREAD_COLOR)
            if img is None:
                skipped["no_files"] += 1
                continue
            h_img, w_img = img.shape[:2]
            K = _read_K(cal_p)
            objs = []
            for line in open(lab_p, encoding="utf-8"):
                p = line.split()
                if len(p) < 15:
                    continue
                cls = p[0].lower()
                if cls not in KEEP_CLASSES:
                    continue
                trunc, occ = float(p[1]), int(float(p[2]))
                x1, y1, x2, y2 = (float(v) for v in p[4:8])
                bh, bw, bl = (float(v) for v in p[8:11])       # dimensions: height, width, length
                X, Y, Z = (float(v) for v in p[11:14])         # location (bottom-face center)
                ry = float(p[14])
                if Z <= 0.1:
                    continue
                half_ext_z = 0.5 * (bl * abs(math.cos(ry)) + bw * abs(math.sin(ry)))
                objs.append({
                    "class": cls,
                    "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "h_px": round(y2 - y1, 1),
                    "trunc": round(trunc, 2), "occ": occ,
                    "difficulty": _difficulty(y2 - y1, occ, trunc),
                    "z_center": round(Z, 3),
                    "z_front": round(max(0.5, Z - half_ext_z), 3),
                    "r_euclid": round(math.sqrt(X * X + (Y - bh / 2.0) ** 2 + Z * Z), 3),
                })
            if not objs:
                skipped["no_objects"] += 1
                continue
            out.write(json.dumps({
                "frame": fr,
                "image": f"training/image_2/{fr}.png",
                "w": w_img, "h": h_img,
                "K": [round(float(v), 4) for v in K.ravel()],
                "objects": objs,
            }) + "\n")
            n_img += 1
            n_obj += len(objs)
            if n_img % 500 == 0:
                print(f"  {n_img} frames / {n_obj} objects", flush=True)
            if limit and n_img >= limit:
                break
    print(f"DONE frames={n_img} objects={n_obj} skipped={skipped}")
    print("cache ->", CACHE)


def box5_sample(depth: np.ndarray, bbox, inset: float = 0.20):
    """Shared single-source 5-point recipe (ocmr_metrics.box5_sample, >=3-of-5 validity)."""
    med, std, _ = ocmr_metrics.box5_sample(depth, bbox, inset=inset)
    return med, std


def evaluate(args):
    import torch
    from benchmark.run_bench import build_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, device, args.gt_cam, args.res_level)
    os.makedirs(RESULTS, exist_ok=True)
    gt_key = {"z_front": "z_front", "z_center": "z_center", "euclid": "r_euclid"}[args.gt_conv]
    want_classes = {c.strip().lower() for c in args.classes.split(",") if c.strip()} or None

    rows = []
    n_img = 0
    for line in open(CACHE, encoding="utf-8"):
        rec = json.loads(line)
        img_path = os.path.join(args.root, rec["image"])
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rec["h"], rec["w"]
        K = np.asarray(rec["K"], dtype=np.float64).reshape(3, 3)
        pred = model.predict(rgb, K)["depth"]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        for obj in rec["objects"]:
            if want_classes and obj["class"] not in want_classes:
                continue
            if (obj["h_px"] < MIN_BOX_H_PX or obj["trunc"] > MAX_TRUNC
                    or obj["occ"] > MAX_OCC):
                continue
            gt_r = float(obj[gt_key])
            if not (RANGE_MIN <= gt_r <= RANGE_MAX):
                continue
            r5, s5 = box5_sample(pred, obj["bbox_xyxy"])
            cx = int(np.clip(round((obj["bbox_xyxy"][0] + obj["bbox_xyxy"][2]) / 2), 0, w - 1))
            cy = int(np.clip(round((obj["bbox_xyxy"][1] + obj["bbox_xyxy"][3]) / 2), 0, h - 1))
            rows.append({
                "frame": rec["frame"], "class": obj["class"],
                "difficulty": obj["difficulty"], "occ": obj["occ"],
                "gt_r": gt_r, "z_center": obj["z_center"], "z_front": obj["z_front"],
                "box5": r5, "box5_std": s5, "center": float(pred[cy, cx]),
                "h_px": obj["h_px"],
            })
        n_img += 1
        if n_img % 200 == 0:
            print(f"  {n_img} frames / {len(rows)} objects", flush=True)
        if args.limit and n_img >= args.limit:
            break
    if not rows:
        raise SystemExit("no objects survived the gates — check cache/filters")

    name = model.name
    tagc = f"-{'_'.join(sorted(want_classes))}" if want_classes else ""
    csv_path = os.path.join(RESULTS, f"kitti{tagc}__{name}__perobj.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    gt = np.array([r["gt_r"] for r in rows])
    classes = np.array([r["class"] for r in rows])
    summary = {"model": name, "gt": f"KITTI object-benchmark 3D boxes ({args.gt_conv})",
               "gt_convention": args.gt_conv, "classes": sorted(set(classes.tolist())),
               "n_images": n_img, "n_objects": len(rows), "protocols": {}}
    for proto in ("box5", "center"):
        pr = np.array([r[proto] for r in rows])
        ok = np.isfinite(pr) & (pr > 0)
        unc = None
        if proto == "box5":
            u = np.array([r["box5_std"] for r in rows])[ok]
            unc = u if np.isfinite(u).all() else None
        summary["protocols"][proto] = ocmr_metrics.summarize(
            gt[ok], pr[ok], unc=unc, classes=classes[ok],
            bands=ocmr_metrics.RANGE_BANDS_OUTDOOR, prox_thresholds=PROX_OUTDOOR)
    js_path = os.path.join(RESULTS, f"kitti{tagc}__{name}__summary.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== OCMR-KITTI {name} ({args.gt_conv}): {len(rows)} objects / {n_img} frames ===")
    for proto, s in summary["protocols"].items():
        print(f"[{proto:6s}] objAbsRel={s['objAbsRel']:.4f} objD1={s['objD1']:.4f} "
              f"MAE={s['MAE']:.2f} bias_log={s['bias_log']:+.4f} "
              f"proxF1@5m={s.get('proxF1@5m')} proxF1@10m={s.get('proxF1@10m')}")
    print("wrote", csv_path, "and", js_path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default=DEFAULT_ROOT)
    b.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, help="v2-s|v2-b|v2-l|v1-l|student:<ckpt>")
    e.add_argument("--root", default=DEFAULT_ROOT)
    e.add_argument("--res-level", type=int, default=None)
    e.add_argument("--gt-cam", action="store_true")
    e.add_argument("--gt-conv", default="z_front", choices=["z_front", "z_center", "euclid"])
    e.add_argument("--classes", default="", help="comma list to restrict (e.g. pedestrian)")
    e.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.root, a.limit)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
