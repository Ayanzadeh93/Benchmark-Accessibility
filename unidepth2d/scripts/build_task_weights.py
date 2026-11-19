"""Precompute Task-Weighted Distillation inputs (S7) for own-domain rows with frozen detections.

For each own image that has a per-object distances JSON (outputs/objects/<stem>.json):
  data_gt/own/taskw/<stem>.png   uint8 weight map on the 640x480 grid: 100 = 1.0 background,
                                 250 = 2.5 inside detection boxes (hazard-relevant regions)
  data_gt/own/boxes/<stem>.npz   boxes (K,4) normalized xyxy + dists (K,) teacher metres
                                 + sizes (K,) known physical HEIGHT in metres (0 = unknown
                                 class), sorted nearest-first (truncation at K keeps closest).
                                 `sizes` feeds the anchor-consistency loss (distill/losses.py);
                                 downstream readers treat a missing/zero size as self-gating.

  python -m scripts.build_task_weights --config config.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from common.io_utils import load_config

OUT_W, OUT_H = 640, 480
BG_W, BOX_W = 100, 250          # /100 -> 1.0 and 2.5

# Physical HEIGHT priors (metres) for reliable-size classes, keyed by lowercased class name.
# Mirrors prototypes/anchor_lock.py SIZE_PRIORS (+ door); small, well-constrained set only.
# Objects whose class is absent get size 0.0 -> the anchor-consistency loss skips them.
ANCHOR_HEIGHTS_M = {
    "person": 1.70, "people": 1.70, "pedestrian": 1.70,
    "door": 2.03, "chair": 0.90, "couch": 0.85, "sofa": 0.85,
    "dining table": 0.75, "table": 0.75, "tv": 0.55, "tvmonitor": 0.42,
    "monitor": 0.42, "television": 0.55, "bottle": 0.25, "vase": 0.30,
    "refrigerator": 1.70, "oven": 0.85, "toilet": 0.75, "sink": 0.80,
}


def _class_name(o: dict) -> str:
    raw = o.get("class_name") or o.get("class") or o.get("label") or o.get("name") or ""
    return " ".join(str(raw).lower().replace("_", " ").replace("-", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["data_gt"]["root"]
    objects_dir = cfg["output"]["objects_dir"]
    tw_dir = os.path.join(root, "own", "taskw")
    bx_dir = os.path.join(root, "own", "boxes")
    os.makedirs(tw_dir, exist_ok=True)
    os.makedirs(bx_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(objects_dir, "*.json")))
    if args.limit:
        files = files[: args.limit]
    written = skipped = 0
    for jp in files:
        stem = os.path.splitext(os.path.basename(jp))[0]
        oj = json.load(open(jp, encoding="utf-8"))
        objs = oj.get("objects", [])
        if not objs:
            skipped += 1
            continue
        gw = oj.get("depth_grid_w", oj.get("image_w", OUT_W))
        gh = oj.get("depth_grid_h", oj.get("image_h", OUT_H))

        wmap = np.full((OUT_H, OUT_W), BG_W, dtype=np.uint8)
        boxes, dists, sizes = [], [], []
        for o in sorted(objs, key=lambda o: o["distance_m"]):
            x1, y1, x2, y2 = o["bbox_xyxy"]
            nb = [x1 / gw, y1 / gh, x2 / gw, y2 / gh]
            nb = [float(np.clip(v, 0.0, 1.0)) for v in nb]
            if nb[2] - nb[0] < 0.01 or nb[3] - nb[1] < 0.01:
                continue
            boxes.append(nb)
            dists.append(float(o["distance_m"]))
            sizes.append(ANCHOR_HEIGHTS_M.get(_class_name(o), 0.0))
            px = [int(round(nb[0] * OUT_W)), int(round(nb[1] * OUT_H)),
                  int(round(nb[2] * OUT_W)), int(round(nb[3] * OUT_H))]
            wmap[px[1]:px[3], px[0]:px[2]] = BOX_W
        if not boxes:
            skipped += 1
            continue
        cv2.imwrite(os.path.join(tw_dir, stem + ".png"), wmap)
        np.savez_compressed(os.path.join(bx_dir, stem + ".npz"),
                            boxes=np.asarray(boxes, dtype=np.float32),
                            dists=np.asarray(dists, dtype=np.float32),
                            sizes=np.asarray(sizes, dtype=np.float32))
        written += 1
    print(f"DONE taskw+boxes written={written} skipped_no_objects={skipped}")


if __name__ == "__main__":
    main()
