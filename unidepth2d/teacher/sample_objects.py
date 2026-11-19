"""Sample per-object distances from ALREADY-SAVED teacher depth maps (CPU only, no GPU).

Decoupled from the depth pass so per-object distances can use any detection source at broad
coverage. Default source is the plain YOLO `labels/` dir (5,229 images) — much wider coverage than
the 10-image rich Detection_ensemble metadata. YOLO boxes are normalized, so they scale directly to
the saved depth PNG grid (resolution-independent); distances are the median of the paper's 5-point
sample.

  python -m teacher.sample_objects --config config.yaml                 # all YOLO-labeled images
  python -m teacher.sample_objects --config config.yaml --limit 20      # quick check
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from common import depth_ops as D
from common.io_utils import load_config, parse_yolo_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels-dir", default="", help="override YOLO labels dir (default: data.yolo_labels_dir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing object JSONs")
    args = ap.parse_args()
    cfg = load_config(args.config)

    labels_dir = args.labels_dir or cfg["data"]["yolo_labels_dir"]
    depth_dir = cfg["output"]["depth_dir"]
    objects_dir = cfg["output"]["objects_dir"]
    os.makedirs(objects_dir, exist_ok=True)
    tcfg = cfg["teacher"]

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if args.limit:
        label_files = label_files[: args.limit]
    print(f"{len(label_files)} YOLO label files under {labels_dir}")

    written = skipped_no_depth = skipped_exist = 0
    for lp in label_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        obj_path = os.path.join(objects_dir, stem + ".json")
        if os.path.exists(obj_path) and not args.overwrite:
            # only skip if it already has objects (don't clobber the rich 10, keep others updatable)
            try:
                if json.load(open(obj_path, encoding="utf-8")).get("num_objects", 0) > 0:
                    skipped_exist += 1
                    continue
            except Exception:
                pass
        dp = os.path.join(depth_dir, stem + ".png")
        if not os.path.exists(dp):
            skipped_no_depth += 1
            continue
        depth_png = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            skipped_no_depth += 1
            continue
        depth_m = depth_png.astype(np.float32) / 1000.0
        dh, dw = depth_m.shape[:2]

        # YOLO boxes are normalized -> scale to the depth PNG grid (dw, dh).
        boxes = parse_yolo_txt(labels_dir, stem + ".jpg", (dw, dh))
        objects = []
        for b in boxes:
            od = D.sample_object_distance(
                depth_m, b.xyxy,
                inset_frac=float(tcfg.get("sample_inset_frac", 0.20)),
                uncertainty_std_m=float(tcfg.get("uncertainty_std_m", 1.0)),
                proximity_warn_m=float(tcfg.get("proximity_warn_m", 1.0)),
                class_id=b.class_id, class_name=b.class_name, confidence=b.confidence,
            )
            if od is not None:
                objects.append(od)
        objects = D.sort_nearest_first(objects)

        with open(obj_path, "w", encoding="utf-8") as f:
            json.dump({
                "image": stem + ".jpg",
                "depth_grid_w": dw, "depth_grid_h": dh,
                "source": "yolo_labels_on_saved_depth",
                "depth_convention": "metric_metres_larger_is_farther",
                "num_objects": len(objects),
                "objects": [o.to_dict() for o in objects],
            }, f, indent=2)
        written += 1
        if written <= 5 or written % 500 == 0:
            near = objects[0].distance_m if objects else None
            print(f"  {stem}: objs={len(objects)} nearest={near}")

    print(f"DONE written={written} skipped_existing={skipped_exist} skipped_no_depth={skipped_no_depth}")


if __name__ == "__main__":
    main()
