"""Run the frozen UniDepth v2 teacher over the dataset.

Produces, per image:
  outputs/depth_mm/<stem>.png   16-bit PNG, millimetres (metric depth, capped to save_longest_side)
  outputs/conf/<stem>.png       8-bit PNG, normalized per-image confidence
  outputs/objects/<stem>.json   per-object distances at the frozen detection boxes (paper recipe)
and a streaming manifest.csv + run_summary.json.

Usage:
  python -m teacher.run_teacher --config config.yaml            # full run (resumes, skips done)
  python -m teacher.run_teacher --config config.yaml --limit 3  # dry run on 3 images
  python -m teacher.run_teacher --config config.yaml --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from common import depth_ops as D
from common.io_utils import get_detections, list_images, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unidepth2d.run_teacher")

MANIFEST_HEADER = [
    "image", "image_w", "image_h", "num_objects", "nearest_class", "nearest_m",
    "any_proximity", "any_uncertain", "depth_png", "conf_png", "objects_json", "checkpoint",
]


def _ensure_dirs(cfg: dict) -> None:
    for key in ("root", "depth_dir", "conf_dir", "objects_dir"):
        os.makedirs(cfg["output"][key], exist_ok=True)


def _outputs_exist(cfg: dict, stem: str) -> bool:
    return (
        os.path.exists(os.path.join(cfg["output"]["depth_dir"], stem + ".png"))
        and os.path.exists(os.path.join(cfg["output"]["objects_dir"], stem + ".json"))
    )


def process_one(teacher, cfg: dict, image_path: str) -> dict:
    tcfg = cfg["teacher"]
    stem = os.path.splitext(os.path.basename(image_path))[0]

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    out = teacher.predict(rgb, infer_longest_side=int(tcfg.get("infer_longest_side", 1280)))
    depth_m = out["depth_m"]           # (H,W) metres, original resolution
    conf = out["conf"]                 # (H,W) or None

    # ── per-object distances at the frozen detection boxes (sampled at ORIGINAL resolution) ──
    boxes, wh = get_detections(image_path, cfg, img_wh=(orig_w, orig_h))
    objects = []
    for b in boxes:
        od = D.sample_object_distance(
            depth_m,
            b.xyxy,
            inset_frac=float(tcfg.get("sample_inset_frac", 0.20)),
            uncertainty_std_m=float(tcfg.get("uncertainty_std_m", 1.0)),
            proximity_warn_m=float(tcfg.get("proximity_warn_m", 1.0)),
            class_id=b.class_id,
            class_name=b.class_name,
            confidence=b.confidence,
        )
        if od is not None:
            objects.append(od)
    objects = D.sort_nearest_first(objects)

    obj_json = {
        "image": os.path.basename(image_path),
        "image_w": orig_w,
        "image_h": orig_h,
        "checkpoint": teacher.loaded_checkpoint,
        "depth_convention": "metric_metres_larger_is_farther",
        "num_objects": len(objects),
        "objects": [o.to_dict() for o in objects],
    }
    objects_path = os.path.join(cfg["output"]["objects_dir"], stem + ".json")
    with open(objects_path, "w", encoding="utf-8") as f:
        json.dump(obj_json, f, indent=2)

    # ── save depth (16-bit PNG, mm) + confidence (8-bit PNG), capped to save_longest_side ──
    save_longest = int(tcfg.get("save_longest_side", 1024))
    png16 = D.depth_m_to_png16(depth_m, clip_m=float(tcfg.get("depth_clip_m", 60.0)))
    png16 = D.resize_longest_side(png16, save_longest, cv2.INTER_NEAREST)
    depth_path = os.path.join(cfg["output"]["depth_dir"], stem + ".png")
    cv2.imwrite(depth_path, png16)

    conf8 = D.confidence_to_png8(conf, png16.shape[:2] if conf is None else conf.shape[:2])
    conf8 = D.resize_longest_side(conf8, save_longest, cv2.INTER_AREA)
    conf_path = os.path.join(cfg["output"]["conf_dir"], stem + ".png")
    cv2.imwrite(conf_path, conf8)

    nearest = objects[0] if objects else None
    return {
        "image": os.path.basename(image_path),
        "image_w": orig_w,
        "image_h": orig_h,
        "num_objects": len(objects),
        "nearest_class": (nearest.class_name if nearest else ""),
        "nearest_m": (round(nearest.distance_m, 3) if nearest else ""),
        "any_proximity": int(any(o.proximity_warn for o in objects)),
        "any_uncertain": int(any(o.uncertain for o in objects)),
        "depth_png": depth_path,
        "conf_png": conf_path,
        "objects_json": objects_path,
        "checkpoint": teacher.loaded_checkpoint,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (dry run)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N images")
    ap.add_argument("--overwrite", action="store_true", help="re-process images that already have outputs")
    ap.add_argument("--checkpoint", default="", help="override: use only this UniDepth checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _ensure_dirs(cfg)

    from teacher.unidepth_teacher import UniDepthTeacher

    ckpts = [args.checkpoint] if args.checkpoint else cfg["teacher"]["checkpoints"]
    teacher = UniDepthTeacher(ckpts, device=cfg["teacher"].get("device", "cuda"))
    teacher.load()

    images = list_images(cfg["data"]["images_dir"], cfg["data"]["image_exts"])
    if args.start:
        images = images[args.start:]
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    logger.info("Processing %d images", len(images))

    manifest_path = cfg["output"]["manifest_csv"]
    manifest_exists = os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=MANIFEST_HEADER)
    if not manifest_exists:
        writer.writeheader()

    processed = skipped = failed = 0
    failures = []
    t0 = time.time()
    for i, ip in enumerate(images):
        stem = os.path.splitext(os.path.basename(ip))[0]
        if not args.overwrite and _outputs_exist(cfg, stem):
            skipped += 1
            continue
        try:
            row = process_one(teacher, cfg, ip)
            writer.writerow(row)
            mf.flush()
            processed += 1
            if processed <= 5 or processed % 50 == 0:
                rate = processed / max(1e-6, time.time() - t0)
                logger.info(
                    "[%d/%d] %s  objs=%d nearest=%sm  (%.2f img/s)",
                    i + 1, len(images), stem, row["num_objects"], row["nearest_m"], rate,
                )
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            failed += 1
            failures.append({"image": ip, "error": str(e)[:300]})
            logger.exception("FAILED %s", stem)
    mf.close()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": teacher.loaded_checkpoint,
        "num_selected": len(images),
        "processed": processed,
        "skipped_existing": skipped,
        "failed": failed,
        "elapsed_s": round(time.time() - t0, 1),
        "failures": failures[:50],
        "config": cfg,
    }
    with open(cfg["output"]["run_summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("DONE processed=%d skipped=%d failed=%d in %.1fs",
                processed, skipped, failed, time.time() - t0)


if __name__ == "__main__":
    main()
