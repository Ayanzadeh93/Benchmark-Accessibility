#!/usr/bin/env python3
"""CLI for Phase-3 segmentation (YOLOv8-seg).

Examples:
  # Segment keyframes produced by phase1
  python -m segmentation.run_segmentation --images-dir output/Airport_BWI/Keyframes --output-dir output/Airport_BWI/Segmentation --model-size x --conf 0.25
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import SegmentationPipeline, SegmentationRunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3: YOLOv8-seg instance segmentation")
    p.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p.add_argument("--output-dir", type=str, default="./segmentation_output", help="Output directory for segmentation results")

    p.add_argument("--model-size", type=str, default="x", choices=["s", "m", "l", "x"], help="YOLOv8-seg model size")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device: auto/cuda/cpu")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold (NMS)")

    p.add_argument("--no-annotated", action="store_true", help="Do not save annotated images")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = SegmentationRunConfig(
        yolo_model_size=args.model_size,
        yolo_device=args.device,
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        save_annotated=not bool(args.no_annotated),
    )

    pipeline = SegmentationPipeline(cfg)
    if not pipeline.enabled:
        raise SystemExit("YOLO-seg is not available. Install ultralytics in this environment.")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pipeline.process_dir(images_dir=str(images_dir), output_dir=str(out_dir), max_images=args.max_images)
    print(f"[OK] Segmentation complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

