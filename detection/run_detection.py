#!/usr/bin/env python3
"""CLI for Phase-2 detection.

Examples:
  # Run detection on keyframes produced by keyfram_analysis.py
  python -m detection.run_detection --images-dir output/Airport_BWI/Keyframes --vlm-json-dir output/Airport_BWI/VLM_analysis --output-dir output/Airport_BWI/Detection

  # Run detection on an arbitrary images directory (will run VLM to build vocab)
  python -m detection.run_detection --images-dir path/to/images --output-dir ./detection_out --vlm-model qwen --vlm-device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import DetectionPipeline, DetectionRunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-2: VLM-assisted open-vocabulary detection")
    p.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p.add_argument("--output-dir", type=str, default="./detection_output", help="Output directory for detection results")
    p.add_argument("--vlm-json-dir", type=str, default=None, help="Directory with existing VLM JSON files (optional)")

    p.add_argument("--vlm-model", type=str, default="qwen", choices=["qwen", "gpt4o"], help="VLM used to build per-image vocab")
    p.add_argument("--vlm-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Qwen VLM")
    p.add_argument("--max-classes", type=int, default=20, help="Max classes per image prompt (keeps detection fast)")
    p.add_argument("--no-use-existing-vlm-json", action="store_true", help="Ignore existing VLM JSON and recompute vocab via VLM")

    p.add_argument("--detector", type=str, default="grounding_dino", choices=["grounding_dino", "yolo_world", "ensemble"], help="Detector: grounding_dino, yolo_world, or ensemble (both combined)")
    p.add_argument("--yolo-model-size", type=str, default="x", choices=["s", "m", "l", "x"], help="YOLO-World model size (only used if --detector yolo_world or ensemble)")
    p.add_argument("--gdino-model", type=str, default="IDEA-Research/grounding-dino-base", help="GroundingDINO model id")
    p.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    p.add_argument("--text-threshold", type=float, default=0.20, help="Text matching threshold")
    p.add_argument("--nms-iou", type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--multi-prompt", action="store_true", help="Run multiple prompt formats (slower, sometimes higher recall, only for grounding_dino)")

    p.add_argument("--no-annotated", action="store_true", help="Do not save annotated images")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = DetectionRunConfig(
        vlm_model=args.vlm_model,
        vlm_device=args.vlm_device,
        detector=args.detector,
        yolo_model_size=args.yolo_model_size,
        gdino_model_id=args.gdino_model,
        conf_threshold=float(args.conf),
        text_threshold=float(args.text_threshold),
        nms_iou=float(args.nms_iou),
        multi_prompt=bool(args.multi_prompt),
        max_classes=int(args.max_classes),
        use_existing_vlm_json=not bool(args.no_use_existing_vlm_json),
        save_annotated=not bool(args.no_annotated),
    )

    pipeline = DetectionPipeline(cfg)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pipeline.process_dir(
        images_dir=str(images_dir),
        output_dir=str(out_dir),
        vlm_json_dir=args.vlm_json_dir,
        max_images=args.max_images,
    )

    print(f"[OK] Detection complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

