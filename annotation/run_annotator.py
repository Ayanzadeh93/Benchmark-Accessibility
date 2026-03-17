#!/usr/bin/env python3
"""CLI for Phase-5 dataset annotation.

Typical usage (use existing signals from Phase 1/2/3/4):
  python -m annotation.run_annotator ^
    --images-dir output/Airport_BWI/Keyframes ^
    --vlm-json-dir output/Airport_BWI/VLM_analysis ^
    --detection-metadata-dir output/Airport_BWI/Detection/metadata ^
    --output-dir output/Airport_BWI/Annotation

Notes:
- This tool NEVER writes model names/paths into the annotation text.
- Exports are written under `output-dir/training_data`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import AnnotationPipeline, AnnotationRunConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-5: Generate textual annotations for VLM fine-tuning")
    p.add_argument("--images-dir", type=str, required=True, help="Directory containing images/keyframes")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for annotations + exports")

    p.add_argument("--detection-metadata-dir", type=str, default=None, help="Phase-2 Detection/metadata directory (optional)")
    p.add_argument("--segmentation-metadata-dir", type=str, default=None, help="Phase-3 Segmentation/metadata directory (optional)")
    p.add_argument("--depth-metadata-dir", type=str, default=None, help="Phase-4 Depth/metadata directory (optional)")
    p.add_argument("--vlm-json-dir", type=str, default=None, help="VLM_analysis directory (optional)")

    p.add_argument("--task", type=str, default="caption", choices=["caption", "navigation"], help="Annotation task type")

    p.add_argument("--caption-model", type=str, default="template", choices=["template", "qwen"], help="Caption generator backend (caption task)")
    p.add_argument("--caption-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Qwen caption model")

    # Navigation knobs (used when --task navigation)
    p.add_argument("--nav-backend", type=str, default="deterministic", choices=["deterministic", "auto", "claude", "qwen"], help="Navigation generation backend")
    p.add_argument("--claude-model", type=str, default="claude-3-7-sonnet-20250219", help="Claude model id (Anthropic)")
    p.add_argument("--nav-max-tokens", type=int, default=1200, help="Max tokens for Claude navigation response")
    p.add_argument("--nav-temperature", type=float, default=0.2, help="Temperature for Claude navigation response")
    p.add_argument("--nav-no-depth-image", action="store_true", help="Do not include depth image in Claude prompt (if depth is available)")
    p.add_argument("--qwen-max-new-tokens", type=int, default=280, help="Max new tokens for Qwen navigation response")
    p.add_argument("--nav-min-distance-m", type=float, default=0.5, help="Minimum distance (meters) used for depth-to-meters mapping")
    p.add_argument("--nav-max-distance-m", type=float, default=8.0, help="Maximum distance (meters) used for depth-to-meters mapping")
    p.add_argument("--nav-max-obstacles", type=int, default=6, help="Maximum obstacles listed in the output")
    p.add_argument("--nav-stop-distance-m", type=float, default=0.8, help="Stop threshold (meters) for immediate hazards/blockers")
    p.add_argument("--nav-caution-distance-m", type=float, default=2.0, help="Caution threshold (meters) for medium risk")

    p.add_argument("--export-formats", type=str, nargs="+", default=["all"], choices=["llava", "alpaca", "sharegpt", "all"])
    p.add_argument("--create-image-copies", action="store_true", help="Copy images into training_data/images for portability")
    p.add_argument("--instruction", type=str, default=None, help="User instruction used in exports (default depends on --task)")
    p.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt used in ShareGPT export")

    p.add_argument("--skip-existing", action="store_true", help="Skip images that already have saved annotations")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = args.instruction
    if instruction is None:
        instruction = (
            "Provide navigation guidance for this scene."
            if str(args.task).lower().strip() == "navigation"
            else "Describe the scene for navigation assistance."
        )

    system_prompt = args.system_prompt
    if system_prompt is None and str(args.task).lower().strip() == "navigation":
        system_prompt = (
            "You are a navigation assistant helping blind and visually impaired users navigate safely. "
            "Use clock positions (12=ahead, 3=right, 9=left) and distances in meters. "
            "Keep responses brief, structured, and safety-first."
        )

    cfg = AnnotationRunConfig(
        task=str(args.task),
        caption_model=args.caption_model,
        caption_device=args.caption_device,
        export_formats=list(args.export_formats),
        create_image_copies=bool(args.create_image_copies),
        instruction=str(instruction),
        system_prompt=system_prompt,
        skip_existing=bool(args.skip_existing),
        nav_backend=str(args.nav_backend),
        claude_model=str(args.claude_model),
        nav_max_tokens=int(args.nav_max_tokens),
        nav_temperature=float(args.nav_temperature),
        nav_include_depth_image=not bool(args.nav_no_depth_image),
        qwen_max_new_tokens=int(args.qwen_max_new_tokens),
        nav_min_distance_m=float(args.nav_min_distance_m),
        nav_max_distance_m=float(args.nav_max_distance_m),
        nav_max_obstacles=int(args.nav_max_obstacles),
        nav_stop_distance_m=float(args.nav_stop_distance_m),
        nav_caution_distance_m=float(args.nav_caution_distance_m),
    )

    pipeline = AnnotationPipeline(cfg)
    summary = pipeline.process_dir(
        images_dir=str(images_dir),
        output_dir=str(out_dir),
        detection_metadata_dir=args.detection_metadata_dir,
        segmentation_metadata_dir=args.segmentation_metadata_dir,
        depth_metadata_dir=args.depth_metadata_dir,
        vlm_json_dir=args.vlm_json_dir,
        max_images=args.max_images,
    )

    print(f"[OK] Annotation complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

