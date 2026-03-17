#!/usr/bin/env python3
"""Project entrypoint (Phase 1 + Phase 2).

Phase 1: keyframe/quality pipeline (existing `keyfram_analysis.py`)
Phase 2: detection on keyframes/images (new `detection/`)

Examples:
  # Phase 2 only (run detection on existing keyframes)
  python main.py phase2 --images-dir output/Airport_BWI/Keyframes --vlm-json-dir output/Airport_BWI/VLM_analysis --output-dir output/Airport_BWI/Detection

  # Phase 1 only (video mode - same as running keyfram_analysis.py directly)
  python main.py phase1 "C:\\path\\to\\video.MOV" --output ./output --enable-vlm --vlm-model florence2 --vlm-device cuda

  # Phase 1 image mode (process existing images with VLM, skip video extraction)
  python main.py phase1 --images-dir "C:\\path\\to\\images" --output-dir ./output --vlm-model florence2 --vlm-device cuda
  
  # Phase 1 image mode with GPT-5 Nano (cheaper API alternative)
  python main.py phase1 --images-dir "C:\\path\\to\\images" --output-dir ./output --vlm-model gpt5nano

  # Both (minimal orchestration): run phase1, then detect on produced keyframes
  python main.py both "C:\\path\\to\\video.MOV" --output ./output --vlm-model florence2 --vlm-device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_phase1_passthrough(argv: list[str]) -> int:
    """Run phase1 by forwarding args to keyfram_analysis.py (keeps full CLI features)."""
    cmd = [sys.executable, "keyfram_analysis.py"] + argv
    return subprocess.call(cmd)


def _run_phase1_images(args) -> int:
    """Run phase1 VLM analysis on existing images (skip video extraction)."""
    import logging
    import json
    from simple_vlm_integration import SimpleVLMIntegration
    from config import get_openai_api_key, get_openrouter_api_key
    
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
    if getattr(args, "max_images", None):
        images = images[: int(args.max_images)]
    
    if not images:
        print(f"ERROR: No images found in {images_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(images)} images in {images_dir}")
    
    # Initialize VLM
    api_key = None
    if args.vlm_model.lower() in {"gpt4o", "gpt-4o", "gpt5nano", "gpt-5-nano", "gpt5-nano", "gpt5mini", "gpt-5-mini", "gpt5-mini"}:
        api_key = get_openai_api_key()
        if not api_key:
            print("ERROR: OpenAI API key not found. Set OPENAI_API_KEY in .env file or environment.", file=sys.stderr)
            return 1
    elif args.vlm_model.lower().startswith("openrouter_"):
        api_key = get_openrouter_api_key()
        if not api_key:
            print("ERROR: OpenRouter API key not found. Set OPENROUTER_API_KEY in .env file or environment.", file=sys.stderr)
            return 1
    
    vlm = SimpleVLMIntegration(
        model_type=args.vlm_model,
        api_key=api_key,
        device=args.vlm_device,
    )
    
    # Create VLM_analysis directory
    vlm_dir = output_dir / "VLM_analysis"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir.mkdir(parents=True, exist_ok=True)
    
    def _blur_flag(image_path: str, threshold: float = 60.0) -> dict:
        """Flag only truly blurry frames (low sensitivity)."""
        try:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"flag": False, "reason": "unknown"}
            score = cv2.Laplacian(img, cv2.CV_64F).var()
            is_blurry = score < float(threshold)
            return {"flag": bool(is_blurry), "reason": "blurry" if is_blurry else "clear"}
        except Exception:
            return {"flag": False, "reason": "unknown"}

    # Process each image
    from tqdm.auto import tqdm
    processed = 0
    failed = 0
    
    for img_path in tqdm(images, desc="VLM Analysis"):
        try:
            # Check if JSON already exists - skip to avoid overwriting and save API costs
            stem = img_path.stem
            vlm_json = vlm_dir / f"{stem}_vlm.json"
            if vlm_json.exists():
                # If JSON already exists, skip (pause/resume friendly; do NOT overwrite).
                # Works for all models (including OpenRouter).
                try:
                    with open(vlm_json, 'r', encoding='utf-8') as f:
                        existing = json.load(f)

                    # Accept any valid prior output (simplified OR comprehensive) and skip.
                    # Simplified: {"objects": [..], "num_objects": N, ...}
                    if isinstance(existing, dict):
                        if isinstance(existing.get("objects"), list) and "num_objects" in existing:
                            processed += 1
                            continue

                        # Alternative simplified legacy key
                        if isinstance(existing.get("objects_list"), list) and "num_objects" in existing:
                            processed += 1
                            continue

                        # Comprehensive: {"keyframe_path": ..., "model": ..., "objects": {"objects":[..], ...}, ...}
                        obj = existing.get("objects")
                        if isinstance(obj, dict) and isinstance(obj.get("objects"), list):
                            processed += 1
                            continue
                except (json.JSONDecodeError, KeyError):
                    # If JSON is invalid or old format, reprocess it
                    pass
            
            # Analyze image with VLM
            # Include artifacts for comprehensive version
            include_artifacts = args.vlm_version == "comprehensive"
            result = vlm.analyze_keyframe(
                str(img_path),
                include_artifacts=include_artifacts,
                include_accessibility=True,
            )
            
            # Check if result has errors (especially authentication errors)
            has_error = False
            if isinstance(result, dict):
                objects = result.get('objects', {})
                
                # Check for explicit error flags
                if isinstance(objects, dict) and objects.get('error'):
                    has_error = True
                
                # Check for authentication errors in error messages
                error_msg = str(result.get('error', ''))
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Unauthorized' in error_msg:
                    print(f"\nERROR: Invalid OpenAI API key detected. Please check your .env file.", file=sys.stderr)
                    print(f"Get a valid API key from: https://platform.openai.com/account/api-keys", file=sys.stderr)
                    return 1
                
                # Don't treat 0 objects as an error if API call succeeded
                # (GPT-5 Nano might legitimately return empty results for some images)
                if isinstance(objects, dict) and objects.get('error') is False and objects.get('num_objects', 0) == 0:
                    # This is fine - just no objects detected
                    has_error = False
            
            if has_error:
                failed += 1
                # Show cost even for failed calls (if available)
                cost_info = ""
                if args.vlm_model.lower() in {"gpt4o", "gpt-4o", "gpt5nano", "gpt-5-nano", "gpt5-nano", "gpt5mini", "gpt-5-mini", "gpt5-mini"}:
                    if hasattr(vlm, 'extractor') and hasattr(vlm.extractor, 'last_call_cost'):
                        cost = vlm.extractor.last_call_cost
                        if cost > 0:
                            cost_info = f" | Cost: ${cost:.6f}"
                # Include backend-provided error message if available
                err_detail = ""
                if isinstance(objects, dict):
                    err_detail = str(objects.get("error_message") or objects.get("error") or "").strip()
                if not err_detail:
                    err_detail = str(result.get("error") or "").strip() if isinstance(result, dict) else ""
                if err_detail:
                    print(f"  ✗ {img_path.name}: VLM analysis failed{cost_info} | {err_detail}", file=sys.stderr)
                else:
                    print(f"  ✗ {img_path.name}: VLM analysis failed{cost_info}", file=sys.stderr)
                continue
            
            # Build output JSON based on version
            if args.vlm_version == "comprehensive":
                # Comprehensive version: full VLM result with all fields
                # Structure matches the prior comprehensive format from keyfram_analysis.py
                from datetime import datetime
                
                # Get objects dict from result (already in correct format from extractor)
                if isinstance(objects, dict):
                    objects_dict = objects.copy()
                    # Ensure all required fields exist
                    if "categories" not in objects_dict:
                        objects_dict["categories"] = {
                            "signs": [],
                            "accessibility": [],
                            "people": [],
                            "furniture": [],
                            "technology": [],
                            "other": []
                        }
                    if "scene_description" not in objects_dict:
                        objects_dict["scene_description"] = ""
                    if "primary_focus" not in objects_dict:
                        objects_dict["primary_focus"] = ""
                    if "model" not in objects_dict:
                        objects_dict["model"] = result.get('model', f"{args.vlm_model.upper()}-VL")
                    if "timestamp" not in objects_dict:
                        objects_dict["timestamp"] = datetime.now().isoformat()
                else:
                    # Convert list to dict format (fallback)
                    objects_list = objects if isinstance(objects, list) else []
                    objects_dict = {
                        "objects": objects_list,
                        "num_objects": len(objects_list),
                        "categories": {
                            "signs": [],
                            "accessibility": [],
                            "people": [],
                            "furniture": [],
                            "technology": [],
                            "other": []
                        },
                        "scene_description": "",
                        "primary_focus": "",
                        "error": False,
                        "model": result.get('model', f"{args.vlm_model.upper()}-VL"),
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Build comprehensive format matching prior version
                comprehensive = {
                    "keyframe_path": str(img_path),
                    "model": args.vlm_model,
                    "objects": objects_dict,
                }
                
                # Add artifacts if available
                if include_artifacts and 'artifacts' in result:
                    comprehensive["artifacts"] = result.get('artifacts', {})
                
                # Add blur flag
                comprehensive["blur"] = _blur_flag(str(img_path))
                
                output_data = comprehensive
            else:
                # Simplified version: objects list only + blur flag
                if isinstance(objects, dict):
                    objects_list = objects.get("objects", [])
                elif isinstance(objects, list):
                    objects_list = objects
                else:
                    objects_list = []

                output_data = {
                    "objects": objects_list,
                    "num_objects": len(objects_list),
                    "blur": _blur_flag(str(img_path)),
                }

            # Save JSON (simplified or comprehensive)
            with open(vlm_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            processed += 1
            
            # Display cost per image if using GPT models
            cost_info = ""
            if args.vlm_model.lower() in {"gpt4o", "gpt-4o", "gpt5nano", "gpt-5-nano", "gpt5-nano", "gpt5mini", "gpt-5-mini", "gpt5-mini"}:
                if hasattr(vlm, 'extractor') and hasattr(vlm.extractor, 'last_call_cost'):
                    cost = vlm.extractor.last_call_cost
                    input_tokens, output_tokens = vlm.extractor.last_call_tokens
                    if cost > 0:
                        cost_info = f" | Cost: ${cost:.6f} | Tokens: {input_tokens}+{output_tokens}={input_tokens+output_tokens}"
            
            print(f"  OK {img_path.name} -> {vlm_json.name}{cost_info}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {img_path.name}: {e}", file=sys.stderr)
    
    print(f"\n[OK] VLM analysis complete: {processed}/{len(images)} processed, {failed} failed")
    print(f"Output: {vlm_dir}")
    
    # Display cost summary if using GPT models
    if args.vlm_model.lower() in {"gpt4o", "gpt-4o", "gpt5nano", "gpt-5-nano", "gpt5-nano", "gpt5mini", "gpt-5-mini", "gpt5-mini"}:
        try:
            if hasattr(vlm, 'extractor') and hasattr(vlm.extractor, 'cost_tracker') and vlm.extractor.cost_tracker:
                cost_summary = vlm.extractor.cost_tracker.get_summary()
                print(cost_summary)
        except Exception as e:
            print(f"Warning: Could not display cost summary: {e}", file=sys.stderr)
    
    return 0


def _run_phase2(args) -> int:
    """Run phase2 detection pipeline."""
    import logging
    from detection.pipeline import DetectionPipeline, DetectionRunConfig
    
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
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = pipeline.process_dir(
        images_dir=str(images_dir),
        output_dir=str(out_dir),
        vlm_json_dir=args.vlm_json_dir,
        max_images=args.max_images,
    )
    
    skipped = summary.get('skipped', 0)
    if skipped > 0:
        print(f"[OK] Detection complete: {summary['processed']}/{summary['num_images']} processed, {skipped} skipped, {summary['failed']} failed")
    else:
        print(f"[OK] Detection complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")
    return 0


def _run_phase3(args) -> int:
    """Run phase3 segmentation pipeline (YOLOv8-seg or SAM3)."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(getattr(args, "backend", "yolo")).lower()
    # Back-compat: if the user provided legacy SAM flags, switch backend to SAM3.
    if backend == "yolo" and (
        getattr(args, "detection_metadata_dir", None)
        or getattr(args, "sam_model", None)
        or getattr(args, "sam3_checkpoint", None)
        or getattr(args, "vlm_json_dir", None)
    ):
        backend = "sam3"

    if backend == "sam3":
        from segmentation.sam_pipeline import SAMSegmentationPipeline, SAMSegmentationRunConfig

        cfg = SAMSegmentationRunConfig(
            device=args.device,
            min_det_conf=float(args.min_det_conf),
            save_annotated=not bool(args.no_annotated),
            sam3_checkpoint=args.sam3_checkpoint,
            sam3_load_from_hf=not bool(args.sam3_no_hf),
            sam3_compile=bool(args.sam3_compile),
        )

        pipeline = SAMSegmentationPipeline(cfg)
        if not pipeline.enabled:
            print("ERROR: SAM3 is not available or failed to load weights (see logs for details).", file=sys.stderr)
            return 1

        det_meta_dir = Path(args.detection_metadata_dir) if args.detection_metadata_dir else None
        # Infer VLM directory from images directory if not provided
        vlm_dir = None
        if images_dir.name.lower() == "keyframes":
            vlm_dir = images_dir.parent / "VLM_analysis"
            if not vlm_dir.exists():
                vlm_dir = None

        summary = pipeline.process_dir(
            images_dir=str(images_dir),
            output_dir=str(out_dir),
            detection_metadata_dir=str(det_meta_dir) if det_meta_dir and det_meta_dir.exists() else None,
            vlm_json_dir=args.vlm_json_dir or (str(vlm_dir) if vlm_dir else None),
            max_images=args.max_images,
        )
    else:
        # Use YOLOv8-seg
        from segmentation.pipeline import SegmentationPipeline, SegmentationRunConfig

        cfg = SegmentationRunConfig(
            yolo_model_size=args.model_size,
            yolo_device=args.device,
            conf_threshold=float(args.conf),
            iou_threshold=float(args.iou),
            save_annotated=not bool(args.no_annotated),
        )

        pipeline = SegmentationPipeline(cfg)
        if not pipeline.enabled:
            print("ERROR: YOLO-seg is not available (install ultralytics in this environment).", file=sys.stderr)
            return 1

        summary = pipeline.process_dir(
            images_dir=str(images_dir),
            output_dir=str(out_dir),
            max_images=args.max_images,
        )

    print(f"[OK] Segmentation complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")
    return 0


def _run_phase4(args) -> int:
    """Run phase4 depth estimation pipeline (DepthAnything V2)."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from depth_estimation.pipeline import DepthEstimationPipeline, DepthRunConfig

    cfg = DepthRunConfig(
        device=args.device,
        encoder=args.encoder,
        checkpoint_path=args.depth_checkpoint,
        save_color=not bool(args.no_color),
        save_raw_depth=bool(args.save_raw_depth),
        cmap=args.cmap,
    )
    pipeline = DepthEstimationPipeline(cfg)
    if not pipeline.enabled:
        print("ERROR: DepthAnything V2 is not available or checkpoint is missing.", file=sys.stderr)
        return 1

    summary = pipeline.process_dir(
        images_dir=str(images_dir),
        output_dir=str(out_dir),
        max_images=args.max_images,
    )

    print(f"[OK] Depth estimation complete: {summary['processed']}/{summary['num_images']} processed, {summary['failed']} failed")
    print(f"Output: {out_dir}")
    return 0


# OpenRouter friendly name -> full model id (for Phase 5 annotation)
_OPENROUTER_MODEL_IDS = {
    "openrouter_qwen3_vl_8b": "qwen/qwen3-vl-8b-instruct",
    "openrouter_qwen3_vl_235b": "qwen/qwen3-vl-235b-a22b-instruct",
    "openrouter_qwen_vl_plus": "qwen/qwen-vl-plus",
    "openrouter_llama4_maverick": "meta-llama/llama-4-maverick",
    "openrouter_llama32_11b_vision": "meta-llama/llama-3.2-11b-vision-instruct",
    "openrouter_trinity": "arcee-ai/trinity-large-preview:free",
    "openrouter_molmo_8b": "allenai/molmo-2-8b:free",
    "openrouter_ministral_3b": "mistralai/ministral-3b-2512",
    "openrouter_gpt_oss_safeguard_20b": "openai/gpt-oss-safeguard-20b",
}


def _resolve_openrouter_model(value: str) -> str:
    """Resolve OpenRouter friendly name to full model id (e.g. openrouter_qwen3_vl_8b -> qwen/qwen3-vl-8b-instruct)."""
    v = (value or "").strip()
    if not v:
        return "qwen/qwen3-vl-8b-instruct"
    if "/" in v:
        return v
    return _OPENROUTER_MODEL_IDS.get(v, v)


def _run_annotate(args) -> int:
    """Run dataset annotation pipeline (Phase 5)."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from annotation.pipeline import AnnotationPipeline, AnnotationRunConfig

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = getattr(args, "instruction", None)
    task_type = str(getattr(args, "task", "caption")).lower().strip()
    if instruction is None:
        if task_type == "navigation":
            instruction = "Provide navigation guidance for this scene."
        elif task_type == "scene":
            instruction = "Provide a detailed, professional description of this scene."
        elif task_type == "accessibility":
            instruction = "Provide accessibility information for this scene including location, time, highlights, and navigation guidance."
        else:
            instruction = "Describe the scene for navigation assistance."

    system_prompt = getattr(args, "system_prompt", None)
    if system_prompt is None:
        if task_type == "navigation":
            system_prompt = (
                "You are a navigation assistant helping blind and visually impaired users navigate safely. "
                "Use clock positions (12=ahead, 3=right, 9=left) and distances in meters. "
                "Keep responses brief, structured, and safety-first."
            )
        elif task_type == "scene":
            system_prompt = (
                "You are a professional scene description assistant. Generate detailed, natural language "
                "descriptions of scenes using spatial positions. Never include file paths, model names, "
                "coordinates, or technical metadata in your descriptions."
            )
        elif task_type == "accessibility":
            system_prompt = (
                "You are an AI assistant helping blind and visually impaired users understand their surroundings. "
                "Provide location type, time of day, accessibility highlights (ramps, stairs, elevators, etc.), "
                "and clear navigation guidance using clock positions and distances in meters."
            )

    # For accessibility task, default to VLM (claude) for rich output; template produces short/minimal output
    caption_model = str(args.caption_model)
    if task_type == "accessibility" and caption_model == "template":
        caption_model = "claude"

    # Resolve OpenRouter model: friendly name -> full model id
    openrouter_arg = str(getattr(args, "openrouter_model", "qwen/qwen3-vl-8b-instruct"))
    openrouter_model_id = _resolve_openrouter_model(openrouter_arg)

    cfg = AnnotationRunConfig(
        task=str(getattr(args, "task", "caption")),
        caption_model=caption_model,
        caption_device=args.caption_device,
        caption_max_tokens=int(getattr(args, "caption_max_tokens", 1000)),
        gpt_model=str(getattr(args, "gpt_model", "gpt-4o")),
        export_formats=list(args.export_formats),
        create_image_copies=bool(args.create_image_copies),
        instruction=str(instruction),
        system_prompt=system_prompt,
        skip_existing=not bool(getattr(args, "no_skip_existing", False)),
        nav_backend=str(getattr(args, "nav_backend", "deterministic")),
        claude_model=str(getattr(args, "claude_model", "claude-3-7-sonnet-20250219")),
        openrouter_model=openrouter_model_id,
        nav_max_tokens=int(getattr(args, "nav_max_tokens", 1000)),
        nav_temperature=float(getattr(args, "nav_temperature", 0.2)),
        nav_include_depth_image=not bool(getattr(args, "nav_no_depth_image", False)),
        qwen_max_new_tokens=int(getattr(args, "qwen_max_new_tokens", 280)),
        nav_min_distance_m=float(getattr(args, "nav_min_distance_m", 0.5)),
        nav_max_distance_m=float(getattr(args, "nav_max_distance_m", 8.0)),
        nav_max_obstacles=int(getattr(args, "nav_max_obstacles", 6)),
        nav_stop_distance_m=float(getattr(args, "nav_stop_distance_m", 0.8)),
        nav_caution_distance_m=float(getattr(args, "nav_caution_distance_m", 2.0)),
        image_detail=str(getattr(args, "image_detail", "auto")),
        resize_720p=bool(getattr(args, "resize_720p", False)),
        use_bboxes=not bool(getattr(args, "no_bboxes", False)),
        annotation_version=str(getattr(args, "annotation_version", "comprehensive")),
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
    return 0


def _run_vqa_grouped(args) -> int:
    """Generate VQA grouped by question type (one file per question)."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ann_dir = Path(args.annotations_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotations-dir not found: {ann_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    from vqa.evaluation.generate_per_question import generate_per_question_vqa

    try:
        summary = generate_per_question_vqa(
            annotations_dir=str(ann_dir),
            output_dir=str(output_dir),
            images_dir=args.images_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            verbose=args.verbose,
            ground_truth_model=getattr(args, "ground_truth_model", None),
            per_image=getattr(args, "per_image", False),
            per_image_dir=getattr(args, "per_image_dir", None),
            skip_existing_per_image=getattr(args, "skip_existing_per_image", False),
        )
    except Exception as e:
        print(f"ERROR: VQA generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n{'='*60}")
    print("VQA GENERATION COMPLETE (GROUPED BY QUESTION)")
    print(f"{'='*60}")
    print(f"Processed: {summary['processed']} annotations")
    print(f"Failed: {summary['failed']}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated files:")
    for qid, count in summary['total_samples_per_question'].items():
        print(f"  {qid}.json: {count} samples")
    print(f"{'='*60}\n")

    return 0


def _run_vqa(args) -> int:
    """Run Phase 6: generate VQA evaluation dataset from Phase-5 annotations."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ann_dir = Path(args.annotations_dir)
    if not ann_dir.exists():
        print(f"ERROR: annotations-dir not found: {ann_dir}", file=sys.stderr)
        return 1

    from vqa.evaluation.pipeline import VQAEvalRunConfig, VQAEvaluationPipeline

    cfg = VQAEvalRunConfig(
        annotations_dir=str(ann_dir),
        output_dir=getattr(args, "output_dir", None),
        images_dir=getattr(args, "images_dir", None),
        max_samples=getattr(args, "max_samples", None),
        seed=int(getattr(args, "seed", 1337)),
        tuning_split=float(getattr(args, "tuning_split", 0.2)),
        ground_truth_model=getattr(args, "ground_truth_model", None),
    )

    pipeline = VQAEvaluationPipeline(cfg)
    try:
        out_dir, summary = pipeline.process()
    except Exception as e:
        print(f"ERROR: VQA generation failed: {e}", file=sys.stderr)
        return 1

    print(f"[OK] VQA dataset generated: {summary['generated']} samples, {summary['failed']} failures")
    print(f"Output: {out_dir}")
    print(f"Dataset: {summary['dataset_path']}")
    if "tuning_path" in summary:
        print(f"Tuning: {summary['tuning_path']}")
    if "evaluation_path" in summary:
        print(f"Evaluation: {summary['evaluation_path']}")
    return 0


def _run_vqa_eval(args) -> int:
    """Run VQA model evaluation: test models on VQA datasets."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    vqa_dataset_path = Path(args.vqa_dataset)
    if not vqa_dataset_path.exists():
        print(f"ERROR: VQA dataset not found: {vqa_dataset_path}", file=sys.stderr)
        return 1

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images-dir not found: {images_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from vqa.evaluation.model_evaluation import run_vqa_evaluation

    # Get API key
    api_key = None
    model_type = args.model_name
    if "openrouter" in model_type.lower():
        try:
            from config import get_openrouter_api_key
            api_key = get_openrouter_api_key()
        except Exception:
            pass

    try:
        result = run_vqa_evaluation(
            vqa_dataset_path=str(vqa_dataset_path),
            images_dir=str(images_dir),
            output_dir=str(output_dir),
            model_name=args.model_name,
            model_type=args.model_name,
            api_key=api_key,
            max_samples=args.max_samples,
            batch_mode=not args.no_batch_mode,
            save_predictions=not args.no_save_predictions,
            verbose=args.verbose,
        )

        print(f"\n{'='*60}")
        print("VQA MODEL EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Model: {result.model_name}")
        print(f"Overall Accuracy: {result.overall_accuracy:.4f}")
        print(f"Total Samples: {result.total_samples}")
        print(f"Correct: {result.total_correct}")
        print(f"Failed: {result.total_failed}")
        
        if result.overall_rouge1_f1 is not None:
            print(f"ROUGE-1 F1: {result.overall_rouge1_f1:.4f}")
        if result.overall_bleu is not None:
            print(f"BLEU: {result.overall_bleu:.4f}")
        if result.overall_bertscore_f1 is not None:
            print(f"BERT Score F1: {result.overall_bertscore_f1:.4f}")
        if result.overall_clip_text_score is not None:
            print(f"CLIP Score (text): {result.overall_clip_text_score:.4f}")
        if result.overall_clip_image_pred_score is not None:
            print(f"CLIP Score (image-pred): {result.overall_clip_image_pred_score:.4f}")
        
        if result.avg_inference_time_s is not None:
            print(f"Avg Inference Time: {result.avg_inference_time_s:.3f}s")
        
        print(f"\nPer-Question Accuracy:")
        for m in result.per_question_metrics:
            print(f"  {m.question_id}: {m.accuracy:.4f} ({m.correct}/{m.total})")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        print(f"ERROR: VQA evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _run_similarity(args) -> int:
    """Run CLIP similarity evaluation over images + VLM JSON."""
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from evaluation.similarity import SimilarityRunConfig, SimilarityPipeline

    cfg = SimilarityRunConfig(
        images_dir=Path(args.images_dir),
        vlm_json_dir=Path(args.vlm_json_dir),
        output_dir=Path(args.output_dir),
        clip_model=str(args.clip_model),
        device=str(args.device),
        min_similarity=float(args.min_similarity),
        top_k=args.top_k,
        max_objects=int(args.max_objects),
        keep_parts=bool(args.keep_parts),
        skip_existing=not bool(args.no_skip_existing),
        max_images=args.max_images,
    )
    pipeline = SimilarityPipeline(cfg)
    summary = pipeline.process_dir()

    print(
        f"[OK] Similarity complete: {summary['processed']} processed, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )
    print(f"Output: {summary['output_dir']}")
    return 0


def _run_llm_judge_eval(args) -> int:
    """Run LLM-as-judge evaluation for accessibility annotations."""
    import logging
    import os

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    from evaluation.llm_as_judge.pipeline import run_evaluation
    from config import get_openai_api_key, get_openrouter_api_key

    # Get API keys from environment or args
    openai_key = args.openai_api_key or get_openai_api_key() or os.getenv("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = args.openrouter_api_key or get_openrouter_api_key() or os.getenv("OPENROUTER_API_KEY")

    # Validate directories
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        print(f"ERROR: Annotations directory not found: {annotations_dir}", file=sys.stderr)
        return 1

    try:
        # Run evaluation
        results_df = run_evaluation(
            annotations_dir=str(annotations_dir),
            output_dir=args.output_dir,
            judge_models=args.judge_models,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            openrouter_api_key=openrouter_key,
            max_annotations=args.max_annotations,
            skip_existing=not args.no_skip_existing,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )

        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total evaluations: {len(results_df)}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nResults saved to:")
        print(f"  - CSV: {args.output_dir}/evaluation_results.csv")
        print(f"  - Excel: {args.output_dir}/evaluation_results.xlsx")
        print(f"  - Plots: {args.output_dir}/plots/")
        print(f"  - Individual results: {args.output_dir}/results/")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
        for criterion in criteria:
            if criterion in results_df.columns:
                mean = results_df[criterion].mean()
                std = results_df[criterion].std()
                print(f"{criterion.replace('_', ' ').title():25s}: {mean:5.2f} ± {std:4.2f}")

        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nature VQA project runner (phase1/phase2/phase3/phase4/both)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1", help="Phase 1: run keyframe/quality pipeline (video) or VLM analysis on images")
    p1.add_argument("--images-dir", type=str, default=None, help="Image directory mode: process images with VLM (skip video extraction)")
    p1.add_argument("--output-dir", type=str, default="./output", help="Output directory (used with --images-dir)")
    p1.add_argument(
        "--vlm-model",
        type=str,
        default="qwen",
        choices=[
            "florence2",
            "qwen",
            "llava",
            "gpt4o",
            "gpt5nano",
            "gpt5mini",
            "openrouter_trinity",
            "openrouter_llama32_11b_vision",
            "openrouter_molmo_8b",
            "openrouter_ministral_3b",
            "openrouter_gpt_oss_safeguard_20b",
            "openrouter_qwen3_vl_235b",
            "openrouter_qwen3_vl_8b",
            "openrouter_qwen_vl_plus",
            "openrouter_llama4_maverick",
        ],
        help="VLM model for analysis (local: florence2/qwen/llava; API: gpt*, openrouter_*)",
    )
    p1.add_argument("--vlm-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Florence-2/Qwen VLM")
    p1.add_argument("--vlm-version", type=str, default="simplified", choices=["simplified", "comprehensive"], help="Output format: simplified (objects list only) or comprehensive (full VLM result with categories, scene_description, artifacts, etc.)")
    p1.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug / rate-limit safe)")
    p1.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p1.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to keyfram_analysis.py (when not using --images-dir)")

    p2 = sub.add_parser("phase2", help="Phase 2: run detection on keyframes/images")
    p2.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p2.add_argument("--output-dir", type=str, default="./detection_output", help="Output directory for detection results")
    p2.add_argument("--vlm-json-dir", type=str, default=None, help="Directory with existing VLM JSON files (optional)")
    p2.add_argument(
        "--vlm-model",
        type=str,
        default="qwen",
        choices=[
            "florence2",
            "qwen",
            "llava",
            "gpt4o",
            "gpt5nano",
            "gpt5mini",
            "openrouter_trinity",
            "openrouter_llama32_11b_vision",
            "openrouter_molmo_8b",
            "openrouter_ministral_3b",
            "openrouter_gpt_oss_safeguard_20b",
            "openrouter_qwen3_vl_235b",
            "openrouter_qwen3_vl_8b",
            "openrouter_qwen_vl_plus",
            "openrouter_llama4_maverick",
        ],
        help="VLM used to build per-image vocab (local: florence2/qwen/llava; API: gpt*, openrouter_*)",
    )
    p2.add_argument("--vlm-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Florence-2/Qwen VLM")
    p2.add_argument("--max-classes", type=int, default=50, help="Max classes per image prompt (default: 50 to match VLM output)")
    p2.add_argument("--detector", type=str, default="grounding_dino", choices=["grounding_dino", "yolo_world", "ensemble"], help="Detector: grounding_dino, yolo_world, or ensemble (both combined)")
    p2.add_argument("--yolo-model-size", type=str, default="x", choices=["s", "m", "l", "x"], help="YOLO-World model size (only used if --detector yolo_world or ensemble)")
    p2.add_argument("--gdino-model", type=str, default="IDEA-Research/grounding-dino-base", help="GroundingDINO model id")
    p2.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    p2.add_argument("--text-threshold", type=float, default=0.20, help="Text matching threshold")
    p2.add_argument("--nms-iou", type=float, default=0.50, help="NMS IoU threshold")
    p2.add_argument("--multi-prompt", action="store_true", help="Run multiple prompt formats (slower, sometimes higher recall, only for grounding_dino)")
    p2.add_argument("--no-annotated", action="store_true", help="Do not save annotated images")
    p2.add_argument("--no-use-existing-vlm-json", action="store_true", help="Ignore existing VLM JSON and recompute vocab via VLM")
    p2.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p2.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p3 = sub.add_parser("phase3", help="Phase 3: run segmentation (YOLOv8-seg or SAM3) on images/keyframes")
    p3.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p3.add_argument("--output-dir", type=str, default="./segmentation_output", help="Output directory for segmentation results")
    p3.add_argument("--backend", type=str, default="yolo", choices=["yolo", "sam3"], help="Segmentation backend: yolo (default) or sam3")
    p3.add_argument("--model-size", type=str, default="x", choices=["s", "m", "l", "x"], help="YOLOv8-seg model size (only used when --backend yolo)")
    p3.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device: auto/cuda/cpu")
    p3.add_argument("--conf", type=float, default=0.25, help="Segmentation confidence threshold")
    p3.add_argument("--iou", type=float, default=0.45, help="Segmentation IoU threshold (NMS)")
    p3.add_argument("--vlm-json-dir", type=str, default=None, help="VLM_analysis directory (optional; inferred from images-dir if omitted)")
    p3.add_argument("--detection-metadata-dir", type=str, default=None, help="Directory with Phase2 Detection/metadata JSON (optional; if omitted, SAM3 will use VLM_analysis object lists as text prompts)")
    # Back-compat shim: this flag used to select SAM(1) variants; now it simply enables SAM3 mode.
    p3.add_argument("--sam-model", type=str, default=None, help="DEPRECATED: kept for backward compatibility; providing this flag enables SAM3 backend")
    p3.add_argument("--sam3-checkpoint", type=str, default=None, help="Optional SAM3 checkpoint path (if omitted, SAM3 will load from HuggingFace)")
    p3.add_argument("--sam3-no-hf", action="store_true", help="Do not load SAM3 weights from HuggingFace (requires --sam3-checkpoint)")
    p3.add_argument("--sam3-compile", action="store_true", help="Enable SAM3 compilation/optimization (may use Triton; slower first run)")
    p3.add_argument("--min-det-conf", type=float, default=0.15, help="SAM3 confidence threshold (also used to filter prompts when detection metadata is provided)")
    p3.add_argument("--no-annotated", action="store_true", help="Do not save annotated images")
    p3.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p3.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p4 = sub.add_parser("phase4", help="Phase 4: run depth estimation (DepthAnything V2) on images/keyframes")
    p4.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p4.add_argument("--output-dir", type=str, default="./depth_output", help="Output directory for depth results")
    p4.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="DepthAnything V2 encoder size")
    p4.add_argument("--depth-checkpoint", type=str, default=None, help="Path to DepthAnything V2 checkpoint (depth_anything_v2_<encoder>.pth)")
    p4.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device: auto/cuda/cpu")
    p4.add_argument("--cmap", type=str, default="turbo", help="Matplotlib colormap for visualization (e.g., turbo, plasma, magma)")
    p4.add_argument("--no-color", action="store_true", help="Do not save colorized depth visualizations")
    p4.add_argument("--save-raw-depth", action="store_true", help="Also save raw depth .npy arrays")
    p4.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p4.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p5 = sub.add_parser("annotate", help="Phase 5: generate textual annotations + export training data")
    p5.add_argument("--images-dir", type=str, required=True, help="Directory containing images (e.g., output/.../Keyframes)")
    p5.add_argument("--output-dir", type=str, default="./annotation_output", help="Output directory for annotations + exports")
    p5.add_argument("--detection-metadata-dir", type=str, default=None, help="Phase-2 Detection/metadata directory (optional)")
    p5.add_argument("--segmentation-metadata-dir", type=str, default=None, help="Phase-3 Segmentation/metadata directory (optional)")
    p5.add_argument("--depth-metadata-dir", type=str, default=None, help="Phase-4 Depth/metadata directory (optional)")
    p5.add_argument("--vlm-json-dir", type=str, default=None, help="VLM_analysis directory (optional)")
    p5.add_argument("--task", type=str, default="caption", choices=["caption", "navigation", "scene", "accessibility"], help="Annotation task. For accessibility, uses Claude by default for rich output (spatial_objects, obstacles, highlights).")
    p5.add_argument("--caption-model", type=str, default="template", choices=["template", "qwen", "gpt", "claude", "openrouter"], help="Caption generator backend (use claude/openrouter for rich accessibility output)")
    p5.add_argument("--caption-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Qwen caption model")
    p5.add_argument("--caption-max-tokens", type=int, default=1000, help="Max tokens for GPT/Qwen caption generation")
    p5.add_argument(
        "--image-detail",
        type=str,
        default="auto",
        choices=["auto", "low", "high"],
        help="Vision detail level for GPT image inputs (auto/low/high). low is cheapest.",
    )
    p5.add_argument(
        "--resize-720p",
        action="store_true",
        help="Resize each image to fit within 1280x720 before sending to GPT/Claude.",
    )
    p5.add_argument(
        "--no-bboxes",
        action="store_true",
        help="Do not use bbox-derived spatial positions in accessibility prompts/outputs (clock+distance only).",
    )
    p5.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4o",
        choices=[
            "gpt-4o",
            "gpt-5.2-2025-12-11",
            "gpt-5.1-2025-11-13",
            "gpt-5-mini-2025-08-07",
        ],
        help="GPT model id for caption/scene/navigation",
    )
    p5.add_argument("--export-formats", type=str, nargs="+", default=["all"], choices=["llava", "alpaca", "sharegpt", "consolidated", "training", "all"])
    p5.add_argument("--create-image-copies", action="store_true", help="Copy images into training_data/images for portability")
    p5.add_argument("--instruction", type=str, default=None, help="User instruction used in exports (default depends on --task)")
    p5.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt used in ShareGPT export")
    p5.add_argument("--nav-backend", type=str, default="openrouter", choices=["deterministic", "auto", "claude", "qwen", "gpt", "openrouter"], help="Navigation generation backend (default: openrouter = Qwen instruct via OpenRouter)")
    p5.add_argument(
        "--claude-model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        choices=[
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        help="Claude model id (Anthropic) - claude-sonnet-4-20250514 is Claude 4.5 Sonnet",
    )
    p5.add_argument(
        "--openrouter-model",
        type=str,
        default="qwen/qwen3-vl-8b-instruct",
        help="OpenRouter model for caption/navigation/accessibility. Default: qwen/qwen3-vl-8b-instruct. Also accepts: openrouter_qwen3_vl_8b, openrouter_qwen3_vl_235b, openai/gpt-oss-safeguard-20b",
    )
    p5.add_argument("--nav-max-tokens", type=int, default=1000, help="Max tokens for Claude/GPT/OpenRouter navigation response")
    p5.add_argument("--nav-temperature", type=float, default=0.2, help="Temperature for Claude navigation response")
    p5.add_argument("--nav-no-depth-image", action="store_true", help="Do not include depth image in Claude prompt (if depth is available)")
    p5.add_argument("--qwen-max-new-tokens", type=int, default=280, help="Max new tokens for Qwen navigation response")
    p5.add_argument("--nav-min-distance-m", type=float, default=0.5, help="Navigation: min distance for depth mapping")
    p5.add_argument("--nav-max-distance-m", type=float, default=8.0, help="Navigation: max distance for depth mapping")
    p5.add_argument("--nav-max-obstacles", type=int, default=6, help="Navigation: max obstacles listed")
    p5.add_argument("--nav-stop-distance-m", type=float, default=0.8, help="Navigation: stop threshold for hazards")
    p5.add_argument("--nav-caution-distance-m", type=float, default=2.0, help="Navigation: caution threshold")
    p5.add_argument("--no-skip-existing", action="store_true", help="Overwrite existing annotations (default: skip already-processed images)")
    p5.add_argument(
        "--annotation-version",
        type=str,
        default="comprehensive",
        choices=["simplified", "comprehensive"],
        help="Annotation export format (simplified removes bbox/centroid fields in exports)",
    )
    p5.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p5.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p_eval = sub.add_parser("evaluation", help="CLIP similarity evaluation for VLM object lists")
    p_eval.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    p_eval.add_argument("--vlm-json-dir", type=str, required=True, help="Directory containing VLM_analysis JSON files")
    p_eval.add_argument("--output-dir", type=str, default="./similarity_output", help="Output directory for similarity results")
    p_eval.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model name (e.g., ViT-B/32)")
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for CLIP model")
    p_eval.add_argument("--min-sim", type=float, default=0.18, dest="min_similarity", help="Minimum similarity threshold")
    p_eval.add_argument("--top-k", type=int, default=None, help="Keep top-K objects by similarity")
    p_eval.add_argument("--max-objects", type=int, default=50, help="Max objects to evaluate per image")
    p_eval.add_argument("--keep-parts", action="store_true", help="Keep granular parts (e.g., button, zipper)")
    p_eval.add_argument("--no-skip-existing", action="store_true", help="Do not skip images with existing similarity JSON")
    p_eval.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")
    p_eval.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p_llm_judge = sub.add_parser("eval", help="LLM-as-judge evaluation for accessibility annotations")
    p_llm_judge.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Directory containing annotation JSON files"
    )
    p_llm_judge.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )
    p_llm_judge.add_argument(
        "--judge-models",
        type=str,
        nargs="+",
        required=True,
        help="Judge model(s) to use (e.g., gpt-4o, claude-sonnet-4-20250514, openrouter:qwen/qwen3-vl-235b-a22b-instruct)"
    )
    p_llm_judge.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    p_llm_judge.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    p_llm_judge.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    p_llm_judge.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Limit number of annotations to evaluate (for testing)"
    )
    p_llm_judge.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip already-evaluated annotations (re-evaluate all)"
    )
    p_llm_judge.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)"
    )
    p_llm_judge.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Max tokens for LLM response (default: 1500)"
    )
    p_llm_judge.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p6 = sub.add_parser("vqa", help="Phase 6: generate VQA evaluation dataset from Phase-5 annotations")
    p6.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Phase-5 annotation output directory (or its `annotations/` subdir)",
    )
    p6.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for VQA datasets (default: <run>/VQA/evaluation inferred from annotations-dir)",
    )
    p6.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional images directory; if provided, VQA items will include absolute image paths",
    )
    p6.add_argument("--seed", type=int, default=1337, help="Deterministic seed for option shuffling/distractors")
    p6.add_argument("--max-samples", type=int, default=None, help="Limit number of generated samples (debug)")
    p6.add_argument("--tuning-split", type=float, default=0.2, help="Fraction of samples reserved for tuning (0-1)")
    p6.add_argument(
        "--ground-truth-model",
        type=str,
        default=None,
        help="VLM to generate reference answers (e.g. openrouter_qwen3_vl_235b, openrouter_qwen3_vl_8b, openrouter_llama32_11b_vision). Requires --images-dir.",
    )
    p6.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p_vqa_grouped = sub.add_parser("vqa-grouped", help="Generate VQA grouped by question type (one file per question)")
    p_vqa_grouped.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Directory containing annotation JSON files"
    )
    p_vqa_grouped.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-question JSON files"
    )
    p_vqa_grouped.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional images directory (adds absolute image paths to output)"
    )
    p_vqa_grouped.add_argument("--max-samples", type=int, default=None, help="Limit number of annotations to process (debug)")
    p_vqa_grouped.add_argument("--seed", type=int, default=1337, help="Deterministic seed for question generation")
    p_vqa_grouped.add_argument(
        "--ground-truth-model",
        type=str,
        default=None,
        help="VLM to generate reference answers (e.g. openrouter_qwen3_vl_235b, openrouter_qwen3_vl_8b, openrouter_llama32_11b_vision). Requires --images-dir.",
    )
    p_vqa_grouped.add_argument(
        "--per-image",
        action="store_true",
        help="Also write per-image JSON files (one file with all questions per image).",
    )
    p_vqa_grouped.add_argument(
        "--per-image-dir",
        type=str,
        default=None,
        help="Optional output directory for per-image JSON files (default: <output-dir>/per_image).",
    )
    p_vqa_grouped.add_argument(
        "--skip-existing-per-image",
        action="store_true",
        help="Reuse existing per-image VQA JSONs when present (do not overwrite, no new model calls).",
    )
    p_vqa_grouped.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p_vqa_eval = sub.add_parser("vqa-eval", help="Evaluate models on VQA datasets (compute accuracy, ROUGE, BLEU)")
    p_vqa_eval.add_argument(
        "--vqa-dataset",
        type=str,
        required=True,
        help="Path to VQA dataset JSON (per-question file, per_image_all.json, or any VQA JSON)",
    )
    p_vqa_eval.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    p_vqa_eval.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results (e.g. .../vqa/T1)",
    )
    p_vqa_eval.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model to evaluate (e.g. openrouter_llama4_maverick, openrouter_qwen3_vl_235b, openrouter_qwen3_vl_8b, openrouter_llama32_11b_vision)",
    )
    p_vqa_eval.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    p_vqa_eval.add_argument(
        "--no-batch-mode",
        action="store_true",
        help="Disable batch mode (1 question per API call instead of all questions per image). Batch mode is ON by default and 6x cheaper.",
    )
    p_vqa_eval.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not save individual predictions (only summary)",
    )
    p_vqa_eval.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    both = sub.add_parser("both", help="Run phase1 then phase2 (minimal orchestration)")
    both.add_argument("source", type=str, help="Video path or YouTube URL")
    both.add_argument("--output", type=str, default="./output", help="Output base directory")
    both.add_argument("--segment-duration", type=int, default=6, help="Segment duration (seconds)")
    both.add_argument("--fps", type=int, default=1, help="FPS to extract")
    both.add_argument("--vlm-model", type=str, default="qwen", choices=["florence2", "qwen", "llava", "gpt4o", "gpt5nano", "gpt5mini"], help="VLM model for phase1/phase2 (florence2=FASTEST free GPU, qwen=free GPU, llava=free GPU VQA, gpt4o=API, gpt5nano=cheaper API, gpt5mini=CHEAPEST API)")
    both.add_argument("--vlm-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for Florence-2/Qwen VLM")
    both.add_argument("--no-visualizations", action="store_true", help="Disable phase1 visualizations")
    both.add_argument("--no-heatmaps", action="store_true", help="Disable phase1 heatmaps")
    both.add_argument("--no-all-frames", action="store_true", help="Phase1: save only keyframes")
    both.add_argument("--conf", type=float, default=0.15, help="Phase2: detection confidence threshold")
    both.add_argument("--text-threshold", type=float, default=0.20, help="Phase2: text threshold")
    both.add_argument("--max-classes", type=int, default=50, help="Phase2: max classes per image prompt")
    both.add_argument("--multi-prompt", action="store_true", help="Phase2: multi prompt (slower)")
    both.add_argument("--no-annotated", action="store_true", help="Phase2: do not save annotated images")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.cmd == "phase1":
        # Check if image directory mode is requested
        if args.images_dir:
            return _run_phase1_images(args)
        else:
            return _run_phase1_passthrough(args.args)
    if args.cmd == "phase2":
        return _run_phase2(args)
    if args.cmd == "phase3":
        return _run_phase3(args)
    if args.cmd == "phase4":
        return _run_phase4(args)
    if args.cmd == "annotate":
        return _run_annotate(args)
    if args.cmd == "evaluation":
        return _run_similarity(args)
    if args.cmd == "eval":
        return _run_llm_judge_eval(args)
    if args.cmd == "vqa":
        return _run_vqa(args)
    if args.cmd == "vqa-grouped":
        return _run_vqa_grouped(args)
    if args.cmd == "vqa-eval":
        return _run_vqa_eval(args)

    # both: run phase1 using library call (so we can discover output_dir), then phase2 on keyframes
    from keyfram_analysis import VideoQualityPipeline, QualityConfig

    cfg = QualityConfig(segment_duration=int(args.segment_duration), fps_target=int(args.fps))
    cfg.save_visualizations = not bool(args.no_visualizations)
    cfg.save_heatmaps = not bool(args.no_heatmaps)
    cfg.save_all_frames = not bool(args.no_all_frames)

    pipeline = VideoQualityPipeline(
        cfg,
        enable_vlm=True,  # ensure VLM_analysis exists for phase2 vocab
        vlm_model=str(args.vlm_model),
        vlm_api_key=None,
        vlm_device=str(args.vlm_device),
    )
    _, _, out_dir = pipeline.process_video(args.source, args.output)

    keyframes_dir = Path(out_dir) / "Keyframes"
    vlm_json_dir = Path(out_dir) / "VLM_analysis"
    det_out = Path(out_dir) / "Detection"

    cmd = [
        sys.executable,
        "-m",
        "detection.run_detection",
        "--images-dir",
        str(keyframes_dir),
        "--output-dir",
        str(det_out),
        "--vlm-json-dir",
        str(vlm_json_dir),
        "--vlm-model",
        str(args.vlm_model),
        "--vlm-device",
        str(args.vlm_device),
        "--conf",
        str(args.conf),
        "--text-threshold",
        str(args.text_threshold),
        "--max-classes",
        str(args.max_classes),
    ]
    if args.multi_prompt:
        cmd.append("--multi-prompt")
    if args.no_annotated:
        cmd.append("--no-annotated")

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

