"""Export annotation records to common fine-tuning formats.

Supported exports:
- llava: LLaVA-style conversations with <image> token
- alpaca: instruction/input/output style (plus image field)
- sharegpt: ShareGPT-style conversations
- consolidated: Merged JSON per image with all metadata
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional
from datetime import datetime

from .schemas import AnnotationRecord, DetectionMetadata, SegmentationMetadata, DepthMetadata


ExportFormat = Literal["llava", "alpaca", "sharegpt", "consolidated", "training", "all"]


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for training data export."""

    export_formats: List[str]
    create_image_copies: bool = False
    instruction: str = "Describe the scene for navigation assistance."
    system_prompt: Optional[str] = None
    annotation_version: str = "comprehensive"  # simplified|comprehensive


def _safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(annotation_dir: Path) -> Iterable[AnnotationRecord]:
    for p in sorted(annotation_dir.glob("*.json")):
        try:
            yield AnnotationRecord.model_validate(_safe_read_json(p))
        except Exception:
            # Skip malformed entries
            continue


def _copy_images(records: List[AnnotationRecord], images_dir: Path, images_out_dir: Path) -> Dict[str, str]:
    """Copy images and return a mapping from original image path to dataset-relative path."""
    images_out_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, str] = {}
    for r in records:
        src = Path(r.image)
        if not src.is_absolute():
            src = (images_dir / src).resolve()
        if not src.exists():
            continue
        dst = images_out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        mapping[str(src)] = str(Path("images") / dst.name)
    return mapping


def export_training_data(
    *,
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    cfg: ExportConfig,
) -> Dict[str, str]:
    """Export dataset files into `output_dir`.

    Returns a mapping {format: file_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(_iter_records(annotations_dir))

    image_path_map: Dict[str, str] = {}
    if cfg.create_image_copies:
        image_path_map = _copy_images(records, images_dir=images_dir, images_out_dir=output_dir / "images")

    def _dataset_image_path(r: AnnotationRecord) -> str:
        # Prefer copied path when enabled, else store relative path to images_dir.
        src = Path(r.image)
        src_abs = str(src.resolve()) if src.is_absolute() else str((images_dir / src).resolve())
        if cfg.create_image_copies and src_abs in image_path_map:
            return image_path_map[src_abs]
        try:
            if src.is_absolute():
                return str(Path(src_abs).relative_to(images_dir.resolve()))
            return str(src)
        except Exception:
            return src.name

    out_files: Dict[str, str] = {}
    formats = cfg.export_formats
    if "all" in formats:
        formats = ["llava", "alpaca", "sharegpt", "consolidated", "training"]
    elif "training" in formats:
        # When training is requested, also generate all other standard formats
        formats = ["llava", "alpaca", "sharegpt", "consolidated", "training"]

    def _filter_obstacles(obstacles: Any) -> List[Dict[str, Any]]:
        """Remove placeholder 'none' obstacles; return list of real obstacles."""
        if not obstacles:
            return []
        out = []
        for o in obstacles:
            d = o.model_dump() if hasattr(o, "model_dump") else (o if isinstance(o, dict) else {})
            obj_name = d.get("object", "")
            if obj_name and str(obj_name).lower() != "none":
                out.append(d)
        return out

    def _build_accessibility_block(record: AnnotationRecord) -> Optional[Dict[str, Any]]:
        if record.accessibility is not None:
            obs = _filter_obstacles(record.accessibility.risk_assessment.obstacles)
            return {
                "location": record.accessibility.location,
                "time": record.accessibility.time,
                "scene_description": _clean_training_text(record.accessibility.scene_description),
                "ground_text": record.accessibility.ground_text,
                "spatial_objects": record.accessibility.spatial_objects,
                "highlight": record.accessibility.highlight,
                "guidance": record.accessibility.guidance,
                "risk_assessment": {
                    "level": record.accessibility.risk_assessment.level,
                    "score": record.accessibility.risk_assessment.score,
                    "reason": record.accessibility.risk_assessment.reason,
                    "scene_summary": record.accessibility.risk_assessment.scene_summary,
                    "obstacles": obs,
                },
            }
        if record.meta.get("task_type") == "accessibility":
            risk_obs = record.meta.get("risk_obstacles", [])
            obs = [o for o in risk_obs if isinstance(o, dict) and str(o.get("object", "")).lower() != "none"]
            return {
                "location": record.meta.get("location", "Unknown"),
                "time": record.meta.get("time", "Unknown"),
                "scene_description": _clean_training_text(record.text),
                "ground_text": record.meta.get("ground_text", ""),
                "spatial_objects": record.meta.get("spatial_objects", []),
                "highlight": record.meta.get("highlight", []),
                "guidance": record.meta.get("guidance", ""),
                "risk_assessment": {
                    "level": record.meta.get("risk_level", "Medium"),
                    "score": record.meta.get("risk_score", 0.5),
                    "reason": record.meta.get("risk_reason", None),
                    "scene_summary": record.meta.get("ground_text", None),
                    "obstacles": obs,
                },
            }
        return None

    # LLaVA
    if "llava" in formats:
        llava_path = output_dir / "llava.json"
        data = []
        for i, r in enumerate(records):
            item: Dict[str, Any] = {
                "id": str(i),
                "image": _dataset_image_path(r),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{cfg.instruction}"},
                    {"from": "gpt", "value": r.text},
                ],
            }
            accessibility_block = _build_accessibility_block(r)
            if accessibility_block:
                item["accessibility"] = accessibility_block
            data.append(item)
        llava_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        out_files["llava"] = str(llava_path)

    # Alpaca-style JSON
    if "alpaca" in formats:
        alpaca_path = output_dir / "alpaca.json"
        data = []
        for r in records:
            item: Dict[str, Any] = {
                "instruction": cfg.instruction,
                "input": "<image>",
                "output": r.text,
                "image": _dataset_image_path(r),
            }
            accessibility_block = _build_accessibility_block(r)
            if accessibility_block:
                item["accessibility"] = accessibility_block
            data.append(item)
        alpaca_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        out_files["alpaca"] = str(alpaca_path)

    # ShareGPT-style
    if "sharegpt" in formats:
        sharegpt_path = output_dir / "sharegpt.json"
        data = []
        for i, r in enumerate(records):
            conversations = []
            if cfg.system_prompt:
                conversations.append({"from": "system", "value": str(cfg.system_prompt)})
            conversations.append({"from": "user", "value": cfg.instruction})
            conversations.append({"from": "assistant", "value": r.text})
            item: Dict[str, Any] = {
                "id": str(i),
                "image": _dataset_image_path(r),
                "conversations": conversations,
            }
            accessibility_block = _build_accessibility_block(r)
            if accessibility_block:
                item["accessibility"] = accessibility_block
            data.append(item)
        sharegpt_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        out_files["sharegpt"] = str(sharegpt_path)

    # Consolidated JSON (one file per image with all metadata merged)
    if "consolidated" in formats:
        consolidated_dir = output_dir / "consolidated"
        consolidated_dir.mkdir(parents=True, exist_ok=True)
        
        for r in records:
            consolidated_data = _build_consolidated_json(
                record=r,
                images_dir=images_dir,
                output_dir=output_dir,
                annotation_version=str(cfg.annotation_version),
            )
            
            # Use image filename (without extension) as JSON filename
            image_path = Path(r.image)
            json_name = image_path.stem + ".json"
            json_path = consolidated_dir / json_name
            
            json_path.write_text(
                json.dumps(consolidated_data, indent=4, ensure_ascii=False),
                encoding="utf-8"
            )
        
        out_files["consolidated"] = str(consolidated_dir)

    # Training JSON (clean version for fine-tuning, excludes unnecessary metadata)
    if "training" in formats:
        training_dir = output_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        for r in records:
            training_data = _build_training_json(
                record=r,
                images_dir=images_dir,
                output_dir=output_dir,
                annotation_version=str(cfg.annotation_version),
            )
            
            # Use image filename (without extension) as JSON filename
            image_path = Path(r.image)
            json_name = image_path.stem + ".json"
            json_path = training_dir / json_name
            
            json_path.write_text(
                json.dumps(training_data, indent=4, ensure_ascii=False),
                encoding="utf-8"
            )
        
        out_files["training"] = str(training_dir)

    return out_files


def _build_consolidated_json(
    record: AnnotationRecord,
    images_dir: Path,
    output_dir: Path,
    annotation_version: str = "comprehensive",
) -> Dict[str, Any]:
    """Build a consolidated JSON structure merging all metadata for a single image."""
    
    # Resolve image path
    image_path = Path(record.image)
    if not image_path.is_absolute():
        image_path = (images_dir / image_path).resolve()
    
    # Image info
    image_info: Dict[str, Any] = {
        "filename": image_path.name,
        "path": str(image_path),
        "metadata": {},
    }
    
    if image_path.exists():
        try:
            stat = image_path.stat()
            image_info["metadata"]["file_size"] = stat.st_size
            image_info["metadata"]["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            pass
    
    # Depth info
    depth_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
    }
    depth_metadata = None
    if record.sources.get("depth_metadata"):
        depth_path = Path(record.sources["depth_metadata"])
        if depth_path.exists():
            try:
                depth_metadata = DepthMetadata.model_validate(_safe_read_json(depth_path))
                if depth_metadata.raw_depth_path:
                    depth_file = Path(depth_metadata.raw_depth_path)
                    depth_info["filename"] = depth_file.name
                    depth_info["path"] = str(depth_file)
                    depth_info["exists"] = depth_file.exists()
            except Exception:
                pass
    
    # Annotation info (text file)
    annotation_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
    }
    annotation_text_path = output_dir / "annotations" / (image_path.stem + ".txt")
    if not annotation_text_path.exists():
        annotation_text_path = output_dir / "annotations" / (image_path.stem + ".json")
    
    if annotation_text_path.exists():
        annotation_info["filename"] = annotation_text_path.name
        annotation_info["path"] = str(annotation_text_path)
        annotation_info["exists"] = True
    
    # Label info (detection/segmentation objects)
    label_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
        "objects": [],
    }
    
    # Load detection metadata
    det_metadata = None
    if record.sources.get("detection_metadata"):
        det_path = Path(record.sources["detection_metadata"])
        if det_path.exists():
            try:
                det_metadata = DetectionMetadata.model_validate(_safe_read_json(det_path))
                label_info["filename"] = det_path.name
                label_info["path"] = str(det_path)
                label_info["exists"] = True
            except Exception:
                pass
    
    # Load segmentation metadata (prefer over detection if available)
    seg_metadata = None
    if record.sources.get("segmentation_metadata"):
        seg_path = Path(record.sources["segmentation_metadata"])
        if seg_path.exists():
            try:
                seg_metadata = SegmentationMetadata.model_validate(_safe_read_json(seg_path))
                if not label_info["exists"]:
                    label_info["filename"] = seg_path.name
                    label_info["path"] = str(seg_path)
                    label_info["exists"] = True
            except Exception:
                pass
    
    # Add image dimensions to image_info if available from metadata
    metadata_to_use = seg_metadata if seg_metadata else det_metadata
    if metadata_to_use and metadata_to_use.image_width and metadata_to_use.image_height:
        image_info["metadata"]["width"] = metadata_to_use.image_width
        image_info["metadata"]["height"] = metadata_to_use.image_height
    
    # Build objects list from detection/segmentation
    objects_list = []
    
    if metadata_to_use:
        img_w = metadata_to_use.image_width or 640
        img_h = metadata_to_use.image_height or 640
        
        for det in metadata_to_use.detections:
            obj_data: Dict[str, Any] = {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
            }
            
            if str(annotation_version).lower().strip() != "simplified":
                # Bounding box (pixel coordinates)
                if det.bbox_xyxy and len(det.bbox_xyxy) == 4:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    obj_data["bbox"] = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                    }
                    obj_data["center"] = {
                        "x": float((x1 + x2) / 2),
                        "y": float((y1 + y2) / 2),
                    }
                
                # Normalized coordinates
                if det.x_center is not None and det.y_center is not None:
                    obj_data["normalized"] = {
                        "cx": float(det.x_center),
                        "cy": float(det.y_center),
                        "width": float(det.width) if det.width else 0.0,
                        "height": float(det.height) if det.height else 0.0,
                    }
                
                # Size
                if det.width is not None and det.height is not None:
                    obj_data["size"] = {
                        "width": int(det.width * img_w) if det.width <= 1.0 else int(det.width),
                        "height": int(det.height * img_h) if det.height <= 1.0 else int(det.height),
                    }
            
            objects_list.append(obj_data)
    
    label_info["objects"] = objects_list
    
    # Build consolidated structure
    consolidated: Dict[str, Any] = {
        "image_info": image_info,
        "depth_info": depth_info,
        "annotation_info": annotation_info,
        "label_info": label_info,
        "model": record.meta.get("model", record.task),
        "model_response_raw": record.text,
        "model_response_parsed": record.meta.get("parsed_response", {}),
        "processed_at": record.timestamp,
    }
    
    # Add depth statistics if available
    if depth_metadata:
        consolidated["depth_stats"] = {
            "depth_min": depth_metadata.depth_min,
            "depth_max": depth_metadata.depth_max,
            "depth_mean": depth_metadata.depth_mean,
            "depth_std": depth_metadata.depth_std,
        }
    
    return consolidated


def _build_training_json(
    record: AnnotationRecord,
    images_dir: Path,
    output_dir: Path,
    annotation_version: str = "comprehensive",
) -> Dict[str, Any]:
    """Build a training-focused JSON structure matching the example format.
    
    Includes relative paths for portability but excludes:
    - Absolute file paths
    - File sizes, timestamps in metadata
    - Model names in descriptions
    """
    
    # Resolve image path
    image_path = Path(record.image)
    if not image_path.is_absolute():
        image_path = (images_dir / image_path).resolve()
    
    # Image info (matches example: filename, path, metadata with width/height)
    image_info: Dict[str, Any] = {
        "filename": image_path.name,
        "path": f"images/{image_path.name}",  # Relative path for training
        "metadata": {},
    }
    
    # Depth info (matches example: filename, path, exists)
    depth_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
    }
    depth_metadata = None
    if record.sources.get("depth_metadata"):
        depth_path = Path(record.sources["depth_metadata"])
        if depth_path.exists():
            try:
                depth_metadata = DepthMetadata.model_validate(_safe_read_json(depth_path))
                if depth_metadata.raw_depth_path:
                    depth_file = Path(depth_metadata.raw_depth_path)
                    depth_info["filename"] = depth_file.name
                    depth_info["path"] = f"depth/{depth_file.name}"  # Relative path
                    depth_info["exists"] = depth_file.exists()
            except Exception:
                pass
    
    # Annotation info (matches example: filename, path, exists)
    annotation_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
    }
    annotation_text_path = output_dir / "annotations" / (image_path.stem + ".txt")
    if not annotation_text_path.exists():
        annotation_text_path = output_dir / "annotations" / (image_path.stem + ".json")
    
    if annotation_text_path.exists():
        annotation_info["filename"] = annotation_text_path.name
        annotation_info["path"] = f"annotations/{annotation_text_path.name}"  # Relative path
        annotation_info["exists"] = True
    
    # Label info (matches example: filename, path, exists, objects)
    label_info: Dict[str, Any] = {
        "filename": None,
        "path": None,
        "exists": False,
        "objects": [],
    }
    
    # Load detection metadata
    det_metadata = None
    if record.sources.get("detection_metadata"):
        det_path = Path(record.sources["detection_metadata"])
        if det_path.exists():
            try:
                det_metadata = DetectionMetadata.model_validate(_safe_read_json(det_path))
                label_info["filename"] = det_path.name
                label_info["path"] = f"label/{det_path.name}"  # Relative path
                label_info["exists"] = True
            except Exception:
                pass
    
    # Load segmentation metadata (prefer over detection if available)
    seg_metadata = None
    if record.sources.get("segmentation_metadata"):
        seg_path = Path(record.sources["segmentation_metadata"])
        if seg_path.exists():
            try:
                seg_metadata = SegmentationMetadata.model_validate(_safe_read_json(seg_path))
                if not label_info["exists"]:
                    label_info["filename"] = seg_path.name
                    label_info["path"] = f"label/{seg_path.name}"  # Relative path
                    label_info["exists"] = True
            except Exception:
                pass
    
    # Add image dimensions to image_info if available from metadata
    metadata_to_use = seg_metadata if seg_metadata else det_metadata
    if metadata_to_use and metadata_to_use.image_width and metadata_to_use.image_height:
        image_info["metadata"]["width"] = metadata_to_use.image_width
        image_info["metadata"]["height"] = metadata_to_use.image_height
    
    # Build objects list from detection/segmentation
    objects_list = []
    
    if metadata_to_use:
        img_w = metadata_to_use.image_width or 640
        img_h = metadata_to_use.image_height or 640
        
        for det in metadata_to_use.detections:
            obj_data: Dict[str, Any] = {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
            }
            
            if str(annotation_version).lower().strip() != "simplified":
                # Bounding box (pixel coordinates)
                if det.bbox_xyxy and len(det.bbox_xyxy) == 4:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    obj_data["bbox"] = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                    }
                    obj_data["center"] = {
                        "x": float((x1 + x2) / 2),
                        "y": float((y1 + y2) / 2),
                    }
                
                # Normalized coordinates
                if det.x_center is not None and det.y_center is not None:
                    obj_data["normalized"] = {
                        "cx": float(det.x_center),
                        "cy": float(det.y_center),
                        "width": float(det.width) if det.width else 0.0,
                        "height": float(det.height) if det.height else 0.0,
                    }
                
                # Size
                if det.width is not None and det.height is not None:
                    obj_data["size"] = {
                        "width": int(det.width * img_w) if det.width <= 1.0 else int(det.width),
                        "height": int(det.height * img_h) if det.height <= 1.0 else int(det.height),
                    }
            
            objects_list.append(obj_data)
    
    label_info["objects"] = objects_list
    
    # Clean the model response text (remove model names, paths, etc.)
    cleaned_text = _clean_training_text(record.text)
    
    # Build training structure (matches example format exactly)
    training: Dict[str, Any] = {
        "image_info": image_info,
        "depth_info": depth_info,
        "annotation_info": annotation_info,
        "label_info": label_info,
        "model": record.meta.get("model", record.task),
        "model_response_raw": cleaned_text,
        "model_response_parsed": record.meta.get("parsed_response", {}),
        "processed_at": record.timestamp,  # Matches example
    }
    
    # Add depth stats if available (useful for training)
    if depth_metadata:
        training["depth_stats"] = {
            "depth_min": depth_metadata.depth_min,
            "depth_max": depth_metadata.depth_max,
            "depth_mean": depth_metadata.depth_mean,
            "depth_std": depth_metadata.depth_std,
        }
    
    # Add accessibility data if available (for blind user assistance)
    def _obstacles_for_training(rec: AnnotationRecord) -> List[Dict[str, Any]]:
        if rec.accessibility is not None:
            return [
                o.model_dump() for o in rec.accessibility.risk_assessment.obstacles
                if o.object and str(o.object).lower() != "none"
            ]
        obs = rec.meta.get("risk_obstacles", [])
        return [o for o in obs if isinstance(o, dict) and str(o.get("object", "")).lower() != "none"]

    if record.accessibility is not None:
        training["accessibility"] = {
            "location": record.accessibility.location,
            "time": record.accessibility.time,
            "scene_description": _clean_training_text(record.accessibility.scene_description),
            "ground_text": record.accessibility.ground_text,
            "spatial_objects": record.accessibility.spatial_objects,
            "highlight": record.accessibility.highlight,
            "guidance": record.accessibility.guidance,
            "risk_assessment": {
                "level": record.accessibility.risk_assessment.level,
                "score": record.accessibility.risk_assessment.score,
                "obstacles": _obstacles_for_training(record),
            },
        }
    elif record.meta.get("task_type") == "accessibility":
        training["accessibility"] = {
            "location": record.meta.get("location", "Unknown"),
            "time": record.meta.get("time", "Unknown"),
            "scene_description": _clean_training_text(record.text),
            "ground_text": record.meta.get("ground_text", ""),
            "spatial_objects": record.meta.get("spatial_objects", []),
            "highlight": record.meta.get("highlight", []),
            "guidance": record.meta.get("guidance", ""),
            "risk_assessment": {
                "level": record.meta.get("risk_level", "Medium"),
                "score": record.meta.get("risk_score", 0.5),
                "obstacles": _obstacles_for_training(record),
            },
        }
    
    return training


def _clean_training_text(text: str) -> str:
    """Clean text for training by removing unnecessary metadata references."""
    if not text:
        return ""
    
    import re
    
    # Remove file paths
    text = re.sub(r"\b[a-zA-Z]:\\[^\s]+", "", text)  # Windows paths
    text = re.sub(r"\b/[^ \n\t]+", "", text)  # Unix paths
    
    # Remove filenames
    text = re.sub(r"\b[\w\-.]+\.(jpg|jpeg|png|webp|bmp|json|txt)\b", "", text, flags=re.IGNORECASE)
    
    # Remove model name mentions (but keep the structure)
    text = re.sub(r"\b(gpt[- ]?4o|qwen|claude|yolo|sam|grounding[- ]?dino)\b", "", text, flags=re.IGNORECASE)
    
    # Remove technical metadata patterns
    text = re.sub(r"\b(width|height|pixels?|resolution):\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s*(px|pixels?)\b", "", text, flags=re.IGNORECASE)
    
    # Remove confidence scores in text
    text = re.sub(r"\bconfidence:\s*[\d.]+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bconf:\s*[\d.]+\b", "", text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

