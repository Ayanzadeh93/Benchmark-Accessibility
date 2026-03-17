"""High-level detection pipeline (Phase 2).

Design goals:
- Reuse existing VLM (Florence-2/Qwen/GPT-4o) to propose an image-specific vocabulary.
- Run open-vocab detection with GroundingDINO (transformers).
- Save outputs in a reproducible, paper-friendly format (YOLO txt + JSON).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import get_huggingface_token, get_openai_api_key
from simple_vlm_integration import SimpleVLMIntegration

from .grounding_dino import GroundingDINOConfig, GroundingDINODetector
from .yolo_world import YOLOWorldDetector
from .ensemble import EnsembleDetector
from .utils import nms_detections
from .vocab import build_detection_classes_from_vlm, load_vlm_result_json

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


@dataclass
class DetectionRunConfig:
    """Config for running detection on a set of images."""

    # VLM for vocabulary (Florence-2, Qwen, or GPT-4o)
    vlm_model: str = "qwen"  # florence2|qwen|gpt4o|gpt5nano|gpt5mini
    vlm_device: str = "auto"  # auto|cuda|cpu
    vlm_api_key: Optional[str] = None

    # Detector (Phase-2)
    detector: str = "grounding_dino"  # grounding_dino, yolo_world, ensemble
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    yolo_model_size: str = "x"  # s, m, l, x
    conf_threshold: float = 0.15
    text_threshold: float = 0.20
    nms_iou: float = 0.50
    multi_prompt: bool = False

    # Vocabulary shaping
    max_classes: int = 50  # Increased to match VLM output limit
    use_existing_vlm_json: bool = True

    # Outputs
    save_annotated: bool = True


class DetectionPipeline:
    """Run VLM-driven open-vocabulary detection on images."""

    def __init__(self, cfg: DetectionRunConfig):
        self.cfg = cfg

        # Detector init (HF token helps on gated models; harmless otherwise)
        hf_token = get_huggingface_token()
        
        # Initialize detector based on config
        if cfg.detector == "ensemble":
            self.detector_impl = EnsembleDetector(
                yolo_model_size=cfg.yolo_model_size,
                gdino_model_id=cfg.gdino_model_id,
                hf_token=hf_token,
            )
        elif cfg.detector == "yolo_world":
            self.detector_impl = YOLOWorldDetector(cfg.yolo_model_size)
        else:  # grounding_dino (default)
            self.detector_impl = GroundingDINODetector(
                GroundingDINOConfig(model_id=cfg.gdino_model_id, device="auto", hf_token=hf_token)
            )

        # Lazy-init VLM (only if we can't load existing VLM JSON)
        self._vlm: Optional[SimpleVLMIntegration] = None

    def _get_vlm(self) -> SimpleVLMIntegration:
        if self._vlm is not None:
            return self._vlm
        api_key = self.cfg.vlm_api_key
        if self.cfg.vlm_model.lower() in {"gpt4o", "gpt-4o"}:
            api_key = api_key or get_openai_api_key()
        self._vlm = SimpleVLMIntegration(
            model_type=self.cfg.vlm_model,
            api_key=api_key,
            device=self.cfg.vlm_device,
        )
        return self._vlm

    def _infer_vlm_json_for_keyframe(self, image_path: Path) -> Optional[Path]:
        """Best-effort mapping: Keyframes/<stem>_keyframe.jpg -> VLM_analysis/<stem>_vlm.json."""
        try:
            if image_path.parent.name.lower() == "keyframes":
                parent = image_path.parent.parent
                vlm_dir = parent / "VLM_analysis"
                if not vlm_dir.exists():
                    return None
                stem = image_path.stem
                if stem.endswith("_keyframe"):
                    base = stem[: -len("_keyframe")]
                else:
                    base = stem
                candidate = vlm_dir / f"{base}_vlm.json"
                return candidate if candidate.exists() else None
        except Exception:
            return None
        return None

    def _load_or_compute_vocab(
        self,
        image_path: Path,
        vlm_json_dir: Optional[Path] = None,
    ) -> Tuple[List[str], Dict[str, Any], Optional[Path]]:
        """Return (classes, vlm_result, vlm_json_path_used)."""
        vlm_json_path: Optional[Path] = None
        vlm_result: Optional[Dict[str, Any]] = None

        if self.cfg.use_existing_vlm_json:
            if vlm_json_dir is not None:
                candidate = vlm_json_dir / f"{image_path.stem.replace('_keyframe', '')}_vlm.json"
                if candidate.exists():
                    vlm_json_path = candidate
            if vlm_json_path is None:
                vlm_json_path = self._infer_vlm_json_for_keyframe(image_path)
            if vlm_json_path is not None:
                vlm_result = load_vlm_result_json(str(vlm_json_path))

        if vlm_result is None:
            # Compute using VLM (objects only; skip artifacts for speed)
            vlm = self._get_vlm()
            vlm_result = vlm.analyze_keyframe(
                str(image_path),
                include_artifacts=False,
                include_accessibility=True,
            )

        classes = build_detection_classes_from_vlm(vlm_result or {}, max_classes=self.cfg.max_classes)
        return classes, (vlm_result or {}), vlm_json_path

    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for a class name (RGB format for PIL)."""
        import hashlib
        # Professional color palette (RGB for PIL)
        palette = [
            (0, 119, 190),    # Blue
            (255, 87, 34),    # Orange
            (76, 175, 80),    # Green
            (156, 39, 176),   # Purple
            (255, 193, 7),    # Amber
            (233, 30, 99),    # Pink
            (3, 169, 244),    # Light Blue
            (255, 152, 0),    # Deep Orange
            (0, 150, 136),    # Teal
            (63, 81, 181),    # Indigo
            (244, 67, 54),    # Red
            (121, 85, 72),    # Brown
        ]
        hash_val = int(hashlib.md5(class_name.lower().encode()).hexdigest(), 16)
        return palette[hash_val % len(palette)]

    def _draw_modern_bbox(
        self,
        image: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        label: str,
        confidence: float,
        color: Tuple[int, int, int],
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw modern bounding box with professional label styling using PIL.
        
        Args:
            image: RGB image array
            x1, y1, x2, y2: Bounding box coordinates
            label: Class label
            confidence: Confidence score
            color: RGB color tuple
            thickness: Box line thickness
            
        Returns:
            Image with drawn bounding box
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw bounding box with smooth lines
        box_thickness = max(2, thickness)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=box_thickness)
        
        # Prepare label text (professional format)
        label_text = f"{label}"
        conf_text = f"{confidence:.2f}"
        
        # Try to load a nice font, fallback to default
        font_size = max(12, int((x2 - x1) / 15))
        font_size = min(font_size, 16)  # Cap at 16
        
        font = None
        # Try multiple font paths (Windows, Linux, macOS)
        font_paths = [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            # Fallback to default font
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        
        # Calculate text dimensions
        bbox_label = draw.textbbox((0, 0), label_text, font=font)
        bbox_conf = draw.textbbox((0, 0), conf_text, font=font)
        
        label_width = bbox_label[2] - bbox_label[0]
        label_height = bbox_label[3] - bbox_label[1]
        conf_width = bbox_conf[2] - bbox_conf[0]
        conf_height = bbox_conf[3] - bbox_conf[1]
        
        # Total text block dimensions
        text_width = max(label_width, conf_width) + 8
        text_height = label_height + conf_height + 4
        
        # Position label above box (or inside if not enough space)
        text_x = x1
        text_y = y1 - text_height - 4
        
        if text_y < 0:
            # Place inside box at top
            text_y = y1 + 2
        
        # Draw semi-transparent background for label
        bg_x1 = text_x
        bg_y1 = text_y
        bg_x2 = text_x + text_width
        bg_y2 = text_y + text_height
        
        # Ensure background doesn't exceed image bounds
        bg_x2 = min(bg_x2, pil_image.width)
        bg_y2 = min(bg_y2, pil_image.height)
        
        # Draw background with darker color for contrast
        bg_color = tuple(max(0, min(255, c - 40)) for c in color)
        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=bg_color)
        
        # Draw border around label background
        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], outline=color, width=1)
        
        # Draw label text (white for contrast)
        draw.text((text_x + 4, text_y), label_text, font=font, fill=(255, 255, 255))
        
        # Draw confidence below label (slightly lighter)
        conf_y = text_y + label_height + 2
        draw.text((text_x + 4, conf_y), conf_text, font=font, fill=(220, 220, 220))
        
        # Convert back to numpy array
        return np.array(pil_image)

    def _annotate_image(self, image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw professional bounding boxes with labels and confidence scores using PIL."""
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            cls = str(det.get("class_name", "object"))
            conf = float(det.get("confidence", 0.0))
            
            # Get color for this class (RGB)
            color = self._get_color_for_class(cls)
            
            # Draw using PIL-based modern bbox
            image_rgb = self._draw_modern_bbox(
                image_rgb,
                x1, y1, x2, y2,
                cls,
                conf,
                color,
                thickness=2
            )
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        vlm_json_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run detection on a single image and write outputs."""
        img_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        vlm_dir = Path(vlm_json_dir) if vlm_json_dir else None

        classes, vlm_result, used_vlm_json = self._load_or_compute_vocab(img_path, vlm_dir)

        if not self.detector_impl.enabled:
            raise RuntimeError(f"{self.cfg.detector} detector is not available.")

        # Call detector with appropriate arguments
        if self.cfg.detector == "ensemble":
            detections, img_w, img_h = self.detector_impl.detect(
                str(img_path),
                classes,
                conf_threshold=self.cfg.conf_threshold,
                text_threshold=self.cfg.text_threshold,
                nms_iou=self.cfg.nms_iou,
            )
        elif self.cfg.detector == "yolo_world":
            detections, img_w, img_h = self.detector_impl.detect(
                str(img_path),
                classes,
                conf_threshold=self.cfg.conf_threshold,
            )
        else:  # grounding_dino
            detections, img_w, img_h = self.detector_impl.detect(
                str(img_path),
                classes,
                conf_threshold=self.cfg.conf_threshold,
                text_threshold=self.cfg.text_threshold,
                multi_prompt=self.cfg.multi_prompt,
                nms_iou=self.cfg.nms_iou,
            )
        
        # Apply additional NMS if needed (ensemble already does this)
        if self.cfg.detector != "ensemble":
            detections = nms_detections(detections, iou_threshold=self.cfg.nms_iou, class_aware=True)

        # Save outputs
        labels_dir = out_dir / "labels"
        meta_dir = out_dir / "metadata"
        vocab_dir = out_dir / "vocab"
        viz_dir = out_dir / "visualizations"
        for d in [labels_dir, meta_dir, vocab_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_annotated:
            viz_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        yolo_txt = labels_dir / f"{stem}.txt"
        meta_json = meta_dir / f"{stem}.json"
        vocab_json = vocab_dir / f"{stem}.json"
        viz_path = viz_dir / f"{stem}_detected.jpg"

        # YOLO format (bbox): class_id x_center y_center width height confidence
        with yolo_txt.open("w", encoding="utf-8") as f:
            for det in detections:
                f.write(
                    f"{int(det['class_id'])} "
                    f"{float(det['x_center']):.6f} "
                    f"{float(det['y_center']):.6f} "
                    f"{float(det['width']):.6f} "
                    f"{float(det['height']):.6f} "
                    f"{float(det['confidence']):.6f}\n"
                )
            f.flush()
        logger.info(f"  ✓ Saved labels: {yolo_txt.name} ({len(detections)} detections)")

        with vocab_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": str(img_path),
                    "timestamp": datetime.now().isoformat(),
                    "vlm_model": self.cfg.vlm_model,
                    "used_vlm_json": str(used_vlm_json) if used_vlm_json else None,
                    "classes": classes,
                },
                f,
                indent=2,
            )
            f.flush()
        logger.info(f"  ✓ Saved vocab: {vocab_json.name}")

        with meta_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": str(img_path),
                    "image_width": img_w,
                    "image_height": img_h,
                    "timestamp": datetime.now().isoformat(),
                    "detector": {
                        "name": self.cfg.detector,
                        "gdino_model_id": self.cfg.gdino_model_id,
                        "yolo_model_size": self.cfg.yolo_model_size if self.cfg.detector in ["yolo_world", "ensemble"] else None,
                        "conf_threshold": self.cfg.conf_threshold,
                        "text_threshold": self.cfg.text_threshold,
                        "multi_prompt": self.cfg.multi_prompt,
                        "nms_iou": self.cfg.nms_iou,
                    },
                    "vocab": classes,
                    "num_detections": len(detections),
                    "detections": detections,
                    "vlm_result": vlm_result,
                },
                f,
                indent=2,
            )
            f.flush()
        logger.info(f"  ✓ Saved metadata: {meta_json.name}")

        if self.cfg.save_annotated:
            img = cv2.imread(str(img_path))
            if img is not None:
                annotated = self._annotate_image(img, detections)
                cv2.imwrite(str(viz_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"  ✓ Saved visualization: {viz_path.name}")

        return {
            "image": str(img_path),
            "labels_txt": str(yolo_txt),
            "metadata_json": str(meta_json),
            "vocab_json": str(vocab_json),
            "num_detections": len(detections),
        }

    def process_dir(
        self,
        images_dir: str,
        output_dir: str,
        vlm_json_dir: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run detection on all images under a directory."""
        in_dir = Path(images_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts and not p.name.endswith("_detected.jpg")])
        if max_images is not None and int(max_images) > 0:
            images = images[: int(max_images)]

        from tqdm.auto import tqdm

        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        skipped = 0
        
        # Pre-create output directories to check for existing files
        labels_dir = out_dir / "labels"
        meta_dir = out_dir / "metadata"
        vocab_dir = out_dir / "vocab"
        viz_dir = out_dir / "visualizations"
        
        for idx, p in enumerate(tqdm(images, desc="Detection"), 1):
            try:
                # Check if visualization already exists (skip if it does)
                stem = p.stem
                viz_path = viz_dir / f"{stem}_detected.jpg"
                yolo_txt = labels_dir / f"{stem}.txt"
                meta_json = meta_dir / f"{stem}.json"
                vocab_json = vocab_dir / f"{stem}.json"
                
                # Skip if visualization exists (and other files exist too)
                if self.cfg.save_annotated and viz_path.exists():
                    if yolo_txt.exists() and meta_json.exists() and vocab_json.exists():
                        skipped += 1
                        logger.info(f"[{idx}/{len(images)}] Skipping {p.name} (already processed)")
                        continue
                
                logger.info(f"[{idx}/{len(images)}] Processing: {p.name}")
                result = self.process_image(str(p), str(out_dir), vlm_json_dir=vlm_json_dir)
                results.append(result)
                logger.info(f"  ✓ Completed: {p.name} → {result['num_detections']} detections saved\n")
            except Exception as e:
                logger.error(f"  ✗ Failed: {p.name} - {e}")
                failures.append({"image": str(p), "error": str(e)})

        summary = {
            "timestamp": datetime.now().isoformat(),
            "images_dir": str(in_dir),
            "output_dir": str(out_dir),
            "num_images": len(images),
            "processed": len(results),
            "skipped": skipped,
            "failed": len(failures),
            "failures": failures[:50],
        }
        with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

