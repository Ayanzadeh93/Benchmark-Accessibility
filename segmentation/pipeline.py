"""High-level segmentation pipeline (Phase 3).

Design goals:
- Keep it simple and batch-friendly
- Use YOLOv8-seg (Ultralytics) for instance segmentation on keyframes/images
- Save YOLO polygon labels + JSON metadata + optional annotated images
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .utils import get_color_for_class, overlay_mask_rgb
from .yolo_seg import YOLOSegConfig, YOLOSegDetector

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


@dataclass
class SegmentationRunConfig:
    """Config for running YOLO segmentation on a set of images."""

    yolo_model_size: str = "x"  # s|m|l|x
    yolo_device: str = "auto"  # auto|cuda|cpu
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    save_annotated: bool = True


class SegmentationPipeline:
    """Run YOLOv8-seg on images and write outputs."""

    def __init__(self, cfg: SegmentationRunConfig):
        self.cfg = cfg
        self.detector = YOLOSegDetector(YOLOSegConfig(model_size=cfg.yolo_model_size, device=cfg.yolo_device))

    @property
    def enabled(self) -> bool:
        return bool(self.detector.enabled)

    def _draw_modern_bbox(
        self,
        image_rgb: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        confidence: float,
        color_rgb,
        thickness: int = 2,
    ) -> np.ndarray:
        """PIL-based bbox rendering (paper-friendly)."""
        from PIL import Image, ImageDraw, ImageFont

        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        box_thickness = max(2, int(thickness))
        draw.rectangle([(x1, y1), (x2, y2)], outline=tuple(color_rgb), width=box_thickness)

        label_text = str(label)
        conf_text = f"{float(confidence):.2f}"

        font_size = max(12, int((x2 - x1) / 15))
        font_size = min(font_size, 16)

        font = None
        font_paths = [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for fp in font_paths:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        bbox_label = draw.textbbox((0, 0), label_text, font=font)
        bbox_conf = draw.textbbox((0, 0), conf_text, font=font)
        label_w = bbox_label[2] - bbox_label[0]
        label_h = bbox_label[3] - bbox_label[1]
        conf_w = bbox_conf[2] - bbox_conf[0]
        conf_h = bbox_conf[3] - bbox_conf[1]

        text_w = max(label_w, conf_w) + 8
        text_h = label_h + conf_h + 4

        text_x = x1
        text_y = y1 - text_h - 4
        if text_y < 0:
            text_y = y1 + 2

        bg_x1, bg_y1 = text_x, text_y
        bg_x2, bg_y2 = min(text_x + text_w, pil_image.width), min(text_y + text_h, pil_image.height)

        bg_color = tuple(max(0, min(255, int(c) - 40)) for c in color_rgb)
        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=bg_color)
        draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], outline=tuple(color_rgb), width=1)

        draw.text((text_x + 4, text_y), label_text, font=font, fill=(255, 255, 255))
        draw.text((text_x + 4, text_y + label_h + 2), conf_text, font=font, fill=(220, 220, 220))

        return np.array(pil_image)

    def _annotate_image(self, image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Overlay masks + draw bboxes."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Overlay masks first
        for det in detections:
            if not det.get("has_segmentation", False):
                continue
            mask = det.get("mask")
            if mask is None:
                continue
            color = get_color_for_class(str(det.get("class_name", "object")))
            image_rgb = overlay_mask_rgb(image_rgb, mask, color_rgb=color, alpha=0.35)

        # Draw boxes/labels
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cls = str(det.get("class_name", "object"))
            conf = float(det.get("confidence", 0.0))
            color = get_color_for_class(cls)
            image_rgb = self._draw_modern_bbox(image_rgb, x1, y1, x2, y2, cls, conf, color, thickness=2)

        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Run segmentation on a single image and write outputs."""
        if not self.enabled:
            raise RuntimeError("YOLO-seg detector is not available (check ultralytics install).")

        img_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dets, img_w, img_h = self.detector.detect(
            str(img_path), conf_threshold=self.cfg.conf_threshold, iou_threshold=self.cfg.iou_threshold
        )

        # Output dirs
        labels_dir = out_dir / "labels"
        meta_dir = out_dir / "metadata"
        viz_dir = out_dir / "visualizations"
        for d in [labels_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_annotated:
            viz_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        yolo_txt = labels_dir / f"{stem}.txt"
        meta_json = meta_dir / f"{stem}.json"
        viz_path = viz_dir / f"{stem}_segmented.jpg"

        # Write YOLO segmentation labels (polygon); fallback to bbox if polygon not available.
        with yolo_txt.open("w", encoding="utf-8") as f:
            for d in dets:
                cid = int(d.get("class_id", 0))
                conf = float(d.get("confidence", 0.0))
                poly = d.get("segmentation") or []
                if d.get("has_segmentation", False) and isinstance(poly, list) and len(poly) >= 6:
                    coords = " ".join([f"{float(v):.6f}" for v in poly])
                    f.write(f"{cid} {coords} {conf:.6f}\n")
                else:
                    f.write(
                        f"{cid} "
                        f"{float(d.get('x_center', 0.0)):.6f} "
                        f"{float(d.get('y_center', 0.0)):.6f} "
                        f"{float(d.get('width', 0.0)):.6f} "
                        f"{float(d.get('height', 0.0)):.6f} "
                        f"{conf:.6f}\n"
                    )

        # Strip non-serializable mask arrays before writing JSON
        dets_json: List[Detection] = []
        for d in dets:
            dd = dict(d)
            if "mask" in dd:
                dd.pop("mask", None)
            dets_json.append(dd)

        with meta_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": str(img_path),
                    "image_width": int(img_w),
                    "image_height": int(img_h),
                    "timestamp": datetime.now().isoformat(),
                    "model": {
                        "name": "yolov8-seg",
                        "model_size": self.cfg.yolo_model_size,
                        "device": str(self.cfg.yolo_device),
                        "conf_threshold": float(self.cfg.conf_threshold),
                        "iou_threshold": float(self.cfg.iou_threshold),
                    },
                    "num_detections": len(dets_json),
                    "detections": dets_json,
                },
                f,
                indent=2,
            )

        if self.cfg.save_annotated:
            img = cv2.imread(str(img_path))
            if img is not None:
                annotated = self._annotate_image(img, dets)
                cv2.imwrite(str(viz_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "image": str(img_path),
            "labels_txt": str(yolo_txt),
            "metadata_json": str(meta_json),
            "num_detections": len(dets_json),
        }

    def process_dir(
        self,
        images_dir: str,
        output_dir: str,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run segmentation on all images under a directory."""
        in_dir = Path(images_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts and not p.name.endswith("_segmented.jpg")])
        if max_images is not None and int(max_images) > 0:
            images = images[: int(max_images)]

        from tqdm.auto import tqdm

        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for p in tqdm(images, desc="Segmentation"):
            try:
                results.append(self.process_image(str(p), str(out_dir)))
            except Exception as e:
                failures.append({"image": str(p), "error": str(e)})

        summary = {
            "timestamp": datetime.now().isoformat(),
            "images_dir": str(in_dir),
            "output_dir": str(out_dir),
            "num_images": len(images),
            "processed": len(results),
            "failed": len(failures),
            "failures": failures[:50],
        }
        with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

