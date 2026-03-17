"""YOLOv8-seg wrapper for instance segmentation (COCO classes).

This is NOT open-vocabulary segmentation. It provides high-quality masks for common objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

Detection = Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YOLOSegConfig:
    model_size: str = "x"  # s|m|l|x
    device: str = "auto"  # auto|cuda|cpu


class YOLOSegDetector:
    """Ultralytics YOLOv8 segmentation detector."""

    def __init__(self, cfg: YOLOSegConfig):
        self.cfg = cfg
        self._enabled = False
        self.device: Union[int, str] = "cpu"  # ultralytics accepts int for GPU
        self.model = None
        self.class_names: Dict[int, str] = {}

        try:
            import torch
            from ultralytics import YOLO

            model_names = {
                "s": "yolov8s-seg.pt",
                "m": "yolov8m-seg.pt",
                "l": "yolov8l-seg.pt",
                "x": "yolov8x-seg.pt",
            }
            model_name = model_names.get(str(cfg.model_size).lower(), "yolov8x-seg.pt")

            # Resolve device
            requested = str(cfg.device).lower()
            use_cuda = False
            if requested == "cuda":
                use_cuda = True
            elif requested == "cpu":
                use_cuda = False
            else:  # auto
                use_cuda = bool(torch.cuda.is_available())

            if use_cuda:
                try:
                    _ = torch.zeros(1).cuda()
                    self.device = 0
                    logger.info(f"[OK] YOLO-seg using CUDA: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    logger.warning(f"CUDA detected but failed test: {e}. Using CPU for YOLO-seg.")
                    self.device = "cpu"
            else:
                self.device = "cpu"

            logger.info(f"Loading YOLO-seg model: {model_name}")
            self.model = YOLO(model_name)
            try:
                self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Could not move YOLO-seg to device {self.device}: {e}. Using CPU.")
                self.device = "cpu"
                self.model.to("cpu")

            # Names mapping
            try:
                self.class_names = {int(k): str(v) for k, v in getattr(self.model, "names", {}).items()}
            except Exception:
                self.class_names = {}

            self._enabled = True
        except Exception as e:
            logger.warning(f"YOLO-seg unavailable: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Tuple[List[Detection], int, int]:
        """Run segmentation on an image.

        Returns: detections, img_w, img_h
        Each detection includes bbox + polygon segmentation (normalized) when available.
        """
        if not self.enabled or self.model is None:
            return [], 0, 0

        img = cv2.imread(str(image_path))
        if img is None:
            return [], 0, 0
        img_h, img_w = img.shape[:2]

        try:
            results = self.model(
                str(image_path),
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                verbose=False,
                device=self.device,
            )
            r0 = results[0]

            detections: List[Detection] = []
            if r0.boxes is None or len(r0.boxes) == 0:
                return [], img_w, img_h

            boxes = r0.boxes.xyxy.detach().cpu().numpy()
            confs = r0.boxes.conf.detach().cpu().numpy()
            clss = r0.boxes.cls.detach().cpu().numpy().astype(int)

            masks = None
            if hasattr(r0, "masks") and r0.masks is not None and hasattr(r0.masks, "data"):
                try:
                    masks = r0.masks.data.detach().cpu().numpy()
                except Exception:
                    masks = None

            from .utils import mask_to_polygon, xyxy_to_yolo_norm, clamp01

            for idx, (box, score, cid) in enumerate(zip(boxes, confs, clss)):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue

                cname = self.class_names.get(int(cid), f"class_{int(cid)}")
                xc, yc, ww, hh = xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h)

                poly: List[float] = []
                has_seg = False
                mask_arr = None
                if masks is not None and idx < len(masks):
                    mask_arr = masks[idx]
                    poly = mask_to_polygon(mask_arr)
                    has_seg = bool(poly)

                detections.append(
                    {
                        "class_id": int(cid),
                        "class_name": str(cname),
                        "confidence": float(score),
                        "x_center": clamp01(xc),
                        "y_center": clamp01(yc),
                        "width": clamp01(ww),
                        "height": clamp01(hh),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "has_segmentation": bool(has_seg),
                        "segmentation": poly,
                        # Not JSON-serializable; pipeline strips this before saving metadata.
                        "mask": mask_arr,
                    }
                )

            return detections, img_w, img_h
        except Exception as e:
            logger.warning(f"YOLO-seg detection failed: {e}")
            return [], img_w, img_h

