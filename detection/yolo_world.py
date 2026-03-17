"""YOLO-World open-vocabulary detector."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import nms_detections, xyxy_to_yolo_norm, clip01

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class YOLOWorldDetector:
    """YOLO-World detector for open vocabulary detection."""

    def __init__(self, model_size: str = "x"):
        """
        Initialize YOLO-World model.

        Args:
            model_size: Model size (s, m, l, x)
        """
        try:
            from ultralytics import YOLO

            model_names = {
                "s": "yolov8s-worldv2.pt",
                "m": "yolov8m-worldv2.pt",
                "l": "yolov8l-worldv2.pt",
                "x": "yolov8x-worldv2.pt",
            }
            model_name = model_names.get(model_size, "yolov8x-worldv2.pt")
            logger.info(f"Loading YOLO-World model: {model_name}")

            # Determine device - ultralytics uses "cuda" or 0 (integer) for CUDA, "cpu" for CPU
            if torch.cuda.is_available():
                try:
                    # Test CUDA actually works
                    test_tensor = torch.zeros(1).cuda()
                    device = "cuda"  # ultralytics uses "cuda" string or 0 (integer)
                    device_int = 0  # For passing to model calls
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    logger.warning(f"CUDA detected but test failed: {e}, using CPU")
                    device = "cpu"
                    device_int = "cpu"
            else:
                device = "cpu"
                device_int = "cpu"
            
            # Load model
            self.model = YOLO(model_name)
            
            # Set device for ultralytics (accepts "cuda", "cpu", or integer 0)
            try:
                if device == "cuda":
                    # Use integer 0 for CUDA device 0, or "cuda" string
                    self.model.to(0)  # Use integer for CUDA device 0
                    logger.info("YOLO-World model loaded on CUDA device: 0")
                else:
                    self.model.to("cpu")
                    logger.info("YOLO-World model loaded on CPU")
            except Exception as device_error:
                logger.warning(f"Could not set YOLO-World device: {device_error}, using CPU")
                device = "cpu"
                device_int = "cpu"
                self.model.to("cpu")

            self.device = device_int  # Store integer 0 or "cpu" for model calls
            logger.info(f"YOLO-World model loaded successfully on device: {device}")
            self._enabled = True
        except Exception as e:
            logger.warning(f"YOLO-World unavailable: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(
        self,
        image_path: str,
        class_names: List[str],
        conf_threshold: float = 0.25,
    ) -> Tuple[List[Detection], int, int]:
        """
        Detect objects using YOLO-World.

        Args:
            image_path: Path to the image
            class_names: List of class names to detect
            conf_threshold: Confidence threshold

        Returns:
            Tuple of (detections, img_width, img_height)
        """
        if not self.enabled:
            return [], 0, 0

        try:
            # Set custom vocabulary with device error handling
            try:
                # Ensure model is on correct device before set_classes
                if hasattr(self, 'device'):
                    self.model.to(self.device)
                
                self.model.set_classes(class_names)
                logger.debug(f"YOLO-World classes set: {len(class_names)} classes")
            except RuntimeError as device_error:
                error_msg = str(device_error).lower()
                if "device" in error_msg or "cuda" in error_msg or "cpu" in error_msg:
                    logger.warning(f"YOLO-World device error during set_classes: {device_error}")
                    # Try to fix by ensuring model is on correct device
                    try:
                        if torch.cuda.is_available() and self.device != "cpu":
                            self.model.to("0")
                            # Clear any cached features that might be on wrong device
                            if hasattr(self.model, "model") and hasattr(self.model.model, "txt_feats"):
                                self.model.model.txt_feats = None
                            self.model.set_classes(class_names)
                            logger.info("YOLO-World device issue fixed, set_classes succeeded")
                        else:
                            self.model.to("cpu")
                            self.model.set_classes(class_names)
                    except Exception as retry_error:
                        logger.error(f"Failed to fix device issue: {retry_error}")
                        import cv2
                        image = cv2.imread(image_path)
                        if image is not None:
                            img_height, img_width = image.shape[:2]
                            return [], img_width, img_height
                        return [], 0, 0
                else:
                    raise

            # Run detection with explicit device
            # ultralytics accepts device as integer (0) or string ("cuda", "cpu")
            results = self.model(image_path, conf=conf_threshold, verbose=False, device=self.device)
            result = results[0]

            import cv2

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return [], 0, 0

            img_height, img_width = image.shape[:2]
            detections = []

            # Process results
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box

                    # Get class name
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    xc, yc, ww, hh = xyxy_to_yolo_norm(x1, y1, x2, y2, img_width, img_height)

                    detection = {
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(conf),
                        "x_center": clip01(xc),
                        "y_center": clip01(yc),
                        "width": clip01(ww),
                        "height": clip01(hh),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "has_segmentation": False,
                    }

                    detections.append(detection)

            logger.info(f"YOLO-World: {len(detections)} detections")
            return detections, img_width, img_height

        except Exception as e:
            logger.error(f"YOLO-World detection error: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            import cv2

            image = cv2.imread(image_path)
            if image is not None:
                img_height, img_width = image.shape[:2]
                return [], img_width, img_height
            return [], 0, 0
