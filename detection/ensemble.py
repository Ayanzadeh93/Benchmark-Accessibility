"""Ensemble detector combining YOLO-World and GroundingDINO."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grounding_dino import GroundingDINOConfig, GroundingDINODetector
from .utils import nms_detections
from .yolo_world import YOLOWorldDetector

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class EnsembleDetector:
    """Ensemble detector combining YOLO-World and GroundingDINO."""

    def __init__(
        self,
        yolo_model_size: str = "x",
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize detectors.

        Args:
            yolo_model_size: YOLO-World model size (s, m, l, x)
            gdino_model_id: GroundingDINO model ID
            hf_token: Hugging Face token
        """
        logger.info("Initializing Ensemble Detector (YOLO-World + GroundingDINO)")

        self.yolo_detector = None
        self.gdino_detector = None

        # Try to initialize YOLO-World
        try:
            self.yolo_detector = YOLOWorldDetector(yolo_model_size)
            if self.yolo_detector.enabled:
                logger.info("✓ YOLO-World initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO-World: {e}")

        # Try to initialize GroundingDINO
        try:
            import torch

            # Verify CUDA actually works before using it
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    gdino_device = "cuda"
                    logger.info(f"CUDA verified: {torch.cuda.get_device_name(0)}")
                except Exception as cuda_test_error:
                    logger.warning(f"CUDA detected but test failed: {cuda_test_error}, using CPU")
                    gdino_device = "cpu"
            else:
                gdino_device = "cpu"
            
            self.gdino_detector = GroundingDINODetector(
                GroundingDINOConfig(model_id=gdino_model_id, device=gdino_device, hf_token=hf_token)
            )
            if self.gdino_detector.enabled:
                logger.info(f"✓ GroundingDINO initialized on {gdino_device}")
        except Exception as e:
            logger.warning(f"Failed to initialize GroundingDINO: {e}")

        if not (self.yolo_detector and self.yolo_detector.enabled) and not (
            self.gdino_detector and self.gdino_detector.enabled
        ):
            raise RuntimeError("Failed to initialize any detector")

    @property
    def enabled(self) -> bool:
        return (self.yolo_detector and self.yolo_detector.enabled) or (
            self.gdino_detector and self.gdino_detector.enabled
        )

    def detect(
        self,
        image_path: str,
        class_names: List[str],
        conf_threshold: float = 0.10,
        text_threshold: float = 0.15,
        nms_iou: float = 0.50,
    ) -> Tuple[List[Detection], int, int]:
        """
        Run both detectors and merge results.

        Args:
            image_path: Path to the image
            class_names: List of class names to detect
            conf_threshold: Confidence threshold
            text_threshold: Text threshold (for GroundingDINO)
            nms_iou: NMS IoU threshold

        Returns:
            Tuple of (detections, img_width, img_height)
        """
        all_detections = []
        img_width, img_height = 0, 0

        # Run YOLO-World
        if self.yolo_detector and self.yolo_detector.enabled:
            try:
                logger.debug("Running YOLO-World detection...")
                yolo_dets, w, h = self.yolo_detector.detect(image_path, class_names, conf_threshold)
                all_detections.extend(yolo_dets)
                if img_width == 0:
                    img_width, img_height = w, h
                logger.debug(f"YOLO-World: {len(yolo_dets)} detections")
            except Exception as e:
                logger.warning(f"YOLO-World detection failed: {e}")

        # Run GroundingDINO
        if self.gdino_detector and self.gdino_detector.enabled:
            try:
                logger.debug("Running GroundingDINO detection...")
                gdino_dets, w, h = self.gdino_detector.detect(
                    image_path,
                    class_names,
                    conf_threshold=conf_threshold,
                    text_threshold=text_threshold,
                    multi_prompt=True,  # Always use multi-prompt for ensemble
                    nms_iou=nms_iou,
                )
                all_detections.extend(gdino_dets)
                if img_width == 0:
                    img_width, img_height = w, h
                logger.debug(f"GroundingDINO: {len(gdino_dets)} detections")
            except Exception as e:
                logger.warning(f"GroundingDINO detection failed: {e}")

        if img_width == 0:
            import cv2

            image = cv2.imread(image_path)
            if image is not None:
                img_height, img_width = image.shape[:2]
            else:
                return [], 0, 0

        logger.debug(f"Total raw detections from ensemble: {len(all_detections)}")

        # Apply NMS to merge overlapping detections
        detections = nms_detections(all_detections, iou_threshold=nms_iou, class_aware=True)
        logger.debug(f"After NMS: {len(detections)} detections")

        # Filter by confidence threshold
        detections = [d for d in detections if float(d.get("confidence", 0.0)) >= conf_threshold]
        logger.debug(f"After confidence filtering (>= {conf_threshold}): {len(detections)} detections")

        logger.info(f"Ensemble: {len(detections)} final detections")
        return detections, img_width, img_height
