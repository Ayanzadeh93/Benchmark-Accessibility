"""Segmentation utilities and pipelines (Phase 3).

This package is intentionally lightweight:
- Use Ultralytics YOLOv8-seg for instance segmentation (COCO classes)
- Save YOLO-seg style polygon labels + JSON metadata + optional visualizations
"""

from .pipeline import SegmentationPipeline, SegmentationRunConfig

__all__ = ["SegmentationPipeline", "SegmentationRunConfig"]

