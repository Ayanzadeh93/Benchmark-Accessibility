"""Phase-2: VLM-assisted open-vocabulary detection.

This package is intentionally lightweight:
- Use existing VLM outputs (Qwen/GPT-4o) to build a per-image vocabulary
- Run an open-vocabulary detector (GroundingDINO via transformers)
- Save YOLO-format labels + JSON metadata + optional annotated images
"""

from .pipeline import DetectionPipeline, DetectionRunConfig

__all__ = ["DetectionPipeline", "DetectionRunConfig"]

