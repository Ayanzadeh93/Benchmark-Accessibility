"""Dataset annotation package.

This package generates high-quality textual annotations for images/keyframes,
optionally leveraging auxiliary signals:
- object detections (Phase 2)
- segmentation masks (Phase 3)
- depth estimation (Phase 4)

The outputs are suitable for fine-tuning vision-language models.
"""

from .pipeline import AnnotationPipeline, AnnotationRunConfig

__all__ = [
    "AnnotationPipeline",
    "AnnotationRunConfig",
]

