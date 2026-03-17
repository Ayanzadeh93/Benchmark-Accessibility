"""Apply a VLM (e.g. OpenRouter Qwen 235B) to generate VQA ground-truth answers.

When --ground-truth-model is set, each MCQ sample is sent to the model with
(image, question, options); the model's chosen letter (A/B/C/D) becomes the
reference answer. The annotation-derived answer is kept in ground_truth for comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from .schemas import VQAMultipleChoiceSample

logger = logging.getLogger(__name__)


def apply_ground_truth_model(
    samples: List[VQAMultipleChoiceSample],
    model_type: str,
    images_dir: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Run the given VLM on each sample (image + question + options) and set
    sample.answer / answer_text to the model's choice. Annotation-derived
    answer is stored in sample.ground_truth for comparison.

    Requires images_dir when samples only have relative image paths.
    Skips samples without a resolvable image path.
    """
    if not samples or not model_type or not model_type.strip():
        return
    model_type = model_type.strip().lower()
    try:
        from vlm_factory import VLMFactory
        extractor = VLMFactory.create_extractor(model_type, api_key=api_key)
    except Exception as e:
        logger.error("Failed to create ground-truth extractor for %s: %s", model_type, e)
        return
    if not getattr(extractor, "answer_multiple_choice", None):
        logger.warning("Model %s has no answer_multiple_choice; skipping ground-truth model step", model_type)
        return
    images_path = Path(images_dir) if images_dir else None
    updated = 0
    skipped = 0
    for sample in samples:
        image_path = sample.image_abs_path
        if not image_path and images_path:
            image_path = str((images_path / sample.image).resolve())
        if not image_path or not Path(image_path).exists():
            skipped += 1
            continue
        orig_answer: Any = sample.answer
        orig_text: str = sample.answer_text
        label = extractor.answer_multiple_choice(image_path, sample.question, sample.options)
        if label and label in sample.options:
            if not sample.ground_truth:
                sample.ground_truth = {}
            sample.ground_truth["annotation_answer"] = orig_answer
            sample.ground_truth["annotation_answer_text"] = orig_text
            sample.ground_truth["ground_truth_model"] = model_type
            sample.answer = label  # type: ignore[assignment]
            sample.answer_text = sample.options[label]
            updated += 1
    if updated or skipped:
        logger.info("Ground-truth model %s: updated %s samples, skipped %s (no image path)", model_type, updated, skipped)
