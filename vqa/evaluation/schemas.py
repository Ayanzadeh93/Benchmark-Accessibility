"""Pydantic schemas for VQA evaluation datasets.

We generate VQA-style evaluation samples derived from Phase-5 annotation outputs.
This file defines stable, explicit JSON schemas for downstream evaluation tooling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ChoiceLabel = Literal["A", "B", "C", "D"]


class VQAMultipleChoiceSample(BaseModel):
    """Single multiple-choice VQA sample (A/B/C/D)."""

    id: str = Field(description="Stable unique id (e.g. '<image_stem>|<question_id>').")
    question_id: str = Field(description="Identifier for the question type (e.g. 'action_command').")

    image: str = Field(description="Image path or image filename (prefer relative path).")
    image_abs_path: Optional[str] = Field(default=None, description="Optional absolute image path if available.")

    question: str
    options: Dict[ChoiceLabel, str]

    answer: ChoiceLabel = Field(description="Correct option label.")
    answer_text: str = Field(description="Text of the correct option.")

    # Optional evaluator guidance (can be shown to humans, or used for analysis).
    feedback: Dict[ChoiceLabel, str] = Field(default_factory=dict)
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    sources: Dict[str, Optional[str]] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def _validate_options(cls, v: Any) -> Dict[ChoiceLabel, str]:
        if not isinstance(v, dict):
            raise ValueError("options must be a dict with keys A/B/C/D")
        keys = set(v.keys())
        if keys != {"A", "B", "C", "D"}:
            raise ValueError("options must contain exactly keys: A, B, C, D")
        for k, txt in v.items():
            if not isinstance(txt, str) or not txt.strip():
                raise ValueError(f"options[{k}] must be a non-empty string")
        return v  # type: ignore[return-value]


class VQADataset(BaseModel):
    """Dataset container for a set of VQA samples."""

    version: str = Field(default="v1")
    created_at: str
    task: str = Field(default="navigation_vqa")
    question_set: str = Field(default="navigation_core_mcq")
    split: Optional[str] = Field(default=None, description="Optional split label: tuning/evaluation")
    samples: List[VQAMultipleChoiceSample] = Field(default_factory=list)
    failures: List[Dict[str, Any]] = Field(default_factory=list)

