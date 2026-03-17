"""Schemas for benchmark inputs (MOS mapping, optional labels)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


QualityClass = Literal["poor", "acceptable", "good"]


class VideoLabel(BaseModel):
    """One labeled video entry for benchmarking."""

    video_id: str = Field(..., min_length=1)
    rel_path: str = Field(..., min_length=1, description="Relative path from dataset_root to the video file.")
    mos: float = Field(..., description="Mean opinion score (any numeric scale).")
    quality_class: Optional[QualityClass] = Field(
        default=None,
        description="Optional discrete label (poor/acceptable/good). If omitted, only correlation metrics are computed.",
    )

    @field_validator("quality_class", mode="before")
    @classmethod
    def _normalize_quality_class(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            # common variants
            if s in {"bad", "low"}:
                return "poor"
            if s in {"ok", "medium", "mid"}:
                return "acceptable"
            if s in {"high", "great"}:
                return "good"
            if s in {"poor", "acceptable", "good"}:
                return s
        # allow numeric encodings: 0/1/2
        try:
            n = int(v)
            if n == 0:
                return "poor"
            if n == 1:
                return "acceptable"
            if n == 2:
                return "good"
        except Exception:
            pass
        raise ValueError(f"Invalid quality_class: {v}")

