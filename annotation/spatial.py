"""Spatial helpers for converting numeric signals into natural language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RelativePosition:
    """Coarse position labels for an object in the image."""

    horizontal: str  # left|center|right
    vertical: str  # upper|middle|lower
    distance: str  # near|mid|far

    def to_phrase(self) -> str:
        """Short phrase like 'upper-right, near'."""
        return f"{self.vertical}-{self.horizontal}, {self.distance}"


def _bucket_3(x: float) -> str:
    if x < 1.0 / 3.0:
        return "left"
    if x < 2.0 / 3.0:
        return "center"
    return "right"


def _bucket_v(y: float) -> str:
    if y < 1.0 / 3.0:
        return "upper"
    if y < 2.0 / 3.0:
        return "middle"
    return "lower"


def _bucket_distance(area: float) -> str:
    """Heuristic distance bucket using normalized bbox area."""
    # Conservative thresholds; tuned for 1080p keyframes but works generally.
    if area >= 0.12:
        return "near"
    if area >= 0.03:
        return "mid"
    return "far"


def bbox_to_relative_position(
    *,
    x_center: Optional[float],
    y_center: Optional[float],
    width: Optional[float],
    height: Optional[float],
) -> Optional[RelativePosition]:
    """Convert normalized bbox to coarse relative position labels.

    Returns None if inputs are missing or invalid.
    """
    if x_center is None or y_center is None or width is None or height is None:
        return None
    try:
        xc = float(x_center)
        yc = float(y_center)
        w = float(width)
        h = float(height)
    except Exception:
        return None
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and w >= 0.0 and h >= 0.0):
        return None
    area = w * h
    return RelativePosition(horizontal=_bucket_3(xc), vertical=_bucket_v(yc), distance=_bucket_distance(area))

