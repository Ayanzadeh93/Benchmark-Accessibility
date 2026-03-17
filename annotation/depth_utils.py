"""Utilities for working with raw depth arrays for prompting."""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def depth_to_grayscale_png_bytes(
    depth: np.ndarray,
    *,
    near_percentile: float = 5.0,
    far_percentile: float = 95.0,
    darker_is_closer: bool = True,
    depth_is_disparity: bool = True,
) -> Optional[bytes]:
    """Convert a raw depth/disparity array to a grayscale PNG for LLM prompting.

    Args:
        depth: Raw depth array (H, W).
        near_percentile: Lower percentile for robust normalization.
        far_percentile: Upper percentile for robust normalization.
        darker_is_closer: If True, closer regions will be darker in the output.
        depth_is_disparity: If True, larger values indicate *closer* (disparity-like output).

    Returns:
        PNG bytes or None on failure.
    """
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return None

    try:
        lo = float(np.percentile(depth, near_percentile))
        hi = float(np.percentile(depth, far_percentile))
        denom = (hi - lo) if (hi - lo) != 0.0 else 1e-6

        # Normalize into [0,1] where 0 corresponds to "near" and 1 to "far".
        # If the model output behaves like disparity (higher=near), invert appropriately.
        if depth_is_disparity:
            t = (depth - lo) / denom  # far->0, near->1
            # Convert to near->0, far->1
            t = 1.0 - t
        else:
            t = (depth - lo) / denom  # near->0, far->1

        t = np.clip(t, 0.0, 1.0)

        if not darker_is_closer:
            # If user wants closer=lighter, invert.
            t = 1.0 - t

        img = (t * 255.0).astype(np.uint8)
        pil = Image.fromarray(img, mode="L")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def depth_basic_stats(depth: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Return (min, max, mean, std) for a depth map."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return None
    try:
        return (
            float(np.min(depth)),
            float(np.max(depth)),
            float(np.mean(depth)),
            float(np.std(depth)),
        )
    except Exception:
        return None

