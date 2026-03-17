"""Segmentation helpers (mask -> polygon, colors, and visualization)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import hashlib

import cv2
import numpy as np

Detection = Dict[str, Any]


_COLOR_PALETTE_RGB: List[Tuple[int, int, int]] = [
    (0, 119, 190),  # Blue
    (255, 87, 34),  # Orange
    (76, 175, 80),  # Green
    (156, 39, 176),  # Purple
    (255, 193, 7),  # Amber
    (233, 30, 99),  # Pink
    (3, 169, 244),  # Light Blue
    (255, 152, 0),  # Deep Orange
    (0, 150, 136),  # Teal
    (63, 81, 181),  # Indigo
    (244, 67, 54),  # Red
    (121, 85, 72),  # Brown
]


def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Get consistent color for a class name (RGB)."""
    h = int(hashlib.md5(str(class_name).lower().encode()).hexdigest(), 16)
    return _COLOR_PALETTE_RGB[h % len(_COLOR_PALETTE_RGB)]


def mask_to_polygon(mask: np.ndarray) -> List[float]:
    """Convert a binary mask to a flattened polygon list with normalized coords.

    Returns: [x1, y1, x2, y2, ...] where x/y are in [0,1].
    """
    if mask is None:
        return []

    if mask.ndim == 3:
        # (N,H,W) -> take first
        mask = mask[0]

    if mask.dtype != np.uint8:
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
    else:
        # assume already 0/255 or 0/1
        mask_bin = (mask > 0).astype(np.uint8) * 255

    h, w = mask_bin.shape[:2]
    if h <= 0 or w <= 0:
        return []

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Use largest contour
    largest = max(contours, key=cv2.contourArea)
    if largest is None or len(largest) < 3:
        return []

    # Simplify polygon a bit
    eps = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    if approx is None or len(approx) < 3:
        return []

    poly: List[float] = []
    for p in approx:
        x, y = p[0]
        poly.extend([float(x) / float(w), float(y) / float(h)])
    return poly


def overlay_mask_rgb(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a single mask on an RGB image."""
    if image_rgb is None or image_rgb.size == 0 or mask is None:
        return image_rgb

    h, w = image_rgb.shape[:2]
    if mask.ndim == 3:
        mask = mask[0]

    mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bin = mask_resized > 0.5

    overlay = image_rgb.copy()
    overlay[mask_bin] = np.array(color_rgb, dtype=np.uint8)
    return cv2.addWeighted(overlay, float(alpha), image_rgb, float(1.0 - alpha), 0)


def xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    xc = float(x1) + w / 2.0
    yc = float(y1) + h / 2.0
    return (
        float(xc / float(img_w)),
        float(yc / float(img_h)),
        float(w / float(img_w)),
        float(h / float(img_h)),
    )


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))

