"""Depth sampling, metric conversions, and depth-map (de)serialization.

Everything here is METRIC METRES, larger = farther. The per-object sampling reproduces the recipe
from *Distance Perception in Vision-Language Models for Blind Navigation* (Sec. III-B): sample the
bbox centre plus four corners inset 20% from each edge, take the median, and flag the object as
uncertain when the standard deviation across the five samples exceeds a threshold (default 1 m).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

M_TO_FT = 3.280839895


@dataclass
class ObjectDistance:
    class_id: Optional[int]
    class_name: Optional[str]
    confidence: Optional[float]
    bbox_xyxy: Tuple[float, float, float, float]   # pixels, original image resolution
    distance_m: float
    distance_ft: float
    std_m: float                                   # spread across the 5 samples
    n_valid_samples: int                           # of 5
    uncertain: bool                                # std_m > uncertainty_std_m
    proximity_warn: bool                           # distance_m < proximity_warn_m

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_xyxy"] = [round(float(v), 2) for v in self.bbox_xyxy]
        d["distance_m"] = round(float(self.distance_m), 3)
        d["distance_ft"] = round(float(self.distance_ft), 2)
        d["std_m"] = round(float(self.std_m), 3)
        return d


def five_point_coords(
    bbox_xyxy: Tuple[float, float, float, float],
    inset_frac: float,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """Centre + four corners inset ``inset_frac`` from each edge, clamped to image bounds.

    Returns integer (x, y) pixel coordinates. Order: centre, TL, TR, BL, BR.
    """
    x1, y1, x2, y2 = bbox_xyxy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = x2 - x1
    bh = y2 - y1
    ix = inset_frac * bw
    iy = inset_frac * bh
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    pts = [
        (cx, cy),
        (x1 + ix, y1 + iy),
        (x2 - ix, y1 + iy),
        (x1 + ix, y2 - iy),
        (x2 - ix, y2 - iy),
    ]
    out: List[Tuple[int, int]] = []
    for px, py in pts:
        xi = int(min(max(round(px), 0), width - 1))
        yi = int(min(max(round(py), 0), height - 1))
        out.append((xi, yi))
    return out


def sample_object_distance(
    depth_m: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    *,
    inset_frac: float = 0.20,
    uncertainty_std_m: float = 1.0,
    proximity_warn_m: float = 1.0,
    class_id: Optional[int] = None,
    class_name: Optional[str] = None,
    confidence: Optional[float] = None,
    valid_min_m: float = 1e-3,
) -> Optional[ObjectDistance]:
    """Median of the 5 sampled depths inside a detection box (paper recipe).

    ``depth_m`` must be the FULL-resolution metric depth map (H, W) in metres. Returns None if no
    sampled point had a valid (finite, > valid_min_m) depth.
    """
    if depth_m is None or depth_m.ndim != 2:
        return None
    h, w = depth_m.shape
    coords = five_point_coords(bbox_xyxy, inset_frac, w, h)
    vals = []
    for xi, yi in coords:
        v = float(depth_m[yi, xi])
        if np.isfinite(v) and v > valid_min_m:
            vals.append(v)
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    distance_m = float(np.median(arr))
    # std across the *valid* samples; with <2 samples spread is undefined -> treat as 0 but the low
    # n_valid_samples still signals low support to downstream consumers.
    std_m = float(np.std(arr)) if arr.size >= 2 else 0.0
    return ObjectDistance(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=tuple(float(v) for v in bbox_xyxy),
        distance_m=distance_m,
        distance_ft=distance_m * M_TO_FT,
        std_m=std_m,
        n_valid_samples=int(arr.size),
        uncertain=bool(std_m > uncertainty_std_m),
        proximity_warn=bool(distance_m < proximity_warn_m),
    )


def sort_nearest_first(objects: List[ObjectDistance]) -> List[ObjectDistance]:
    return sorted(objects, key=lambda o: o.distance_m)


# ── depth map (de)serialization ─────────────────────────────────────────────
# Stored as 16-bit PNG in millimetres: value_mm = round(depth_m * 1000), clipped to uint16 range.
# 1 mm quantization is negligible vs. UniDepth's own error; 65.535 m ceiling is far past our range.

def depth_m_to_png16(depth_m: np.ndarray, clip_m: float = 60.0) -> np.ndarray:
    d = np.nan_to_num(depth_m, nan=0.0, posinf=clip_m, neginf=0.0)
    d = np.clip(d, 0.0, min(clip_m, 65.535))
    return (d * 1000.0).astype(np.uint16)


def png16_to_depth_m(png16: np.ndarray) -> np.ndarray:
    return png16.astype(np.float32) / 1000.0


def resize_longest_side(arr: np.ndarray, longest: int, interp: int) -> np.ndarray:
    """Resize so the longer side == ``longest`` (no-op if already <= longest or longest <= 0)."""
    if longest is None or longest <= 0:
        return arr
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= longest:
        return arr
    scale = longest / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def confidence_to_png8(conf: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Normalize a (relative, per-image) confidence map to 0-255 uint8. Falls back to all-255."""
    if conf is None:
        return np.full(shape, 255, dtype=np.uint8)
    c = np.nan_to_num(conf.astype(np.float32), nan=0.0)
    lo, hi = float(np.min(c)), float(np.max(c))
    if hi <= lo + 1e-9:
        return np.full(shape, 255, dtype=np.uint8)
    c = (c - lo) / (hi - lo)
    return (c * 255.0).astype(np.uint8)
