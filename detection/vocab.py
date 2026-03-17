"""Build per-image detection vocabularies from VLM outputs.

Goal: keep prompts short (for speed/accuracy) while prioritizing useful classes
(signs + accessibility + people + technology).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
from pathlib import Path


_STOP_OBJECTS = {
    # overly generic / not useful for detection training
    "floor",
    "ceiling",
    "wall",
    "walls",
    "stone",
    "stones",
    "rock",
    "rocks",
    "brick",
    "bricks",
    "concrete",
    "lighting",
    "light",
    "lamp",
    "beam",
    "beams",
    # Note: "column" removed - pillars/columns are important obstacles for blind users
    "building",
    "buildings",
    "room",
    "rooms",
    "indoor",
    "outdoor",
    "sky",
    "grass",
    "pavement",
    "road",
    "roads",
    "carpet",
    "carpets",
    "platform",
    "platforms",
    "background",
    "ground",
    "surface",
    "surfaces",
    "texture",
    "textures",
    "material",
    "materials",
}


_SIGN_NORMALIZE = {
    "exit": "exit sign",
    "restroom": "restroom sign",
    "bathroom": "restroom sign",
    "toilet": "restroom sign",
    "accessible": "accessibility sign",
    "wheelchair": "accessibility sign",
    "elevator": "elevator sign",
    "lift": "elevator sign",
    "emergency": "emergency exit sign",
    "fire exit": "emergency exit sign",
    "no smoking": "no smoking sign",
    "information": "information sign",
    "direction": "direction sign",
    "warning": "warning sign",
    "safety": "safety sign",
    "braille": "braille sign",
    "hearing loop": "hearing loop sign",
}

# Normalize obstacle terms for accessibility (important for blind users)
_OBSTACLE_NORMALIZE = {
    "pillar": "pillar",
    "pillars": "pillar",
    "column": "pillar",  # Structural columns are obstacles
    "columns": "pillar",
    "post": "pillar",
    "posts": "pillar",
    "pole": "pole",
    "poles": "pole",
    "barrier": "barrier",
    "barriers": "barrier",
    "obstacle": "obstacle",
    "obstacles": "obstacle",
    "block": "obstacle",
    "blocks": "obstacle",
    "protrusion": "obstacle",
    "protrusions": "obstacle",
    "overhang": "overhang",
    "overhangs": "overhang",
    "low hanging": "low hanging obstacle",
    "low ceiling": "low hanging obstacle",
}


def _norm_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def normalize_detection_label(label: str) -> str:
    """Normalize labels (especially sign types and obstacles) for consistent prompting."""
    l = _norm_label(label)
    if not l:
        return ""
    # if it's already a specific sign, keep
    if "sign" in l:
        return l
    # map common obstacle terms (important for accessibility)
    for k, v in _OBSTACLE_NORMALIZE.items():
        if k in l:
            return v
    # map common sign tokens to sign names
    for k, v in _SIGN_NORMALIZE.items():
        if k in l:
            return v
    return l


def load_vlm_result_json(vlm_json_path: str) -> Optional[Dict[str, Any]]:
    """Load VLM analysis JSON from disk."""
    p = Path(vlm_json_path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_objects_dict(vlm_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(vlm_result, dict):
        return None
    obj = vlm_result.get("objects")
    if isinstance(obj, dict):
        return obj
    return None


def build_detection_classes_from_vlm(
    vlm_result: Dict[str, Any],
    max_classes: int = 20,
) -> List[str]:
    """Build an ordered list of class names for open-vocab detection.
    
    Now handles simplified format: only objects list, no categories.
    """
    objects_dict = _extract_objects_dict(vlm_result) or {}

    # Simplified format: objects is directly in the result or in objects_dict
    if isinstance(vlm_result, dict) and "objects" in vlm_result:
        objects = vlm_result.get("objects", [])
    else:
        objects = objects_dict.get("objects") or []
    
    # Legacy support: check for categories (may not exist in simplified format)
    categories = objects_dict.get("categories") or {}
    if not categories and isinstance(vlm_result, dict):
        categories = vlm_result.get("categories", {})

    # Extract obstacles from objects list and prioritize them
    obstacle_keywords = ["pillar", "column", "post", "pole", "barrier", "obstacle", "block", "protrusion", "overhang"]
    obstacles_found = []
    if isinstance(objects, list):
        for obj in objects:
            obj_lower = str(obj).lower()
            for keyword in obstacle_keywords:
                if keyword in obj_lower:
                    obstacles_found.append(normalize_detection_label(str(obj)))
                    break

    out: List[str] = []
    seen = set()

    def _maybe_add(x: str) -> None:
        x = normalize_detection_label(x)
        if not x:
            return
        if x in _STOP_OBJECTS:
            return
        if x in seen:
            return
        seen.add(x)
        out.append(x)

    # CRITICAL: Add obstacles FIRST (pillars, barriers, etc.) - important for blind users
    for obstacle in obstacles_found:
        _maybe_add(obstacle)

    # Add prioritized categories if they exist (legacy support)
    if categories:
        priority_keys = ["signs", "accessibility", "people", "technology", "furniture"]
        for k in priority_keys:
            vals = categories.get(k, [])
            if not isinstance(vals, list):
                continue
            for v in vals:
                _maybe_add(str(v))

    # Then add the global objects list (filtered)
    if isinstance(objects, list):
        for v in objects:
            _maybe_add(str(v))

    # Fallback (never empty)
    if not out:
        out = ["person", "sign"]

    return out[: max(1, int(max_classes))]

