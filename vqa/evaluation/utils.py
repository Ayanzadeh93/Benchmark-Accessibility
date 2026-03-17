"""Shared utilities for VQA question generation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .schemas import ChoiceLabel


_DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", flags=re.IGNORECASE)
_CLOCK_RE = re.compile(r"\b(1[0-2]|[1-9])\s*o'?clock\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ObstacleInfo:
    name: str
    raw: str
    distance_m: Optional[float] = None
    clock: Optional[int] = None
    direction: Optional[str] = None  # left/right/ahead/behind/center


def stable_int_seed(*parts: str, base_seed: int = 1337) -> int:
    """Return a deterministic 32-bit seed derived from string parts."""
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8", errors="ignore"))
    return int(h.hexdigest()[:8], 16)


def normalize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_primary_obstacle(obstacles_lines: Sequence[str]) -> Optional[str]:
    """Extract an obstacle name from the first obstacle line."""
    if not obstacles_lines:
        return None
    line = str(obstacles_lines[0]).strip()
    if not line:
        return None
    m = re.match(r"^(.+?)\s+(?:at|on\s+your)\s+", line, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip(" ,.;:")
        return name if name else None
    name = line.split(",", 1)[0].strip(" ,.;:")
    return name if name else None


def extract_distance_m(line: str) -> Optional[float]:
    if not line:
        return None
    m = _DIST_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_clock(line: str) -> Optional[int]:
    if not line:
        return None
    m = _CLOCK_RE.search(line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_direction(line: str) -> Optional[str]:
    if not line:
        return None
    s = line.lower()
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if "behind" in s or "back" in s:
        return "behind"
    if "ahead" in s or "front" in s or "straight" in s:
        return "ahead"
    if "center" in s or "centre" in s:
        return "center"
    return None


def parse_obstacle_line(line: str) -> ObstacleInfo:
    name = extract_primary_obstacle([line]) or (line.split(",", 1)[0].strip() if line else "obstacle")
    return ObstacleInfo(
        name=name or "obstacle",
        raw=str(line or "").strip(),
        distance_m=extract_distance_m(line),
        clock=extract_clock(line),
        direction=extract_direction(line),
    )


def assign_labels(texts: Sequence[str]) -> Dict[ChoiceLabel, str]:
    if len(texts) != 4:
        raise ValueError("Expected exactly 4 option texts")
    labels: Tuple[ChoiceLabel, ChoiceLabel, ChoiceLabel, ChoiceLabel] = ("A", "B", "C", "D")
    return {lab: str(txt) for lab, txt in zip(labels, texts)}


class PathSafe:
    """Tiny helper to safely extract a stem from a filename-like string."""

    @staticmethod
    def stem(p: str) -> str:
        s = str(p or "").replace("\\", "/").rstrip("/")
        name = s.split("/")[-1] if s else "image"
        if "." in name:
            name = name.rsplit(".", 1)[0]
        return name or "image"
