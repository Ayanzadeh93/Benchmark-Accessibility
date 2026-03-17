"""Parsing and normalization for navigation-style responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional

# Risk levels: Low (path clear), Medium (potential hazard), High (hazardous in few sec if not aligned),
# Extreme (falling, hit in <1 sec, dangerous - stop immediately)
RiskLevel = Literal["Low", "Medium", "High", "Extreme"]


@dataclass(frozen=True)
class ParsedNavigation:
    scene: str
    risk: RiskLevel
    obstacles_lines: List[str]  # raw obstacle lines (no leading '- ' required)
    guidance: str


# Updated to handle multi-line scene descriptions (up to 120 tokens)
_SCENE_RE = re.compile(r"^\s*Scene:\s*(.+?)(?=\n\s*Risk:|\n\s*Obstacles:|$)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_RISK_RE = re.compile(r"^\s*Risk:\s*(Low|Medium|High|Extreme)\s*$", re.IGNORECASE | re.MULTILINE)
_GUIDANCE_RE = re.compile(r"^\s*Guidance:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_navigation_response(text: str) -> Optional[ParsedNavigation]:
    """Parse a navigation response into structured fields.

    Returns None if the response can't be parsed reliably.
    """
    if not text or not isinstance(text, str):
        return None

    s = text.strip()
    if not s:
        return None

    scene_m = _SCENE_RE.search(s)
    risk_m = _RISK_RE.search(s)
    guidance_m = _GUIDANCE_RE.search(s)

    if not scene_m or not risk_m or not guidance_m:
        return None

    scene = scene_m.group(1).strip()
    # Normalize multi-line scene descriptions to single line with spaces
    scene = re.sub(r"\s+", " ", scene)
    risk = risk_m.group(1).strip().capitalize()
    if risk not in {"Low", "Medium", "High", "Extreme"}:
        risk = "Low"

    guidance = guidance_m.group(1).strip()

    # Extract obstacle block between "Obstacles:" and "Guidance:"
    obstacles_lines: List[str] = []
    obstacles_block = _extract_section(s, header="Obstacles:", until_header="Guidance:")
    if obstacles_block:
        block = obstacles_block.strip()
        if re.search(r"\bpath\s+clear\b", block, re.IGNORECASE):
            obstacles_lines = []
        else:
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    line = line[1:].strip()
                # Drop any lingering bullets or numbering
                line = re.sub(r"^\*+\s*", "", line)
                if line:
                    obstacles_lines.append(line)

    return ParsedNavigation(scene=scene, risk=risk, obstacles_lines=obstacles_lines, guidance=guidance)


def normalize_navigation_response(parsed: ParsedNavigation) -> str:
    """Normalize to the canonical output format."""
    scene = _single_line(parsed.scene)
    risk: RiskLevel = parsed.risk
    guidance = _single_line(parsed.guidance)

    if parsed.obstacles_lines:
        obs_block = "\n".join([f"- {o}" for o in parsed.obstacles_lines])
    else:
        obs_block = "Path clear"

    return f"Scene: {scene}\n\nRisk: {risk}\n\nObstacles:\n{obs_block}\n\nGuidance: {guidance}"


def _single_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_section(text: str, *, header: str, until_header: str) -> Optional[str]:
    # Find header
    idx = text.lower().find(header.lower())
    if idx == -1:
        return None
    start = idx + len(header)
    rest = text[start:]

    # Find until header
    end_idx = rest.lower().find(until_header.lower())
    if end_idx == -1:
        return rest
    return rest[:end_idx]

