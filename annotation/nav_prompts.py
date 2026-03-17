"""Prompt templates for navigation-style annotation generation."""

from __future__ import annotations

from typing import Iterable, Optional

from .navigation import NavObstacle


NAV_SYSTEM_PROMPT = (
    "You are a navigation assistant helping blind and visually impaired users navigate safely. "
    "Analyze the image and any provided depth/object cues comprehensively. "
    "Provide extremely detailed and comprehensive scene descriptions (up to 3000 tokens) that include: "
    "spatial layout, key features, lighting conditions, surface types, obstacle details, "
    "navigable paths, environmental context, and all notable details. "
    "Be thorough, descriptive, and comprehensive - do not limit yourself. "
    "Always prioritize safety, be extremely detailed, and follow the required output format exactly."
)


NAV_GENERATION_INSTRUCTION = """You are a navigation assistant for blind users. Analyze the scene comprehensively and provide extremely detailed navigation guidance.

RESPONSE FORMAT (use exactly this structure):

Scene: [Extremely comprehensive and detailed description up to 3000 tokens covering: complete spatial layout, all key features, detailed obstacle descriptions, lighting conditions, surface types (floor, walls, ceiling), environmental context, navigable paths, distances, relationships between objects, and any other relevant details. Be thorough and descriptive - use as many sentences as needed to fully describe the scene.]

Risk: [Low/Medium/High/Extreme]

Obstacles (use exact phrasing - list every object in front, left, or right):
- Obstacle in your front: [object name] at [clock], [distance]m away
- Obstacle in your left: [object name] at [clock], [distance]m away
- Obstacle in your right: [object name] at [clock], [distance]m away
(If both sides: "Obstacle in your left: ..." and "Obstacle in your right: ...")
(If no obstacles in front, left, or right: "Path clear (no obstacle in your front, left, or right)")

Guidance: [Single action command]

RISK LEVELS (4 types) - LANGUAGE MODEL MUST FOLLOW:
- Low: ONLY when path is clear - NO object/obstacle in your front, left, OR right. If ANY object (door, wall, person, etc.) is in your front or left or right, risk CANNOT be Low.
- Medium: Potential hazard; obstacle(s) in front/left/right but not immediately dangerous
- High: Can be hazardous in a few seconds if person doesn't get aligned
- Extreme: Falling risk, can hit in less than a few seconds or even 1 second; dangerous - stop immediately

GUIDANCE COMMANDS (choose one clear command):
Safe: "Path clear, move forward." | "Proceed straight ahead."
Adjust: "Move to [clock] o'clock to avoid [object]." | "Shift [left/right] to pass [object]."
Caution: "Slow down, [object] ahead." | "[Object] blocking, go around [direction]."
Blocked: "Turn around, path blocked!" | "Stop! [hazard] ahead. Turn back."
Danger: "Stop! [hazard] ahead." | "Caution! [hazard], proceed carefully."
Extreme: "Stop immediately! [hazard] directly ahead - dangerous. Do not proceed."

CLOCK POSITIONS (map to obstacle phrasing): 12/11/1=front, 9/10=left, 2/3=right. Always list obstacles as "Obstacle in your [front|left|right]: [name] at [clock], [distance]m away".

SCENE DESCRIPTION GUIDELINES:
- Be extremely comprehensive: describe the complete space, detailed layout, and all environmental features
- Include extensive details: mention floor type, wall materials, ceiling height, lighting conditions, colors, textures
- Note all obstacles: describe their exact positions, sizes, shapes, materials, and potential navigation impact
- Spatial context: explain detailed relationships between all objects, distances, and the navigable path
- Environmental details: describe air quality indicators, sounds, temperature cues, and any other sensory information
- Be thorough: use as many sentences as needed (up to 3000 tokens) to fully and comprehensively describe the scene
- Do not limit yourself - provide the most detailed and comprehensive description possible

RULES:
- Output must contain ONLY the structured response (no extra commentary, no JSON).
- Do NOT include file paths, filenames, model names, or coordinates.
- Do NOT invent objects that are not in the image or not in the provided object list.
- Use the provided object distances and clock positions; do not change numeric values.
- Scene description should be detailed and informative, helping users understand the full context.
"""


def build_navigation_prompt(
    *,
    obstacle_candidates: Iterable[NavObstacle],
    extra_notes: Optional[str] = None,
) -> str:
    """Build a prompt for LLM navigation guidance generation."""
    lines = [NAV_GENERATION_INSTRUCTION.strip(), ""]
    
    lines.append("IMPORTANT: The Scene description should be EXTREMELY comprehensive and detailed (up to 3000 tokens).")
    lines.append("Include extensive information about: complete spatial layout, floor/wall/ceiling materials and textures,")
    lines.append("lighting conditions and sources, detailed obstacle positions/sizes/shapes/materials, navigable paths,")
    lines.append("distances between objects, environmental context, colors, and ALL notable features.")
    lines.append("Be thorough - do not limit yourself. Provide the most detailed description possible.")
    lines.append("")

    if extra_notes:
        lines.append("Additional Context:")
        lines.append(str(extra_notes).strip())
        lines.append("")

    cand = list(obstacle_candidates)
    if cand:
        lines.append("Detected objects you may reference (use only these; values are fixed):")
        for o in cand:
            if o.distance_m is None:
                lines.append(f"- {o.name}: {o.clock_phrase}")
            else:
                lines.append(f"- {o.name}: {o.clock_phrase}, {float(o.distance_m):.1f} m")
        lines.append("")
    else:
        lines.append('Detected objects: (none) — if no obstacles are present, output "Path clear".')
        lines.append("")

    return "\n".join(lines).strip() + "\n"

