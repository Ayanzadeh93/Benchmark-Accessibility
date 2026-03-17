"""Navigation-style annotation generation.

Generates structured navigation guidance suitable for LoRA fine-tuning:

Scene: ...
Risk: Low | Medium | High | Extreme
Obstacles (explicit phrasing with clock before distance):
- Obstacle in your front: [name] at [clock], [distance]m away
- Obstacle in your left: [name] at [clock], [distance]m away
- Obstacle in your right: [name] at [clock], [distance]m away
Guidance: ...

Risk levels:
- Low: ONLY when path is clear - no object/obstacle in front, left, or right. Any object in front or left or right -> risk cannot be Low.
- Medium: Potential hazard; obstacle(s) in front/left/right but not immediately dangerous.
- High: Can be hazardous in a few seconds if person doesn't get aligned.
- Extreme: Falling risk, can hit in <1 sec; dangerous - stop immediately.

This module is deterministic and does NOT call external APIs.
It uses detections + (optional) raw depth maps to estimate coarse distances.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .captioners import CaptionContext, ObjectMention


@dataclass(frozen=True)
class NavigationConfig:
    """Configuration knobs for navigation generation."""

    min_distance_m: float = 0.5
    max_distance_m: float = 8.0
    near_percentile: float = 5.0
    far_percentile: float = 95.0
    # DepthAnything V2 raw output is closer to disparity (higher values ~ closer).
    depth_increases_with_distance: bool = False
    max_obstacles: int = 6
    stop_distance_m: float = 0.8
    caution_distance_m: float = 2.0


@dataclass(frozen=True)
class NavObstacle:
    name: str
    clock: int  # 1..12
    distance_m: Optional[float]

    @property
    def clock_phrase(self) -> str:
        return f"{self.clock} o'clock"


_NON_OBSTACLE_HINTS = {
    "sign",
    "exit sign",
    "restroom sign",
    "elevator sign",
    "information board",
    "poster",
    "screen",
    "display",
    "display screen",
    "monitor",
    "lighting",
    "light",
    "ceiling",
    "control panel",
    "button",
    "reflector",
    "latch",
}

# High risk: hazardous in a few seconds if person doesn't get aligned
_HIGH_RISK = {
    "stairs",
    "stair",
    "stairway",
    "escalator",
}

# Extreme risk: falling, can hit in <1 sec, dangerous - stop immediately
_EXTREME_RISK = {
    "stairs",
    "stair",
    "stairway",
    "escalator",
    "dropoff",
    "drop-off",
    "hole",
    "gap",
    "cliff",
    "edge",
}

_COMMON_BLOCKERS = {
    "wall",
    "closed door",
    "door",
}


def generate_navigation_guidance(
    *,
    ctx: CaptionContext,
    depth: Optional[np.ndarray] = None,
    cfg: Optional[NavigationConfig] = None,
) -> Tuple[str, str, List[NavObstacle], str]:
    """Generate (scene, risk, obstacles, guidance) deterministically."""
    cfg = cfg or NavigationConfig()

    scene = _scene_line(ctx)
    obstacles = _select_obstacles(ctx.mentions, depth=depth, cfg=cfg)
    risk = _risk_level(obstacles, cfg=cfg)
    guidance = _guidance_command(risk=risk, obstacles=obstacles, cfg=cfg)

    return scene, risk, obstacles, guidance


def format_navigation_output(scene: str, risk: str, obstacles: Sequence[NavObstacle], guidance: str) -> str:
    """Format the navigation output with the required structure."""
    scene = scene.strip()
    risk = risk.strip()
    guidance = guidance.strip()

    if not scene:
        scene = "Indoor space with a few visible objects."
    if risk not in {"Low", "Medium", "High", "Extreme"}:
        risk = "Low"
    if not guidance:
        guidance = "Path clear, move forward."

    if obstacles:
        obs_lines = "\n".join([f"- {format_obstacle_line(o)}" for o in obstacles])
    else:
        obs_lines = "Path clear (no obstacle in your front, left, or right)"

    return (
        f"Scene: {scene}\n\n"
        f"Risk: {risk}\n\n"
        f"Obstacles:\n{obs_lines}\n\n"
        f"Guidance: {guidance}"
    )


def _clock_to_position_phrase(clock: int) -> str:
    """Map clock position to 'front', 'left', 'right', or 'behind' for obstacle phrasing."""
    if clock in {9, 10}:
        return "left"
    if clock in {11, 12, 1}:
        return "front"
    if clock in {2, 3}:
        return "right"
    return "behind"


def format_obstacle_line(o: NavObstacle) -> str:
    """Render obstacle line as 'Obstacle in your [left/right/front]: [name] at [clock], [distance]m away'.

    Includes both left/right/front and the clock direction before distance.
    """
    name = (o.name or "object").strip() or "object"
    position = _clock_to_position_phrase(o.clock)
    if o.distance_m is None:
        return f"Obstacle in your {position}: {name} at {o.clock_phrase}"
    dist = max(0.0, float(o.distance_m))
    return f"Obstacle in your {position}: {name} at {o.clock_phrase}, {dist:.1f}m away"


def clock_from_normalized_xy(x_center: float, y_center: float) -> int:
    """Convert normalized (x,y) into a clock position.

    12 o'clock is straight ahead. 3 is right, 9 is left.

    Note: For forward-facing keyframes, the image does not contain "behind" content.
    We therefore map points in the image to the *front* semicircle only (9..3).
    """
    # Center coordinates; treat vertical displacement as "forward" magnitude only.
    dx = float(x_center) - 0.5  # right positive
    dy = abs(0.5 - float(y_center))  # forward magnitude (no "behind" in a single image)
    dy = max(dy, 1e-3)

    # Signed angle from straight ahead: -90 (left) .. +90 (right)
    angle = math.degrees(math.atan2(dx, dy))
    angle = max(-90.0, min(90.0, angle))

    step = int(round(angle / 30.0))  # -3..+3
    step = max(-3, min(3, step))

    # Map steps to clock: -3->9, -2->10, -1->11, 0->12, +1->1, +2->2, +3->3
    if step == 0:
        return 12
    if step > 0:
        return step  # 1..3
    return 12 + step  # 11,10,9


def estimate_distance_m_from_depth(
    *,
    depth: np.ndarray,
    x_center: float,
    y_center: float,
    cfg: NavigationConfig,
) -> Optional[float]:
    """Estimate distance in meters from a raw depth map.

    DepthAnything V2 produces relative depth values (not metric). We map values
    to a pseudo-metric range [min_distance_m, max_distance_m] using robust percentiles.
    This keeps labels consistent for fine-tuning even if absolute meters are approximate.
    """
    if depth is None:
        return None
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return None

    h, w = depth.shape[:2]
    px = int(round(float(x_center) * (w - 1)))
    py = int(round(float(y_center) * (h - 1)))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))

    v = float(depth[py, px])

    near = float(np.percentile(depth, cfg.near_percentile))
    far = float(np.percentile(depth, cfg.far_percentile))
    denom = (far - near) if (far - near) != 0.0 else 1e-6

    if cfg.depth_increases_with_distance:
        t = (v - near) / denom
    else:
        t = (far - v) / denom
    t = max(0.0, min(1.0, float(t)))

    return float(cfg.min_distance_m + t * (cfg.max_distance_m - cfg.min_distance_m))


def estimate_distance_m_from_bbox_area(area: Optional[float], cfg: NavigationConfig) -> Optional[float]:
    """Fallback distance estimate from normalized bbox area."""
    if area is None:
        return None
    try:
        a = float(area)
    except Exception:
        return None
    if a <= 0.0:
        return None
    # Heuristic: larger bbox area => closer.
    # Map area in [0.001..0.2] roughly into distance range.
    a = max(0.001, min(0.2, a))
    t = 1.0 - (math.sqrt(a) - math.sqrt(0.001)) / (math.sqrt(0.2) - math.sqrt(0.001) + 1e-6)
    t = max(0.0, min(1.0, t))
    return float(cfg.min_distance_m + t * (cfg.max_distance_m - cfg.min_distance_m))


def _scene_line(ctx: CaptionContext) -> str:
    """Extremely comprehensive scene description, up to ~3000 tokens (~2250 words)."""
    base = (ctx.scene_description or "").strip()
    if not base:
        # Enhanced fallback from primary focus / objects
        if ctx.primary_focus:
            base = f"This is a {ctx.primary_focus} scene with various features and obstacles. The space contains multiple elements that require careful navigation."
        elif ctx.objects:
            obj_list = ", ".join(ctx.objects[:5])
            base = f"Scene contains {obj_list} and other environmental features. The layout includes various obstacles and navigable paths."
        else:
            base = "Indoor space with a clear view and standard architectural elements. The environment includes various features that affect navigation."

    # Normalize whitespace and allow multiple sentences/paragraphs
    base = re.sub(r"\s+", " ", base.replace("\n", " ")).strip()
    
    # Allow up to ~3000 tokens (approximately 2250 words)
    # Rough estimate: 1 token ≈ 0.75 words, so 3000 tokens ≈ 2250 words
    words = [w for w in base.split(" ") if w]
    max_words = 2500  # Very generous limit for extremely comprehensive descriptions
    
    if len(words) > max_words:
        # Truncate at sentence boundary if possible
        truncated = " ".join(words[:max_words])
        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        last_exclamation = truncated.rfind("!")
        last_question = truncated.rfind("?")
        last_sentence_end = max(last_period, last_exclamation, last_question)
        if last_sentence_end > max_words * 0.7:  # If we found a sentence end in the last 30%
            out = truncated[:last_sentence_end + 1].strip()
        else:
            # Otherwise, add ellipsis to indicate truncation
            out = truncated.rstrip(" ,.;:") + "..."
    else:
        out = base
    
    # Ensure it ends with punctuation
    if out and not out.endswith((".", "!", "?", "...")):
        out += "."
    
    return out.strip()


def _select_obstacles(mentions: Iterable[ObjectMention], *, depth: Optional[np.ndarray], cfg: NavigationConfig) -> List[NavObstacle]:
    # Score mentions by closeness (distance) then by size.
    candidates: List[Tuple[float, NavObstacle]] = []

    depth_near: Optional[float] = None
    depth_far: Optional[float] = None
    if isinstance(depth, np.ndarray) and depth.ndim == 2:
        try:
            depth_near = float(np.percentile(depth, cfg.near_percentile))
            depth_far = float(np.percentile(depth, cfg.far_percentile))
        except Exception:
            depth_near = None
            depth_far = None

    for m in mentions:
        name = (m.name or "").strip()
        if not name:
            continue
        lname = name.lower()

        # Skip purely non-actionable UI-like items unless they are very close/large.
        if lname in _NON_OBSTACLE_HINTS and float(m.size or 0.0) < 0.05:
            continue

        # Prefer exact normalized centers when available.
        clock = 12
        if m.x_center is not None and m.y_center is not None:
            try:
                clock = clock_from_normalized_xy(float(m.x_center), float(m.y_center))
            except Exception:
                clock = 12
        elif m.position is not None:
            # Fallback to bucket-based clock
            clock = _clock_from_bucket(m.position.vertical, m.position.horizontal)

        # Distance: prefer raw depth lookup, else fallback to bbox area.
        dist_m: Optional[float] = None
        if depth is not None and m.x_center is not None and m.y_center is not None and depth_near is not None and depth_far is not None:
            dist_m = _estimate_distance_m_from_depth_precomputed(
                depth=depth,
                x_center=float(m.x_center),
                y_center=float(m.y_center),
                cfg=cfg,
                near=depth_near,
                far=depth_far,
            )
        if dist_m is None:
            dist_m = estimate_distance_m_from_bbox_area(m.size, cfg)

        # Score: prefer closer distances; if unknown, push back.
        score = float(dist_m) if dist_m is not None else cfg.max_distance_m + 1.0
        candidates.append((score, NavObstacle(name=name, clock=clock, distance_m=dist_m)))

    candidates.sort(key=lambda x: x[0])
    obstacles = [o for _, o in candidates][: int(cfg.max_obstacles)]

    # Filter out very far items (keeps labels tight)
    out: List[NavObstacle] = []
    for o in obstacles:
        if o.distance_m is not None and o.distance_m > cfg.max_distance_m:
            continue
        out.append(o)
    return out


def _estimate_distance_m_from_depth_precomputed(
    *,
    depth: np.ndarray,
    x_center: float,
    y_center: float,
    cfg: NavigationConfig,
    near: float,
    far: float,
) -> Optional[float]:
    """Same as `estimate_distance_m_from_depth`, but uses precomputed near/far percentiles."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return None
    h, w = depth.shape[:2]
    px = int(round(float(x_center) * (w - 1)))
    py = int(round(float(y_center) * (h - 1)))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    v = float(depth[py, px])

    denom = (far - near) if (far - near) != 0.0 else 1e-6
    if cfg.depth_increases_with_distance:
        t = (v - near) / denom
    else:
        t = (far - v) / denom
    t = max(0.0, min(1.0, float(t)))
    return float(cfg.min_distance_m + t * (cfg.max_distance_m - cfg.min_distance_m))


def _clock_from_bucket(vertical: str, horizontal: str) -> int:
    v = (vertical or "").lower()
    h = (horizontal or "").lower()
    if v == "upper":
        if h == "left":
            return 11
        if h == "right":
            return 1
        return 12
    if v == "middle":
        if h == "left":
            return 9
        if h == "right":
            return 3
        return 12
    # lower
    if h == "left":
        return 7
    if h == "right":
        return 5
    return 6


# Clock positions that count as front, left, or right (any object here means risk cannot be Low)
_FRONT_LEFT_RIGHT_CLOCKS = {9, 10, 11, 12, 1, 2, 3}


def _risk_level(obstacles: Sequence[NavObstacle], *, cfg: NavigationConfig) -> str:
    """Compute risk level: Low | Medium | High | Extreme.

    Low: ONLY when path is clear - no object/obstacle in your front, left, or right.
    Any obstacle in front or left or right means risk cannot be Low.
    """
    if not obstacles:
        return "Low"

    # Any obstacle in front, left, or right -> cannot be Low (clear space required for Low)
    has_front_left_right = any(o.clock in _FRONT_LEFT_RIGHT_CLOCKS for o in obstacles)
    if not has_front_left_right:
        return "Low"

    # Extreme: fall-risk or drop-off very close (can hit in <1 sec)
    for o in obstacles:
        lname = o.name.lower()
        if lname in _EXTREME_RISK and o.distance_m is not None and o.distance_m <= float(cfg.stop_distance_m):
            return "Extreme"
        if lname in _EXTREME_RISK and o.clock in {5, 6, 7}:
            return "Extreme"

    # High: high-risk hazard close, or wall/door directly ahead at stop distance
    for o in obstacles:
        lname = o.name.lower()
        if lname in _HIGH_RISK and (o.distance_m is not None and o.distance_m <= float(cfg.caution_distance_m)):
            return "High"

    for o in obstacles:
        if o.distance_m is None:
            continue
        if o.clock in {11, 12, 1} and o.distance_m <= float(cfg.stop_distance_m):
            if o.name.lower() in _COMMON_BLOCKERS:
                return "High"

    # Medium: obstacle in front or side (potential hazard) or close
    for o in obstacles:
        if o.distance_m is None:
            continue
        if o.clock in {10, 11, 12, 1, 2} and o.distance_m <= float(cfg.caution_distance_m):
            return "Medium"
        if o.distance_m <= 1.0 and o.clock == 6:
            return "Medium"

    # Obstacle(s) in front/left/right but far -> still not Low; use Medium
    return "Medium"


def _guidance_command(*, risk: str, obstacles: Sequence[NavObstacle], cfg: NavigationConfig) -> str:
    if not obstacles:
        return "Path clear, move forward."

    # Extreme: stop immediately - falling or hit in <1 sec
    if risk == "Extreme":
        for o in obstacles:
            lname = o.name.lower()
            if lname in _EXTREME_RISK:
                return f"Stop immediately! {o.name} directly ahead - dangerous. Do not proceed."
        return "Stop immediately! Danger ahead - do not proceed."

    # High: hazardous in few sec if not aligned
    if risk == "High":
        for o in obstacles:
            lname = o.name.lower()
            if lname in _HIGH_RISK:
                return f"Stop! {o.name} ahead. Proceed carefully or turn back."
        for o in obstacles:
            if o.name.lower() in _COMMON_BLOCKERS and o.distance_m is not None and o.distance_m <= float(cfg.stop_distance_m):
                return "Turn around, path blocked!"
        return "Stop! Obstacle ahead."

    # Medium: steer away from nearest front obstacle (potential hazard)
    if risk == "Medium":
        front = [o for o in obstacles if o.clock in {10, 11, 12, 1, 2} and o.distance_m is not None]
        front.sort(key=lambda o: float(o.distance_m))
        if front:
            o = front[0]
            # If obstacle is on right-ish, move left (10 o'clock); if left-ish, move right (2 o'clock).
            if o.clock in {1, 2, 3}:
                return f"Move to 10 o'clock to avoid {o.name}."
            if o.clock in {10, 11, 9}:
                return f"Move to 2 o'clock to avoid {o.name}."
            return f"Slow down, {o.name} ahead."
        return "Slow down, obstacle nearby."

    # Low: if a door is present, guide toward it; else proceed.
    for o in obstacles:
        if o.name.lower() == "door":
            return "Proceed forward toward door."
    return "Proceed straight ahead."

