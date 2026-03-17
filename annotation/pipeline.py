"""High-level dataset annotation pipeline (Phase 5).

This module generates per-image textual annotations suitable for VLM fine-tuning.
It can leverage auxiliary signals saved by existing pipelines:
- Phase 2: `Detection/metadata/*.json`
- Phase 3: `Segmentation/metadata/*.json` or `SAM_Segmentation/metadata/*.json`
- Phase 4: `Depth/metadata/*.json`

Outputs:
- `annotations/*.json` (per-image annotation records)
- `training_data/*` exports (LLaVA/Alpaca/ShareGPT)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .captioners import CaptionContext, ObjectMention, QwenCaptioner, GPTCaptioner, TemplateCaptioner
from .formatter import ExportConfig, export_training_data
from .prompts import (
    build_prompt, build_scene_description_prompt, bbox_to_spatial_position, bbox_to_distance_estimate,
    build_accessibility_prompt, parse_accessibility_response, infer_location, infer_time_of_day,
    generate_highlights, generate_ground_text, shorten_guidance
)
from .schemas import AnnotationRecord, DepthMetadata, DetectionMetadata, SegmentationMetadata, AccessibilityData, RiskAssessment, RiskObstacle
from .spatial import bbox_to_relative_position
from .navigation import NavigationConfig, format_navigation_output, generate_navigation_guidance, NavObstacle
from .nav_prompts import NAV_SYSTEM_PROMPT, build_navigation_prompt
from .nav_parse import normalize_navigation_response, parse_navigation_response
from .nav_llm import ClaudeNavGenerator, QwenNavGenerator, GPTNavGenerator, OpenRouterNavGenerator
from .depth_utils import depth_to_grayscale_png_bytes
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)

# For type checking only
if TYPE_CHECKING:
    import numpy as np

# =============================================================================
# Accessibility risk assessment (deterministic safety rules)
# =============================================================================

_DYNAMIC_TOKENS = {
    "person", "people", "man", "woman", "child", "boy", "girl",
    "dog", "cat", "pet",
    "cart", "trolley", "stroller", "wheelchair",
    "bicycle", "bike", "scooter",
    "vehicle", "car", "bus", "truck", "forklift",
}
_DYNAMIC_PHRASES = {
    "hand truck",
    "shopping cart",
    "baby stroller",
}
_FALL_RISK_TOKENS = {
    "stairs", "stair", "staircase",
    "escalator",
    "dropoff", "drop-off", "drop off",
    "hole", "gap", "missing",
}
_FALL_RISK_PHRASES = {
    "moving stairs",
    "escalator roller",
    "missing floor",
}
_WALL_TOKENS = {"wall"}


def _tokenize_name(name: str) -> List[str]:
    import re
    tokens = re.split(r"[^a-z0-9]+", (name or "").lower())
    return [t for t in tokens if t]


def _has_phrase(name: str, phrases: set) -> bool:
    lname = (name or "").lower()
    return any(p in lname for p in phrases)


def _is_dynamic_object(name: str) -> bool:
    tokens = _tokenize_name(name)
    if _has_phrase(name, _DYNAMIC_PHRASES):
        return True
    return any(t in _DYNAMIC_TOKENS for t in tokens)


def _is_fall_risk_object(name: str) -> bool:
    tokens = _tokenize_name(name)
    if _has_phrase(name, _FALL_RISK_PHRASES):
        return True
    return any(t in _FALL_RISK_TOKENS for t in tokens)


def _is_wall_object(name: str) -> bool:
    tokens = _tokenize_name(name)
    return any(t in _WALL_TOKENS for t in tokens)


def _parse_spatial_objects_to_risk_obstacles(spatial_objects: List[str]) -> List[RiskObstacle]:
    """Parse spatial_objects strings into RiskObstacle list (one per object).

    Expects: 'Obstacle in your [front|left|right]: [object name] at [clock], [distance]m away'
    """
    result: List[RiskObstacle] = []
    for line in spatial_objects or []:
        s = (line or "").strip()
        if not s or "path clear" in s.lower():
            continue
        m = re.search(
            r"obstacle\s+in\s+your\s+(front|left|right)\s*:\s*(.+?)(?:\s+at\s+\d+\s*o['']?clock)?(?:\s*,\s*([\d.]+)\s*m\s+away)?\s*$",
            s,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue
        position = m.group(1).strip().lower()
        object_name = (m.group(2) or "").strip().rstrip(",")
        if not object_name:
            continue
        dist_m: Optional[float] = None
        if m.lastindex >= 3 and m.group(3):
            try:
                dist_m = float(m.group(3))
            except (ValueError, TypeError):
                pass
        distance = _distance_bucket(dist_m)
        dynamic = _is_dynamic_object(object_name)
        fall_risk = _is_fall_risk_object(object_name)
        result.append(
            RiskObstacle(
                type="dynamic" if dynamic else "static",
                object=object_name,
                position=position,
                distance=distance,
                motion="moving" if dynamic else "stationary",
                fall_risk="yes" if fall_risk else "no",
            )
        )
    return result


def _position_from_clock(clock: int) -> str:
    if clock in {11, 12, 1}:
        return "front"
    if clock in {9, 10}:
        return "left"
    if clock in {2, 3}:
        return "right"
    return "none"


def _distance_bucket(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "medium"
    d = float(distance_m)
    if d <= 0.8:
        return "very near"
    if d <= 1.6:
        return "near"
    if d <= 3.2:
        return "medium"
    return "far"


def _risk_score_from_level(level: str) -> float:
    """Map risk level to score 0–1. Extreme=0.98, High=0.8, Medium=0.5, Low=0.2."""
    level = (level or "").upper()
    if level == "EXTREME":
        return 0.98
    if level == "HIGH":
        return 0.8
    if level == "MEDIUM":
        return 0.5
    return 0.2


def _risk_reason(obstacles: List[RiskObstacle], *, level: str) -> str:
    level = (level or "").upper()
    if not obstacles:
        return "Path is clear with no nearby hazards."

    def pos_phrase(pos: str) -> str:
        if pos == "front":
            return "ahead"
        if pos == "left":
            return "on the left"
        if pos == "right":
            return "on the right"
        return "nearby"

    for o in obstacles:
        if o.fall_risk == "yes":
            return f"Fall risk from {o.object} {pos_phrase(o.position)}."
    # Extreme: falling, hit in <1 sec, dangerous
    if level == "EXTREME":
        for o in obstacles:
            if o.position == "front" and o.distance in {"very near", "near"} and _is_wall_object(o.object):
                return "Wall directly ahead at close distance."
        for o in obstacles:
            if o.position == "front" and o.distance == "very near":
                return f"Very close obstacle ahead ({o.object})."
        return "Immediate collision or fall risk ahead."
    # High: hazardous in few sec if not aligned
    if level == "HIGH":
        for o in obstacles:
            if o.motion == "moving" and o.position in {"left", "right"}:
                return f"Moving {o.object} close on the {o.position} poses collision risk."
        for o in obstacles:
            if o.position == "front":
                return f"{o.object.capitalize()} directly ahead requires immediate attention."
        return "Multiple nearby obstacles narrow the walking corridor."
    # Medium: potential hazard
    if level == "MEDIUM":
        for o in obstacles:
            if o.position == "front":
                return "Obstacle ahead at some distance; proceed carefully."
        return "Obstacle on the side with sufficient clearance."
    # Low: path clear
    return "Path is clear with no nearby hazards."


def _compute_accessibility_risk(
    obstacles: List["NavObstacle"],
    *,
    scene_summary: Optional[str] = None,
    spatial_objects: Optional[List[str]] = None,
) -> RiskAssessment:
    """Compute risk from nav obstacles; fallback to spatial_objects when obstacles empty."""
    risk_obstacles: List[RiskObstacle] = []
    for o in obstacles:
        name = (o.name or "object").strip()
        position = _position_from_clock(int(o.clock))
        distance = _distance_bucket(o.distance_m)
        dynamic = _is_dynamic_object(name)
        fall_risk = _is_fall_risk_object(name)
        risk_obstacles.append(
            RiskObstacle(
                type="dynamic" if dynamic else "static",
                object=name,
                position=position,
                distance=distance,
                motion="moving" if dynamic else "stationary",
                fall_risk="yes" if fall_risk else "no",
            )
        )

    if not risk_obstacles and spatial_objects:
        risk_obstacles = _parse_spatial_objects_to_risk_obstacles(spatial_objects)

    # No placeholder "none" when empty; use empty list
    # (obstacles list remains [] when path is clear)

    # Extreme: falling, hit in <1 sec, dangerous
    if any(o.fall_risk == "yes" for o in risk_obstacles):
        level = "Extreme"
    elif any(
        o.position == "front"
        and o.distance in {"very near", "near"}
        and _is_wall_object(o.object)
        for o in risk_obstacles
    ):
        level = "Extreme"
    elif any(o.position == "front" and o.distance == "very near" for o in risk_obstacles):
        level = "Extreme"
    else:
        # High: hazardous in few sec if not aligned
        dynamic_side = any(
            o.motion == "moving" and o.position in {"left", "right"} and o.distance in {"very near", "near", "medium"}
            for o in risk_obstacles
        )
        front_near = any(o.position == "front" and o.distance in {"very near", "near"} for o in risk_obstacles)
        front_moving = any(o.position == "front" and o.motion == "moving" for o in risk_obstacles)
        close_count = sum(
            1 for o in risk_obstacles
            if o.position in {"front", "left", "right"} and o.distance in {"very near", "near"}
        )
        # Moving objects escalate risk: front_moving or dynamic_side -> High
        moving_count = sum(
            1 for o in risk_obstacles
            if o.motion == "moving" and o.position in {"front", "left", "right"}
        )
        if front_moving or dynamic_side or front_near or close_count >= 2 or moving_count >= 2:
            level = "High"
        elif moving_count >= 1:
            # Single moving object in front/left/right -> at least Medium, High if near
            level = "High" if front_near or close_count >= 1 else "Medium"
        else:
            # Low only when no obstacle in front, left, or right (clear space)
            front_any = any(o.position == "front" for o in risk_obstacles)
            side_any = any(o.position in {"left", "right"} for o in risk_obstacles)
            if front_any or side_any:
                level = "Medium"
            else:
                level = "Low"

    reason = _risk_reason(risk_obstacles, level=level)
    return RiskAssessment(
        level=level,
        score=_risk_score_from_level(level),
        reason=reason,
        scene_summary=scene_summary,
        obstacles=risk_obstacles,
    )


@dataclass
class AnnotationRunConfig:
    """Config for running dataset annotation generation."""

    task: str = "caption"  # caption|navigation|scene|accessibility
    caption_model: str = "template"  # template|qwen|gpt|claude
    caption_device: str = "auto"  # for qwen
    caption_max_tokens: int = 1000  # max tokens for GPT/Qwen/Claude caption generation
    gpt_model: str = "gpt-4o"  # GPT model id for caption/scene/nav
    max_mentions: int = 10  # max mentions included in aux summary
    max_objects_in_template: int = 6
    export_formats: List[str] = None  # ["all"] by default
    create_image_copies: bool = False
    instruction: str = "Describe the scene for navigation assistance."
    system_prompt: Optional[str] = None  # used for ShareGPT exports (optional)
    skip_existing: bool = False
    annotation_version: str = "comprehensive"  # simplified|comprehensive (controls export fields)

    # Navigation task knobs
    nav_backend: str = "openrouter"  # deterministic|claude|qwen|gpt|openrouter|auto (default openrouter = Qwen instruct)
    claude_model: str = "claude-3-7-sonnet-20250219"
    openrouter_model: str = "qwen/qwen3-vl-8b-instruct"
    nav_max_tokens: int = 1000
    nav_temperature: float = 0.2
    nav_include_depth_image: bool = True
    qwen_max_new_tokens: int = 280

    nav_min_distance_m: float = 0.5
    nav_max_distance_m: float = 8.0
    nav_max_obstacles: int = 6
    nav_stop_distance_m: float = 0.8
    nav_caution_distance_m: float = 2.0

    # Vision cost controls (only affect GPT/Claude API calls)
    image_detail: str = "auto"  # auto|low|high (GPT only)
    resize_720p: bool = False  # if True, resize to fit within 1280x720 before sending
    use_bboxes: bool = True  # if False, avoid bbox-based spatial positions (clock+distance only)


class AnnotationPipeline:
    """Generate dataset annotations and exports."""

    def __init__(self, cfg: AnnotationRunConfig):
        self.cfg = cfg
        if self.cfg.export_formats is None:
            self.cfg.export_formats = ["all"]

        self._template = TemplateCaptioner(max_objects=self.cfg.max_objects_in_template)
        self._qwen: Optional[QwenCaptioner] = None
        self._gpt: Optional[GPTCaptioner] = None
        self._claude_nav: Optional[ClaudeNavGenerator] = None
        self._openrouter_nav: Optional[OpenRouterNavGenerator] = None
        self._qwen_nav: Optional[QwenNavGenerator] = None
        self._gpt_nav: Optional[GPTNavGenerator] = None
        self._cost_tracker = CostTracker()

    def _get_qwen(self) -> QwenCaptioner:
        if self._qwen is None:
            self._qwen = QwenCaptioner(device=self.cfg.caption_device)
        return self._qwen

    def _get_gpt(self) -> GPTCaptioner:
        if self._gpt is None:
            self._gpt = GPTCaptioner(model=self.cfg.gpt_model, max_tokens=self.cfg.caption_max_tokens)
        return self._gpt

    def _get_claude_nav(self) -> ClaudeNavGenerator:
        if self._claude_nav is None:
            self._claude_nav = ClaudeNavGenerator()
        return self._claude_nav

    def _get_openrouter_nav(self) -> OpenRouterNavGenerator:
        if self._openrouter_nav is None:
            self._openrouter_nav = OpenRouterNavGenerator(
                model_id=str(self.cfg.openrouter_model),
                max_tokens=int(self.cfg.nav_max_tokens),
            )
        return self._openrouter_nav

    def _get_qwen_nav(self) -> QwenNavGenerator:
        if self._qwen_nav is None:
            self._qwen_nav = QwenNavGenerator(device=self.cfg.caption_device)
        return self._qwen_nav

    def _get_gpt_nav(self) -> GPTNavGenerator:
        if self._gpt_nav is None:
            self._gpt_nav = GPTNavGenerator(model=self.cfg.gpt_model, max_tokens=self.cfg.nav_max_tokens)
        return self._gpt_nav

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_detection(self, meta_path: Optional[Path]) -> Optional[DetectionMetadata]:
        if not meta_path or not meta_path.exists():
            return None
        try:
            return DetectionMetadata.model_validate(self._read_json(meta_path))
        except Exception as e:
            logger.debug(f"Failed to parse detection metadata: {meta_path} ({e})")
            return None

    def _load_segmentation(self, meta_path: Optional[Path]) -> Optional[SegmentationMetadata]:
        if not meta_path or not meta_path.exists():
            return None
        try:
            return SegmentationMetadata.model_validate(self._read_json(meta_path))
        except Exception as e:
            logger.debug(f"Failed to parse segmentation metadata: {meta_path} ({e})")
            return None

    def _load_depth(self, meta_path: Optional[Path]) -> Optional[DepthMetadata]:
        if not meta_path or not meta_path.exists():
            return None
        try:
            return DepthMetadata.model_validate(self._read_json(meta_path))
        except Exception as e:
            logger.debug(f"Failed to parse depth metadata: {meta_path} ({e})")
            return None

    def _load_vlm_json(self, vlm_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not vlm_path or not vlm_path.exists():
            return None
        try:
            return self._read_json(vlm_path)
        except Exception:
            return None

    def _vlm_objects_from_payload(self, payload: Optional[Dict[str, Any]]) -> Tuple[List[str], str, str]:
        """Return (objects, scene_description, primary_focus) from a VLM payload."""
        if not payload or not isinstance(payload, dict):
            return [], "", ""
        obj_block = payload.get("objects")
        if not isinstance(obj_block, dict):
            return [], "", ""
        objs = obj_block.get("objects") or []
        if not isinstance(objs, list):
            objs = []
        objs = [str(o).strip() for o in objs if str(o).strip()]
        scene = str(obj_block.get("scene_description") or "").strip()
        focus = str(obj_block.get("primary_focus") or "").strip()
        return objs, scene, focus

    def _build_mentions(
        self,
        det: Optional[DetectionMetadata],
        seg: Optional[SegmentationMetadata],
    ) -> List[ObjectMention]:
        mentions: List[ObjectMention] = []

        def _add(items: List[Any], img_w: Optional[int], img_h: Optional[int]):
            for d in items:
                try:
                    name = str(d.class_name).strip()
                    if not name:
                        continue
                    xc = d.x_center
                    yc = d.y_center
                    w = d.width
                    h = d.height
                    if (xc is None or yc is None or w is None or h is None) and d.bbox_xyxy and img_w and img_h:
                        x1, y1, x2, y2 = [float(v) for v in d.bbox_xyxy[:4]]
                        xc = ((x1 + x2) / 2.0) / float(img_w)
                        yc = ((y1 + y2) / 2.0) / float(img_h)
                        w = (x2 - x1) / float(img_w)
                        h = (y2 - y1) / float(img_h)
                    pos = bbox_to_relative_position(x_center=xc, y_center=yc, width=w, height=h)
                    size = float(w) * float(h) if (w is not None and h is not None) else None
                    mentions.append(
                        ObjectMention(
                            name=name,
                            position=pos,
                            confidence=float(d.confidence) if d.confidence is not None else None,
                            size=size,
                            x_center=float(xc) if xc is not None else None,
                            y_center=float(yc) if yc is not None else None,
                        )
                    )
                except Exception:
                    continue

        if det is not None:
            _add(det.detections, det.image_width, det.image_height)
        if seg is not None:
            _add(seg.detections, seg.image_width, seg.image_height)

        # Deduplicate by keeping the biggest/most confident mention per class name.
        best: Dict[str, ObjectMention] = {}
        for m in mentions:
            k = m.name.lower()
            cur = best.get(k)
            if cur is None:
                best[k] = m
                continue
            if float(m.size or 0.0) > float(cur.size or 0.0):
                best[k] = m
                continue
            if float(m.confidence or 0.0) > float(cur.confidence or 0.0):
                best[k] = m
        return list(best.values())

    def _load_depth_array(
        self,
        *,
        depth_meta: Optional[DepthMetadata],
        depth_metadata_dir: Optional[Path],
        image_stem: str,
    ) -> Optional["np.ndarray"]:
        """Load a raw depth array if available."""
        try:
            import numpy as np

            raw_path: Optional[Path] = None
            if depth_meta is not None and depth_meta.raw_depth_path:
                raw_path = Path(str(depth_meta.raw_depth_path))
                if not raw_path.is_absolute():
                    raw_path = raw_path.resolve()

            # If metadata didn't include raw path, infer from depth output directory.
            if (raw_path is None or not raw_path.exists()) and depth_metadata_dir is not None:
                candidate = (depth_metadata_dir.parent / "depth_raw" / f"{image_stem}.npy").resolve()
                if candidate.exists():
                    raw_path = candidate

            if raw_path is None or not raw_path.exists():
                return None

            arr = np.load(str(raw_path))
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr
        except Exception:
            return None
        return None

    def _build_aux_summary(
        self,
        ctx: CaptionContext,
        depth: Optional[DepthMetadata],
    ) -> str:
        lines: List[str] = []

        if ctx.primary_focus:
            lines.append(f"Primary focus: {ctx.primary_focus}")
        if ctx.scene_description:
            lines.append(f"Scene hint: {ctx.scene_description}")

        # Mentions with coarse locations (no numeric coordinates)
        mention_lines = []
        mentions_sorted = sorted(
            ctx.mentions,
            key=lambda m: (-float(m.size or 0.0), -float(m.confidence or 0.0), m.name),
        )
        for m in mentions_sorted[: int(self.cfg.max_mentions)]:
            if m.position is None:
                mention_lines.append(f"{m.name}: visible")
            else:
                mention_lines.append(f"{m.name}: {m.position.to_phrase()}")
        if mention_lines:
            lines.append("Detected objects (coarse positions): " + "; ".join(mention_lines))

        # Depth availability note (avoid numbers)
        if depth is not None:
            if depth.raw_depth_path:
                lines.append("Depth map: available (raw depth saved).")
            else:
                lines.append("Depth map: available (summary stats only).")

        return "\n".join(lines).strip()

    def _build_caption_context(
        self,
        det: Optional[DetectionMetadata],
        seg: Optional[SegmentationMetadata],
        vlm_payload: Optional[Dict[str, Any]],
    ) -> CaptionContext:
        vlm_objects, scene_desc, primary_focus = self._vlm_objects_from_payload(vlm_payload)

        # Merge object names from detections/segmentation
        det_names = [d.class_name for d in det.detections] if det is not None else []
        seg_names = [d.class_name for d in seg.detections] if seg is not None else []
        merged = []
        seen = set()
        for name in [*vlm_objects, *det_names, *seg_names]:
            s = str(name).strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            merged.append(s)

        mentions = self._build_mentions(det, seg)
        return CaptionContext(
            scene_description=scene_desc,
            primary_focus=primary_focus,
            objects=merged,
            mentions=mentions,
        )

    def _build_objects_with_positions(
        self,
        det: Optional[DetectionMetadata],
        seg: Optional[SegmentationMetadata],
        depth: Optional[DepthMetadata],
    ) -> List[Dict[str, Any]]:
        """Build list of objects with spatial positions from detection/segmentation metadata."""
        objects: List[Dict[str, Any]] = []
        seen_names: set = set()
        
        # Process detections first (usually have better bbox info)
        if det is not None:
            for d in det.detections:
                name = d.class_name.strip()
                if not name or name.lower() in seen_names:
                    continue
                
                # Get spatial position from bbox
                if d.x_center is not None and d.y_center is not None and d.width is not None and d.height is not None:
                    position = bbox_to_spatial_position(
                        float(d.x_center), float(d.y_center), 
                        float(d.width), float(d.height)
                    )
                    distance = bbox_to_distance_estimate(float(d.width), float(d.height))
                else:
                    position = "visible in scene"
                    distance = ""
                
                objects.append({
                    "name": name,
                    "position": position,
                    "distance": distance,
                    "confidence": float(d.confidence) if d.confidence else 0.0,
                })
                seen_names.add(name.lower())
        
        # Add segmentation objects not already in detections
        if seg is not None:
            for s in seg.detections:
                name = s.class_name.strip()
                if not name or name.lower() in seen_names:
                    continue
                
                if s.x_center is not None and s.y_center is not None and s.width is not None and s.height is not None:
                    position = bbox_to_spatial_position(
                        float(s.x_center), float(s.y_center),
                        float(s.width), float(s.height)
                    )
                    distance = bbox_to_distance_estimate(float(s.width), float(s.height))
                else:
                    position = "visible in scene"
                    distance = ""
                
                objects.append({
                    "name": name,
                    "position": position,
                    "distance": distance,
                    "confidence": float(s.confidence) if s.confidence else 0.0,
                })
                seen_names.add(name.lower())
        
        # Sort by confidence (highest first) and limit
        objects.sort(key=lambda x: -x.get("confidence", 0.0))
        return objects[:15]  # Limit to top 15 objects

    def _generate_scene_template(
        self,
        objects_with_positions: List[Dict[str, Any]],
        ctx: CaptionContext,
        depth_info: Optional[str],
    ) -> str:
        """Generate a template-based scene description (fallback when VLM unavailable)."""
        parts = []
        
        # Scene context
        if ctx.scene_description:
            parts.append(ctx.scene_description)
        elif ctx.primary_focus:
            parts.append(f"A scene featuring {ctx.primary_focus}.")
        else:
            parts.append("A scene with various objects visible.")
        
        # Object descriptions with positions
        if objects_with_positions:
            obj_descs = []
            for obj in objects_with_positions[:8]:  # Limit for readability
                name = obj.get("name", "object")
                position = obj.get("position", "visible")
                distance = obj.get("distance", "")
                if distance:
                    obj_descs.append(f"{name} ({position}, {distance})")
                else:
                    obj_descs.append(f"{name} ({position})")
            
            if len(obj_descs) == 1:
                parts.append(f"The scene contains {obj_descs[0]}.")
            elif len(obj_descs) == 2:
                parts.append(f"The scene contains {obj_descs[0]} and {obj_descs[1]}.")
            else:
                last = obj_descs[-1]
                rest = ", ".join(obj_descs[:-1])
                parts.append(f"Key objects include {rest}, and {last}.")
        
        # Depth information
        if depth_info:
            parts.append(depth_info + ".")
        
        return " ".join(parts)

    def process_image(
        self,
        *,
        image_path: Path,
        output_dir: Path,
        images_dir: Path,
        detection_metadata_dir: Optional[Path] = None,
        segmentation_metadata_dir: Optional[Path] = None,
        depth_metadata_dir: Optional[Path] = None,
        vlm_json_dir: Optional[Path] = None,
    ) -> Optional[AnnotationRecord]:
        """Generate and save annotation for a single image."""
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        ann_path = annotations_dir / f"{image_path.stem}.json"
        if self.cfg.skip_existing and ann_path.exists():
            try:
                rec = AnnotationRecord.model_validate(self._read_json(ann_path))
                logger.debug("Skipping %s (already annotated)", image_path.name)
                return rec
            except Exception:
                return None

        det_path = detection_metadata_dir / f"{image_path.stem}.json" if detection_metadata_dir else None
        seg_path = segmentation_metadata_dir / f"{image_path.stem}.json" if segmentation_metadata_dir else None
        depth_path = depth_metadata_dir / f"{image_path.stem}.json" if depth_metadata_dir else None

        det = self._load_detection(det_path)
        seg = self._load_segmentation(seg_path)
        depth = self._load_depth(depth_path)
        depth_arr = self._load_depth_array(depth_meta=depth, depth_metadata_dir=depth_metadata_dir, image_stem=image_path.stem)

        # Prefer VLM payload already embedded in detection metadata (Phase 2).
        vlm_payload = det.vlm_result if (det and det.vlm_result) else None

        # Otherwise try loading from VLM_analysis directory.
        if vlm_payload is None and vlm_json_dir is not None:
            base = image_path.stem.replace("_keyframe", "")
            vlm_path = vlm_json_dir / f"{base}_vlm.json"
            vlm_payload = self._load_vlm_json(vlm_path)

        ctx = self._build_caption_context(det, seg, vlm_payload)
        aux_summary = self._build_aux_summary(ctx, depth)

        # Generate text using selected task/backend.
        task = str(self.cfg.task).lower().strip()
        text = ""
        accessibility_data = None  # Will be set only for accessibility task

        meta: Dict[str, Any] = {}
        if task == "navigation":
            nav_cfg = NavigationConfig(
                min_distance_m=float(self.cfg.nav_min_distance_m),
                max_distance_m=float(self.cfg.nav_max_distance_m),
                max_obstacles=int(self.cfg.nav_max_obstacles),
                stop_distance_m=float(self.cfg.nav_stop_distance_m),
                caution_distance_m=float(self.cfg.nav_caution_distance_m),
            )
            # Build candidate obstacles (also used for deterministic fallback).
            scene, risk, obstacles, guidance = generate_navigation_guidance(ctx=ctx, depth=depth_arr, cfg=nav_cfg)

            backend = str(self.cfg.nav_backend).lower().strip()
            if backend not in {"deterministic", "claude", "qwen", "gpt", "openrouter", "auto"}:
                backend = "deterministic"

            # Auto selection: prefer Claude if available, else GPT, else OpenRouter, else Qwen, else deterministic.
            if backend == "auto":
                c = self._get_claude_nav()
                if c.enabled:
                    backend = "claude"
                else:
                    g = self._get_gpt_nav()
                    if g.enabled:
                        backend = "gpt"
                    else:
                        o = self._get_openrouter_nav()
                        if o.enabled:
                            backend = "openrouter"
                        else:
                            q = self._get_qwen_nav()
                            backend = "qwen" if q.enabled else "deterministic"

            depth_png = None
            if self.cfg.nav_include_depth_image and depth_arr is not None:
                depth_png = depth_to_grayscale_png_bytes(depth_arr, darker_is_closer=True, depth_is_disparity=True)

            # LLM-backed navigation generation
            if backend in {"claude", "qwen", "gpt", "openrouter"}:
                # Build comprehensive context for better scene descriptions
                extra_notes_parts = []
                if ctx.scene_description:
                    extra_notes_parts.append(f"Scene context: {ctx.scene_description}")
                if ctx.primary_focus:
                    extra_notes_parts.append(f"Primary focus: {ctx.primary_focus}")
                if ctx.objects:
                    obj_list = ", ".join(ctx.objects[:10])  # Include up to 10 objects
                    extra_notes_parts.append(f"Detected objects in scene: {obj_list}")
                if depth is not None:
                    extra_notes_parts.append(
                        f"Depth statistics: min={depth.depth_min:.2f}m, max={depth.depth_max:.2f}m, "
                        f"mean={depth.depth_mean:.2f}m (relative depth values)"
                    )
                
                extra_notes = "\n".join(extra_notes_parts) if extra_notes_parts else ""

                prompt = build_navigation_prompt(obstacle_candidates=obstacles, extra_notes=extra_notes)
                raw = None
                try:
                    if backend == "claude":
                        client = self._get_claude_nav()
                        if not client.enabled:
                            raise RuntimeError("Claude backend not available.")
                        raw = client.generate(
                            image_path=image_path,
                            prompt=prompt,
                            system_prompt=NAV_SYSTEM_PROMPT,
                            model=str(self.cfg.claude_model),
                            max_tokens=int(self.cfg.nav_max_tokens),
                            temperature=float(self.cfg.nav_temperature),
                            depth_png_bytes=depth_png,
                            cost_tracker=self._cost_tracker,
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                    elif backend == "qwen":
                        qwen = self._get_qwen_nav()
                        if not qwen.enabled:
                            raise RuntimeError("Qwen backend not available.")
                        raw = qwen.generate(image_path=image_path, prompt=prompt, max_new_tokens=int(self.cfg.qwen_max_new_tokens))
                        self._cost_tracker.add_qwen_call()
                    elif backend == "openrouter":
                        openrouter = self._get_openrouter_nav()
                        if not openrouter.enabled:
                            raise RuntimeError("OpenRouter backend not available.")
                        raw = openrouter.generate(
                            image_path=image_path,
                            prompt=prompt,
                            system_prompt=NAV_SYSTEM_PROMPT,
                            model=str(self.cfg.openrouter_model),
                            max_tokens=int(self.cfg.nav_max_tokens),
                            temperature=float(self.cfg.nav_temperature),
                            depth_png_bytes=depth_png,
                            cost_tracker=self._cost_tracker,
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                    else:  # gpt
                        gpt = self._get_gpt_nav()
                        if not gpt.enabled:
                            raise RuntimeError("GPT backend not available.")
                        raw = gpt.generate(
                            image_path=image_path,
                            prompt=prompt,
                            system_prompt=NAV_SYSTEM_PROMPT,
                            max_tokens=int(self.cfg.nav_max_tokens),
                            depth_png_bytes=depth_png,
                            cost_tracker=self._cost_tracker,
                            image_detail=str(self.cfg.image_detail),
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                except Exception as e:
                    logger.warning(f"Navigation LLM backend '{backend}' failed for {image_path.name}: {e}. Falling back to deterministic.")
                    raw = None

                parsed = parse_navigation_response(raw or "")
                if parsed is not None:
                    text = normalize_navigation_response(parsed)
                    if backend == "claude":
                        model_name = str(self.cfg.claude_model)
                    elif backend == "gpt":
                        model_name = str(self.cfg.gpt_model)
                    elif backend == "openrouter":
                        model_name = str(self.cfg.openrouter_model)
                    else:
                        model_name = f"{backend}-navigation"
                    meta = {
                        "scene": parsed.scene,
                        "risk": parsed.risk,
                        "obstacles": parsed.obstacles_lines,
                        "guidance": parsed.guidance,
                        "backend": backend,
                        "model": model_name,
                    }
                else:
                    # Deterministic fallback output
                    text = format_navigation_output(scene, risk, obstacles, guidance)
                    meta = {
                        "scene": scene,
                        "risk": risk,
                        "obstacles": [
                            {
                                "name": o.name,
                                "clock": int(o.clock),
                                "distance_m": float(o.distance_m) if o.distance_m is not None else None,
                            }
                            for o in obstacles
                        ],
                        "guidance": guidance,
                        "backend": "deterministic_fallback",
                        "model": "deterministic-navigation",
                    }
            else:
                text = format_navigation_output(scene, risk, obstacles, guidance)
                meta = {
                    "scene": scene,
                    "risk": risk,
                    "obstacles": [
                        {
                            "name": o.name,
                            "clock": int(o.clock),
                            "distance_m": float(o.distance_m) if o.distance_m is not None else None,
                        }
                        for o in obstacles
                    ],
                    "guidance": guidance,
                    "backend": "deterministic",
                    "model": "deterministic-navigation",
                }
        elif task == "scene":
            # Professional scene description using bounding boxes for localization
            objects_with_positions = self._build_objects_with_positions(det, seg, depth)
            
            # Build depth info summary
            depth_info = None
            if depth is not None:
                depth_info = f"Objects range from close ({depth.depth_min:.1f}m) to distant ({depth.depth_max:.1f}m)"
            
            scene_prompt = build_scene_description_prompt(
                objects_with_positions=objects_with_positions,
                scene_context=ctx.scene_description or ctx.primary_focus,
                depth_info=depth_info,
            )
            
            model = str(self.cfg.caption_model).lower().strip()
            if model == "qwen":
                try:
                    qwen = self._get_qwen()
                    text = qwen.generate(str(image_path), ctx=ctx, prompt=scene_prompt)
                    self._cost_tracker.add_qwen_call()
                    model_name = "qwen-scene"
                except Exception as e:
                    logger.warning(f"Qwen scene generation failed for {image_path.name}: {e}. Falling back to template.")
                    text = self._generate_scene_template(objects_with_positions, ctx, depth_info)
                    model_name = "template-scene"
            elif model == "gpt":
                try:
                    gpt = self._get_gpt()
                    if not gpt.enabled:
                        # More detailed diagnostics
                        import sys
                        from annotation.captioners import _OPENAI_AVAILABLE
                        diag_msg = (
                            f"GPT captioner not enabled. "
                            f"API key: {bool(gpt.api_key)}, "
                            f"Client: {gpt._client is not None}, "
                            f"OpenAI available: {_OPENAI_AVAILABLE}, "
                            f"Python: {sys.executable}"
                        )
                        logger.error(diag_msg)
                        raise RuntimeError(diag_msg)
                    text = gpt.generate(
                        str(image_path),
                        ctx=ctx,
                        prompt=scene_prompt,
                        cost_tracker=self._cost_tracker,
                        image_detail=str(self.cfg.image_detail),
                        resize_720p=bool(self.cfg.resize_720p),
                    )
                    model_name = f"{self.cfg.gpt_model}-scene"
                except Exception as e:
                    logger.warning(f"GPT scene generation failed for {image_path.name}: {e}. Falling back to template.")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    text = self._generate_scene_template(objects_with_positions, ctx, depth_info)
                    model_name = "template-scene"
            elif model == "openrouter":
                try:
                    openrouter = self._get_openrouter_nav()
                    if not openrouter.enabled:
                        raise RuntimeError("OpenRouter not enabled")
                    text = openrouter.generate(
                        image_path=image_path,
                        prompt=scene_prompt,
                        system_prompt="Generate a professional, detailed scene description. No file paths, model names, or coordinates.",
                        model=str(self.cfg.openrouter_model),
                        max_tokens=int(self.cfg.caption_max_tokens),
                        temperature=0.2,
                        cost_tracker=self._cost_tracker,
                        resize_720p=bool(self.cfg.resize_720p),
                    )
                    model_name = f"{self.cfg.openrouter_model}-scene"
                except Exception as e:
                    logger.warning(f"OpenRouter scene generation failed for {image_path.name}: {e}. Falling back to template.")
                    text = self._generate_scene_template(objects_with_positions, ctx, depth_info)
                    model_name = "template-scene"
            else:
                text = self._generate_scene_template(objects_with_positions, ctx, depth_info)
                model_name = "template-scene"
            
            meta = {
                "task_type": "scene_description",
                "num_objects": len(objects_with_positions),
                "has_depth": depth is not None,
                "model": model_name,
            }
        elif task == "accessibility":
            # Accessibility-focused annotation for blind users
            objects_with_positions = self._build_objects_with_positions(det, seg, depth)
            
            # Build depth info summary
            depth_info = None
            if depth is not None:
                depth_info = f"Objects range from close ({depth.depth_min:.1f}m) to distant ({depth.depth_max:.1f}m)"

            # Build obstacle lines (clock + distance) for no-bbox mode and risk assessment
            nav_cfg = NavigationConfig(
                min_distance_m=float(self.cfg.nav_min_distance_m),
                max_distance_m=float(self.cfg.nav_max_distance_m),
                max_obstacles=int(self.cfg.nav_max_obstacles),
                stop_distance_m=float(self.cfg.nav_stop_distance_m),
                caution_distance_m=float(self.cfg.nav_caution_distance_m),
            )
            _, _, nav_obstacles, _ = generate_navigation_guidance(ctx=ctx, depth=depth_arr, cfg=nav_cfg)
            obstacle_lines: List[str] = []
            for o in nav_obstacles:
                pos = _position_from_clock(int(o.clock))
                if pos == "none":
                    continue
                name = (o.name or "object").strip() or "object"
                clock_phrase = f"{int(o.clock)} o'clock"
                if o.distance_m is None:
                    obstacle_lines.append(f"Obstacle in your {pos}: {name} at {clock_phrase}")
                else:
                    dist = float(o.distance_m)
                    obstacle_lines.append(f"Obstacle in your {pos}: {name} at {clock_phrase}, {dist:.1f}m away")
            
            accessibility_prompt = build_accessibility_prompt(
                objects_with_positions=objects_with_positions,
                scene_context=ctx.scene_description or ctx.primary_focus,
                depth_info=depth_info,
                obstacle_lines=obstacle_lines if not self.cfg.use_bboxes else None,
                use_bboxes=bool(self.cfg.use_bboxes),
            )
            
            # Generate accessibility data using VLM or template
            model = str(self.cfg.caption_model).lower().strip()
            accessibility_data = None
            raw_response = ""
            
            if model in ["gpt", "claude", "openrouter"]:
                try:
                    if model == "gpt":
                        gpt = self._get_gpt()
                        if not gpt.enabled:
                            raise RuntimeError("GPT not enabled")
                        raw_response = gpt.generate(
                            str(image_path),
                            ctx=ctx,
                            prompt=accessibility_prompt,
                            cost_tracker=self._cost_tracker,
                            image_detail=str(self.cfg.image_detail),
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                        model_name = f"{self.cfg.gpt_model}-accessibility"
                    elif model == "claude":
                        claude = self._get_claude_nav()
                        if not claude.enabled:
                            raise RuntimeError("Claude not enabled")
                        raw_response = claude.generate(
                            image_path=image_path,
                            prompt=accessibility_prompt,
                            system_prompt="You are an AI assistant helping blind users understand their surroundings.",
                            model=str(self.cfg.claude_model),
                            max_tokens=int(self.cfg.caption_max_tokens),
                            temperature=0.2,
                            cost_tracker=self._cost_tracker,
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                        model_name = f"{self.cfg.claude_model}-accessibility"
                    else:  # openrouter
                        openrouter = self._get_openrouter_nav()
                        if not openrouter.enabled:
                            raise RuntimeError("OpenRouter not enabled")
                        raw_response = openrouter.generate(
                            image_path=image_path,
                            prompt=accessibility_prompt,
                            system_prompt="You are an AI assistant helping blind users understand their surroundings.",
                            model=str(self.cfg.openrouter_model),
                            max_tokens=int(self.cfg.caption_max_tokens),
                            temperature=0.2,
                            cost_tracker=self._cost_tracker,
                            resize_720p=bool(self.cfg.resize_720p),
                        )
                        model_name = f"{self.cfg.openrouter_model}-accessibility"
                except Exception as e:
                    logger.warning(f"VLM accessibility generation failed for {image_path.name}: {e}. Falling back to template.")
                    model_name = "template-accessibility"
                    raw_response = ""
            elif model == "qwen":
                try:
                    qwen = self._get_qwen()
                    raw_response = qwen.generate(str(image_path), ctx=ctx, prompt=accessibility_prompt)
                    self._cost_tracker.add_qwen_call()
                    model_name = "qwen-accessibility"
                except Exception as e:
                    logger.warning(f"Qwen accessibility generation failed for {image_path.name}: {e}. Falling back to template.")
                    model_name = "template-accessibility"
                    raw_response = ""
            else:
                model_name = "template-accessibility"

            object_names = [obj.get("name", "") for obj in objects_with_positions]
            spatial_objects_fallback = []
            for obj in objects_with_positions[:8]:
                name = obj.get("name", "object")
                position = obj.get("position", "visible")
                distance = obj.get("distance", "")
                if distance:
                    spatial_objects_fallback.append(f"{name}: {position}, {distance}")
                else:
                    spatial_objects_fallback.append(f"{name}: {position}")

            if not self.cfg.use_bboxes:
                spatial_objects_fallback = obstacle_lines if obstacle_lines else ["Path clear (no obstacle in your front, left, or right)"]

            # Fallback anchors based on detections and context.
            location_fallback = infer_location(object_names, ctx.scene_description or "")
            time_fallback = infer_time_of_day(object_names, ctx.scene_description or "")
            highlights_fallback = generate_highlights(object_names)
            scene_desc_fallback = ctx.scene_description or self._generate_scene_template(objects_with_positions, ctx, depth_info)
            ground_text_fallback = generate_ground_text(scene_desc_fallback, object_names, location_fallback)

            # Parse the structured response if available.
            if raw_response:
                parsed = parse_accessibility_response(raw_response)
                location = parsed.get("location") or location_fallback
                time_of_day = parsed.get("time") or time_fallback
                scene_desc = parsed.get("scene_description") or scene_desc_fallback
                ground_text = parsed.get("ground_text") or generate_ground_text(scene_desc, object_names, location)
                spatial_objects = parsed.get("spatial_objects") or spatial_objects_fallback
                if not self.cfg.use_bboxes:
                    spatial_objects = obstacle_lines if obstacle_lines else spatial_objects
                highlights = parsed.get("highlight") or highlights_fallback
                guidance = shorten_guidance(parsed.get("guidance", ""))
                if not guidance:
                    guidance = ""
                risk_data = parsed.get("risk_assessment", {}) if isinstance(parsed.get("risk_assessment", {}), dict) else {}
                risk_level = str(risk_data.get("level") or "Medium")
                risk_score = float(risk_data.get("score") or 0.5)
            else:
                location = location_fallback
                time_of_day = time_fallback
                scene_desc = scene_desc_fallback
                ground_text = ground_text_fallback
                spatial_objects = spatial_objects_fallback
                highlights = highlights_fallback
                guidance = ""
                risk_level = "Medium"
                risk_score = 0.5

            if not guidance:
                names_lower = [o.lower() for o in object_names]
                if any("stairs" in n for n in names_lower):
                    guidance = "Stairs ahead; keep a handrail and slow down before stepping."
                elif any("escalator" in n for n in names_lower):
                    guidance = "Escalator nearby; approach slowly and hold the handrail as you step on."
                elif any("ramp" in n for n in names_lower):
                    guidance = "Ramp available; take the gentle slope and keep a steady pace."
                elif highlights and any(k in " ".join(h.lower() for h in highlights) for k in ["obstacle", "hazard", "wall", "pillar"]):
                    guidance = "Keep to the clearer side of the path and avoid nearby obstacles."
                else:
                    guidance = "Proceed forward on the clearer side of the path and stay aware of obstacles."
                guidance = shorten_guidance(guidance)

            # Clamp risk score to [0, 1]
            if risk_score < 0.0:
                risk_score = 0.0
            if risk_score > 1.0:
                risk_score = 1.0

            accessibility_data = AccessibilityData(
                location=location,
                time=time_of_day,
                scene_description=scene_desc,
                ground_text=ground_text,
                spatial_objects=spatial_objects,
                highlight=highlights,
                guidance=guidance,
                risk_assessment=RiskAssessment(level=risk_level, score=risk_score),
            )

            # Deterministic risk assessment: use nav_obstacles, fallback to spatial_objects
            accessibility_data.risk_assessment = _compute_accessibility_risk(
                nav_obstacles,
                scene_summary=accessibility_data.ground_text or accessibility_data.scene_description,
                spatial_objects=spatial_objects if spatial_objects else None,
            )

            text = accessibility_data.scene_description or accessibility_data.ground_text or raw_response
            meta = {
                "task_type": "accessibility",
                "num_objects": len(objects_with_positions),
                "has_depth": depth is not None,
                "model": model_name,
                "bbox_mode": "bbox" if self.cfg.use_bboxes else "no_bbox",
                "location": accessibility_data.location,
                "time": accessibility_data.time,
                "ground_text": accessibility_data.ground_text,
                "highlight": accessibility_data.highlight,
                "guidance": accessibility_data.guidance,
                "risk_level": accessibility_data.risk_assessment.level,
                "risk_score": accessibility_data.risk_assessment.score,
                "risk_reason": accessibility_data.risk_assessment.reason,
                "risk_obstacles": [o.model_dump() for o in accessibility_data.risk_assessment.obstacles],
            }
        else:
            # Default caption task
            model = str(self.cfg.caption_model).lower().strip()
            if model == "qwen":
                try:
                    qwen = self._get_qwen()
                    prompt = build_prompt(aux_summary)
                    text = qwen.generate(str(image_path), ctx=ctx, prompt=prompt)
                    self._cost_tracker.add_qwen_call()
                    model_name = "qwen-caption"
                except Exception as e:
                    logger.warning(f"Qwen caption generation failed for {image_path.name}: {e}. Falling back to template.")
                    text = self._template.generate(str(image_path), ctx)
                    self._cost_tracker.add_template_call()
                    model_name = "template-caption"
            elif model == "gpt":
                try:
                    gpt = self._get_gpt()
                    if not gpt.enabled:
                        raise RuntimeError(f"GPT captioner not enabled. API key available: {bool(gpt.api_key)}, Client available: {gpt._client is not None}")
                    prompt = build_prompt(aux_summary)
                    text = gpt.generate(
                        str(image_path),
                        ctx=ctx,
                        prompt=prompt,
                        cost_tracker=self._cost_tracker,
                        image_detail=str(self.cfg.image_detail),
                        resize_720p=bool(self.cfg.resize_720p),
                    )
                    model_name = f"{self.cfg.gpt_model}-caption"
                except Exception as e:
                    logger.warning(f"GPT caption generation failed for {image_path.name}: {e}. Falling back to template.")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    text = self._template.generate(str(image_path), ctx)
                    self._cost_tracker.add_template_call()
                    model_name = "template-caption"
            elif model == "openrouter":
                try:
                    openrouter = self._get_openrouter_nav()
                    if not openrouter.enabled:
                        raise RuntimeError("OpenRouter not enabled")
                    prompt = build_prompt(aux_summary)
                    text = openrouter.generate(
                        image_path=image_path,
                        prompt=prompt,
                        system_prompt=None,
                        model=str(self.cfg.openrouter_model),
                        max_tokens=int(self.cfg.caption_max_tokens),
                        temperature=0.2,
                        cost_tracker=self._cost_tracker,
                        resize_720p=bool(self.cfg.resize_720p),
                    )
                    model_name = f"{self.cfg.openrouter_model}-caption"
                except Exception as e:
                    logger.warning(f"OpenRouter caption generation failed for {image_path.name}: {e}. Falling back to template.")
                    text = self._template.generate(str(image_path), ctx)
                    self._cost_tracker.add_template_call()
                    model_name = "template-caption"
            else:
                text = self._template.generate(str(image_path), ctx)
                model_name = "template-caption"
            
            meta = {
                "model": model_name,
            }

        # Store image path as relative to images_dir when possible (safer for dataset portability).
        try:
            rel_image = str(image_path.resolve().relative_to(images_dir.resolve()))
        except Exception:
            rel_image = str(image_path.name)

        record = AnnotationRecord(
            image=rel_image,
            text=text,
            timestamp=datetime.now().isoformat(),
            task=task if task in {"caption", "navigation", "scene", "accessibility"} else "caption",
            sources={
                "detection_metadata": str(det_path) if det_path and det_path.exists() else None,
                "segmentation_metadata": str(seg_path) if seg_path and seg_path.exists() else None,
                "depth_metadata": str(depth_path) if depth_path and depth_path.exists() else None,
                "vlm_json_dir": str(vlm_json_dir) if vlm_json_dir else None,
            },
            meta=meta,
            accessibility=accessibility_data if task == "accessibility" else None,
        )

        ann_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        return record

    def process_dir(
        self,
        *,
        images_dir: str,
        output_dir: str,
        detection_metadata_dir: Optional[str] = None,
        segmentation_metadata_dir: Optional[str] = None,
        depth_metadata_dir: Optional[str] = None,
        vlm_json_dir: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate annotations for all images under `images_dir`."""
        in_dir = Path(images_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        det_dir = Path(detection_metadata_dir) if detection_metadata_dir else None
        seg_dir = Path(segmentation_metadata_dir) if segmentation_metadata_dir else None
        dep_dir = Path(depth_metadata_dir) if depth_metadata_dir else None
        vlm_dir = Path(vlm_json_dir) if vlm_json_dir else None

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts and not p.name.endswith("_detected.jpg")])
        if max_images is not None and int(max_images) > 0:
            images = images[: int(max_images)]

        from tqdm.auto import tqdm

        records: List[AnnotationRecord] = []
        failures: List[Dict[str, str]] = []
        for img in tqdm(images, desc="Annotate"):
            try:
                rec = self.process_image(
                    image_path=img,
                    output_dir=out_dir,
                    images_dir=in_dir,
                    detection_metadata_dir=det_dir,
                    segmentation_metadata_dir=seg_dir,
                    depth_metadata_dir=dep_dir,
                    vlm_json_dir=vlm_dir,
                )
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                failures.append({"image": str(img), "error": str(e)})

        # Export training data
        exports = export_training_data(
            annotations_dir=out_dir / "annotations",
            images_dir=in_dir,
            output_dir=out_dir / "training_data",
            cfg=ExportConfig(
                export_formats=list(self.cfg.export_formats),
                create_image_copies=bool(self.cfg.create_image_copies),
                instruction=str(self.cfg.instruction),
                system_prompt=self.cfg.system_prompt,
                annotation_version=str(self.cfg.annotation_version),
            ),
        )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "images_dir": str(in_dir),
            "output_dir": str(out_dir),
            "num_images": len(images),
            "processed": len(records),
            "failed": len(failures),
            "failures": failures[:50],
            "exports": exports,
        }
        
        # Add cost summary
        costs = self._cost_tracker.calculate_cost()
        summary["costs"] = costs
        summary["cost_summary"] = self._cost_tracker.get_summary()
        
        with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Print cost summary to console
        print(self._cost_tracker.get_summary())
        
        return summary

