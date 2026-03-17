"""Prompt templates for annotation generation."""

from __future__ import annotations
from typing import List, Optional, Dict, Any


# =============================================================================
# ACCESSIBILITY FEATURES FOR BLIND USER ASSISTANCE
# =============================================================================

# Location keywords to infer scene type
LOCATION_KEYWORDS = {
    "airport": ["terminal", "gate", "boarding", "luggage", "flight", "airline", "airport"],
    "grocery_store": ["grocery", "supermarket", "produce", "cart", "shelf", "aisle", "checkout"],
    "shopping_mall": ["mall", "store", "shop", "escalator", "directory", "food_court"],
    "street": ["sidewalk", "crosswalk", "traffic_light", "pedestrian", "road", "car", "bus"],
    "subway_station": ["metro", "subway", "platform", "train", "turnstile", "ticket"],
    "bus_station": ["bus", "bus_stop", "station", "schedule", "bench"],
    "train_station": ["train", "platform", "track", "departure", "arrival"],
    "hospital": ["hospital", "medical", "doctor", "nurse", "wheelchair", "stretcher"],
    "office_building": ["office", "elevator", "lobby", "reception", "desk", "cubicle"],
    "restaurant": ["restaurant", "table", "chair", "menu", "waiter", "food", "dining"],
    "school": ["classroom", "desk", "blackboard", "student", "teacher", "school"],
    "park": ["tree", "bench", "grass", "playground", "fountain", "path"],
    "parking_lot": ["parking", "car", "vehicle", "parking_meter", "lot"],
    "restroom": ["restroom", "bathroom", "toilet", "sink", "mirror"],
    "elevator": ["elevator", "button", "floor_indicator", "control_panel"],
    "stairway": ["stairs", "staircase", "handrail", "steps", "landing"],
}

# Highlight keywords for accessibility alerts
ACCESSIBILITY_HIGHLIGHTS = {
    "stairs": {"keywords": ["stairs", "staircase", "steps", "stair"], "alert": "Stairs ahead - use handrail, watch footing"},
    "escalator": {"keywords": ["escalator", "moving_stairs"], "alert": "Escalator present - hold handrail, mind the step"},
    "elevator": {"keywords": ["elevator", "lift"], "alert": "Elevator available - accessible vertical transport"},
    "ramp": {"keywords": ["ramp", "wheelchair_ramp", "incline"], "alert": "Ramp available - accessible route"},
    "accessible_restroom": {"keywords": ["accessible_restroom", "handicap_restroom", "wheelchair_accessible"], "alert": "Accessible restroom nearby"},
    "restroom": {"keywords": ["restroom", "bathroom", "toilet", "wc"], "alert": "Restroom sign visible"},
    "door": {"keywords": ["door", "entrance", "exit", "doorway"], "alert": "Door or entrance ahead"},
    "automatic_door": {"keywords": ["automatic_door", "sliding_door"], "alert": "Automatic door - may open unexpectedly"},
    "wall": {"keywords": ["wall"], "alert": "Wall nearby - potential collision risk"},
    "pillar": {"keywords": ["pillar", "column", "post"], "alert": "Support pillar in path"},
    "curb": {"keywords": ["curb", "curb_cut"], "alert": "Curb present - watch step height"},
    "crosswalk": {"keywords": ["crosswalk", "pedestrian_crossing", "zebra_crossing"], "alert": "Crosswalk - check traffic before crossing"},
    "traffic_light": {"keywords": ["traffic_light", "signal", "pedestrian_signal"], "alert": "Traffic signal - wait for walk signal"},
    "wet_floor": {"keywords": ["wet_floor", "caution_sign", "slippery"], "alert": "Wet floor warning - slippery surface"},
    "construction": {"keywords": ["construction", "barrier", "cone"], "alert": "Construction area - path may be obstructed"},
    "bench": {"keywords": ["bench", "seat", "seating"], "alert": "Seating available"},
    "handrail": {"keywords": ["handrail", "railing", "grab_bar"], "alert": "Handrail available for support"},
}

# Time/lighting indicators
DAYTIME_INDICATORS = ["sunlight", "bright", "sunny", "daylight", "clear_sky", "shadow", "outdoor_light"]
NIGHTTIME_INDICATORS = ["dark", "night", "artificial_light", "lamp", "streetlight", "neon", "dim"]
INDOOR_INDICATORS = ["indoor", "ceiling", "fluorescent", "interior", "room"]


def infer_location(objects: List[str], scene_context: str = "") -> str:
    """Infer location type from detected objects and scene context."""
    all_text = " ".join([o.lower().replace("_", " ") for o in objects])
    if scene_context:
        all_text += " " + scene_context.lower()
    
    location_scores = {}
    for location, keywords in LOCATION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in all_text)
        if score > 0:
            location_scores[location] = score
    
    if location_scores:
        best_location = max(location_scores, key=location_scores.get)
        return best_location.replace("_", " ").title()
    
    return "Unknown Location"


def infer_time_of_day(objects: List[str], scene_context: str = "") -> str:
    """Infer time of day (Day/Night/Indoor) from lighting clues."""
    all_text = " ".join([o.lower() for o in objects])
    if scene_context:
        all_text += " " + scene_context.lower()
    
    day_score = sum(1 for ind in DAYTIME_INDICATORS if ind in all_text)
    night_score = sum(1 for ind in NIGHTTIME_INDICATORS if ind in all_text)
    indoor_score = sum(1 for ind in INDOOR_INDICATORS if ind in all_text)
    
    if indoor_score > max(day_score, night_score):
        return "Indoor"
    elif day_score > night_score:
        return "Day"
    elif night_score > day_score:
        return "Night"
    else:
        return "Unknown"


def generate_highlights(objects: List[str], obstacles: List[Dict[str, Any]] = None) -> List[str]:
    """Generate accessibility highlights based on detected objects."""
    highlights = []
    objects_lower = [o.lower().replace(" ", "_") for o in objects]
    
    for highlight_type, info in ACCESSIBILITY_HIGHLIGHTS.items():
        for keyword in info["keywords"]:
            if any(keyword in obj for obj in objects_lower):
                highlights.append(info["alert"])
                break
    
    # Add obstacle-based highlights
    if obstacles:
        close_obstacles = [o for o in obstacles if o.get("distance_m", 10) < 1.5]
        if len(close_obstacles) >= 3:
            highlights.append("Crowded area ahead")
        
        for obs in obstacles:
            if obs.get("distance_m", 10) < 0.8:
                name = obs.get("name", "obstacle")
                highlights.append(f"Immediate obstacle: {name}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_highlights = []
    for h in highlights:
        if h not in seen:
            seen.add(h)
            unique_highlights.append(h)
    
    return unique_highlights[:5]  # Limit to top 5 highlights


def generate_ground_text(scene_description: str, objects: List[str], location: str) -> str:
    """Generate a very short summary (ground text) of the scene."""
    if scene_description:
        # Extract first sentence or truncate
        first_sentence = scene_description.split(".")[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."
        return first_sentence + "."
    
    # Fallback: generate from objects
    if objects:
        obj_str = ", ".join(objects[:3])
        return f"Scene with {obj_str}."
    
    return f"Scene at {location}."


def shorten_guidance(text: str, max_words: int = 22) -> str:
    """Shorten guidance to a single helpful sentence."""
    if not text:
        return ""
    import re

    s = re.sub(r"\s+", " ", str(text)).strip()
    if not s:
        return ""

    # Use only the first sentence.
    parts = re.split(r"(?<=[.!?])\s+", s)
    s = parts[0].strip() if parts else s

    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",;:")

    if s and s[-1] not in ".!?":
        s += "."
    return s


# =============================================================================
# ACCESSIBILITY ANNOTATION PROMPT (VLM-based)
# =============================================================================

ACCESSIBILITY_ANNOTATION_INSTRUCTIONS = """You are an AI assistant helping blind and visually impaired users understand their surroundings.

Analyze this image and provide a structured accessibility report with the following fields:

1. **Location**: Identify the type of location (e.g., Airport Terminal, Grocery Store, Street, Mall, etc.)

2. **Time**: Determine if it's Day, Night, or Indoor based on lighting conditions

3. **Scene Description**: A detailed description of the scene with spatial positions using clock directions (12=ahead, 3=right, 6=behind, 9=left) and distances in meters

4. **Ground Text**: A ONE sentence summary of the scene (max 20 words)

5. **Spatial Object Description**: List key objects using obstacle-in-position phrasing, clock direction, and distance
   Format: "Obstacle in your [front|left|right]: [object name] at [clock], [distance]m away"
   Examples: "Obstacle in your front: wall at 12 o'clock, 2m away"; "Obstacle in your left: person at 9 o'clock, 1.5m away"; "Obstacle in your right: door at 3 o'clock, 3m away"
   List every object in front, left, or right. If both sides have objects, list both (Obstacle in your left: ...; Obstacle in your right: ...).

6. **Highlight**: List accessibility-relevant items that would help a blind user:
   - Ramps, escalators, elevators, lifts (accessible routes)
   - Stairs (fall risk)
   - Doors, automatic doors
   - Restrooms, accessible restrooms
   - Obstacles, walls, pillars
   - Open spaces (safe navigation areas)

7. **Guidance**: One short sentence (12-22 words) focusing on the safest path or obstacle avoidance

8. **Risk Assessment**: Use ONLY these 4 levels. LANGUAGE MODEL MUST BE STRICT:
   - LOW: ONLY when path is clear - NO object/obstacle in the user's front, left, OR right. If ANY object (door, wall, person, pillar, etc.) is in front or left or right, risk CANNOT be Low. Low = clear space, go ahead.
   - MEDIUM: Obstacle(s) in front/left/right but not immediately dangerous; potential hazard.
   - HIGH: Can be hazardous in a few seconds if the person doesn't get aligned; requires immediate attention.
   - EXTREME: Falling risk, can hit in less than a few seconds or even 1 second; dangerous - stop immediately.

IMPORTANT RULES:
- Use "Obstacle in your [front|left|right]: [name] at [clock], [distance]m away" for spatial objects. Clock directions should appear before distance (12=ahead, 3=right, 9=left).
- Estimate distances in meters
- Focus on navigation-relevant information
- DO NOT include file paths, model names, or technical metadata
- Keep descriptions clear and actionable
- If no bounding-box positions are provided, avoid upper-left/center-right phrases and use only clock direction + distance

OUTPUT FORMAT (use this exact structure):
Location: [location type]
Time: [Day/Night/Indoor]
Scene Description: [detailed description]
Ground Text: [one sentence summary]
Spatial Objects: [list of objects with positions]
Highlight: [accessibility alerts, comma-separated]
Guidance: [navigation instruction]
Risk Assessment: [LOW|MEDIUM|HIGH|EXTREME] ([0.0-1.0])
"""


def build_accessibility_prompt(
    objects_with_positions: List[Dict[str, Any]],
    scene_context: Optional[str] = None,
    depth_info: Optional[str] = None,
    obstacle_lines: Optional[List[str]] = None,
    use_bboxes: bool = True,
) -> str:
    """Build a prompt for accessibility-focused annotation generation."""
    lines = [ACCESSIBILITY_ANNOTATION_INSTRUCTIONS.strip(), ""]
    
    if scene_context:
        lines.append(f"Scene Context: {scene_context}")
        lines.append("")
    
    if use_bboxes and objects_with_positions:
        lines.append("Detected Objects:")
        for obj in objects_with_positions[:15]:
            name = obj.get("name", "object")
            position = obj.get("position", "visible")
            distance = obj.get("distance", "")
            if distance:
                lines.append(f"- {name}: {position}, {distance}")
            else:
                lines.append(f"- {name}: {position}")
        lines.append("")
    elif obstacle_lines:
        lines.append("Detected Obstacles (front/left/right + clock + distance):")
        for line in obstacle_lines[:15]:
            lines.append(f"- {line}")
        lines.append("")
    
    if depth_info:
        lines.append(f"Depth Info: {depth_info}")
        lines.append("")
    
    lines.append("Generate the accessibility report using the format specified above.")
    
    return "\n".join(lines)


def parse_accessibility_response(response_text: str) -> Dict[str, Any]:
    """Parse the structured accessibility response from VLM."""
    result = {
        "location": "Unknown",
        "time": "Unknown",
        "scene_description": "",
        "ground_text": "",
        "spatial_objects": [],
        "highlight": [],
        "guidance": "",
        "risk_assessment": {"level": "Medium", "score": 0.5},
    }
    
    if not response_text:
        return result
    
    lines = response_text.strip().split("\n")
    current_field = None
    current_value = []
    
    field_mapping = {
        "location": "location",
        "time": "time",
        "scene description": "scene_description",
        "ground text": "ground_text",
        "spatial objects": "spatial_objects",
        "highlight": "highlight",
        "guidance": "guidance",
        "risk assessment": "risk_assessment",
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a field header
        found_field = False
        for key, field_name in field_mapping.items():
            if line.lower().startswith(key + ":"):
                # Save previous field
                if current_field and current_value:
                    _set_field_value(result, current_field, current_value)
                
                current_field = field_name
                value = line[len(key) + 1:].strip()
                current_value = [value] if value else []
                found_field = True
                break
        
        if not found_field and current_field:
            current_value.append(line)
    
    # Save last field
    if current_field and current_value:
        _set_field_value(result, current_field, current_value)
    
    return result


def _set_field_value(result: Dict[str, Any], field: str, values: List[str]) -> None:
    """Set a field value in the result dictionary."""
    combined = " ".join(values).strip()
    
    if field in ["location", "time", "guidance", "scene_description", "ground_text"]:
        result[field] = combined
    elif field == "spatial_objects":
        # Parse as list of objects
        objects = []
        for v in values:
            v = v.strip().lstrip("-•").strip()
            if v:
                if ";" in v:
                    parts = [p.strip() for p in v.split(";") if p.strip()]
                    objects.extend(parts)
                else:
                    objects.append(v)
        result[field] = objects
    elif field == "highlight":
        # Parse as comma-separated or list
        highlights = []
        for v in values:
            v = v.strip().lstrip("-•").strip()
            if ";" in v:
                highlights.extend([h.strip() for h in v.split(";") if h.strip()])
            elif "," in v:
                highlights.extend([h.strip() for h in v.split(",") if h.strip()])
            elif v:
                highlights.append(v)
        result[field] = highlights
    elif field == "risk_assessment":
        # Parse risk level and score (Low, Medium, High, Extreme)
        level = "Medium"
        score = 0.5
        combined_lower = combined.lower()
        if "extreme" in combined_lower or "critical" in combined_lower:
            level = "Extreme"
            score = 0.98
        elif "low" in combined_lower:
            level = "Low"
            score = 0.2
        elif "high" in combined_lower:
            level = "High"
            score = 0.8
        elif "medium" in combined_lower or "moderate" in combined_lower or "mediocre" in combined_lower:
            level = "Medium"
            score = 0.5
        
        # Try to extract numeric score
        import re
        score_match = re.search(r"(\d+\.?\d*)", combined)
        if score_match:
            try:
                parsed_score = float(score_match.group(1))
                if 0 <= parsed_score <= 1:
                    score = parsed_score
                elif 0 <= parsed_score <= 100:
                    score = parsed_score / 100
            except ValueError:
                pass
        
        result[field] = {"level": level, "score": score}


# =============================================================================
# PROFESSIONAL SCENE DESCRIPTION PROMPT (for fine-tuning data generation)
# =============================================================================

SCENE_DESCRIPTION_INSTRUCTIONS = """You are a professional AI assistant generating high-quality scene descriptions for Vision-Language Model fine-tuning.

Your task is to generate a comprehensive, detailed scene description using the detected objects and their spatial positions.

STRICT EXCLUSION RULES - DO NOT include:
- Image file paths or filenames
- Model names (GPT-4o, Qwen, Claude, YOLO, SAM, etc.)
- Image dimensions/sizes (width, height, pixels)
- Internal metadata (timestamps, model versions, confidence scores)
- Bounding box coordinates or normalized values
- Segmentation mask IDs or technical identifiers
- JSON, code, or structured data

REQUIRED OUTPUT FORMAT:
Write a professional, fluent scene description that:
1. Starts with overall scene context (environment type, setting)
2. Describes key objects with their spatial positions (upper-left, center-right, foreground, background, etc.)
3. Includes spatial relationships between objects (near, beside, behind, in front of)
4. Mentions depth/distance when relevant (close, distant, nearby)
5. Uses natural language only - no technical terms or coordinates

QUALITY STANDARDS:
- Professional, detailed English prose
- Only describe objects that are actually present
- Use spatial descriptions derived from object positions
- Avoid repetition and hallucinations
- Suitable for VLM training datasets

Produce ONLY the final scene description text. No headers, no labels, no metadata."""


def build_scene_description_prompt(
    objects_with_positions: List[Dict[str, Any]],
    scene_context: Optional[str] = None,
    depth_info: Optional[str] = None,
) -> str:
    """Build a prompt for professional scene description generation.
    
    Args:
        objects_with_positions: List of dicts with 'name', 'position' (spatial), 'distance' keys
        scene_context: Optional scene context from VLM analysis
        depth_info: Optional depth information summary
    
    Returns:
        Complete prompt for scene description generation
    """
    lines = [SCENE_DESCRIPTION_INSTRUCTIONS.strip(), ""]
    
    if scene_context:
        lines.append(f"Scene Context: {scene_context}")
        lines.append("")
    
    if objects_with_positions:
        lines.append("Detected Objects with Spatial Positions:")
        for obj in objects_with_positions:
            name = obj.get("name", "object")
            position = obj.get("position", "visible in scene")
            distance = obj.get("distance", "")
            if distance:
                lines.append(f"- {name}: {position}, {distance}")
            else:
                lines.append(f"- {name}: {position}")
        lines.append("")
    
    if depth_info:
        lines.append(f"Depth Information: {depth_info}")
        lines.append("")
    
    lines.append("Generate a professional scene description using the above information.")
    lines.append("Remember: NO coordinates, NO metadata, NO model names - only natural language.")
    
    return "\n".join(lines)


def bbox_to_spatial_position(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> str:
    """Convert normalized bounding box to spatial position description.
    
    Args:
        x_center: Normalized x center (0-1)
        y_center: Normalized y center (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
    
    Returns:
        Spatial position description string
    """
    # Horizontal position
    if x_center < 0.33:
        h_pos = "left"
    elif x_center < 0.66:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Vertical position
    if y_center < 0.33:
        v_pos = "upper"
    elif y_center < 0.66:
        v_pos = "middle"
    else:
        v_pos = "lower"
    
    # Combined position
    if h_pos == "center" and v_pos == "middle":
        position = "in the center of the scene"
    elif h_pos == "center":
        position = f"in the {v_pos} area"
    elif v_pos == "middle":
        position = f"on the {h_pos} side"
    else:
        position = f"in the {v_pos}-{h_pos} area"
    
    # Size-based prominence
    area = width * height
    if area > 0.15:
        position = f"prominently {position}"
    elif area < 0.02:
        position = f"small, {position}"
    
    return position


def bbox_to_distance_estimate(width: float, height: float) -> str:
    """Estimate distance based on bounding box size.
    
    Args:
        width: Normalized width (0-1)
        height: Normalized height (0-1)
    
    Returns:
        Distance description string
    """
    area = width * height
    if area > 0.12:
        return "close to the viewer"
    elif area > 0.04:
        return "at medium distance"
    elif area > 0.01:
        return "in the background"
    else:
        return "distant"


# =============================================================================
# ORIGINAL ANNOTATION INSTRUCTIONS
# =============================================================================

ANNOTATION_INSTRUCTIONS = """You are a Vision-Language Model used for dataset annotation.

You will receive a single keyframe image extracted from a video.
Auxiliary signals such as segmentation masks, depth maps, and object detections
may be available to you to improve reasoning.

Your task is to generate a clean, high-quality textual annotation
that is suitable for fine-tuning a Vision-Language Model (e.g., via LoRA).

Follow these rules strictly:

1. DO NOT include:
   - Image file paths
   - File names
   - Model names
   - Bounding box coordinates
   - Segmentation mask IDs
   - Any internal or metadata

2. USE auxiliary information ONLY to:
   - Improve object identification
   - Improve spatial reasoning
   - Improve safety and navigation understanding

3. The annotation MUST include:
   - A concise but informative scene description
   - Key objects and their functional roles
   - Relative spatial relationships (left/right/front/near/far) when relevant
   - Depth-aware reasoning when relevant (close obstacles vs far)
   - Safety- or task-relevant observations if applicable

4. Write in natural, human-readable language.
   - No bullet lists unless necessary
   - No JSON
   - No repetition
   - No hallucinated objects

5. The output should be model-agnostic and reusable across datasets.

Produce ONLY the final annotation text.
"""


def build_prompt(aux_summary: str) -> str:
    """Build a single user prompt containing rules + auxiliary summary."""
    aux = aux_summary.strip()
    if aux:
        return f"{ANNOTATION_INSTRUCTIONS}\n\nAuxiliary summary (non-sensitive):\n{aux}\n"
    return ANNOTATION_INSTRUCTIONS

