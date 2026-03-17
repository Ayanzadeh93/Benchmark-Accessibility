"""Core 5-question VQA generation for navigation evaluation."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

from annotation.nav_parse import ParsedNavigation

from .action_command import _build_distractor_pool
from .schemas import ChoiceLabel, VQAMultipleChoiceSample
from .utils import (
    ObstacleInfo,
    PathSafe,
    assign_labels,
    normalize,
    parse_obstacle_line,
    stable_int_seed,
)

_DISTANCE_TIE_M = 0.2

_GENERIC_OBSTACLES = [
    "wall",
    "door",
    "person",
    "chair",
    "table",
    "stairs",
    "box",
    "cart",
]


def build_navigation_core_samples(
    *,
    image: str,
    parsed: ParsedNavigation,
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> List[VQAMultipleChoiceSample]:
    """Build the 5 core MCQ samples for a single scene."""
    obstacles = _parse_obstacles(parsed.obstacles_lines)
    return [
        build_main_obstacle_mcq(
            image=image,
            parsed=parsed,
            obstacles=obstacles,
            sources=sources,
            image_abs_path=image_abs_path,
            base_seed=base_seed,
        ),
        build_closest_obstacle_mcq(
            image=image,
            parsed=parsed,
            obstacles=obstacles,
            sources=sources,
            image_abs_path=image_abs_path,
            base_seed=base_seed,
        ),
        build_risk_assessment_mcq(
            image=image,
            parsed=parsed,
            sources=sources,
            image_abs_path=image_abs_path,
        ),
        build_spatial_clock_mcq(
            image=image,
            parsed=parsed,
            obstacles=obstacles,
            sources=sources,
            image_abs_path=image_abs_path,
            base_seed=base_seed,
        ),
        build_action_suggestion_mcq(
            image=image,
            parsed=parsed,
            obstacles=obstacles,
            sources=sources,
            image_abs_path=image_abs_path,
            base_seed=base_seed,
        ),
    ]


def build_main_obstacle_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    obstacles: Sequence[ObstacleInfo],
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> VQAMultipleChoiceSample:
    question_id = "main_obstacle"
    question = "What is the main obstacle or barrier in this scene?"

    if not obstacles:
        options = assign_labels(["No obstacle", "Wall", "Door", "Person"])
        answer = "A"
        return _build_sample(
            image=image,
            question_id=question_id,
            question=question,
            options=options,
            answer=answer,
            image_abs_path=image_abs_path,
            parsed=parsed,
            sources=sources,
            ground_truth={"main_obstacle": "no_obstacle"},
        )

    closest = _closest_obstacles(obstacles)
    combined = _combined_obstacle_text(closest)
    use_combined = combined is not None
    correct_text = combined if use_combined else closest[0].name

    if use_combined:
        # Keep combined answer in D (as suggested by user).
        opts = [closest[0].name, closest[1].name, _fill_generic([], 1)[0], correct_text]
        options = assign_labels(opts)
        answer = "D"
    else:
        pool = [o.name for o in obstacles if o.name]
        pool = _dedup(pool)
        opts = [correct_text]
        for name in pool:
            if normalize(name) == normalize(correct_text):
                continue
            if len(opts) >= 3:
                break
            opts.append(name)
        while len(opts) < 4:
            opts.append(_next_generic(opts))
        opts = _deterministic_shuffle(opts, seed=stable_int_seed(image, question_id, base_seed=base_seed))
        options = assign_labels(opts)
        answer = _find_answer_label(options, correct_text)

    return _build_sample(
        image=image,
        question_id=question_id,
        question=question,
        options=options,
        answer=answer,
        image_abs_path=image_abs_path,
        parsed=parsed,
        sources=sources,
        ground_truth={
            "main_obstacle": correct_text,
            "obstacles": [o.name for o in obstacles],
        },
    )


def build_closest_obstacle_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    obstacles: Sequence[ObstacleInfo],
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> VQAMultipleChoiceSample:
    question_id = "closest_obstacle"
    question = "Which object is closest to the user?"

    if not obstacles:
        options = assign_labels(["No obstacle nearby", "Wall", "Door", "Person"])
        answer = "A"
        return _build_sample(
            image=image,
            question_id=question_id,
            question=question,
            options=options,
            answer=answer,
            image_abs_path=image_abs_path,
            parsed=parsed,
            sources=sources,
            ground_truth={"closest_obstacle": "no_obstacle"},
        )

    closest = _closest_obstacles(obstacles)
    use_equal = _distances_close(closest)
    correct_text = "Both are about the same distance" if use_equal else closest[0].name

    opts = [closest[0].name]
    if len(closest) > 1:
        opts.append(closest[1].name)
    if use_equal:
        opts.append("Both are about the same distance")
    while len(opts) < 4:
        opts.append(_next_generic(opts))
    opts = _deterministic_shuffle(opts, seed=stable_int_seed(image, question_id, base_seed=base_seed))
    options = assign_labels(opts)
    answer = _find_answer_label(options, correct_text)

    return _build_sample(
        image=image,
        question_id=question_id,
        question=question,
        options=options,
        answer=answer,
        image_abs_path=image_abs_path,
        parsed=parsed,
        sources=sources,
        ground_truth={
            "closest_obstacle": correct_text,
            "obstacles": [o.name for o in obstacles],
            "distances_m": {o.name: o.distance_m for o in obstacles},
        },
    )


def build_risk_assessment_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
) -> VQAMultipleChoiceSample:
    question_id = "risk_assessment"
    question = "How safe is this scene for blind users?"

    # 4 risk levels: Low (path clear), Medium (potential hazard), High (hazardous in few sec), Extreme (immediate danger)
    options = assign_labels(
        [
            "Low risk - path clear, safe to proceed",
            "Medium risk - potential hazard, caution needed",
            "High risk - hazardous in few seconds if not aligned, stop or reroute",
            "Extreme risk - immediate danger, stop immediately",
        ]
    )
    risk = str(parsed.risk).strip().capitalize()
    if risk == "Low":
        answer = "A"
    elif risk == "Medium":
        answer = "B"
    elif risk == "High":
        answer = "C"
    elif risk == "Extreme":
        answer = "D"
    else:
        answer = "A"

    return _build_sample(
        image=image,
        question_id=question_id,
        question=question,
        options=options,
        answer=answer,
        image_abs_path=image_abs_path,
        parsed=parsed,
        sources=sources,
        ground_truth={"risk": risk},
    )


def build_spatial_clock_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    obstacles: Sequence[ObstacleInfo],
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> VQAMultipleChoiceSample:
    question_id = "spatial_clock"

    target = _select_target_obstacle(obstacles)
    object_name = target.name if target else "obstacle"
    question = f"Locate the {object_name} based on clock direction."

    clock_text = _clock_text(target)
    if not clock_text:
        correct_text = "Not specified"
        opts = ["Not specified", "12 o'clock", "3 o'clock", "9 o'clock"]
        options = assign_labels(opts)
        answer = "A"
    else:
        distractors = _clock_distractors(clock_text)
        opts = [clock_text, *distractors]
        opts = _deterministic_shuffle(opts, seed=stable_int_seed(image, question_id, base_seed=base_seed))
        options = assign_labels(opts)
        answer = _find_answer_label(options, clock_text)

    return _build_sample(
        image=image,
        question_id=question_id,
        question=question,
        options=options,
        answer=answer,
        image_abs_path=image_abs_path,
        parsed=parsed,
        sources=sources,
        ground_truth={
            "object": object_name,
            "clock": clock_text or "not_specified",
            "obstacles": [o.name for o in obstacles],
        },
    )


def build_action_suggestion_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    obstacles: Sequence[ObstacleInfo],
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> VQAMultipleChoiceSample:
    question_id = "action_suggestion"
    question = "What action do you suggest to the user in this scene?"

    obstacle = obstacles[0].name if obstacles else "obstacle"
    correct = _short_action(parsed.guidance)
    pool = _build_distractor_pool(parsed=parsed, obstacle=obstacle)

    seed = stable_int_seed(image, question_id, base_seed=base_seed)
    scored = []
    for cand in pool:
        hh = hashlib.sha256()
        hh.update(str(seed).encode("utf-8"))
        hh.update(b"|")
        hh.update(cand.encode("utf-8", errors="ignore"))
        scored.append((hh.hexdigest(), cand))
    scored.sort(key=lambda x: x[0])
    distractors = [c for _, c in scored][:3]

    opts = [correct, *distractors]
    scored2 = []
    for t in opts:
        hh = hashlib.sha256()
        hh.update(b"opt|")
        hh.update(str(seed).encode("utf-8"))
        hh.update(b"|")
        hh.update(t.encode("utf-8", errors="ignore"))
        scored2.append((hh.hexdigest(), t))
    scored2.sort(key=lambda x: x[0])
    shuffled = [t for _, t in scored2]

    options = assign_labels(shuffled)
    answer = _find_answer_label(options, correct)

    feedback: Dict[ChoiceLabel, str] = {}
    for lab, txt in options.items():
        if lab == answer:
            feedback[lab] = "Correct."
        else:
            feedback[lab] = f"Incorrect. Ground truth action is: {correct}"

    return _build_sample(
        image=image,
        question_id=question_id,
        question=question,
        options=options,
        answer=answer,
        image_abs_path=image_abs_path,
        parsed=parsed,
        sources=sources,
        ground_truth={"guidance": parsed.guidance},
        feedback=feedback,
    )


def _build_sample(
    *,
    image: str,
    question_id: str,
    question: str,
    options: Dict[ChoiceLabel, str],
    answer: ChoiceLabel,
    parsed: ParsedNavigation,
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    ground_truth: Optional[Dict[str, object]] = None,
    feedback: Optional[Dict[ChoiceLabel, str]] = None,
) -> VQAMultipleChoiceSample:
    sample_id = f"{PathSafe.stem(image)}|{question_id}"
    return VQAMultipleChoiceSample(
        id=sample_id,
        question_id=question_id,
        image=image,
        image_abs_path=image_abs_path,
        question=question,
        options=options,
        answer=answer,
        answer_text=str(options[answer]),
        feedback=feedback or {},
        ground_truth={
            "scene": parsed.scene,
            "risk": parsed.risk,
            "obstacles": list(parsed.obstacles_lines),
            **(ground_truth or {}),
        },
        sources=sources,
    )


def _parse_obstacles(lines: Sequence[str]) -> List[ObstacleInfo]:
    out: List[ObstacleInfo] = []
    for line in lines or []:
        if not line or not str(line).strip():
            continue
        out.append(parse_obstacle_line(str(line)))
    return out


def _closest_obstacles(obstacles: Sequence[ObstacleInfo]) -> List[ObstacleInfo]:
    def score(o: ObstacleInfo) -> float:
        return o.distance_m if o.distance_m is not None else 9999.0

    return sorted(list(obstacles), key=score)


def _distances_close(obstacles: Sequence[ObstacleInfo]) -> bool:
    if len(obstacles) < 2:
        return False
    d1, d2 = obstacles[0].distance_m, obstacles[1].distance_m
    if d1 is None or d2 is None:
        return False
    return abs(d1 - d2) <= _DISTANCE_TIE_M


def _combined_obstacle_text(obstacles: Sequence[ObstacleInfo]) -> Optional[str]:
    if len(obstacles) < 2:
        return None
    if not _distances_close(obstacles):
        return None
    return f"Multiple: {obstacles[0].name} and {obstacles[1].name}"


def _next_generic(existing: Sequence[str]) -> str:
    existing_norm = {normalize(s) for s in existing}
    for g in _GENERIC_OBSTACLES:
        if normalize(g) not in existing_norm:
            return g.title() if g.islower() else g
    return "Obstacle"


def _fill_generic(existing: Sequence[str], count: int) -> List[str]:
    out: List[str] = []
    while len(out) < count:
        out.append(_next_generic([*existing, *out]))
    return out


def _dedup(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in items:
        k = normalize(s)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s.strip())
    return out


def _deterministic_shuffle(items: Sequence[str], *, seed: int) -> List[str]:
    scored = []
    for t in items:
        hh = hashlib.sha256()
        hh.update(str(seed).encode("utf-8"))
        hh.update(b"|")
        hh.update(str(t).encode("utf-8", errors="ignore"))
        scored.append((hh.hexdigest(), t))
    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored]


def _find_answer_label(options: Dict[ChoiceLabel, str], correct_text: str) -> ChoiceLabel:
    for lab, txt in options.items():
        if normalize(txt) == normalize(correct_text):
            return lab
    return "A"


def _select_target_obstacle(obstacles: Sequence[ObstacleInfo]) -> Optional[ObstacleInfo]:
    for o in obstacles:
        if o.clock is not None or o.direction is not None:
            return o
    return obstacles[0] if obstacles else None


def _clock_text(obstacle: Optional[ObstacleInfo]) -> Optional[str]:
    if obstacle is None:
        return None
    if obstacle.clock is not None:
        return f"{obstacle.clock} o'clock"
    if obstacle.direction:
        mapping = {
            "ahead": "12 o'clock",
            "center": "12 o'clock",
            "left": "9 o'clock",
            "right": "3 o'clock",
            "behind": "6 o'clock",
        }
        return mapping.get(obstacle.direction)
    return None


def _clock_distractors(correct: str) -> List[str]:
    clocks = [
        "12 o'clock",
        "1 o'clock",
        "2 o'clock",
        "3 o'clock",
        "4 o'clock",
        "5 o'clock",
        "6 o'clock",
        "7 o'clock",
        "8 o'clock",
        "9 o'clock",
        "10 o'clock",
        "11 o'clock",
    ]
    clocks = [c for c in clocks if normalize(c) != normalize(correct)]
    return clocks[:3]


def _short_action(guidance: str) -> str:
    g = str(guidance or "").strip()
    if not g:
        return "Proceed carefully."
    if ". " in g:
        g = g.split(". ", 1)[0].strip()
    if len(g) > 120:
        g = g[:117].rstrip() + "..."
    return g
