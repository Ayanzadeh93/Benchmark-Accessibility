"""Action-command multiple-choice question generation.

We derive a 4-choice (A/B/C/D) question from the ground-truth "Guidance" line
in a navigation annotation (Scene/Risk/Obstacles/Guidance).

Goal:
- Create a VQA evaluation item where a model must pick the correct navigation action.
- Provide plausible distractors (wrong but reasonable commands).
- Keep generation deterministic via stable seeding.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from annotation.nav_parse import ParsedNavigation

from .schemas import ChoiceLabel, VQAMultipleChoiceSample
from .utils import PathSafe, assign_labels, extract_primary_obstacle, normalize, stable_int_seed


_QUESTION_ID = "action_command"
_QUESTION_TEXT = "What is the recommended navigation action for a blind user in this situation?"


def _action_category(guidance: str) -> str:
    g = normalize(guidance)
    if "turn around" in g:
        return "turn_around"
    if g.startswith("stop") or "stop!" in g:
        return "stop"
    if "slow down" in g:
        return "slow"
    if "move to" in g and "o'clock" in g:
        return "move_clock"
    if "shift left" in g:
        return "shift_left"
    if "shift right" in g:
        return "shift_right"
    if "proceed" in g or "path clear" in g:
        return "proceed"
    return "other"


def _opposite_clock_action(correct: str, *, obstacle: str) -> Optional[str]:
    """If correct says move to 10 or 2 o'clock, propose the opposite."""
    g = normalize(correct)
    if "move to 10 o'clock" in g:
        return f"Move to 2 o'clock to avoid {obstacle}."
    if "move to 2 o'clock" in g:
        return f"Move to 10 o'clock to avoid {obstacle}."
    if "move to 11 o'clock" in g:
        return f"Move to 1 o'clock to avoid {obstacle}."
    if "move to 1 o'clock" in g:
        return f"Move to 11 o'clock to avoid {obstacle}."
    return None


def _build_distractor_pool(
    *,
    parsed: ParsedNavigation,
    obstacle: str,
) -> List[str]:
    """Build a pool of plausible distractor actions."""
    risk = str(parsed.risk)
    cat = _action_category(parsed.guidance)

    pool: List[str] = []

    # Very common commands (project-style)
    pool.extend(
        [
            "Path clear, move forward.",
            "Proceed straight ahead.",
            f"Slow down, {obstacle} ahead.",
            "Slow down, obstacle nearby.",
            "Stop! Obstacle ahead.",
            "Turn around, path blocked!",
            f"Shift left to pass {obstacle}.",
            f"Shift right to pass {obstacle}.",
            f"Move to 10 o'clock to avoid {obstacle}.",
            f"Move to 2 o'clock to avoid {obstacle}.",
        ]
    )

    # Risk-aware nudges (Low, Medium, High, Extreme)
    if risk == "Low":
        pool.extend(
            [
                "Proceed forward at normal speed.",
                "Proceed forward toward door.",
            ]
        )
    elif risk == "Medium":
        pool.extend(
            [
                f"Slow down, {obstacle} ahead.",
                f"Move to 10 o'clock to avoid {obstacle}.",
                f"Move to 2 o'clock to avoid {obstacle}.",
            ]
        )
    elif risk == "High":
        pool.extend(
            [
                f"Stop! {obstacle} ahead. Proceed carefully or turn back.",
                "Stop! Obstacle ahead.",
                "Turn around, path blocked!",
            ]
        )
    else:  # Extreme
        pool.extend(
            [
                f"Stop immediately! {obstacle} ahead - dangerous. Do not proceed.",
                "Stop immediately! Danger ahead - do not proceed.",
                "Turn around, path blocked!",
            ]
        )

    # Category-specific opposites
    if cat == "move_clock":
        opp = _opposite_clock_action(parsed.guidance, obstacle=obstacle)
        if opp:
            pool.append(opp)
    if cat == "shift_left":
        pool.append(f"Shift right to pass {obstacle}.")
    if cat == "shift_right":
        pool.append(f"Shift left to pass {obstacle}.")
    if cat == "stop":
        pool.extend(
            [
                f"Slow down, {obstacle} ahead.",
                f"Move to 10 o'clock to avoid {obstacle}.",
            ]
        )
    if cat == "proceed":
        pool.extend(
            [
                f"Slow down, {obstacle} ahead.",
                f"Move to 10 o'clock to avoid {obstacle}.",
                "Stop! Obstacle ahead.",
            ]
        )

    # Dedup (preserve order)
    seen = set()
    out: List[str] = []
    correct_norm = normalize(parsed.guidance)
    for s in pool:
        if not isinstance(s, str) or not s.strip():
            continue
        if normalize(s) == correct_norm:
            continue
        k = normalize(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s.strip())
    return out


def build_action_command_mcq(
    *,
    image: str,
    parsed: ParsedNavigation,
    sources: Dict[str, Optional[str]],
    image_abs_path: Optional[str] = None,
    base_seed: int = 1337,
) -> VQAMultipleChoiceSample:
    """Create an Action Command MCQ sample from parsed navigation ground truth."""
    obstacle = extract_primary_obstacle(parsed.obstacles_lines) or "obstacle"
    correct = str(parsed.guidance).strip()

    pool = _build_distractor_pool(parsed=parsed, obstacle=obstacle)
    # Take first 3 distractors deterministically after a stable shuffle.
    seed = stable_int_seed(image, correct, base_seed=base_seed)
    # Stable "shuffle": sort by sha256(seed|candidate)
    scored = []
    for cand in pool:
        hh = hashlib.sha256()
        hh.update(str(seed).encode("utf-8"))
        hh.update(b"|")
        hh.update(cand.encode("utf-8", errors="ignore"))
        scored.append((hh.hexdigest(), cand))
    scored.sort(key=lambda x: x[0])
    distractors = [c for _, c in scored][:3]

    # Compose options then shuffle deterministically.
    options_texts = [correct, *distractors]
    scored2 = []
    for t in options_texts:
        hh = hashlib.sha256()
        hh.update(b"opt|")
        hh.update(str(seed).encode("utf-8"))
        hh.update(b"|")
        hh.update(t.encode("utf-8", errors="ignore"))
        scored2.append((hh.hexdigest(), t))
    scored2.sort(key=lambda x: x[0])
    shuffled = [t for _, t in scored2]

    options = assign_labels(shuffled)

    # Find correct label
    answer: ChoiceLabel = "A"
    for lab, txt in options.items():
        if normalize(txt) == normalize(correct):
            answer = lab
            break

    feedback: Dict[ChoiceLabel, str] = {}
    for lab, txt in options.items():
        if lab == answer:
            feedback[lab] = "Correct."
        else:
            feedback[lab] = f"Incorrect. Ground truth guidance is: {correct}"

    sample_id = f"{PathSafe.stem(image)}|{_QUESTION_ID}"

    return VQAMultipleChoiceSample(
        id=sample_id,
        question_id=_QUESTION_ID,
        image=image,
        image_abs_path=image_abs_path,
        question=_QUESTION_TEXT,
        options=options,
        answer=answer,
        answer_text=str(options[answer]),
        feedback=feedback,
        ground_truth={
            "scene": parsed.scene,
            "risk": parsed.risk,
            "obstacles": list(parsed.obstacles_lines),
            "guidance": correct,
        },
        sources=sources,
    )

