"""Caption generation backends.

We provide a deterministic template-based captioner (always available),
and optionally a VLM-based captioner (Qwen/GPT-4o) for higher quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Protocol

from .spatial import RelativePosition

# Try to import OpenAI at module level to catch import errors early
try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAIClient = None
    _OPENAI_AVAILABLE = False


@dataclass(frozen=True)
class ObjectMention:
    """Object mention with an optional coarse relative position."""

    name: str
    position: Optional[RelativePosition] = None
    confidence: Optional[float] = None
    size: Optional[float] = None  # normalized area, if available
    x_center: Optional[float] = None  # normalized [0,1]
    y_center: Optional[float] = None  # normalized [0,1]


@dataclass(frozen=True)
class CaptionContext:
    """All structured info we expose to captioners."""

    scene_description: str
    primary_focus: str
    objects: List[str]
    mentions: List[ObjectMention]


class Captioner(Protocol):
    """Captioner interface."""

    def generate(self, image_path: str, ctx: CaptionContext) -> str:  # pragma: no cover
        ...


def _clean_text(text: str) -> str:
    """Conservative sanitizer for generated captions (strip paths/model names)."""
    s = (text or "").strip()
    if not s:
        return ""

    # Remove obvious file paths and filenames.
    s = re.sub(r"\b[a-zA-Z]:\\[^\s]+", "", s)  # Windows paths
    s = re.sub(r"\b/[^ \n\t]+", "", s)  # Unix-like absolute paths
    s = re.sub(r"\b[\w\-.]+\.(jpg|jpeg|png|webp|bmp)\b", "", s, flags=re.IGNORECASE)

    # Remove common model-name mentions if they appear.
    s = re.sub(r"\b(gpt[- ]?4o|qwen|claude)\b", "", s, flags=re.IGNORECASE)

    # Collapse whitespace.
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


class TemplateCaptioner:
    """Deterministic caption generator (no external calls)."""

    def __init__(self, max_objects: int = 6):
        self.max_objects = int(max_objects)

    def generate(self, image_path: str, ctx: CaptionContext) -> str:
        # Base scene description
        scene = (ctx.scene_description or "").strip()
        if not scene:
            # Fallback: infer a minimal scene statement from objects
            env = _infer_environment(ctx.objects)
            if env:
                scene = f"{env.capitalize()} scene."
            elif ctx.objects:
                scene = f"A scene containing {', '.join(ctx.objects[:3])}."
            else:
                scene = "A scene with a few visible objects."

        scene = _ensure_period(scene)

        # Choose key mentions (prefer detections with size/confidence)
        mentions = sorted(
            ctx.mentions,
            key=lambda m: (
                -float(m.size or 0.0),
                -float(m.confidence or 0.0),
                m.name,
            ),
        )
        key = _dedupe_preserve([m for m in mentions if m.name], key=lambda m: m.name)[: self.max_objects]

        # Build object sentences with coarse spatial cues.
        obj_sentence = _render_key_objects_sentence(key)

        safety_sentence = _render_safety_sentence(ctx.objects, key)

        parts = [scene]
        if obj_sentence:
            parts.append(obj_sentence)
        if safety_sentence:
            parts.append(safety_sentence)

        # Keep it concise (2-4 sentences).
        out = " ".join([p.strip() for p in parts if p.strip()])
        out = _clean_text(out)
        return out


class QwenCaptioner:
    """Qwen3-VL captioner using local generation (if available)."""

    def __init__(self, device: str = "auto"):
        from vlm_qwen import Qwen3VLExtractor

        self.extractor = Qwen3VLExtractor(device=device)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.extractor, "enabled", False))

    def generate(self, image_path: str, ctx: CaptionContext, prompt: str) -> str:
        if not self.enabled:
            raise RuntimeError("Qwen captioner is not available.")
        text = self.extractor.generate_freeform_text(image_path=image_path, prompt=prompt, max_new_tokens=220)
        return _clean_text(text)


class GPTCaptioner:
    """GPT captioner using OpenAI API (if available)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", max_tokens: int = 1000):
        from config import get_openai_api_key

        self.api_key = api_key or get_openai_api_key()
        self.model = str(model)
        self.max_tokens = max_tokens
        self._client = None
        self._enabled = False

        if not self.api_key:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("OPENAI_API_KEY not found; GPT captioner disabled.")
            return

        import logging
        logger = logging.getLogger(__name__)
        
        if not _OPENAI_AVAILABLE:
            logger.error("OpenAI SDK not available. Install with: pip install openai")
            logger.error(f"Python executable: {__import__('sys').executable}")
            self._enabled = False
            return
        
        try:
            self._client = _OpenAIClient(api_key=self.api_key)
            self._enabled = True
            logger.info(f"GPT captioner initialized with model: {self.model}, API key length: {len(self.api_key) if self.api_key else 0}")
        except Exception as e:
            logger.error(f"OpenAI client initialization failed: {e}")
            logger.error(f"API key present: {bool(self.api_key)}, API key length: {len(self.api_key) if self.api_key else 0}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._client is not None)

    def generate(
        self,
        image_path: str,
        ctx: CaptionContext,
        prompt: str,
        cost_tracker=None,
        image_detail: str = "auto",
        resize_720p: bool = False,
    ) -> str:
        if not self.enabled:
            raise RuntimeError(f"GPT captioner is not available. API key: {bool(self.api_key)}, Client: {self._client is not None}")
        
        import base64
        from pathlib import Path
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Encode image
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img_bytes = img_path.read_bytes()

        # Optionally resize in-memory to fit within 1280x720 (keeps aspect ratio).
        if resize_720p:
            try:
                from PIL import Image
                import io

                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()
            except Exception:
                # If resize fails, fall back to original bytes.
                pass

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Estimate image tokens (gpt-4o family: base 85 + 170 per 512px tile for detail=high/auto)
        # detail=low is fixed 85 tokens.
        image_tokens = 0
        detail = str(image_detail or "auto").lower()
        if detail == "low":
            image_tokens = 85
        else:
            try:
                from PIL import Image
                import io

                with Image.open(io.BytesIO(img_bytes)) as img2:
                    w, h = img2.size
                    if max(w, h) > 2048:
                        scale = 2048 / max(w, h)
                        w, h = int(w * scale), int(h * scale)
                    if min(w, h) > 768:
                        scale = 768 / min(w, h)
                        w, h = int(w * scale), int(h * scale)
                    tiles = ((w + 511) // 512) * ((h + 511) // 512)
                    image_tokens = 85 + (tiles * 170)
            except Exception:
                image_tokens = 1105
        
        # Determine media type
        suffix = img_path.suffix.lower()
        # If we resized we re-encoded as JPEG.
        media_type = "image/jpeg"
        
        # Call GPT
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(prompt)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{img_b64}",
                                    "detail": detail,
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.2
            )
            
            # Track usage
            if cost_tracker and hasattr(response, 'usage'):
                usage = response.usage
                # OpenAI's prompt_tokens includes image tokens, but we track image separately for display
                # For pricing, we use actual prompt_tokens (which includes images)
                text_input_tokens = max(0, usage.prompt_tokens - image_tokens)
                output_tokens = usage.completion_tokens
                cost_tracker.add_gpt_usage(self.model, text_input_tokens, output_tokens, image_tokens)
            
            text = response.choices[0].message.content
            return _clean_text(text or "")
        except Exception as e:
            logger.error(f"OpenAI API call failed with model {self.model}: {e}")
            raise


def _ensure_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        return s + "."
    return s


def _dedupe_preserve(items: List, key):
    seen = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _infer_environment(objects: List[str]) -> str:
    oset = {o.lower() for o in objects}
    if "elevator" in oset:
        return "an elevator interior"
    if "airport terminal" in oset:
        return "an airport terminal"
    if "hallway" in oset or "corridor" in oset:
        return "a hallway"
    if "staircase" in oset or "stairs" in oset:
        return "a stair area"
    return ""


def _render_key_objects_sentence(key: List[ObjectMention]) -> str:
    if not key:
        return ""

    phrases: List[str] = []
    # Special-case a couple of common pairs for nicer wording
    names = {m.name.lower() for m in key}
    if "control panel" in names and "button" in names:
        phrases.append("A control panel with buttons is visible")
        # Remove "button" from remaining to avoid repetition
        key = [m for m in key if m.name.lower() not in {"button"}]

    for m in key:
        pos = ""
        if m.position is not None:
            # Translate "upper-right" into natural phrase
            pos = _pos_to_phrase(m.position)
        if pos:
            phrases.append(f"{_a_or_an(m.name)} {m.name} is {pos}")
        else:
            phrases.append(f"{_a_or_an(m.name)} {m.name} is visible")

    # Join into a single sentence, keeping it readable.
    if not phrases:
        return ""
    if len(phrases) == 1:
        return _ensure_period(phrases[0])
    if len(phrases) == 2:
        return _ensure_period(f"{phrases[0]}, and {phrases[1]}")
    return _ensure_period(", ".join(phrases[:-1]) + f", and {phrases[-1]}")


def _render_safety_sentence(objects: List[str], key: List[ObjectMention]) -> str:
    oset = {o.lower() for o in objects}
    if "floor mat" in oset:
        return "A floor mat on the ground may require attention to avoid tripping."
    if "stairs" in oset or "stair" in oset:
        return "Stairs can present a fall risk; use handrails and watch footing."
    if "escalator" in oset:
        return "An escalator may be nearby; approach carefully and hold the handrail."
    # If we know there is a wheelchair ramp, that's a navigation affordance.
    if "ramp" in oset or "wheelchair ramp" in oset:
        return "A ramp provides an accessible route compared with stairs."
    return ""


def _pos_to_phrase(pos: RelativePosition) -> str:
    h = pos.horizontal
    v = pos.vertical
    dist = pos.distance
    parts = []
    # Keep it simple and non-numeric.
    if v == "upper":
        parts.append("in the upper")
    elif v == "middle":
        parts.append("around the middle")
    else:
        parts.append("in the lower")
    if h == "left":
        parts.append("left")
    elif h == "center":
        parts.append("center")
    else:
        parts.append("right")
    loc = " ".join(parts)
    if dist == "near":
        return f"{loc} and appears close"
    if dist == "far":
        return f"{loc} and appears farther away"
    return f"{loc}"


def _a_or_an(noun: str) -> str:
    n = (noun or "").strip().lower()
    if not n:
        return "a"
    if n[0] in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"

