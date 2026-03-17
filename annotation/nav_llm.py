"""LLM-backed navigation annotation generation.

Supports:
- Claude (Anthropic) vision models
- Qwen3-VL (local) vision-language model
- GPT (OpenAI) vision models
- OpenRouter (openai/gpt-oss-safeguard-20b, etc.)
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import get_anthropic_api_key, get_openrouter_api_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMNavConfig:
    backend: str = "auto"  # auto|claude|qwen|deterministic
    claude_model: str = "claude-3-7-sonnet-20250219"  # Also supports claude-sonnet-4-20250514 (Claude 4.5 Sonnet)
    max_tokens: int = 1200
    temperature: float = 0.2
    include_depth_image: bool = True


# Supported Claude models with their capabilities
CLAUDE_MODELS = {
    "claude-3-7-sonnet-20250219": {"name": "Claude 3.7 Sonnet", "vision": True},
    "claude-sonnet-4-20250514": {"name": "Claude 4.5 Sonnet", "vision": True},
    "claude-3-5-sonnet-20241022": {"name": "Claude 3.5 Sonnet", "vision": True},
    "claude-3-opus-20240229": {"name": "Claude 3 Opus", "vision": True},
}


class ClaudeNavGenerator:
    """Claude vision backend (Anthropic)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_anthropic_api_key()
        self._client = None
        self._enabled = False

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found; Claude navigation backend disabled.")
            return

        try:
            from anthropic import Anthropic  # type: ignore

            self._client = Anthropic(api_key=self.api_key)
            self._enabled = True
        except Exception as e:
            logger.warning(f"Anthropic SDK not available or failed to init: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._client is not None)

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        max_tokens: int,
        temperature: float,
        depth_png_bytes: Optional[bytes] = None,
        cost_tracker=None,
        resize_720p: bool = False,
    ) -> str:
        """Generate navigation response from Claude."""
        if not self.enabled:
            raise RuntimeError("Claude backend not available.")

        img_bytes = image_path.read_bytes()

        # Optionally resize in-memory to fit within 1280x720 (keeps aspect ratio).
        if resize_720p:
            try:
                from PIL import Image
                import io

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()
            except Exception:
                pass

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        content = [
            {"type": "text", "text": str(prompt)},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
            },
        ]

        if depth_png_bytes:
            content.append(
                {
                    "type": "text",
                    "text": "Depth map (grayscale): darker is closer, lighter is farther.",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(depth_png_bytes).decode("utf-8"),
                    },
                }
            )

        msg = self._client.messages.create(
            model=str(model),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=str(system_prompt) if system_prompt else None,
            messages=[{"role": "user", "content": content}],
        )

        # Track usage
        if cost_tracker and hasattr(msg, 'usage'):
            usage = msg.usage
            cost_tracker.add_claude_usage(str(model), usage.input_tokens, usage.output_tokens)

        # Anthropic returns list of content blocks; first is usually text.
        try:
            return str(msg.content[0].text).strip()
        except Exception:
            return str(getattr(msg, "content", "")).strip()


class QwenNavGenerator:
    """Qwen3-VL local backend."""

    def __init__(self, device: str = "auto"):
        self._enabled = False
        self._extractor = None
        try:
            from vlm_qwen import Qwen3VLExtractor

            self._extractor = Qwen3VLExtractor(device=device)
            self._enabled = bool(self._extractor and getattr(self._extractor, "enabled", False))
        except Exception as e:
            logger.warning(f"Qwen3-VL backend unavailable: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._extractor is not None)

    def generate(self, *, image_path: Path, prompt: str, max_new_tokens: int = 280) -> str:
        if not self.enabled:
            raise RuntimeError("Qwen backend not available.")
        return str(self._extractor.generate_freeform_text(str(image_path), prompt=str(prompt), max_new_tokens=int(max_new_tokens))).strip()


class GPTNavGenerator:
    """GPT navigation backend (OpenAI API)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", max_tokens: int = 1000):
        from config import get_openai_api_key

        self.api_key = api_key or get_openai_api_key()
        self.model = str(model)
        self.max_tokens = max_tokens
        self._client = None
        self._enabled = False

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found; GPT navigation backend disabled.")
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            self._enabled = True
        except Exception as e:
            logger.warning(f"OpenAI SDK not available or failed to init: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._client is not None)

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        depth_png_bytes: Optional[bytes] = None,
        cost_tracker=None,
        image_detail: str = "auto",
        resize_720p: bool = False,
    ) -> str:
        """Generate navigation response from GPT."""
        if not self.enabled:
            raise RuntimeError("GPT backend not available.")

        img_bytes = image_path.read_bytes()

        # Optionally resize in-memory to fit within 1280x720 (keeps aspect ratio).
        if resize_720p:
            try:
                from PIL import Image
                import io

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()
            except Exception:
                pass

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Estimate image tokens (4o family: base 85 + 170 per 512px tile for high/auto; low is fixed 85)
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

        content = [
            {"type": "text", "text": str(prompt)},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": detail,
                }
            },
        ]

        if depth_png_bytes:
            content.append(
                {
                    "type": "text",
                    "text": "Depth map (grayscale): darker is closer, lighter is farther.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(depth_png_bytes).decode('utf-8')}"
                    }
                }
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=int(max_tokens or self.max_tokens),
            temperature=0.2
        )

        # Track usage
        if cost_tracker and hasattr(response, 'usage'):
            usage = response.usage
            # OpenAI includes image tokens in prompt_tokens, so we track separately
            text_input_tokens = max(0, usage.prompt_tokens - image_tokens)
            output_tokens = usage.completion_tokens
            cost_tracker.add_gpt_usage(self.model, text_input_tokens, output_tokens, image_tokens)

        return str(response.choices[0].message.content).strip()


class OpenRouterNavGenerator:
    """OpenRouter navigation backend (OpenAI-compatible API, e.g. openai/gpt-oss-safeguard-20b)."""

    def __init__(self, *, api_key: Optional[str] = None, model_id: str = "openai/gpt-oss-safeguard-20b", max_tokens: int = 1200):
        self.api_key = api_key or get_openrouter_api_key()
        self.model_id = str(model_id)
        self.max_tokens = max_tokens
        self._enabled = bool(self.api_key)

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found; OpenRouter navigation backend disabled.")

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        depth_png_bytes: Optional[bytes] = None,
        cost_tracker=None,
        resize_720p: bool = False,
    ) -> str:
        """Generate navigation/accessibility response from OpenRouter."""
        if not self.enabled:
            raise RuntimeError("OpenRouter backend not available.")

        img_bytes = image_path.read_bytes()

        if resize_720p:
            try:
                from PIL import Image
                import io

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()
            except Exception:
                pass

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        content = [
            {"type": "text", "text": str(prompt)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]

        if depth_png_bytes:
            content.append(
                {"type": "text", "text": "Depth map (grayscale): darker is closer, lighter is farther."}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(depth_png_bytes).decode('utf-8')}"
                    },
                }
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})

        model_id = model or self.model_id

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "nature-annotation",
            }

            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": int(max_tokens or self.max_tokens),
                "temperature": float(temperature),
            }

            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if r.status_code in (429, 500, 502, 503, 504):
                wait_s = 5
                logger.warning(f"OpenRouter HTTP {r.status_code}; retrying after {wait_s}s")
                time.sleep(wait_s)
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

            r.raise_for_status()
            data = r.json()

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if cost_tracker and hasattr(cost_tracker, "add_gpt_usage"):
                usage = data.get("usage", {})
                inp = int(usage.get("prompt_tokens", 0))
                out = int(usage.get("completion_tokens", 0))
                cost_tracker.add_gpt_usage(model_id, inp, out, 0)

            return str(text).strip()
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise


def _guess_media_type(path: Path) -> str:
    s = path.suffix.lower()
    if s in {".png"}:
        return "image/png"
    if s in {".webp"}:
        return "image/webp"
    return "image/jpeg"

