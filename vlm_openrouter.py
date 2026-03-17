#!/usr/bin/env python3
"""
OpenRouter VLM Object Extractor (OpenAI-compatible API).

Notes:
- DO NOT hardcode API keys. Use environment variable: OPENROUTER_API_KEY
- Endpoint: https://openrouter.ai/api/v1/chat/completions
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import time
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from vlm_base import BaseVLMExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    api_key: str
    # Keep outputs short for speed + rate-limit friendliness
    max_tokens: int = 220
    temperature: float = 0.0
    # Fail fast on slow providers; retries will backoff.
    timeout_s: int = 25
    max_retries: int = 6


class OpenRouterExtractor(BaseVLMExtractor):
    """OpenRouter-backed extractor (works for vision models that accept image_url)."""

    def __init__(self, *, api_key: Optional[str], model_id: str):
        super().__init__(model_name=f"OpenRouter:{model_id}")

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("No OpenRouter API key found (set OPENROUTER_API_KEY).")
            self.enabled = False
            return

        self.cfg = OpenRouterConfig(model=model_id, api_key=api_key)
        self.enabled = True

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 JPEG (downscale for speed)."""
        max_dim = 768
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def extract_objects_from_keyframe(
        self,
        image_path: str,
        focus_areas: List[str] = None,
        include_accessibility: bool = True,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._empty_result()

        prompt = self._build_extraction_prompt(focus_areas=focus_areas, include_accessibility=include_accessibility)
        b64 = self._encode_image(image_path)

        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
            # Best-effort JSON mode (supported by many OpenAI-compatible providers)
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter headers (safe defaults)
            "HTTP-Referer": "http://localhost",
            "X-Title": "nature-vlm",
        }

        try:
            import requests  # dependency is added to requirements.txt

            last_err: Optional[Exception] = None
            for attempt in range(int(self.cfg.max_retries) + 1):
                try:
                    logger.info(f"OpenRouter request -> {self.cfg.model} (attempt {attempt+1}/{self.cfg.max_retries+1})")
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.cfg.timeout_s,
                    )
                    # Explicit backoff on rate-limit / transient errors
                    if r.status_code in (429, 500, 502, 503, 504):
                        wait_s = min(60, 2 ** attempt)
                        logger.warning(f"OpenRouter HTTP {r.status_code}; backing off {wait_s}s (attempt {attempt+1}/{self.cfg.max_retries+1})")
                        time.sleep(wait_s)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    last_err = e
                    wait_s = min(60, 2 ** attempt)
                    logger.warning(f"OpenRouter request failed: {e}; backing off {wait_s}s (attempt {attempt+1}/{self.cfg.max_retries+1})")
                    time.sleep(wait_s)
            else:
                # exhausted retries
                raise last_err or RuntimeError("OpenRouter request failed after retries")

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = self._parse_objects_json(text)
            # If parsing failed, keep the raw response to aid debugging (truncated).
            if isinstance(parsed, dict) and parsed.get("error"):
                preview = str(text)[:500]
                parsed["raw_response_preview"] = preview
                parsed["error_message"] = f"Failed to parse objects from model response. Preview: {preview[:200]}"
            parsed["model"] = self.model_name
            parsed["timestamp"] = datetime.now().isoformat()
            return parsed
        except Exception as e:
            logger.error(f"OpenRouter extraction error: {e}")
            out = self._empty_result()
            # surface error to caller so main.py can print it
            if isinstance(out, dict):
                out["error"] = True
                out["error_message"] = str(e)
            return out

    def classify_frame_artifacts(self, image_path: str) -> Dict[str, Any]:
        """Classify artifacts via OpenRouter (JSON-only)."""
        if not self.enabled:
            return self._empty_artifact_result()

        prompt = """Return ONLY valid JSON. No markdown. Use double quotes.

Schema:
{"has_artifacts":true/false,"artifact_type":"1|2|3|none","confidence":0.0,"description":"","severity":"low|medium|high"}

Rules:
- artifact_type: "1" visual hallucinations, "2" image artifacts (blur/noise/compression/pixelation), "3" AI inconsistencies, "none" if clean
- description: <= 120 chars
"""

        try:
            import requests

            b64 = self._encode_image(image_path)
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} ,
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 180,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "nature-vlm",
            }

            last_err: Optional[Exception] = None
            data = None
            for attempt in range(int(self.cfg.max_retries) + 1):
                try:
                    logger.info(f"OpenRouter artifact request -> {self.cfg.model} (attempt {attempt+1}/{self.cfg.max_retries+1})")
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.cfg.timeout_s,
                    )
                    if r.status_code in (429, 500, 502, 503, 504):
                        wait_s = min(60, 2 ** attempt)
                        logger.warning(f"OpenRouter HTTP {r.status_code}; backing off {wait_s}s (attempt {attempt+1}/{self.cfg.max_retries+1})")
                        time.sleep(wait_s)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    last_err = e
                    wait_s = min(60, 2 ** attempt)
                    logger.warning(f"OpenRouter artifact request failed: {e}; backing off {wait_s}s (attempt {attempt+1}/{self.cfg.max_retries+1})")
                    time.sleep(wait_s)
            if data is None:
                raise last_err or RuntimeError("OpenRouter artifact request failed after retries")

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = self._parse_artifact_json(text)
            parsed["model"] = self.model_name
            parsed["timestamp"] = datetime.now().isoformat()
            return parsed
        except Exception as e:
            out = self._empty_artifact_result()
            out["error"] = True
            out["error_message"] = str(e)
            return out

    def answer_multiple_choice(
        self,
        image_path: str,
        question: str,
        options: Union[Dict[str, str], Any],
    ) -> Optional[str]:
        """
        Answer a single multiple-choice VQA question (A/B/C/D) from image + question.
        Used for VQA ground-truth creation with a powerful model (e.g. Qwen 235B).

        Args:
            image_path: Path to the image (must exist).
            question: The question text.
            options: Dict with keys "A", "B", "C", "D" and option text values.

        Returns:
            One of "A", "B", "C", "D" or None if parsing failed.
        """
        if not self.enabled:
            return None
        opts = {k: str(v).strip() for k, v in (options or {}).items() if k in ("A", "B", "C", "D")}
        if len(opts) != 4:
            return None
        prompt = (
            "You are answering a multiple-choice question about this image. "
            "Reply with exactly one letter: A, B, C, or D. No explanation.\n\n"
            f"Question: {question}\n\n"
            "Options:\n"
            f"A: {opts.get('A', '')}\n"
            f"B: {opts.get('B', '')}\n"
            f"C: {opts.get('C', '')}\n"
            f"D: {opts.get('D', '')}\n\n"
            "Answer (one letter only):"
        )
        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return None
        b64 = self._encode_image(image_path)
        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 10,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "nature-vqa-groundtruth",
        }
        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    wait_s = min(60, 2 ** attempt)
                    logger.warning(
                        "OpenRouter VQA HTTP %s; backing off %ss (attempt %s)",
                        r.status_code,
                        wait_s,
                        attempt + 1,
                    )
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                data = r.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                text = (text or "").strip().upper()
                for c in text:
                    if c in "ABCD":
                        return c
                logger.warning("OpenRouter VQA: no A/B/C/D in response %r", (text or "")[:200])
                return None
            except Exception as e:
                last_err = e
                wait_s = min(60, 2 ** attempt)
                logger.warning("OpenRouter VQA request failed: %s; backing off %ss", e, wait_s)
                time.sleep(wait_s)
        if last_err:
            logger.error("OpenRouter VQA failed after retries: %s", last_err)
        return None

    def generate_freeform_text(self, image_path: str, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate free-form text from image + prompt (for VQA, captions, etc.).
        
        Args:
            image_path: Path to image file
            prompt: Question or instruction text
            max_new_tokens: Max tokens to generate (default 100, ~75 words)
        
        Returns:
            Generated text, or empty string on failure
        """
        if not self.enabled:
            return ""
        try:
            import requests
            b64 = self._encode_image(image_path)
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(prompt)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": min(int(max_new_tokens), 500),
            }
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "nature-vqa",
            }
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.cfg.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return (text or "").strip()
        except Exception as e:
            logger.error("OpenRouter generate_freeform_text failed: %s", e)
            return ""

    def generate_freeform_text_batch(
        self,
        image_path: str,
        questions: List[Dict[str, Any]],
        max_words_per_answer: int = 50,
        include_reference_in_prompt: bool = False,
    ) -> Dict[str, str]:
        """
        Answer multiple free-form VQA questions for one image in ONE API call.
        Returns Dict mapping question id -> answer text.
        
        Args:
            image_path: Path to image
            questions: List of {"id": "...", "question": "...", "answer": "..." (optional)}
            max_words_per_answer: Max words per answer (default 50)
            include_reference_in_prompt: If True, include each question's "answer" as a reference
                for the model to use as a helper (refinement mode - produce best possible answer)
        
        Returns:
            Dict[qid, answer_text] - empty string if parse failed
        """
        if not self.enabled or not questions:
            return {q.get("id", ""): "" for q in questions}
        try:
            import requests
            import re
            if include_reference_in_prompt:
                from vlm_refinement_prompts import REFINEMENT_SYSTEM_INSTRUCTIONS
                parts = [
                    REFINEMENT_SYSTEM_INSTRUCTIONS,
                    f"Keep each answer to {max_words_per_answer} words or less. Respond in exactly this format: Q1: [answer]\nQ2: [answer]\n...\n\n"
                ]
                for i, q in enumerate(questions, 1):
                    parts.append(f"Q{i}: {q.get('question', '')}")
                parts.append("\nYour answers:")
            else:
                parts = [
                    f"Answer these questions about the image. Keep each answer to {max_words_per_answer} words or less. "
                    "Respond in exactly this format:\nQ1: [answer]\nQ2: [answer]\n...\n\n"
                ]
                for i, q in enumerate(questions, 1):
                    parts.append(f"Q{i}: {q.get('question', '')}")
                parts.append("\nYour answers:")
            prompt = "\n".join(parts)
            max_tokens = min(600, 50 + len(questions) * (max_words_per_answer * 2))
            b64 = self._encode_image(image_path)
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "nature-vqa-batch",
            }
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.cfg.timeout_s,
            )
            r.raise_for_status()
            text = (
                r.json().get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            result = {}
            text = (text or "").strip()
            for i, q in enumerate(questions, 1):
                qid = q.get("id", "")
                ans = ""
                # Primary: Q1: answer, Q2: answer, ...
                match = re.search(rf"Q{i}\s*:\s*(.+?)(?=Q{i+1}\s*:|$)", text, re.DOTALL | re.IGNORECASE)
                if match:
                    ans = match.group(1).strip()
                else:
                    # Fallback: "1. answer", "1) answer", or numbered lines
                    for pat in [rf"^{i}\s*[\.\)]\s*(.+?)(?=\n{i+1}\s*[\.\)]|\n\n|\Z)", rf"^{i}\s*[\.\)]\s*(.+)$"]:
                        m = re.search(pat, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                        if m:
                            ans = m.group(1).strip()
                            break
                if not ans and i <= len(questions):
                    # Last resort: split by newlines, take line i-1 (1-based index)
                    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
                    if i <= len(lines):
                        ans = lines[i - 1]
                if ans:
                    words = ans.split()
                    if len(words) > max_words_per_answer:
                        ans = " ".join(words[:max_words_per_answer])
                result[qid] = ans
            return result
        except Exception as e:
            logger.error("OpenRouter generate_freeform_text_batch failed: %s", e)
            return {q.get("id", ""): "" for q in questions}

    def answer_multiple_choice_batch(
        self,
        image_path: str,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        """
        Answer multiple VQA questions for a single image in one API call (economical).
        
        Args:
            image_path: Path to the image.
            questions: List of dicts, each with:
                - "id": question identifier (e.g. "Q1", "main_obstacle")
                - "question": question text
                - "options": Dict with "A", "B", "C", "D"
        
        Returns:
            Dict mapping question id to answer ("A"/"B"/"C"/"D" or None if failed)
        
        Example:
            questions = [
                {"id": "Q1", "question": "What is the main obstacle?", "options": {...}},
                {"id": "Q2", "question": "What is the closest obstacle?", "options": {...}},
            ]
            result = {"Q1": "A", "Q2": "C"}
        """
        if not self.enabled:
            return {q["id"]: None for q in questions}
        
        if not questions:
            return {}
        
        # Build compact prompt with all questions
        prompt_parts = [
            "Answer these multiple-choice questions about the image. "
            "Respond with ONLY the format: Q1: A, Q2: B, Q3: C, etc. No explanation.\n"
        ]
        
        for i, q in enumerate(questions, 1):
            opts = {k: str(v).strip() for k, v in (q.get("options", {}) or {}).items() if k in ("A", "B", "C", "D")}
            if len(opts) != 4:
                continue
            
            prompt_parts.append(f"\nQ{i}: {q['question']}")
            prompt_parts.append(f"A) {opts['A']}")
            prompt_parts.append(f"B) {opts['B']}")
            prompt_parts.append(f"C) {opts['C']}")
            prompt_parts.append(f"D) {opts['D']}")
        
        prompt_parts.append("\n\nYour answer (format: Q1: A, Q2: B, etc.):")
        prompt = "\n".join(prompt_parts)
        
        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return {q["id"]: None for q in questions}
        
        b64 = self._encode_image(image_path)
        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 50,  # Enough for "Q1: A, Q2: B, Q3: C, Q4: D, Q5: A, Q6: B"
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "nature-vqa-batch",
        }
        
        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    wait_s = min(60, 2 ** attempt)
                    logger.warning(
                        "OpenRouter batch VQA HTTP %s; backing off %ss (attempt %s)",
                        r.status_code,
                        wait_s,
                        attempt + 1,
                    )
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                data = r.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                
                # Parse response like "Q1: A, Q2: B, Q3: C, Q4: D, Q5: A, Q6: B"
                result = self._parse_batch_response(text, questions)
                return result
                
            except Exception as e:
                last_err = e
                wait_s = min(60, 2 ** attempt)
                logger.warning("OpenRouter batch VQA request failed: %s; backing off %ss", e, wait_s)
                time.sleep(wait_s)
        
        if last_err:
            logger.error("OpenRouter batch VQA failed after retries: %s", last_err)
        return {q["id"]: None for q in questions}
    
    def _parse_batch_response(self, text: str, questions: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Parse batch response like "Q1: A, Q2: B, Q3: C" into dict.
        
        Handles various formats:
        - "Q1: A, Q2: B, Q3: C"
        - "Q1:A Q2:B Q3:C"
        - "1:A 2:B 3:C"
        - "A B C D A B" (positional)
        """
        import re
        
        text = (text or "").strip().upper()
        result = {}
        
        # Try pattern: Q1: A, Q2: B
        matches = re.findall(r'Q?(\d+)\s*:\s*([ABCD])', text)
        if matches:
            for idx_str, answer in matches:
                idx = int(idx_str) - 1  # Q1 -> index 0
                if 0 <= idx < len(questions):
                    result[questions[idx]["id"]] = answer
        else:
            # Fallback: extract all A/B/C/D letters in order
            letters = re.findall(r'[ABCD]', text)
            for i, letter in enumerate(letters):
                if i < len(questions):
                    result[questions[i]["id"]] = letter
        
        # Fill missing with None
        for q in questions:
            if q["id"] not in result:
                result[q["id"]] = None
        
        return result

    def _build_extraction_prompt(self, *, focus_areas: Optional[List[str]], include_accessibility: bool) -> str:
        priority = ", ".join(focus_areas) if focus_areas else ""
        access = "YES" if include_accessibility else "NO"
        # Comprehensive schema (works for ALL OpenRouter models; simplified output writer will still
        # only store objects list if --vlm-version simplified is used).
        prompt = f"""Return ONLY valid JSON. No markdown. Use double quotes.

Schema:
{{
  "objects": ["..."],
  "categories": {{
    "signs": ["..."],
    "accessibility": ["..."],
    "people": ["..."],
    "furniture": ["..."],
    "technology": ["..."],
    "other": ["..."]
  }},
  "scene_description": "",
  "primary_focus": ""
}}

Rules:
- objects: 1-50 short singular nouns (no adjectives/colors/materials)
- Use "person" for any human
- No duplicates
- scene_description: <= 200 chars
- primary_focus: one short noun from objects
- Accessibility mode: {access}
"""
        if include_accessibility:
            prompt += "- Include obstacles/hazards first (cable, cord, cart, bag, box, trash can, cone, wet floor sign, step, stairs, curb, pillar, pole, barrier)\n"
        if priority:
            prompt += f"- Priority objects: {priority}\n"
        return prompt

    def _parse_objects_json(self, text: str) -> Dict[str, Any]:
        """Parse object extraction output (supports comprehensive schema)."""
        try:
            s = (text or "").strip()
            # Strip common fenced wrappers
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
                s = re.sub(r"\s*```$", "", s).strip()

            # Try to extract a balanced JSON object (robust to extra text)
            def _extract_balanced_obj(t: str) -> Optional[str]:
                start = t.find("{")
                if start == -1:
                    return None
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(t)):
                    ch = t[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                return t[start : i + 1]
                return None

            obj_str = _extract_balanced_obj(s) or ""
            if obj_str:
                data = json.loads(obj_str)
                if isinstance(data, dict) and isinstance(data.get("objects"), list):
                    objs = self._normalize_objects([str(x) for x in data.get("objects") or []])
                    categories_in = data.get("categories") or {}
                    categories: Dict[str, List[str]] = {
                        "signs": [],
                        "accessibility": [],
                        "people": [],
                        "furniture": [],
                        "technology": [],
                        "other": [],
                    }
                    if isinstance(categories_in, dict):
                        for k in list(categories.keys()):
                            v = categories_in.get(k, [])
                            if isinstance(v, list):
                                categories[k] = self._normalize_objects([str(x) for x in v])
                    return {
                        "objects": objs,
                        "categories": categories,
                        "scene_description": str(data.get("scene_description") or "")[:500],
                        "primary_focus": str(data.get("primary_focus") or "")[:200],
                        "num_objects": len(objs),
                        "error": False,
                    }

            # Sometimes models return a JSON array directly
            arr_match = re.search(r"\[[\s\S]*\]", s)
            if arr_match:
                arr = json.loads(arr_match.group(0))
                if isinstance(arr, list):
                    objs = self._normalize_objects([str(x) for x in arr])
                    return {"objects": objs, "num_objects": len(objs), "error": False}

            # Last resort: extract quoted tokens as object candidates
            quoted = re.findall(r'["\']([^"\']+)["\']', s)
            if quoted:
                objs = self._normalize_objects([str(x) for x in quoted])
                if objs:
                    return {"objects": objs, "num_objects": len(objs), "error": False}

            # Last resort #2: try comma/newline separated plain text lists (no quotes)
            # e.g. "chair, table, person" or "- chair\n- table"
            plain = re.sub(r"^[\s\-•\d\.\)]*", "", s, flags=re.MULTILINE)
            parts = re.split(r"[,;\n]+", plain)
            cand = [p.strip() for p in parts if p.strip()]
            # avoid grabbing long sentences; keep short tokens only
            cand = [c for c in cand if 1 <= len(c.split()) <= 3 and len(c) <= 30]
            if cand:
                objs = self._normalize_objects(cand)
                if objs:
                    return {"objects": objs, "num_objects": len(objs), "error": False}
        except Exception:
            pass
        return self._empty_result()

    def _parse_artifact_json(self, text: str) -> Dict[str, Any]:
        """Parse artifact JSON output."""
        try:
            s = (text or "").strip()
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
                s = re.sub(r"\s*```$", "", s).strip()
            # try balanced object
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]
            data = json.loads(s)
            if isinstance(data, dict):
                return {
                    "has_artifacts": bool(data.get("has_artifacts", False)),
                    "artifact_type": str(data.get("artifact_type", "none")),
                    "confidence": float(data.get("confidence", 0.0) or 0.0),
                    "description": str(data.get("description", "") or "")[:200],
                    "severity": str(data.get("severity", "low") or "low"),
                    "affected_regions": [],
                    "error": False,
                }
        except Exception:
            pass
        return self._empty_artifact_result()

