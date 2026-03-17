#!/usr/bin/env python3
"""
GPT-4o Object Extractor (API-based)
Uses OpenAI GPT-4o for cloud-based inference

Requirements:
    pip install openai python-dotenv
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64

from vlm_base import BaseVLMExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT4oExtractor(BaseVLMExtractor):
    """
    GPT-4o Vision-Language Model for object extraction.
    Uses OpenAI API (requires API key).
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        """
        Initialize GPT extractor (supports GPT-4o, GPT-5 Nano, etc.).
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-5-nano")
        """
        # Map friendly names to OpenAI model names
        model_map = {
            "gpt-4o": "gpt-4o",
            "gpt4o": "gpt-4o",
            "gpt-5-nano": "gpt-5-nano-2025-08-07",
            "gpt5nano": "gpt-5-nano-2025-08-07",
            "gpt5-nano": "gpt-5-nano-2025-08-07",
            "gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07",
            "gpt-5-mini": "gpt-5-mini-2025-08-07",
            "gpt5mini": "gpt-5-mini-2025-08-07",
            "gpt5-mini": "gpt-5-mini-2025-08-07",
            "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07",
        }
        self.openai_model = model_map.get(model_name.lower(), model_name)
        display_name = self.openai_model.replace("-", " ").title()
        
        super().__init__(model_name=display_name)
        
        # Load API key
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.error("No OpenAI API key found")
            self.enabled = False
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.enabled = True
            
            # Initialize cost tracker
            from annotation.cost_tracker import CostTracker, GPT_PRICING
            self.cost_tracker = CostTracker()
            self.gpt_pricing = GPT_PRICING
            self.last_call_cost = 0.0  # Track cost of last API call
            self.last_call_tokens = (0, 0)  # Track (input, output) tokens of last call
            
            logger.info(f"✅ {display_name} initialized (model: {self.openai_model})")
        except ImportError as e:
            logger.error("openai package not installed: pip install openai")
            self.enabled = False
            self.cost_tracker = None
            self.gpt_pricing = {}
        except Exception as e:
            logger.warning(f"Cost tracker initialization failed: {e}")
            # Still enable the extractor even if cost tracking fails
            if not hasattr(self, 'cost_tracker'):
                self.cost_tracker = None
            if not hasattr(self, 'gpt_pricing'):
                self.gpt_pricing = {}
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 (downscale to reduce cost while keeping analyzable quality)."""
        from io import BytesIO
        from PIL import Image

        max_dim = 512  # 512px is sufficient for GPT-5 nano object detection, saves ~75% on image tokens
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)  # Quality 70 saves more while keeping objects visible
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def extract_objects_from_keyframe(
        self, 
        image_path: str,
        focus_areas: List[str] = None,
        include_accessibility: bool = True
    ) -> Dict[str, Any]:
        """
        Extract objects using GPT-4o.
        
        Args:
            image_path: Path to keyframe image
            focus_areas: Optional focus areas
            include_accessibility: Include accessibility features
            
        Returns:
            Dictionary with extracted objects
        """
        if not self.enabled:
            return self._empty_result()
        
        try:
            # Build prompt
            prompt = self._build_extraction_prompt(focus_areas, include_accessibility)
            
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Call OpenAI model
            # GPT-5 Nano/Mini requires max_completion_tokens instead of max_tokens
            # GPT-5 Nano/Mini only supports default temperature (1), not custom values
            is_gpt5_nano = "gpt-5-nano" in self.openai_model.lower()
            is_gpt5_mini = "gpt-5-mini" in self.openai_model.lower()
            is_gpt5 = is_gpt5_nano or is_gpt5_mini
            api_params = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            }
            if is_gpt5:
                # GPT-5 Nano/Mini uses tokens for reasoning - need much more room for output
                # Reasoning can take 1000-2000 tokens, so we need 3000+ total
                api_params["max_completion_tokens"] = 3000  # Increased to handle reasoning + output
                # GPT-5 Nano/Mini only supports default temperature (1), so we omit it
                logger.info(f"Using GPT-5 model: {self.openai_model} (max_completion_tokens=3000)")
            else:
                api_params["max_tokens"] = 1000
                api_params["temperature"] = 0.1
                logger.info(f"Using GPT-4o model: {self.openai_model}")
            
            response = self.client.chat.completions.create(**api_params)
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            output = choice.message.content
            
            # Debug logging for empty responses
            if not output or not output.strip():
                logger.warning(f"Empty response from {self.openai_model}. Finish reason: {finish_reason}")
                if hasattr(response, 'usage'):
                    logger.warning(f"Token usage: {response.usage}")
                # For length errors (token limit), return empty result instead of failing
                if finish_reason == "length":
                    logger.warning("Response truncated due to token limit - returning empty result")
                    return self._empty_result()
                # For other errors, still raise
                raise ValueError(f"Empty response content from OpenAI model. Finish reason: {finish_reason}")
            
            # Parse response
            parsed = self._parse_gpt_response(output)
            parsed['model'] = self.model_name
            parsed['timestamp'] = datetime.now().isoformat()
            
            # Track cost per API call (after parsing so we can show num_objects)
            if hasattr(response, 'usage') and self.cost_tracker:
                usage = response.usage
                input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                
                # Image tokens are already included in prompt_tokens for vision models
                image_tokens = 0  # Already counted in prompt_tokens
                
                # Calculate cost for this call
                pricing = self.gpt_pricing.get(self.openai_model, self.gpt_pricing.get("gpt-4o", {"input": 2.50, "output": 10.00}))
                input_cost = (input_tokens / 1_000_000) * pricing["input"]
                output_cost = (output_tokens / 1_000_000) * pricing["output"]
                call_cost = input_cost + output_cost
                
                # Store last call info for display
                self.last_call_cost = call_cost
                self.last_call_tokens = (input_tokens, output_tokens)
                
                # Track usage
                self.cost_tracker.add_gpt_usage(self.openai_model, input_tokens, output_tokens, image_tokens)
                
                # Log cost per call with object count
                logger.info(f"Extracted {parsed['num_objects']} objects | Cost: ${call_cost:.6f} | Tokens: {input_tokens}+{output_tokens}={input_tokens+output_tokens}")
            else:
                # Reset if no usage info
                self.last_call_cost = 0.0
                self.last_call_tokens = (0, 0)
            
            return parsed
            
        except Exception as e:
            error_str = str(e)
            # Check for authentication errors
            if "401" in error_str or "invalid_api_key" in error_str or "Unauthorized" in error_str:
                logger.error(f"Authentication failed: Invalid API key. Please check your OPENAI_API_KEY.")
                raise RuntimeError("Invalid OpenAI API key. Please update your .env file with a valid key.")
            logger.error(f"GPT extraction error: {e}")
            return self._empty_result()
    
    def classify_frame_artifacts(
        self, 
        image_path: str
    ) -> Dict[str, Any]:
        """
        Classify frame artifacts using GPT-4o.
        
        Args:
            image_path: Path to frame image
            
        Returns:
            Dictionary with artifact classification
        """
        if not self.enabled:
            return self._empty_artifact_result()
        
        try:
            prompt = """Examine this image for quality issues and artifacts.

Classify into ONE of these categories:
1. Visual hallucinations - artificial or impossible elements
2. Image artifacts - compression, blur, pixelation, noise
3. AI inconsistencies - unrealistic lighting, impossible shadows
none - no significant artifacts

Respond ONLY with JSON (no markdown):
{
    "has_artifacts": true/false,
    "artifact_type": "1" or "2" or "3" or "none",
    "confidence": 0.0-1.0,
    "description": "brief description",
    "severity": "low" or "medium" or "high"
}"""
            
            base64_image = self._encode_image(image_path)
            
            # GPT-5 Nano/Mini requires max_completion_tokens instead of max_tokens
            # GPT-5 Nano/Mini only supports default temperature (1), not custom values
            is_gpt5_nano = "gpt-5-nano" in self.openai_model.lower()
            is_gpt5_mini = "gpt-5-mini" in self.openai_model.lower()
            is_gpt5 = is_gpt5_nano or is_gpt5_mini
            api_params = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            }
            if is_gpt5:
                # GPT-5 Nano/Mini needs more tokens - it uses tokens for reasoning, leaving room for output
                api_params["max_completion_tokens"] = 1000  # Reduced for faster response
                # GPT-5 Nano/Mini only supports default temperature (1), so we omit it
                logger.info(f"Using GPT-5 model: {self.openai_model} (max_completion_tokens=1000)")
            else:
                api_params["max_tokens"] = 300
                api_params["temperature"] = 0.1
                logger.info(f"Using GPT-4o model: {self.openai_model}")
            
            response = self.client.chat.completions.create(**api_params)
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            output = choice.message.content
            
            # Debug logging for empty responses
            if not output or not output.strip():
                logger.warning(f"Empty response from {self.openai_model}. Finish reason: {finish_reason}")
                if hasattr(response, 'usage'):
                    logger.warning(f"Token usage: {response.usage}")
                # For length errors (token limit), return empty result instead of failing
                if finish_reason == "length":
                    logger.warning("Response truncated due to token limit - returning empty artifact result")
                    return self._empty_artifact_result()
                # For other errors, still raise
                raise ValueError(f"Empty response content from OpenAI model. Finish reason: {finish_reason}")
            
            # Parse JSON
            parsed = self._parse_artifact_response(output)
            parsed['model'] = self.model_name
            parsed['timestamp'] = datetime.now().isoformat()
            
            # Track cost for artifact classification (if enabled)
            if hasattr(response, 'usage') and self.cost_tracker:
                usage = response.usage
                input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                pricing = self.gpt_pricing.get(self.openai_model, self.gpt_pricing.get("gpt-4o", {"input": 2.50, "output": 10.00}))
                input_cost = (input_tokens / 1_000_000) * pricing["input"]
                output_cost = (output_tokens / 1_000_000) * pricing["output"]
                call_cost = input_cost + output_cost
                self.cost_tracker.add_gpt_usage(self.openai_model, input_tokens, output_tokens, 0)
                # Update last call cost (artifact calls are separate)
                self.last_call_cost = call_cost
                self.last_call_tokens = (input_tokens, output_tokens)
                logger.info(f"Artifact type={parsed.get('artifact_type', 'none')} | Cost: ${call_cost:.6f}")
            
            return parsed
            
        except Exception as e:
            error_str = str(e)
            # Check for authentication errors
            if "401" in error_str or "invalid_api_key" in error_str or "Unauthorized" in error_str:
                logger.error(f"Authentication failed: Invalid API key. Please check your OPENAI_API_KEY.")
                raise RuntimeError("Invalid OpenAI API key. Please update your .env file with a valid key.")
            logger.error(f"GPT artifact error: {e}")
            return self._empty_artifact_result()
    
    def _build_extraction_prompt(
        self, 
        focus_areas: Optional[List[str]], 
        include_accessibility: bool
    ) -> str:
        """Build ultra-short prompt to minimize token usage."""
        # Ultra-minimal prompt to reduce reasoning tokens
        prompt = "List objects as JSON: {\"objects\":[\"name1\",\"name2\"]}. Max 50. Singular nouns only."
        
        if include_accessibility:
            prompt += " Priority: pillar/barrier/obstacle/step/curb. Signs: exit/restroom/elevator."
        
        return prompt
        
        if focus_areas:
            prompt += f"FOCUS AREAS: {', '.join(focus_areas)}\n"
        
        return prompt
    
    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response (works for GPT-4o and GPT-5 Nano)."""
        try:
            # Clean markdown code blocks
            response = response.strip()
            
            # Remove markdown code block markers
            if response.startswith("```json"):
                response = response[7:].strip()
                if response.endswith("```"):
                    response = response[:-3].strip()
            elif response.startswith("```"):
                # Generic code block
                lines = response.split("\n")
                if len(lines) > 1 and lines[0].startswith("```"):
                    response = "\n".join(lines[1:])
                if response.endswith("```"):
                    response = response[:-3].strip()
            
            # Remove leading "json" keyword if present
            if response.lower().startswith("json"):
                response = response[4:].strip()
                if response.startswith(":"):
                    response = response[1:].strip()
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            data = json.loads(response)
            if isinstance(data, list):
                objects = self._normalize_objects(data)
            else:
                objects = self._normalize_objects(data.get('objects', []))
            
            # Enforce 50 object limit
            if len(objects) > 50:
                logger.warning(f"Truncating {len(objects)} objects to 50")
                objects = objects[:50]
            
            # Filter out invalid object names
            objects = self._filter_clean_objects(objects)
            
            # SIMPLIFIED OUTPUT - Only objects list, no extra fields
            return {
                'objects': objects,
                'num_objects': len(objects),
                'error': False
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:500]}...")  # Log first 500 chars for debugging
            return self._empty_result()
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            logger.debug(f"Response was: {response[:500]}...")  # Log first 500 chars for debugging
            return self._empty_result()
    
    def _filter_clean_objects(self, objects: List[str]) -> List[str]:
        """Filter out invalid object names (verbs, long names, etc.)."""
        clean_objects = []
        
        # Common verbs and bad patterns to exclude
        bad_patterns = [
            # Verbs and actions
            r'\b(walking|running|sitting|standing|holding|wearing|looking|moving)\b',
            # Vague terms
            r'\b(thing|stuff|item|object)\b',
            # Descriptions/adjectives at start
            r'^(beautiful|modern|old|new|large|small|big|red|blue|green)\s',
        ]
        
        for obj in objects:
            obj_lower = obj.lower()
            
            # Skip if too long (more than 4 words)
            if len(obj.split()) > 4:
                continue
            
            # Skip if matches bad patterns
            is_bad = False
            for pattern in bad_patterns:
                if re.search(pattern, obj_lower):
                    is_bad = True
                    break
            
            if not is_bad:
                clean_objects.append(obj)
        
        return clean_objects
    
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
            base64_image = self._encode_image(image_path)
            is_gpt5 = "gpt-5" in self.openai_model.lower()
            api_params = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(prompt)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
            }
            if is_gpt5:
                api_params["max_completion_tokens"] = min(max_new_tokens, 500)
            else:
                api_params["max_tokens"] = min(max_new_tokens, 500)
                api_params["temperature"] = 0.0
            response = self.client.chat.completions.create(**api_params)
            text = (response.choices[0].message.content or "").strip()
            return text
        except Exception as e:
            logger.error(f"GPT generate_freeform_text failed: {e}")
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
        """
        if not self.enabled or not questions:
            return {q.get("id", ""): "" for q in questions}
        try:
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
            base64_image = self._encode_image(image_path)
            is_gpt5 = "gpt-5" in self.openai_model.lower()
            api_params = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
            }
            if is_gpt5:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
                api_params["temperature"] = 0.0
            response = self.client.chat.completions.create(**api_params)
            text = (response.choices[0].message.content or "").strip()
            result = {}
            for i, q in enumerate(questions, 1):
                qid = q.get("id", "")
                match = re.search(rf"Q{i}\s*:\s*(.+?)(?=Q{i+1}\s*:|$)", text, re.DOTALL | re.IGNORECASE)
                if match:
                    ans = match.group(1).strip()
                    words = ans.split()
                    if len(words) > max_words_per_answer:
                        ans = " ".join(words[:max_words_per_answer])
                    result[qid] = ans
                else:
                    result[qid] = ""
            return result
        except Exception as e:
            logger.error("GPT generate_freeform_text_batch failed: %s", e)
            return {q.get("id", ""): "" for q in questions}

    def _parse_artifact_response(self, response: str) -> Dict[str, Any]:
        """Parse artifact response."""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = "\n".join(response.split("\n")[1:-1])
            if response.startswith("json"):
                response = response[4:].strip()
            
            data = json.loads(response)
            
            return {
                'has_artifacts': data.get('has_artifacts', False),
                'artifact_type': data.get('artifact_type', 'none'),
                'confidence': float(data.get('confidence', 0.0)),
                'description': data.get('description', ''),
                'severity': data.get('severity', 'low'),
                'affected_regions': [],
                'error': False
            }
        except:
            return self._empty_artifact_result()


def test_gpt4o_extractor():
    """Test GPT-4o extractor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vlm_gpt4o.py <image_path>")
        return
    
    print("\n" + "="*60)
    print("Testing GPT-4o Extractor")
    print("="*60)
    
    extractor = GPT4oExtractor()
    
    if not extractor.enabled:
        print("❌ GPT-4o not available (check API key)")
        return
    
    result = extractor.extract_objects_from_keyframe(sys.argv[1])
    
    print(f"\n✅ Objects: {result['num_objects']}")
    for obj in result['objects'][:10]:
        print(f"  - {obj}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_gpt4o_extractor()