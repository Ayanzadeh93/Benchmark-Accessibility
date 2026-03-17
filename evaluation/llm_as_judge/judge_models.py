"""LLM judge model implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from .schemas import EvaluationResult, CriterionScore
from .prompts import EVALUATION_SYSTEM_PROMPT, format_evaluation_prompt

logger = logging.getLogger(__name__)


class BaseLLMJudge(ABC):
    """Base class for LLM judge models."""
    
    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 1500):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def evaluate(self, image_name: str, annotation_text: str) -> EvaluationResult:
        """Evaluate an annotation and return structured results.
        
        Args:
            image_name: Name of the image being evaluated
            annotation_text: The annotation content to evaluate
            
        Returns:
            EvaluationResult with scores and feedback
        """
        pass
    
    def _parse_json_response(self, response_text: str, image_name: str) -> EvaluationResult:
        """Parse JSON response from LLM and create EvaluationResult.
        
        Args:
            response_text: Raw text response from LLM
            image_name: Name of the evaluated image
            
        Returns:
            Parsed EvaluationResult
            
        Raises:
            ValueError: If response cannot be parsed
        """
        # Try to extract JSON from markdown code blocks
        text = response_text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Normalize scores to half-points only (e.g. 7.0, 7.5, 8.0)
        def _round_to_half(v: float) -> float:
            return round(float(v) * 2) / 2.0
        for key in ("clarity", "completeness", "robustness", "user_friendliness", "accuracy"):
            if key in data and isinstance(data[key], dict) and "score" in data[key]:
                data[key]["score"] = max(1.0, min(10.0, _round_to_half(data[key]["score"])))
        if "overall_score" in data:
            data["overall_score"] = max(1.0, min(10.0, _round_to_half(data["overall_score"])))
        
        # Parse criterion scores
        try:
            result = EvaluationResult(
                image_name=image_name,
                model_name=self.model_name,
                clarity=CriterionScore(**data["clarity"]),
                completeness=CriterionScore(**data["completeness"]),
                robustness=CriterionScore(**data["robustness"]),
                user_friendliness=CriterionScore(**data["user_friendliness"]),
                accuracy=CriterionScore(**data["accuracy"]),
                overall_score=float(data["overall_score"]),
                feedback=str(data["feedback"]),
            )
            return result
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            logger.debug(f"Data: {data}")
            raise ValueError(f"Invalid evaluation response structure: {e}")


class OpenAIJudge(BaseLLMJudge):
    """OpenAI-based LLM judge (GPT-4, GPT-4o, GPT-5 models)."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.2, max_tokens: int = 1500):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.client = None
    
    def _init_client(self):
        """Initialize OpenAI client lazily."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
    
    def evaluate(self, image_name: str, annotation_text: str) -> EvaluationResult:
        """Evaluate annotation using OpenAI model."""
        self._init_client()
        
        prompt = format_evaluation_prompt(image_name, annotation_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            result = self._parse_json_response(response_text, image_name)
            result.annotation_text = annotation_text
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error for {image_name}: {e}")
            raise


class AnthropicJudge(BaseLLMJudge):
    """Anthropic Claude-based LLM judge."""
    
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-20250514", temperature: float = 0.2, max_tokens: int = 1500):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.client = None
    
    def _init_client(self):
        """Initialize Anthropic client lazily."""
        if self.client is None:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def evaluate(self, image_name: str, annotation_text: str) -> EvaluationResult:
        """Evaluate annotation using Claude model."""
        self._init_client()
        
        prompt = format_evaluation_prompt(image_name, annotation_text)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=EVALUATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            result = self._parse_json_response(response_text, image_name)
            result.annotation_text = annotation_text
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API error for {image_name}: {e}")
            raise


class OpenRouterJudge(BaseLLMJudge):
    """OpenRouter-based LLM judge (supports many models)."""
    
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o", temperature: float = 0.2, max_tokens: int = 1500):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.client = None
    
    def _init_client(self):
        """Initialize OpenRouter client (uses OpenAI SDK)."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
    
    def evaluate(self, image_name: str, annotation_text: str) -> EvaluationResult:
        """Evaluate annotation using OpenRouter model."""
        self._init_client()
        
        prompt = format_evaluation_prompt(image_name, annotation_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            response_text = response.choices[0].message.content
            result = self._parse_json_response(response_text, image_name)
            result.annotation_text = annotation_text
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter API error for {image_name}: {e}")
            raise


def create_judge(
    model_name: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> BaseLLMJudge:
    """Factory function to create appropriate judge model.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4', 'openrouter:qwen/qwen3-vl')
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        openrouter_api_key: OpenRouter API key
        temperature: Sampling temperature
        max_tokens: Max tokens for response
        
    Returns:
        Configured judge model instance
        
    Raises:
        ValueError: If model_name is invalid or required API key is missing
    """
    model_lower = model_name.lower()
    
    # OpenAI models
    if model_lower.startswith(("gpt-", "o1-")):
        if not openai_api_key:
            raise ValueError(f"OpenAI API key required for model: {model_name}")
        return OpenAIJudge(
            api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    # Anthropic models
    elif model_lower.startswith("claude"):
        if not anthropic_api_key:
            raise ValueError(f"Anthropic API key required for model: {model_name}")
        return AnthropicJudge(
            api_key=anthropic_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    # OpenRouter models (prefix with 'openrouter:' or just use the full model path)
    elif "openrouter:" in model_lower or "/" in model_name:
        if not openrouter_api_key:
            raise ValueError(f"OpenRouter API key required for model: {model_name}")
        # Remove 'openrouter:' prefix if present
        actual_model = model_name.replace("openrouter:", "").replace("OpenRouter:", "")
        return OpenRouterJudge(
            api_key=openrouter_api_key,
            model_name=actual_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Use 'gpt-4o', 'claude-sonnet-4-20250514', or 'openrouter:<model-path>'"
        )
