"""Schemas for LLM-as-judge evaluation."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, validator


class CriterionScore(BaseModel):
    """Score and reasoning for a single evaluation criterion."""
    
    score: float = Field(..., ge=1.0, le=10.0, description="Score from 1-10")
    reasoning: str = Field(..., min_length=1, description="Brief explanation for the score")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single annotation."""
    
    image_name: str = Field(..., description="Name of the evaluated image")
    model_name: str = Field(..., description="Name of the LLM judge model")
    
    clarity: CriterionScore = Field(..., description="Clarity score and reasoning")
    completeness: CriterionScore = Field(..., description="Completeness score and reasoning")
    robustness: CriterionScore = Field(..., description="Robustness score and reasoning")
    user_friendliness: CriterionScore = Field(..., description="User-friendliness score and reasoning")
    accuracy: CriterionScore = Field(..., description="Accuracy score and reasoning")
    
    overall_score: float = Field(..., ge=1.0, le=10.0, description="Overall average score")
    feedback: str = Field(..., description="Concise overall feedback")
    
    annotation_text: Optional[str] = Field(None, description="Original annotation text")
    
    @validator("overall_score")
    def validate_overall_score(cls, v: float, values: dict) -> float:
        """Ensure overall score is reasonable average of component scores."""
        if all(k in values for k in ["clarity", "completeness", "robustness", "user_friendliness", "accuracy"]):
            scores = [
                values["clarity"].score,
                values["completeness"].score,
                values["robustness"].score,
                values["user_friendliness"].score,
                values["accuracy"].score,
            ]
            expected_avg = sum(scores) / len(scores)
            # Allow small tolerance for rounding
            if abs(v - expected_avg) > 0.5:
                return expected_avg
        return v
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for CSV export."""
        return {
            "image_name": self.image_name,
            "model": self.model_name,
            "clarity": self.clarity.score,
            "clarity_reasoning": self.clarity.reasoning,
            "completeness": self.completeness.score,
            "completeness_reasoning": self.completeness.reasoning,
            "robustness": self.robustness.score,
            "robustness_reasoning": self.robustness.reasoning,
            "user_friendliness": self.user_friendliness.score,
            "user_friendliness_reasoning": self.user_friendliness.reasoning,
            "accuracy": self.accuracy.score,
            "accuracy_reasoning": self.accuracy.reasoning,
            "overall_score": self.overall_score,
            "feedback": self.feedback,
        }


class EvaluationConfig(BaseModel):
    """Configuration for LLM-as-judge evaluation."""
    
    annotations_dir: str = Field(..., description="Directory containing annotation JSON files")
    output_dir: str = Field(..., description="Output directory for evaluation results")
    judge_models: list[str] = Field(..., min_items=1, description="List of LLM judge models to use")
    
    # API configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")
    
    # Evaluation settings
    max_annotations: Optional[int] = Field(None, description="Limit number of annotations to evaluate")
    skip_existing: bool = Field(True, description="Skip already evaluated annotations")
    
    # Output settings
    save_csv: bool = Field(True, description="Save results to CSV")
    save_excel: bool = Field(True, description="Save results to Excel")
    save_json: bool = Field(True, description="Save individual JSON results")
    generate_plots: bool = Field(True, description="Generate visualization plots")
    
    # LLM settings
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Temperature for LLM sampling")
    max_tokens: int = Field(1500, ge=100, description="Max tokens for LLM response")
    
    verbose: bool = Field(False, description="Verbose logging")
    
    class Config:
        arbitrary_types_allowed = True
