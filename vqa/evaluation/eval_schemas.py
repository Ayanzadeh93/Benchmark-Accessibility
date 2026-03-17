"""Pydantic schemas for VQA model evaluation results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VQAPrediction(BaseModel):
    """Single VQA prediction from a model."""
    
    id: str = Field(description="Sample ID")
    question_id: str = Field(description="Question type")
    image: str = Field(description="Image filename")
    
    question: str
    options: Dict[str, str]
    
    predicted_answer: str = Field(description="Model's predicted label (A/B/C/D)")
    predicted_answer_text: str = Field(description="Model's predicted answer text")
    
    ground_truth_answer: str = Field(description="Reference answer label")
    ground_truth_answer_text: str = Field(description="Reference answer text")
    
    is_correct: bool = Field(description="Whether prediction matches ground truth")
    
    model_response_raw: Optional[str] = Field(default=None, description="Raw model response (for debugging)")
    inference_time_s: Optional[float] = Field(default=None, description="Inference time in seconds")


class QuestionTypeMetrics(BaseModel):
    """Metrics for a specific question type."""
    
    question_id: str
    question_text: Optional[str] = None
    
    accuracy: float = Field(description="Exact match accuracy (0-1)")
    correct: int = Field(description="Number of correct predictions")
    total: int = Field(description="Total number of samples")
    
    rouge1_f1: Optional[float] = None
    rouge2_f1: Optional[float] = None
    rouge3_f1: Optional[float] = None
    rouge4_f1: Optional[float] = None
    rougeL_f1: Optional[float] = None
    rougeLsum_f1: Optional[float] = None
    bleu: Optional[float] = None
    bertscore_f1: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    clip_text_score: Optional[float] = None


class VQAEvaluationResult(BaseModel):
    """Complete VQA evaluation results for a model."""
    
    model_name: str = Field(description="Model identifier")
    dataset_path: str = Field(description="Path to VQA dataset used")
    timestamp: str = Field(description="Evaluation timestamp")
    
    # Overall metrics
    overall_accuracy: float
    total_samples: int
    total_correct: int
    total_failed: int = Field(default=0, description="Samples that failed inference")
    
    # Text-based metrics (optional)
    overall_rouge1_f1: Optional[float] = None
    overall_rouge2_f1: Optional[float] = None
    overall_rouge3_f1: Optional[float] = None
    overall_rouge4_f1: Optional[float] = None
    overall_rougeL_f1: Optional[float] = None
    overall_rougeLsum_f1: Optional[float] = None
    overall_bleu: Optional[float] = None
    overall_bertscore_f1: Optional[float] = None
    overall_bertscore_precision: Optional[float] = None
    overall_bertscore_recall: Optional[float] = None
    overall_clip_text_score: Optional[float] = None
    overall_clip_image_pred_score: Optional[float] = None
    overall_clip_image_ref_score: Optional[float] = None
    
    # Per-question-type breakdown
    per_question_metrics: List[QuestionTypeMetrics] = Field(default_factory=list)
    
    # Individual predictions (optional, can be large)
    predictions: List[VQAPrediction] = Field(default_factory=list)
    
    # Summary stats
    avg_inference_time_s: Optional[float] = None
    total_inference_time_s: Optional[float] = None


class VQAEvaluationSummary(BaseModel):
    """Compact summary for multiple model evaluations (for comparison)."""
    
    model_name: str
    overall_accuracy: float
    total_samples: int
    
    per_question_accuracy: Dict[str, float] = Field(default_factory=dict)
    
    overall_rouge1_f1: Optional[float] = None
    overall_rouge2_f1: Optional[float] = None
    overall_rouge3_f1: Optional[float] = None
    overall_rouge4_f1: Optional[float] = None
    overall_rougeL_f1: Optional[float] = None
    overall_rougeLsum_f1: Optional[float] = None
    overall_bleu: Optional[float] = None
    overall_bertscore_f1: Optional[float] = None
    overall_clip_text_score: Optional[float] = None
    
    avg_inference_time_s: Optional[float] = None
