"""VQA model evaluation pipeline: run models on VQA datasets and compute metrics."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from .eval_schemas import (
    VQAPrediction,
    QuestionTypeMetrics,
    VQAEvaluationResult,
    VQAEvaluationSummary,
)
from .metrics import (
    exact_match_accuracy,
    compute_rouge_scores,
    compute_bleu_score,
    compute_bertscore,
    compute_clip_score_text,
    compute_clip_score_image_text,
    compute_per_question_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass
class VQAEvalConfig:
    """Configuration for VQA model evaluation."""
    
    vqa_dataset_path: str  # Path to VQA JSON (per-question or per-image or per_image_all.json)
    images_dir: str  # Directory with images
    output_dir: str  # Where to save results (e.g. .../vqa/T1)
    
    model_name: str  # e.g. "openrouter_llama4_maverick"
    model_type: str  # e.g. "openrouter"
    api_key: Optional[str] = None
    
    max_samples: Optional[int] = None
    batch_mode: bool = True  # If True, ask all questions per image in one call (6x cheaper)
    save_predictions: bool = True  # Save individual predictions to JSON
    verbose: bool = False


class VQAModelEvaluator:
    """Evaluate a model on a VQA dataset."""
    
    def __init__(self, config: VQAEvalConfig):
        self.config = config
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the model (OpenRouter or others)."""
        try:
            from vlm_factory import VLMFactory
            self.model = VLMFactory.create_extractor(
                self.config.model_type,
                api_key=self.config.api_key,
            )
            if not hasattr(self.model, "answer_multiple_choice"):
                raise ValueError(f"Model {self.config.model_type} does not support answer_multiple_choice")
            
            # Check batch support
            self.supports_batch = hasattr(self.model, "answer_multiple_choice_batch")
            if self.config.batch_mode and not self.supports_batch:
                logger.warning(f"Model {self.config.model_type} does not support batch mode, falling back to sequential")
                self.config.batch_mode = False
        except Exception as e:
            logger.error(f"Failed to initialize model {self.config.model_type}: {e}")
            raise
    
    @staticmethod
    def _load_vqa_dataset(path: Path) -> List[Dict[str, Any]]:
        """
        Load VQA dataset from JSON.
        
        Supports:
        - per-question JSON ({"question_id": "...", "samples": [...]})
        - per_image_all.json ([{"image": "...", "questions": [...]}])
        - flat list of samples
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        
        # Check format
        if isinstance(data, list):
            # Could be per_image_all.json or flat list
            if data and "questions" in data[0]:
                # per_image_all.json format
                for entry in data:
                    for q in entry.get("questions", []):
                        samples.append(q)
            else:
                # Flat list of samples
                samples = data
        
        elif isinstance(data, dict):
            # Could be per-question JSON
            if "samples" in data:
                samples = data["samples"]
            elif "questions" in data:
                samples = data["questions"]
            else:
                # Single sample?
                samples = [data]
        
        return samples
    
    @staticmethod
    def _load_vqa_dataset_per_image(path: Path) -> List[Dict[str, Any]]:
        """
        Load VQA dataset grouped by image (for batch mode).
        
        Returns:
            List of dicts with {"image": "...", "questions": [...]}
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if already in per-image format
        if isinstance(data, list) and data and "questions" in data[0]:
            return data
        
        # Otherwise, convert flat samples to per-image format
        samples = []
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        elif isinstance(data, dict) and "questions" in data:
            samples = data["questions"]
        
        # Group by image
        from collections import defaultdict
        by_image = defaultdict(list)
        for sample in samples:
            img = sample.get("image", "unknown")
            by_image[img].append(sample)
        
        # Convert to list of per-image dicts
        per_image = []
        for img, questions in by_image.items():
            per_image.append({"image": img, "questions": questions})
        
        return per_image
    
    def _predict_sample(
        self,
        sample: Dict[str, Any],
        image_path: str,
    ) -> Optional[VQAPrediction]:
        """
        Run model on a single VQA sample.
        
        Args:
            sample: Dict with id, question, options, answer, etc.
            image_path: Full path to the image
        
        Returns:
            VQAPrediction or None if inference failed
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        question = sample.get("question", "")
        options = sample.get("options", {})
        gt_answer = str(sample.get("answer", "")).strip().upper()
        gt_answer_text = str(sample.get("answer_text") or options.get(gt_answer) or "").strip()
        if not gt_answer_text and gt_answer:
            gt_answer_text = f"Option {gt_answer}"
        
        if not question or not options or len(options) != 4:
            logger.warning(f"Invalid sample format: {sample.get('id', 'unknown')}")
            return None
        
        start_time = time.time()
        try:
            predicted_label = self.model.answer_multiple_choice(
                image_path,
                question,
                options,
            )
            inference_time = time.time() - start_time
            
            if not predicted_label or predicted_label not in options:
                logger.warning(f"Invalid prediction for {sample.get('id', 'unknown')}: {predicted_label}")
                return None
            
            predicted_label = predicted_label.upper()
            predicted_text = str(options.get(predicted_label) or "").strip()
            if not predicted_text:
                predicted_text = f"Option {predicted_label}"
            is_correct = (predicted_label == gt_answer)
            
            return VQAPrediction(
                id=sample.get("id", "unknown"),
                question_id=sample.get("question_id", "unknown"),
                image=sample.get("image", ""),
                question=question,
                options=options,
                predicted_answer=predicted_label,
                predicted_answer_text=predicted_text,
                ground_truth_answer=gt_answer,
                ground_truth_answer_text=gt_answer_text,
                is_correct=is_correct,
                model_response_raw=predicted_label,
                inference_time_s=inference_time,
            )
        
        except Exception as e:
            logger.error(f"Inference failed for {sample.get('id', 'unknown')}: {e}")
            return None
    
    def _predict_image_batch(
        self,
        image_path: str,
        questions: List[Dict[str, Any]],
    ) -> List[VQAPrediction]:
        """
        Run model on all questions for one image in batch mode (1 API call).
        
        Args:
            image_path: Full path to the image
            questions: List of question dicts with id, question, options, answer, etc.
        
        Returns:
            List of VQAPrediction (one per question)
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return []
        
        if not questions:
            return []
        
        # Prepare batch input
        batch_questions = []
        for q in questions:
            if not q.get("question") or not q.get("options") or len(q.get("options", {})) != 4:
                continue
            batch_questions.append({
                "id": q.get("question_id", q.get("id", "unknown")),
                "question": q["question"],
                "options": q["options"],
            })
        
        if not batch_questions:
            return []
        
        # Call model once with all questions
        start_time = time.time()
        try:
            answers = self.model.answer_multiple_choice_batch(image_path, batch_questions)
            inference_time = time.time() - start_time
            
            # Build predictions
            predictions = []
            for q in questions:
                qid = q.get("question_id", q.get("id", "unknown"))
                predicted_label = answers.get(qid)
                
                if not predicted_label or predicted_label not in q.get("options", {}):
                    logger.debug(f"Invalid batch prediction for {qid}: {predicted_label}")
                    continue
                
                predicted_label = predicted_label.upper()
                opts = q.get("options", {})
                predicted_text = str(opts.get(predicted_label) or "").strip()
                if not predicted_text:
                    predicted_text = f"Option {predicted_label}"
                gt_answer = str(q.get("answer", "")).strip().upper()
                gt_answer_text = str(q.get("answer_text") or opts.get(gt_answer) or "").strip()
                if not gt_answer_text and gt_answer:
                    gt_answer_text = f"Option {gt_answer}"
                is_correct = (predicted_label == gt_answer)
                
                predictions.append(VQAPrediction(
                    id=q.get("id", qid),
                    question_id=qid,
                    image=q.get("image", ""),
                    question=q["question"],
                    options=q["options"],
                    predicted_answer=predicted_label,
                    predicted_answer_text=predicted_text,
                    ground_truth_answer=gt_answer,
                    ground_truth_answer_text=gt_answer_text,
                    is_correct=is_correct,
                    model_response_raw=predicted_label,
                    inference_time_s=inference_time / len(questions) if questions else None,
                ))
            
            return predictions
        
        except Exception as e:
            logger.error(f"Batch inference failed for {image_path}: {e}")
            return []
    
    def evaluate(self) -> VQAEvaluationResult:
        """Run evaluation on the VQA dataset."""
        # Load dataset
        dataset_path = Path(self.config.vqa_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"VQA dataset not found: {dataset_path}")
        
        logger.info(f"Loading VQA dataset from {dataset_path}")
        
        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(self.config.images_dir)
        
        # Run predictions
        predictions: List[VQAPrediction] = []
        failed = 0
        
        if self.config.batch_mode:
            # Batch mode: load per-image, 1 API call per image
            logger.info("Using batch mode (1 API call per image)")
            per_image_data = self._load_vqa_dataset_per_image(dataset_path)
            
            if self.config.max_samples:
                # Limit images, not individual samples
                per_image_data = per_image_data[:self.config.max_samples]
            
            total_samples = sum(len(entry["questions"]) for entry in per_image_data)
            logger.info(f"Evaluating {len(per_image_data)} images ({total_samples} questions) with model {self.config.model_name}")
            
            for entry in tqdm(per_image_data, desc=f"Evaluating {self.config.model_name} (batch)"):
                image_name = entry.get("image", "")
                image_path = images_dir / image_name
                questions = entry.get("questions", [])
                
                batch_preds = self._predict_image_batch(str(image_path), questions)
                predictions.extend(batch_preds)
                
                if len(batch_preds) < len(questions):
                    failed += (len(questions) - len(batch_preds))
        else:
            # Sequential mode: 1 API call per question
            logger.info("Using sequential mode (1 API call per question)")
            samples = self._load_vqa_dataset(dataset_path)
            
            if self.config.max_samples:
                samples = samples[:self.config.max_samples]
            
            logger.info(f"Evaluating {len(samples)} samples with model {self.config.model_name}")
            
            for sample in tqdm(samples, desc=f"Evaluating {self.config.model_name}"):
                # Build image path
                image_name = sample.get("image", "")
                image_path = images_dir / image_name
                if sample.get("image_path"):
                    image_path = Path(sample["image_path"])
                
                pred = self._predict_sample(sample, str(image_path))
                if pred:
                    predictions.append(pred)
                else:
                    failed += 1
        
        logger.info(f"Completed: {len(predictions)} predictions, {failed} failed")
        
        # Compute metrics
        pred_labels = [p.predicted_answer for p in predictions]
        gt_labels = [p.ground_truth_answer for p in predictions]
        pred_texts = [p.predicted_answer_text for p in predictions]
        gt_texts = [p.ground_truth_answer_text for p in predictions]
        
        overall_accuracy = exact_match_accuracy(pred_labels, gt_labels)
        
        rouge_scores = compute_rouge_scores(pred_texts, gt_texts)
        bleu_score = compute_bleu_score(pred_texts, gt_texts)
        
        # BERT Score (skip if no valid text; use 0.0 on error so CSV is complete)
        logger.info("Computing BERT Score...")
        bertscore = compute_bertscore(pred_texts, gt_texts)
        if "error" in bertscore:
            logger.warning("BERT Score: %s", bertscore.get("error", "unknown"))
            bertscore = {}
        
        # CLIP Score excluded (avoids GPU OOM; re-enable by uncommenting and setting clip_* below)
        clip_text_score = None
        clip_image_scores = {}

        # Per-question metrics
        pred_dicts = [{"question_id": p.question_id, "answer": p.predicted_answer} for p in predictions]
        gt_dicts = [{"question_id": p.question_id, "answer": p.ground_truth_answer} for p in predictions]
        per_q_acc = compute_per_question_accuracy(pred_dicts, gt_dicts)
        
        per_question_metrics = []
        for qid, metrics in per_q_acc.items():
            per_question_metrics.append(QuestionTypeMetrics(
                question_id=qid,
                accuracy=metrics["accuracy"],
                correct=metrics["correct"],
                total=metrics["total"],
            ))
        
        # Inference time stats
        inference_times = [p.inference_time_s for p in predictions if p.inference_time_s is not None]
        avg_time = sum(inference_times) / len(inference_times) if inference_times else None
        total_time = sum(inference_times) if inference_times else None
        
        # Build result
        result = VQAEvaluationResult(
            model_name=self.config.model_name,
            dataset_path=str(dataset_path),
            timestamp=datetime.now().isoformat(),
            overall_accuracy=overall_accuracy,
            total_samples=len(predictions),
            total_correct=sum(1 for p in predictions if p.is_correct),
            total_failed=failed,
            overall_rouge1_f1=rouge_scores.get("rouge1_f1"),
            overall_rouge2_f1=rouge_scores.get("rouge2_f1"),
            overall_rouge3_f1=rouge_scores.get("rouge3_f1"),
            overall_rouge4_f1=rouge_scores.get("rouge4_f1"),
            overall_rougeL_f1=rouge_scores.get("rougeL_f1"),
            overall_rougeLsum_f1=rouge_scores.get("rougeLsum_f1"),
            overall_bleu=bleu_score,
            overall_bertscore_f1=bertscore.get("bertscore_f1"),
            overall_bertscore_precision=bertscore.get("bertscore_precision"),
            overall_bertscore_recall=bertscore.get("bertscore_recall"),
            overall_clip_text_score=clip_text_score if (clip_text_score is not None and clip_text_score > 0) else None,
            overall_clip_image_pred_score=clip_image_scores.get("clip_image_pred_score"),
            overall_clip_image_ref_score=clip_image_scores.get("clip_image_ref_score"),
            per_question_metrics=per_question_metrics,
            predictions=predictions if self.config.save_predictions else [],
            avg_inference_time_s=avg_time,
            total_inference_time_s=total_time,
        )
        
        # Save results
        self._save_results(result, output_dir)
        
        return result
    
    def _save_results(self, result: VQAEvaluationResult, output_dir: Path):
        """Save evaluation results to JSON, summary, and per-image files."""
        # Full results with predictions
        full_path = output_dir / f"{self.config.model_name}_evaluation.json"
        with full_path.open("w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Saved full results to {full_path}")
        
        # Per-image results: one JSON per image in output_dir/per_image/
        if result.predictions:
            per_image_dir = output_dir / "per_image"
            per_image_dir.mkdir(parents=True, exist_ok=True)
            by_image: Dict[str, List[VQAPrediction]] = {}
            for p in result.predictions:
                by_image.setdefault(p.image, []).append(p)
            for image_name, preds in by_image.items():
                stem = Path(image_name).stem
                out_path = per_image_dir / f"{stem}.json"
                entry = {
                    "image": image_name,
                    "model": self.config.model_name,
                    "predictions": [p.model_dump() for p in preds],
                    "correct": sum(1 for p in preds if p.is_correct),
                    "total": len(preds),
                }
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved per-image results to {per_image_dir} ({len(by_image)} images)")
        
        # Summary (without predictions)
        summary = VQAEvaluationSummary(
            model_name=result.model_name,
            overall_accuracy=result.overall_accuracy,
            total_samples=result.total_samples,
            per_question_accuracy={
                m.question_id: m.accuracy for m in result.per_question_metrics
            },
            overall_rouge1_f1=result.overall_rouge1_f1,
            overall_rouge2_f1=result.overall_rouge2_f1,
            overall_rouge3_f1=result.overall_rouge3_f1,
            overall_rouge4_f1=result.overall_rouge4_f1,
            overall_rougeL_f1=result.overall_rougeL_f1,
            overall_rougeLsum_f1=result.overall_rougeLsum_f1,
            overall_bleu=result.overall_bleu,
            overall_bertscore_f1=result.overall_bertscore_f1,
            overall_clip_text_score=result.overall_clip_text_score,
            avg_inference_time_s=result.avg_inference_time_s,
        )
        summary_path = output_dir / f"{self.config.model_name}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        logger.info(f"Saved summary to {summary_path}")
        
        # CSV for easy viewing
        self._save_csv_summary(result, output_dir)
    
    def _save_csv_summary(self, result: VQAEvaluationResult, output_dir: Path):
        """Save summary as CSV."""
        csv_path = output_dir / f"{self.config.model_name}_summary.csv"
        
        lines = []
        lines.append("metric,value")
        lines.append(f"model,{result.model_name}")
        lines.append(f"overall_accuracy,{result.overall_accuracy:.4f}")
        lines.append(f"total_samples,{result.total_samples}")
        lines.append(f"total_correct,{result.total_correct}")
        lines.append(f"total_failed,{result.total_failed}")
        
        # Always write all text metrics (use 0.0 when None so CSV is complete)
        lines.append(f"rouge1_f1,{result.overall_rouge1_f1:.4f}" if result.overall_rouge1_f1 is not None else "rouge1_f1,0.0")
        lines.append(f"rouge2_f1,{result.overall_rouge2_f1:.4f}" if result.overall_rouge2_f1 is not None else "rouge2_f1,0.0")
        lines.append(f"rouge3_f1,{result.overall_rouge3_f1:.4f}" if result.overall_rouge3_f1 is not None else "rouge3_f1,0.0")
        lines.append(f"rouge4_f1,{result.overall_rouge4_f1:.4f}" if result.overall_rouge4_f1 is not None else "rouge4_f1,0.0")
        lines.append(f"rougeL_f1,{result.overall_rougeL_f1:.4f}" if result.overall_rougeL_f1 is not None else "rougeL_f1,0.0")
        lines.append(f"rougeLsum_f1,{result.overall_rougeLsum_f1:.4f}" if result.overall_rougeLsum_f1 is not None else "rougeLsum_f1,0.0")
        lines.append(f"bleu,{result.overall_bleu:.4f}" if result.overall_bleu is not None else "bleu,0.0")
        
        lines.append(f"bertscore_f1,{result.overall_bertscore_f1:.4f}" if result.overall_bertscore_f1 is not None else "bertscore_f1,0.0")
        lines.append(f"bertscore_precision,{result.overall_bertscore_precision:.4f}" if result.overall_bertscore_precision is not None else "bertscore_precision,0.0")
        lines.append(f"bertscore_recall,{result.overall_bertscore_recall:.4f}" if result.overall_bertscore_recall is not None else "bertscore_recall,0.0")
        
        lines.append(f"clip_text_score,{result.overall_clip_text_score:.4f}" if result.overall_clip_text_score is not None else "clip_text_score,0.0")
        lines.append(f"clip_image_pred_score,{result.overall_clip_image_pred_score:.4f}" if result.overall_clip_image_pred_score is not None else "clip_image_pred_score,0.0")
        lines.append(f"clip_image_ref_score,{result.overall_clip_image_ref_score:.4f}" if result.overall_clip_image_ref_score is not None else "clip_image_ref_score,0.0")
        
        if result.avg_inference_time_s is not None:
            lines.append(f"avg_inference_time_s,{result.avg_inference_time_s:.3f}")
        
        lines.append("")
        lines.append("question_type,accuracy,correct,total")
        for m in result.per_question_metrics:
            lines.append(f"{m.question_id},{m.accuracy:.4f},{m.correct},{m.total}")
        
        csv_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved CSV summary to {csv_path}")


def run_vqa_evaluation(
    vqa_dataset_path: str,
    images_dir: str,
    output_dir: str,
    model_name: str,
    model_type: str = "openrouter",
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_mode: bool = True,
    save_predictions: bool = True,
    verbose: bool = False,
) -> VQAEvaluationResult:
    """
    Convenience function to run VQA evaluation.
    
    Args:
        vqa_dataset_path: Path to VQA JSON dataset
        images_dir: Directory with images
        output_dir: Where to save results (e.g. .../vqa/T1)
        model_name: Model identifier (e.g. "openrouter_llama4_maverick")
        model_type: Model type for VLMFactory (default: "openrouter")
        api_key: API key if needed
        max_samples: Limit samples (for testing)
        batch_mode: If True, ask all questions per image in one call (6x cheaper)
        save_predictions: Save individual predictions
        verbose: Verbose logging
    
    Returns:
        VQAEvaluationResult
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    config = VQAEvalConfig(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_dir=output_dir,
        model_name=model_name,
        model_type=model_type,
        api_key=api_key,
        max_samples=max_samples,
        batch_mode=batch_mode,
        save_predictions=save_predictions,
        verbose=verbose,
    )
    
    evaluator = VQAModelEvaluator(config)
    return evaluator.evaluate()
