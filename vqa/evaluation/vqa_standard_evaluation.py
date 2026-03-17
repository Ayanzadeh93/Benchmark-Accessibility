"""Standard VQA evaluation without multiple choice.

This script evaluates VLM models on standard VQA tasks where:
- Input: image + question
- Output: free-form text answer (not multiple choice)
- Evaluation: text similarity metrics (ROUGE, BLEU, BERTScore, CLIP)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class StandardVQAEvaluator:
    """Evaluate models on standard VQA (free-form answers, no multiple choice)."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        api_key: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize evaluator with a VLM model.
        
        Args:
            model_name: Model identifier (e.g. "florence2", "qwen", "gpt4o")
            model_type: Model type for VLMFactory
            api_key: API key if needed
            device: Device for local models ("cuda", "cpu", "auto")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.device = device
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the VLM model."""
        try:
            from vlm_factory import VLMFactory
            self.model = VLMFactory.create_extractor(
                self.model_type,
                api_key=self.api_key,
                device=self.device,
            )
            logger.info(f"Initialized model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_type}: {e}")
            raise
    
    @staticmethod
    def _load_vqa_dataset(path: Path) -> List[Dict[str, Any]]:
        """
        Load standard VQA dataset from JSON.
        
        Supports:
        - per-question JSON ({"question_id": "...", "samples": [...]})
        - per_image_all.json ([{"image": "...", "questions": [...]}])
        - flat list of samples
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        
        if isinstance(data, list):
            # Could be per_image_all.json or flat list
            if data and "questions" in data[0]:
                # per_image_all.json format
                for entry in data:
                    for q in entry.get("questions", []):
                        # Add image path if not present
                        if "image" not in q and "image" in entry:
                            q["image"] = entry["image"]
                        if "image_path" not in q and "image_path" in entry:
                            q["image_path"] = entry["image_path"]
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
                # Single sample
                samples = [data]
        
        return samples
    
    def _ask_question(
        self,
        image_path: str,
        question: str,
    ) -> Optional[str]:
        """
        Ask the model a question about an image.
        
        Args:
            image_path: Path to the image
            question: Question text
        
        Returns:
            Model's answer as text, or None if inference failed
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        try:
            # Use extract_text method (standard VQA interface)
            if hasattr(self.model, "extract_text"):
                answer = self.model.extract_text(image_path, question)
            elif hasattr(self.model, "answer_question"):
                answer = self.model.answer_question(image_path, question)
            else:
                logger.error(f"Model {self.model_name} does not support standard VQA (no extract_text or answer_question method)")
                return None
            
            return answer.strip() if answer else None
        
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            return None
    
    def evaluate(
        self,
        vqa_dataset_path: str,
        images_dir: str,
        output_dir: str,
        max_samples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on a standard VQA dataset.
        
        Args:
            vqa_dataset_path: Path to VQA JSON dataset
            images_dir: Directory with images
            output_dir: Where to save results
            max_samples: Limit number of samples (for testing)
            save_predictions: Save individual predictions to JSON
        
        Returns:
            Evaluation results dict
        """
        dataset_path = Path(vqa_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"VQA dataset not found: {dataset_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_path = Path(images_dir)
        
        # Load dataset
        logger.info(f"Loading VQA dataset from {dataset_path}")
        samples = self._load_vqa_dataset(dataset_path)
        
        if max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"Evaluating {len(samples)} samples with model {self.model_name}")
        
        # Run inference
        predictions = []
        failed = 0
        
        for sample in tqdm(samples, desc=f"Evaluating {self.model_name}"):
            # Get image path
            image_name = sample.get("image", "")
            image_path = images_path / image_name
            if sample.get("image_path"):
                image_path = Path(sample["image_path"])
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                failed += 1
                continue
            
            question = sample.get("question", "")
            reference_answer = sample.get("answer", "")
            
            if not question or not reference_answer:
                logger.warning(f"Invalid sample: {sample.get('id', 'unknown')}")
                failed += 1
                continue
            
            # Run inference
            start_time = time.time()
            predicted_answer = self._ask_question(str(image_path), question)
            inference_time = time.time() - start_time
            
            if predicted_answer is None:
                failed += 1
                continue
            
            # Store prediction
            prediction = {
                "id": sample.get("id", "unknown"),
                "question_id": sample.get("question_id", "unknown"),
                "image": image_name,
                "question": question,
                "predicted_answer": predicted_answer,
                "reference_answer": reference_answer,
                "inference_time_s": inference_time,
            }
            
            predictions.append(prediction)
        
        logger.info(f"Completed: {len(predictions)} predictions, {failed} failed")
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self._compute_metrics(predictions)
        
        # Build result
        result = {
            "model_name": self.model_name,
            "dataset_path": str(dataset_path),
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(predictions),
            "total_failed": failed,
            "metrics": metrics,
        }
        
        if save_predictions:
            result["predictions"] = predictions
        
        # Save results
        self._save_results(result, output_path)
        
        return result
    
    def _compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of prediction dicts with predicted_answer and reference_answer
        
        Returns:
            Metrics dict
        """
        if not predictions:
            return {}
        
        predicted_texts = [p["predicted_answer"] for p in predictions]
        reference_texts = [p["reference_answer"] for p in predictions]
        
        # Import metrics
        try:
            from .metrics import (
                exact_match_accuracy,
                compute_rouge_scores,
                compute_bleu_score,
                compute_bertscore,
                compute_clip_score_text,
                compute_per_question_accuracy,
            )
        except ImportError:
            logger.warning("Metrics module not found, computing basic metrics only")
            return {"exact_match": self._exact_match(predicted_texts, reference_texts)}
        
        metrics = {}
        
        # Exact match
        metrics["exact_match"] = exact_match_accuracy(predicted_texts, reference_texts)
        
        # ROUGE
        logger.info("Computing ROUGE scores...")
        rouge_scores = compute_rouge_scores(predicted_texts, reference_texts)
        metrics.update(rouge_scores)
        
        # BLEU
        logger.info("Computing BLEU score...")
        bleu_score = compute_bleu_score(predicted_texts, reference_texts)
        metrics["bleu"] = bleu_score
        
        # BERTScore
        logger.info("Computing BERTScore...")
        bertscore = compute_bertscore(predicted_texts, reference_texts)
        if "error" not in bertscore:
            metrics["bertscore_f1"] = bertscore.get("bertscore_f1")
            metrics["bertscore_precision"] = bertscore.get("bertscore_precision")
            metrics["bertscore_recall"] = bertscore.get("bertscore_recall")
        else:
            logger.warning(f"BERTScore failed: {bertscore.get('error')}")
        
        # CLIP Score (text similarity)
        logger.info("Computing CLIP score...")
        try:
            clip_score = compute_clip_score_text(predicted_texts, reference_texts)
            metrics["clip_text_score"] = clip_score
        except Exception as e:
            logger.warning(f"CLIP score failed: {e}")
        
        # Per-question metrics
        pred_dicts = [{"question_id": p["question_id"], "answer": p["predicted_answer"]} for p in predictions]
        ref_dicts = [{"question_id": p["question_id"], "answer": p["reference_answer"]} for p in predictions]
        per_q_metrics = compute_per_question_accuracy(pred_dicts, ref_dicts)
        metrics["per_question"] = per_q_metrics
        
        # Inference time
        inference_times = [p["inference_time_s"] for p in predictions if "inference_time_s" in p]
        if inference_times:
            metrics["avg_inference_time_s"] = sum(inference_times) / len(inference_times)
            metrics["total_inference_time_s"] = sum(inference_times)
        
        return metrics
    
    @staticmethod
    def _exact_match(predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy (fallback if metrics module not available)."""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0
        matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        return matches / len(predictions)
    
    def _save_results(self, result: Dict[str, Any], output_dir: Path):
        """Save evaluation results to JSON and CSV."""
        # Full results JSON
        full_path = output_dir / f"{self.model_name}_evaluation.json"
        with full_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved full results to {full_path}")
        
        # Summary CSV
        csv_path = output_dir / f"{self.model_name}_summary.csv"
        self._save_csv_summary(result, csv_path)
        logger.info(f"Saved CSV summary to {csv_path}")
    
    def _save_csv_summary(self, result: Dict[str, Any], csv_path: Path):
        """Save summary as CSV."""
        lines = []
        lines.append("metric,value")
        lines.append(f"model,{result['model_name']}")
        lines.append(f"total_samples,{result['total_samples']}")
        lines.append(f"total_failed,{result['total_failed']}")
        
        metrics = result.get("metrics", {})
        
        # Overall metrics
        for key in ["exact_match", "rouge1_f1", "rouge2_f1", "rouge3_f1", "rouge4_f1",
                    "rougeL_f1", "rougeLsum_f1", "bleu",
                    "bertscore_f1", "bertscore_precision", "bertscore_recall",
                    "clip_text_score", "avg_inference_time_s"]:
            if key in metrics and metrics[key] is not None:
                lines.append(f"{key},{metrics[key]:.4f}")
        
        # Per-question metrics
        per_q = metrics.get("per_question", {})
        if per_q:
            lines.append("")
            lines.append("question_type,accuracy,correct,total")
            for qid, qmetrics in per_q.items():
                acc = qmetrics.get("accuracy", 0.0)
                correct = qmetrics.get("correct", 0)
                total = qmetrics.get("total", 0)
                lines.append(f"{qid},{acc:.4f},{correct},{total}")
        
        csv_path.write_text("\n".join(lines), encoding="utf-8")


def run_multi_model_evaluation(
    vqa_dataset_path: str,
    images_dir: str,
    output_base_dir: str,
    models: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
    save_predictions: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation on multiple models.
    
    Args:
        vqa_dataset_path: Path to VQA JSON dataset
        images_dir: Directory with images
        output_base_dir: Base directory to save results (creates subdirs per model)
        models: List of model configs, each with:
            - name: Model name (e.g. "florence2")
            - type: Model type (e.g. "florence2")
            - api_key: Optional API key
            - device: Optional device ("cuda", "cpu", "auto")
        max_samples: Limit number of samples (for testing)
        save_predictions: Save individual predictions
        verbose: Verbose logging
    
    Returns:
        Summary dict with results for all models
    
    Example:
        models = [
            {"name": "florence2", "type": "florence2", "device": "cuda"},
            {"name": "qwen", "type": "qwen", "device": "cuda"},
            {"name": "gpt4o", "type": "gpt4o", "api_key": "sk-..."},
        ]
        results = run_multi_model_evaluation(
            vqa_dataset_path="path/to/vqa.json",
            images_dir="path/to/images",
            output_base_dir="path/to/output",
            models=models,
        )
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_config in models:
        model_name = model_config.get("name", "unknown")
        model_type = model_config.get("type", model_name)
        api_key = model_config.get("api_key")
        device = model_config.get("device", "auto")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create evaluator
            evaluator = StandardVQAEvaluator(
                model_name=model_name,
                model_type=model_type,
                api_key=api_key,
                device=device,
            )
            
            # Create output dir for this model
            model_output_dir = output_base / model_name
            
            # Run evaluation
            result = evaluator.evaluate(
                vqa_dataset_path=vqa_dataset_path,
                images_dir=images_dir,
                output_dir=str(model_output_dir),
                max_samples=max_samples,
                save_predictions=save_predictions,
            )
            
            all_results[model_name] = result
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save comparison summary
    comparison_path = output_base / "comparison_summary.json"
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "dataset": vqa_dataset_path,
        "models_evaluated": len(all_results),
        "results": {
            name: {
                "total_samples": r.get("total_samples", 0),
                "metrics": r.get("metrics", {}),
            }
            for name, r in all_results.items()
            if "error" not in r
        }
    }
    
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved comparison summary to {comparison_path}")
    
    # Save comparison CSV
    comparison_csv_path = output_base / "comparison_summary.csv"
    _save_comparison_csv(all_results, comparison_csv_path)
    logger.info(f"Saved comparison CSV to {comparison_csv_path}")
    
    return all_results


def _save_comparison_csv(results: Dict[str, Any], csv_path: Path):
    """Save comparison of all models as CSV."""
    lines = []
    
    # Header
    metric_names = ["exact_match", "rouge1_f1", "rouge2_f1", "rouge3_f1", "rouge4_f1",
                    "rougeL_f1", "rougeLsum_f1", "bleu",
                    "bertscore_f1", "clip_text_score", "avg_inference_time_s"]
    header = ["model", "total_samples"] + metric_names
    lines.append(",".join(header))
    
    # Rows
    for model_name, result in results.items():
        if "error" in result:
            lines.append(f"{model_name},ERROR: {result['error']}")
            continue
        
        row = [model_name, str(result.get("total_samples", 0))]
        metrics = result.get("metrics", {})
        
        for metric in metric_names:
            val = metrics.get(metric)
            if val is not None:
                row.append(f"{val:.4f}")
            else:
                row.append("N/A")
        
        lines.append(",".join(row))
    
    csv_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Evaluate VLM models on standard VQA (no multiple choice)"
    )
    parser.add_argument(
        "--vqa-dataset",
        required=True,
        help="Path to VQA JSON dataset (per-question or per_image_all.json)"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory with images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["florence2", "qwen", "llava"],
        help="Models to evaluate (e.g. florence2 qwen llava gpt4o)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for local models"
    )
    parser.add_argument(
        "--api-key",
        help="API key for API-based models (GPT-4o, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Build model configs
    model_configs = []
    for model_name in args.models:
        config = {
            "name": model_name,
            "type": model_name,
            "device": args.device,
        }
        if args.api_key:
            config["api_key"] = args.api_key
        model_configs.append(config)
    
    # Run evaluation
    run_multi_model_evaluation(
        vqa_dataset_path=args.vqa_dataset,
        images_dir=args.images_dir,
        output_base_dir=args.output_dir,
        models=model_configs,
        max_samples=args.max_samples,
        save_predictions=True,
        verbose=args.verbose,
    )
