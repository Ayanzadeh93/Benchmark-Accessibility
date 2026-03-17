#!/usr/bin/env python3
"""
Dual-Mode VQA Evaluation Script

Supports two evaluation modes:
1. MMVQA (multiple choice): A/B/C/D accuracy. Ground truth from VQA per-image folder.
2. VQA2 (open-ended): ROUGE-1, ROUGE-L, BLEU, BERTScore. Reference = answer text per question.

Ground truth (same for both modes):
  C:\\...\\vqa\\qwen-3-235b  (per-image JSON)
- MMVQA uses: annotation_answer (A/B/C/D)
- VQA2 uses: annotation_answer_text (reference text for ROUGE/BLEU/BERTScore)

Usage:
    python -m vqa.evaluation.evaluate_dual_mode --mode mmvqa \\
        --predictions-dir "path/to/model/per_image" \\
        --ground-truth "C:\\...\\vqa\\qwen-3-235b" \\
        --output-dir "path/to/results"
    
    python -m vqa.evaluation.evaluate_dual_mode --mode vqa2 \\
        --predictions-dir "path/to/model/per_image" \\
        --ground-truth "C:\\...\\vqa\\qwen-3-235b" \\
        --output-dir "path/to/results"
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DualModeEvaluator:
    """Evaluator supporting MMVQA and VQA2 modes."""
    
    def __init__(self, mode: str = "vqa2"):
        """
        Args:
            mode: "mmvqa" for multiple choice, "vqa2" for open-ended
        """
        self.mode = mode.lower()
        if self.mode not in ("mmvqa", "vqa2"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'mmvqa' or 'vqa2'")
        logger.info(f"Initialized DualModeEvaluator in {self.mode.upper()} mode")
    
    def load_predictions(self, predictions_dir: Path) -> Dict[str, Any]:
        """
        Load per-image predictions.
        
        Returns:
            Dict mapping image_stem -> {predictions: [...], model: ...}
        """
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
        
        predictions = {}
        json_files = list(predictions_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {predictions_dir}")
        
        logger.info(f"Loading predictions from {len(json_files)} files")
        
        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                stem = json_file.stem
                predictions[stem] = data
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"✓ Loaded {len(predictions)} prediction files")
        return predictions
    
    def load_ground_truth(self, gt_path: Path) -> Dict[str, Any]:
        """
        Load ground truth. Supports:
        - File (per_image_all.json): list of {image, questions: [...]}
        - Directory (VQA per-image): e.g. vqa/qwen-3-235b/*.json with questions per image
        
        Returns:
            Dict mapping image_stem -> {questions: [...]}
        """
        gt_dict = {}
        
        if gt_path.is_dir():
            # VQA format or per_image_all_models
            for json_file in gt_path.glob("*.json"):
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    stem = json_file.stem
                    if "questions" in data:
                        gt_dict[stem] = data
                    elif "predictions_by_model" in data:
                        # per_image_all_models: use reference_answer as ground truth
                        by_model = data["predictions_by_model"]
                        preds = list(by_model.values())[0] if by_model else []
                        questions = [
                            {
                                "question_id": p.get("question_id", ""),
                                "question": p.get("question", ""),
                                "answer": p.get("reference_answer", ""),
                                "answer_text": p.get("reference_answer", ""),
                            }
                            for p in preds
                        ]
                        gt_dict[stem] = {"image": data.get("image", stem + ".jpg"), "questions": questions}
                    else:
                        gt_dict[stem] = {"image": data.get("image", stem + ".jpg"), "questions": data.get("questions", [])}
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
            logger.info(f"✓ Loaded ground truth from directory: {len(gt_dict)} images")
        else:
            # File: per_image_all.json
            with gt_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    image = entry.get("image", "")
                    stem = Path(image).stem
                    gt_dict[stem] = entry
            logger.info(f"✓ Loaded ground truth from file: {len(gt_dict)} images")
        
        return gt_dict
    
    def evaluate_mmvqa(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate MMVQA (multiple choice) mode.
        
        Computes:
        - Overall accuracy
        - Per-question accuracy
        - Confusion matrix
        """
        total_correct = 0
        total_questions = 0
        per_question = defaultdict(lambda: {"correct": 0, "total": 0})
        confusion = defaultdict(lambda: defaultdict(int))  # predicted -> reference -> count
        
        for stem, pred_data in predictions.items():
            if stem not in ground_truth:
                logger.warning(f"No ground truth for {stem}")
                continue
            
            gt_data = ground_truth[stem]
            gt_questions = {q["question_id"]: q for q in gt_data.get("questions", [])}
            
            for pred in pred_data.get("predictions", []):
                qid = pred.get("question_id", "")
                predicted_raw = pred.get("predicted_answer", "") or pred.get("answer", "")
                predicted = str(predicted_raw).strip().upper()
                
                if qid not in gt_questions:
                    continue
                
                # Extract reference choice (A/B/C/D)
                # VQA format: ground_truth.annotation_answer or metadata.ground_truth.annotation_answer
                gt_q = gt_questions[qid]
                reference = None
                if "ground_truth" in gt_q:
                    reference = gt_q["ground_truth"].get("annotation_answer", "")
                if not reference and "metadata" in gt_q and "ground_truth" in gt_q["metadata"]:
                    reference = gt_q["metadata"]["ground_truth"].get("annotation_answer", "")
                if not reference:
                    reference = gt_q.get("annotation_answer", "")
                
                if not reference:
                    continue
                
                reference = reference.strip().upper()
                
                # Extract A/B/C/D from prediction: "Answer: C", "C", or match to option text
                pred_choice = ""
                if len(predicted) == 1 and predicted in ("A", "B", "C", "D"):
                    pred_choice = predicted
                elif predicted:
                    # Try "Answer: X" or first char
                    for prefix in ("ANSWER:", "ANSWER ", "ANS:"):
                        if prefix in predicted:
                            rest = predicted.split(prefix, 1)[-1].strip()
                            if rest and rest[0] in ("A", "B", "C", "D"):
                                pred_choice = rest[0]
                                break
                    if not pred_choice and predicted[0] in ("A", "B", "C", "D"):
                        pred_choice = predicted[0]
                    # Match to option text if we have options
                    if not pred_choice and "options" in gt_q:
                        pred_lower = predicted_raw.strip().lower()
                        for opt, text in (gt_q.get("options") or {}).items():
                            if opt in ("A", "B", "C", "D") and text and text.strip().lower() in pred_lower:
                                pred_choice = opt
                                break
                predicted = pred_choice
                
                if predicted in ("A", "B", "C", "D"):
                    total_questions += 1
                    per_question[qid]["total"] += 1
                    confusion[predicted][reference] += 1
                    
                    if predicted == reference:
                        total_correct += 1
                        per_question[qid]["correct"] += 1
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Compute per-question accuracy
        per_q_acc = {}
        for qid, stats in per_question.items():
            per_q_acc[qid] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"]
            }
        
        results = {
            "mode": "mmvqa",
            "overall_accuracy": accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "per_question_accuracy": per_q_acc,
            "confusion_matrix": dict(confusion),
        }
        
        logger.info(f"MMVQA Accuracy: {accuracy:.2%} ({total_correct}/{total_questions})")
        return results
    
    def evaluate_vqa2(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate VQA2 (open-ended) mode.
        
        Computes:
        - ROUGE-1, ROUGE-L
        - BLEU
        - BERTScore (F1, Precision, Recall)
        - Per-question metrics
        """
        # Try importing metrics libraries
        try:
            from rouge_score import rouge_scorer
            rouge_available = True
        except ImportError:
            logger.warning("rouge-score not installed. Install: pip install rouge-score")
            rouge_available = False
        
        try:
            from bert_score import score as bert_score
            bertscore_available = True
        except ImportError:
            logger.warning("bert-score not installed. Install: pip install bert-score")
            bertscore_available = False
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            bleu_available = True
        except ImportError:
            logger.warning("nltk not installed. Install: pip install nltk")
            bleu_available = False
        
        # Collect all predictions and references
        all_predictions = []
        all_references = []
        per_question_data = defaultdict(lambda: {"predictions": [], "references": []})
        
        for stem, pred_data in predictions.items():
            if stem not in ground_truth:
                logger.warning(f"No ground truth for {stem}")
                continue
            
            gt_data = ground_truth[stem]
            gt_questions = {q["question_id"]: q for q in gt_data.get("questions", [])}
            
            for pred in pred_data.get("predictions", []):
                qid = pred.get("question_id", "")
                predicted = pred.get("predicted_answer", "").strip()
                
                if qid not in gt_questions:
                    continue
                
                gt_q = gt_questions[qid]
                # Reference: answer_text, or ground_truth.annotation_answer_text, or options[annotation_answer]
                reference = ""
                if "ground_truth" in gt_q:
                    ref_gt = gt_q["ground_truth"]
                    reference = ref_gt.get("annotation_answer_text", "") or ref_gt.get("answer_text", "")
                if not reference:
                    reference = gt_q.get("answer_text", "") or gt_q.get("answer", "")
                # If reference is A/B/C/D, use options text
                if reference.strip().upper() in ("A", "B", "C", "D"):
                    opts = gt_q.get("options") or {}
                    reference = opts.get(reference.strip().upper(), reference)
                reference = str(reference).strip()
                
                if not reference or not predicted:
                    continue
                
                all_predictions.append(predicted)
                all_references.append(reference)
                per_question_data[qid]["predictions"].append(predicted)
                per_question_data[qid]["references"].append(reference)
        
        if not all_predictions:
            logger.error("No valid prediction-reference pairs found")
            return {"mode": "vqa2", "error": "No valid pairs"}
        
        logger.info(f"Evaluating {len(all_predictions)} prediction-reference pairs")
        
        results = {
            "mode": "vqa2",
            "total_samples": len(all_predictions),
        }
        
        # ROUGE
        if rouge_available:
            # Include all available n-gram types (1-4) + LCS variants
            rouge_types = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum']
            scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
            rouge_scores_by_type = {rt: [] for rt in rouge_types}
            
            for pred, ref in zip(all_predictions, all_references):
                scores = scorer.score(ref, pred)
                for rt in rouge_types:
                    rouge_scores_by_type[rt].append(scores[rt].fmeasure)
            
            # Save all ROUGE metrics
            for rt in rouge_types:
                metric_name = f"{rt}_f1"
                results[metric_name] = float(np.mean(rouge_scores_by_type[rt]))
            
            # Log main metrics
            logger.info(f"  ROUGE-1: {results['rouge1_f1']:.4f}")
            logger.info(f"  ROUGE-2: {results['rouge2_f1']:.4f}")
            logger.info(f"  ROUGE-3: {results['rouge3_f1']:.4f}")
            logger.info(f"  ROUGE-4: {results['rouge4_f1']:.4f}")
            logger.info(f"  ROUGE-L: {results['rougeL_f1']:.4f}")
            logger.info(f"  ROUGE-Lsum: {results['rougeLsum_f1']:.4f}")
        
        # BLEU
        if bleu_available:
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            for pred, ref in zip(all_predictions, all_references):
                pred_tokens = pred.lower().split()
                ref_tokens = [ref.lower().split()]
                try:
                    bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(bleu)
                except:
                    pass
            
            if bleu_scores:
                results["bleu"] = float(np.mean(bleu_scores))
                logger.info(f"  BLEU: {results['bleu']:.4f}")
        
        # BERTScore
        if bertscore_available:
            logger.info("  Computing BERTScore (this may take a while)...")
            try:
                P, R, F1 = bert_score(all_predictions, all_references, lang="en", verbose=False)
                results["bertscore_f1"] = float(F1.mean().item())
                results["bertscore_precision"] = float(P.mean().item())
                results["bertscore_recall"] = float(R.mean().item())
                logger.info(f"  BERTScore F1: {results['bertscore_f1']:.4f}")
            except Exception as e:
                logger.error(f"BERTScore failed: {e}")
        
        # Per-question metrics
        per_q_metrics = {}
        if rouge_available:
            for qid, data in per_question_data.items():
                scores_by_type = {rt: [] for rt in rouge_types}
                for pred, ref in zip(data["predictions"], data["references"]):
                    scores = scorer.score(ref, pred)
                    for rt in rouge_types:
                        scores_by_type[rt].append(scores[rt].fmeasure)
                
                per_q_metrics[qid] = {
                    f"{rt}_f1": float(np.mean(scores_by_type[rt])) if scores_by_type[rt] else 0.0
                    for rt in rouge_types
                }
                per_q_metrics[qid]["total"] = len(data["predictions"])
        
        results["per_question_metrics"] = per_q_metrics
        
        return results
    
    def evaluate(
        self,
        predictions_dir: Path,
        ground_truth_path: Path
    ) -> Dict[str, Any]:
        """Run evaluation."""
        predictions = self.load_predictions(predictions_dir)
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        if self.mode == "mmvqa":
            return self.evaluate_mmvqa(predictions, ground_truth)
        else:
            return self.evaluate_vqa2(predictions, ground_truth)
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"evaluation_{self.mode}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved results: {json_path}")
        
        # Save CSV summary
        csv_path = output_dir / f"evaluation_{self.mode}_summary.csv"
        lines = []
        lines.append("metric,value")
        lines.append(f"mode,{results['mode']}")
        
        if self.mode == "mmvqa":
            lines.append(f"overall_accuracy,{results['overall_accuracy']:.4f}")
            lines.append(f"total_correct,{results['total_correct']}")
            lines.append(f"total_questions,{results['total_questions']}")
            
            lines.append("")
            lines.append("question_id,accuracy,correct,total")
            for qid, metrics in results.get("per_question_accuracy", {}).items():
                lines.append(f"{qid},{metrics['accuracy']:.4f},{metrics['correct']},{metrics['total']}")
        else:
            lines.append(f"total_samples,{results['total_samples']}")
            # All ROUGE types + other metrics
            for key in ["rouge1_f1", "rouge2_f1", "rouge3_f1", "rouge4_f1", "rougeL_f1", "rougeLsum_f1", 
                       "bleu", "bertscore_f1", "bertscore_precision", "bertscore_recall"]:
                if key in results:
                    lines.append(f"{key},{results[key]:.4f}")
            
            if "per_question_metrics" in results and results["per_question_metrics"]:
                lines.append("")
                # CSV header for per-question with all ROUGE types
                lines.append("question_id,rouge1_f1,rouge2_f1,rouge3_f1,rouge4_f1,rougeL_f1,rougeLsum_f1,total")
                for qid, metrics in results["per_question_metrics"].items():
                    line_parts = [
                        qid,
                        f"{metrics.get('rouge1_f1', 0):.4f}",
                        f"{metrics.get('rouge2_f1', 0):.4f}",
                        f"{metrics.get('rouge3_f1', 0):.4f}",
                        f"{metrics.get('rouge4_f1', 0):.4f}",
                        f"{metrics.get('rougeL_f1', 0):.4f}",
                        f"{metrics.get('rougeLsum_f1', 0):.4f}",
                        str(metrics.get('total', 0))
                    ]
                    lines.append(",".join(line_parts))
        
        csv_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"✓ Saved summary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Mode VQA Evaluation (MMVQA + VQA2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both use same ground truth: vqa/qwen-3-235b (per-image)
  python -m vqa.evaluation.evaluate_dual_mode --mode mmvqa \\
      --predictions-dir "path/to/model/per_image" \\
      --ground-truth "C:\\...\\vqa\\qwen-3-235b" --output-dir "path/to/results"
  python -m vqa.evaluation.evaluate_dual_mode --mode vqa2 \\
      --predictions-dir "path/to/model/per_image" \\
      --ground-truth "C:\\...\\vqa\\qwen-3-235b" --output-dir "path/to/results"
        """
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["mmvqa", "vqa2"],
        help="Evaluation mode: 'mmvqa' for multiple choice, 'vqa2' for open-ended"
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing per-image prediction JSON files"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground truth: per-image folder (e.g. vqa/qwen-3-235b). Same for both modes."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    evaluator = DualModeEvaluator(mode=args.mode)
    results = evaluator.evaluate(
        predictions_dir=Path(args.predictions_dir),
        ground_truth_path=Path(args.ground_truth)
    )
    evaluator.save_results(results, Path(args.output_dir))
    
    logger.info("\n" + "="*60)
    logger.info("✓ Evaluation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
