"""Simple VQA Evaluation Script

This script:
1. Loads your VQA2 ground truth (per_image_all.json or per-question files)
2. BATCH MODE: Sends all questions per image in ONE API call (like MMVQA - fast!)
3. Evaluates model text responses (not multiple choice)
4. Limits response to ~50 words per answer
5. Computes metrics: ROUGE-1/2/L, BLEU, BERTScore
6. Saves results per model

Usage:
    python -m vqa.evaluation.evaluate_vqa_simple \\
        --ground-truth "path/to/per_image_all.json" \\
        --output-dir "path/to/results" \\
        --models florence2 qwen openrouter_qwen3_vl_8b

    # Run all factory models (local GPU only)
    python -m vqa.evaluation.evaluate_vqa_simple \\
        --ground-truth "path/to/per_image_all.json" \\
        --output-dir "path/to/results" \\
        --all-models
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# All models available in VLM factory (for --all-models)
ALL_FACTORY_MODELS = [
    "florence2", "qwen", "llava",
    "gpt4o", "gpt5nano", "gpt5mini",
    "openrouter_trinity", "openrouter_llama32_11b_vision",
    "openrouter_llama4_maverick", "openrouter_molmo_8b",
    "openrouter_ministral_3b", "openrouter_qwen3_vl_235b",
    "openrouter_qwen3_vl_8b", "openrouter_qwen_vl_plus",
]

# Local GPU models only (no API key needed)
LOCAL_MODELS = ["florence2", "qwen", "llava"]


class SimpleVQAEvaluator:
    """Simple VQA evaluator for your ground truth files."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        api_key: Optional[str] = None,
        device: str = "auto",
        max_words: int = 50,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.device = device
        self.max_words = max_words
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
            logger.info(f"✓ Initialized model: {self.model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize model {self.model_type}: {e}")
            raise
    
    @staticmethod
    def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
        """
        Load ground truth from JSON file.
        
        Supports:
        - per_image_all.json: [{"image": "...", "questions": [...]}]
        - per-question files: {"question_id": "...", "samples": [...]}
        
        Returns:
            List of samples with: id, image, image_path, question, answer_text
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        
        if isinstance(data, list):
            # per_image_all.json format
            for entry in data:
                image = entry.get("image", "")
                image_path = entry.get("image_path", "")
                
                for q in entry.get("questions", []):
                    # VQA2 uses "answer", MMVQA uses "answer_text"
                    answer = q.get("answer") or q.get("answer_text", "")
                    sample = {
                        "id": q.get("id", ""),
                        "question_id": q.get("question_id", ""),
                        "image": image,
                        "image_path": image_path or q.get("image_path", ""),
                        "question": q.get("question", ""),
                        "answer": answer,  # Ground truth answer
                    }
                    samples.append(sample)
        
        elif isinstance(data, dict):
            # per-question file format
            for sample in data.get("samples", []):
                answer = sample.get("answer") or sample.get("answer_text", "")
                s = {
                    "id": sample.get("id", ""),
                    "question_id": sample.get("question_id", ""),
                    "image": sample.get("image", ""),
                    "image_path": sample.get("image_path", ""),
                    "question": sample.get("question", ""),
                    "answer": answer,
                }
                samples.append(s)
        
        return samples
    
    @staticmethod
    def load_ground_truth_per_image(path: Path, images_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Load ground truth grouped by image for batch evaluation.
        Supports: file (per_image_all.json) or directory (vqa/qwen-3-235b per-image JSONs).
        If image_path is missing, uses images_dir / image_name when images_dir is provided.
        Returns: [{"image": "...", "image_path": "...", "questions": [{id, question_id, question, answer}, ...]}, ...]
        """
        def _resolve_image_path(entry: Dict[str, Any]) -> str:
            """Resolve image_path; use images_dir/image when path is empty."""
            ip = entry.get("image_path", "")
            if ip and Path(ip).exists():
                return ip
            if images_dir and entry.get("image"):
                candidate = Path(images_dir) / entry["image"]
                if candidate.exists():
                    return str(candidate)
            return ip

        # Directory: VQA per-image format (e.g. vqa/qwen-3-235b) or per_image_all_models
        if path.is_dir():
            result = []
            for json_file in sorted(path.glob("*.json")):
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    image = data.get("image", json_file.stem + ".jpg")
                    image_path = data.get("image_path", "")
                    questions = []

                    # per_image_all_models: predictions_by_model -> use reference_answer as ground truth
                    if "predictions_by_model" in data:
                        by_model = data["predictions_by_model"]
                        preds = list(by_model.values())[0] if by_model else []  # first model
                        for p in preds:
                            ref = p.get("reference_answer", "")
                            if ref or p.get("question"):
                                questions.append({
                                    "id": p.get("id", ""),
                                    "question_id": p.get("question_id", ""),
                                    "question": p.get("question", ""),
                                    "answer": ref,
                                })
                    else:
                        # Standard VQA: questions with ground_truth or answer_text
                        if not image_path and "questions" in data and data["questions"]:
                            image_path = data["questions"][0].get("image_path", "")
                        for q in data.get("questions", []):
                            ref = ""
                            if "ground_truth" in q:
                                ref = q["ground_truth"].get("annotation_answer_text", "") or q["ground_truth"].get("answer_text", "")
                            if not ref:
                                ref = q.get("answer_text", "") or q.get("answer", "")
                            questions.append({
                                "id": q.get("id", ""),
                                "question_id": q.get("question_id", ""),
                                "question": q.get("question", ""),
                                "answer": ref,
                            })
                    if questions and (image_path or image):
                        entry = {"image": image, "image_path": image_path, "questions": questions}
                        entry["image_path"] = _resolve_image_path(entry)
                        result.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
            return result
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Fast path: already per-image format (per_image_all.json)
        if isinstance(data, list) and data and "questions" in (data[0] or {}):
            result = []
            for entry in data:
                image_path = entry.get("image_path", "")
                image = entry.get("image", "")
                if not image_path and entry.get("questions"):
                    image_path = entry["questions"][0].get("image_path", "")
                questions = []
                for q in entry.get("questions", []):
                    answer = q.get("answer") or q.get("answer_text", "")
                    questions.append({
                        "id": q.get("id", ""),
                        "question_id": q.get("question_id", ""),
                        "question": q.get("question", ""),
                        "answer": answer,
                    })
                if questions and (image_path or image):
                    e = {"image": image, "image_path": image_path, "questions": questions}
                    e["image_path"] = _resolve_image_path(e)
                    result.append(e)
            return result
        
        # Convert from flat or per-question format
        samples = SimpleVQAEvaluator.load_ground_truth(path)
        by_image: Dict[str, Dict[str, Any]] = {}
        for s in samples:
            key = s.get("image_path") or s.get("image", "")
            if not key:
                continue
            if key not in by_image:
                by_image[key] = {
                    "image": s.get("image", ""),
                    "image_path": s.get("image_path", ""),
                    "questions": [],
                }
            by_image[key]["questions"].append({
                "id": s.get("id", ""),
                "question_id": s.get("question_id", ""),
                "question": s.get("question", ""),
                "answer": s.get("answer", ""),
            })
        out = list(by_image.values())
        for e in out:
            e["image_path"] = _resolve_image_path(e)
        return out
    
    def ask_question(self, image_path: str, question: str) -> Optional[str]:
        """
        Ask the model a question about an image.
        Uses generate_freeform_text with max_words limit.
        
        Args:
            image_path: Path to the image
            question: Question text
        
        Returns:
            Model's answer as text, or None if failed
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        try:
            # Build prompt with length limit (~1.5 tokens per word)
            max_tokens = max(20, int(self.max_words * 1.5))
            prompt = f"{question}\n\nAnswer in {self.max_words} words or less. Be concise."
            
            if hasattr(self.model, "generate_freeform_text"):
                answer = self.model.generate_freeform_text(
                    image_path, prompt, max_new_tokens=max_tokens
                )
            elif hasattr(self.model, "extract_text"):
                answer = self.model.extract_text(image_path, prompt)
            elif hasattr(self.model, "answer_question"):
                answer = self.model.answer_question(image_path, prompt)
            else:
                logger.error(f"Model {self.model_name} does not support VQA")
                return None
            
            # Truncate to max_words if model ignored instruction
            if answer:
                words = answer.split()
                if len(words) > self.max_words:
                    answer = " ".join(words[:self.max_words])
            
            return answer.strip() if answer else None
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        batch_mode: bool = True,
        output_dir: Optional[Path] = None,
        refinement_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on samples.
        Uses batch mode by default: 1 API/model call per image (all questions).
        
        Args:
            samples: List of samples OR per-image list [{"image_path", "questions": [...]}]
            max_samples: Limit number of samples (for testing) - limits images in batch mode
            batch_mode: If True, one call per image (fast). If False, one call per question.
            output_dir: If set, save per-image JSON after each image (incremental save).
            refinement_mode: If True, send reference answers to the model as a helper - model
                can improve upon them. Good for generating refined ground truth (e.g. qwen 235b).
        
        Returns:
            Results dict with predictions and metrics
        """
        use_batch = batch_mode and hasattr(self.model, "generate_freeform_text_batch")
        if use_batch:
            return self._evaluate_batch(samples, max_samples, output_dir, refinement_mode)
        return self._evaluate_sequential(samples, max_samples)
    
    def _evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        output_dir: Optional[Path] = None,
        refinement_mode: bool = False,
    ) -> Dict[str, Any]:
        """Batch mode: 1 call per image with all questions."""
        # Expect per-image format: [{"image_path", "questions": [...]}]
        per_image = samples
        if per_image and "questions" not in (per_image[0] or {}):
            per_image = self._samples_to_per_image(samples)
        
        if max_samples:
            cum = 0
            trimmed = []
            for entry in per_image:
                nq = len(entry.get("questions", []))
                if cum + nq > max_samples:
                    # Take partial: only first (max_samples - cum) questions
                    qs = entry.get("questions", [])[: max_samples - cum]
                    trimmed.append({**entry, "questions": qs})
                    break
                trimmed.append(entry)
                cum += nq
            per_image = trimmed
        
        total_questions = sum(len(e.get("questions", [])) for e in per_image)
        mode_str = " [REFINEMENT - reference as helper]" if refinement_mode else ""
        logger.info(f"Evaluating {len(per_image)} images ({total_questions} questions) with {self.model_name} [BATCH MODE - 1 call per image]{mode_str}...")
        
        predictions = []
        failed = 0
        per_image_dir = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            per_image_dir = output_dir / "per_image"
            per_image_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Incremental save: per-image JSON → {per_image_dir}")
        
        for entry in tqdm(per_image, desc=f"Evaluating {self.model_name}"):
            image_path = entry.get("image_path", "")
            questions = entry.get("questions", [])
            if not image_path or not Path(image_path).exists():
                failed += len(questions)
                continue
            if not questions:
                continue
            
            start_time = time.time()
            try:
                answers = self.model.generate_freeform_text_batch(
                    image_path,
                    questions,
                    max_words_per_answer=self.max_words,
                    include_reference_in_prompt=refinement_mode,
                )
            except Exception as e:
                logger.error(f"Batch failed for {image_path}: {e}")
                failed += len(questions)
                continue
            
            inference_time = time.time() - start_time
            per_q_time = inference_time / len(questions)
            
            image_preds = []
            for q in questions:
                qid = q.get("id", "")
                reference = q.get("answer", "")
                predicted = answers.get(qid, "")
                if not reference:
                    failed += 1
                    continue
                pred = {
                    "id": qid,
                    "question_id": q.get("question_id", ""),
                    "image": entry.get("image", ""),
                    "question": q.get("question", ""),
                    "predicted_answer": predicted or "",
                    "reference_answer": reference,
                    "inference_time_s": per_q_time,
                }
                predictions.append(pred)
                image_preds.append(pred)
            
            # Incremental save: write per-image JSON right after each image
            if per_image_dir and image_preds:
                image_name = entry.get("image", Path(image_path).name)
                stem = Path(image_name).stem
                out_path = per_image_dir / f"{stem}.json"
                entry_data = {
                    "image": image_name,
                    "model": self.model_name,
                    "predictions": image_preds,
                    "num_questions": len(image_preds),
                    "correct": sum(
                        1 for p in image_preds
                        if p.get("predicted_answer", "").strip().lower()
                        == p.get("reference_answer", "").strip().lower()
                    ),
                }
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(entry_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Completed: {len(predictions)} predictions, {failed} failed [batch mode]")
        return self._finish_evaluate(predictions, failed)
    
    def _samples_to_per_image(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert flat samples to per-image format."""
        from collections import defaultdict
        by_image: Dict[str, Dict[str, Any]] = {}
        for s in samples:
            key = s.get("image_path") or s.get("image", "")
            if key not in by_image:
                by_image[key] = {"image": s.get("image", ""), "image_path": s.get("image_path", ""), "questions": []}
            by_image[key]["questions"].append({
                "id": s.get("id", ""), "question_id": s.get("question_id", ""),
                "question": s.get("question", ""), "answer": s.get("answer", ""),
            })
        return list(by_image.values())
    
    def _evaluate_sequential(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sequential: 1 call per question."""
        if max_samples:
            samples = samples[:max_samples]
        logger.info(f"Evaluating {len(samples)} samples with {self.model_name} [sequential]...")
        predictions = []
        failed = 0
        for sample in tqdm(samples, desc=f"Evaluating {self.model_name}"):
            image_path = sample.get("image_path", "")
            question = sample.get("question", "")
            reference = sample.get("answer", "")
            if not image_path or not question or not reference:
                failed += 1
                continue
            start_time = time.time()
            predicted = self.ask_question(image_path, question)
            inference_time = time.time() - start_time
            if predicted is None:
                failed += 1
                continue
            predictions.append({
                "id": sample.get("id", ""),
                "question_id": sample.get("question_id", ""),
                "image": sample.get("image", ""),
                "question": question,
                "predicted_answer": predicted,
                "reference_answer": reference,
                "inference_time_s": inference_time,
            })
        logger.info(f"✓ Completed: {len(predictions)} predictions, {failed} failed")
        return self._finish_evaluate(predictions, failed)
    
    def _finish_evaluate(self, predictions: List[Dict[str, Any]], failed: int) -> Dict[str, Any]:
        metrics = self._compute_metrics(predictions)
        return {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(predictions),
            "total_failed": failed,
            "metrics": metrics,
            "predictions": predictions,
        }
    
    def _compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if not predictions:
            return {}
        
        predicted_texts = [p["predicted_answer"] for p in predictions]
        reference_texts = [p["reference_answer"] for p in predictions]
        
        metrics = {}
        
        # Exact match
        metrics["exact_match"] = self._exact_match(predicted_texts, reference_texts)
        
        # ROUGE
        try:
            try:
                from vqa.evaluation.metrics import compute_rouge_scores
            except ImportError:
                from metrics import compute_rouge_scores
            rouge_scores = compute_rouge_scores(predicted_texts, reference_texts)
            if "error" not in rouge_scores:
                metrics.update(rouge_scores)
            else:
                logger.warning(f"ROUGE: {rouge_scores.get('error')}")
        except Exception as e:
            logger.warning(f"ROUGE failed: {e}")
        
        # BLEU
        try:
            try:
                from vqa.evaluation.metrics import compute_bleu_score
            except ImportError:
                from metrics import compute_bleu_score
            bleu_score = compute_bleu_score(predicted_texts, reference_texts)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
        
        # BERTScore
        try:
            try:
                from vqa.evaluation.metrics import compute_bertscore
            except ImportError:
                from metrics import compute_bertscore
            logger.info("Computing BERTScore (this may take a while)...")
            bertscore = compute_bertscore(predicted_texts, reference_texts)
            if "error" not in bertscore:
                metrics["bertscore_f1"] = bertscore.get("bertscore_f1")
                metrics["bertscore_precision"] = bertscore.get("bertscore_precision")
                metrics["bertscore_recall"] = bertscore.get("bertscore_recall")
            else:
                logger.warning(f"BERTScore: {bertscore.get('error')}")
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
        
        # Per-question metrics
        try:
            try:
                from vqa.evaluation.metrics import compute_per_question_accuracy
            except ImportError:
                from metrics import compute_per_question_accuracy
            pred_dicts = [{"question_id": p["question_id"], "answer": p["predicted_answer"]} for p in predictions]
            ref_dicts = [{"question_id": p["question_id"], "answer": p["reference_answer"]} for p in predictions]
            per_q_metrics = compute_per_question_accuracy(pred_dicts, ref_dicts)
            metrics["per_question"] = per_q_metrics
        except Exception as e:
            logger.warning(f"Per-question metrics failed: {e}")
        
        # Inference time
        inference_times = [p["inference_time_s"] for p in predictions]
        if inference_times:
            metrics["avg_inference_time_s"] = sum(inference_times) / len(inference_times)
            metrics["total_inference_time_s"] = sum(inference_times)
        
        return metrics
    
    @staticmethod
    def _exact_match(predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy."""
        if not predictions or not references or len(predictions) != len(references):
            return 0.0
        matches = sum(1 for p, r in zip(predictions, references) 
                     if p.strip().lower() == r.strip().lower())
        return matches / len(predictions)
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Full results JSON
        full_path = output_dir / f"{self.model_name}_results.json"
        with full_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved results: {full_path}")
        
        # Summary CSV
        csv_path = output_dir / f"{self.model_name}_summary.csv"
        self._save_csv(results, csv_path)
        logger.info(f"✓ Saved summary: {csv_path}")
        
        # Per-image results (one JSON per image for this model)
        predictions = results.get("predictions", [])
        if predictions:
            per_image_dir = output_dir / "per_image"
            per_image_dir.mkdir(parents=True, exist_ok=True)
            by_image: Dict[str, List[Dict[str, Any]]] = {}
            for p in predictions:
                img = p.get("image", "unknown")
                by_image.setdefault(img, []).append(p)
            for image_name, preds in by_image.items():
                stem = Path(image_name).stem
                out_path = per_image_dir / f"{stem}.json"
                entry = {
                    "image": image_name,
                    "model": self.model_name,
                    "predictions": preds,
                    "num_questions": len(preds),
                    "correct": sum(1 for p in preds if p.get("predicted_answer", "").strip().lower() == p.get("reference_answer", "").strip().lower()),
                }
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved per-image results: {per_image_dir} ({len(by_image)} images)")
    
    def _save_csv(self, results: Dict[str, Any], csv_path: Path):
        """Save summary as CSV."""
        lines = []
        lines.append("metric,value")
        lines.append(f"model,{results['model_name']}")
        lines.append(f"total_samples,{results['total_samples']}")
        lines.append(f"total_failed,{results['total_failed']}")
        
        metrics = results.get("metrics", {})
        
        # Overall metrics
        for key in ["exact_match", "rouge1_f1", "rouge2_f1", "rouge3_f1", "rouge4_f1", 
                    "rougeL_f1", "rougeLsum_f1", "bleu",
                    "bertscore_f1", "bertscore_precision", "bertscore_recall",
                    "avg_inference_time_s"]:
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


def _resolve_per_image_dir(output_path: Path, model_name: str) -> Optional[Path]:
    """Resolve per_image dir - supports flat (model/per_image) and nested (model/model/per_image)."""
    flat = output_path / model_name / "per_image"
    if flat.exists():
        return flat
    nested = output_path / model_name / model_name / "per_image"
    if nested.exists():
        return nested
    return None


def _run_dual_mode_evaluation(
    output_path: Path,
    ground_truth_path: Path,
    models: List[str],
    eval_mmvqa: bool = True,
    eval_vqa2: bool = True,
) -> None:
    """Run MMVQA and/or VQA2 evaluation on model outputs."""
    from vqa.evaluation.evaluate_dual_mode import DualModeEvaluator
    gt_path = Path(ground_truth_path)
    for model_name in models:
        pred_dir = _resolve_per_image_dir(output_path, model_name)
        if pred_dir is None:
            logger.warning(f"Skipping dual eval for {model_name}: no per_image dir (checked {output_path / model_name / 'per_image'} and nested)")
            continue
        out_dir = output_path / model_name
        if eval_mmvqa:
            ev = DualModeEvaluator(mode="mmvqa")
            try:
                res = ev.evaluate(pred_dir, gt_path)
                ev.save_results(res, out_dir)
            except Exception as e:
                logger.error(f"MMVQA eval failed for {model_name}: {e}")
        if eval_vqa2:
            ev = DualModeEvaluator(mode="vqa2")
            try:
                res = ev.evaluate(pred_dir, gt_path)
                ev.save_results(res, out_dir)
            except Exception as e:
                logger.error(f"VQA2 eval failed for {model_name}: {e}")

def _discover_models_from_output(output_path: Path) -> List[str]:
    """Find model subfolders that contain per_image results. Supports flat and nested structures."""
    models = []
    for item in output_path.iterdir():
        if not item.is_dir():
            continue
        # Flat: output/model/per_image
        if (item / "per_image").is_dir():
            models.append(item.name)
            continue
        # Nested: output/model/model/per_image
        if (item / item.name / "per_image").is_dir():
            models.append(item.name)
    return sorted(models)


def evaluate_multiple_models(
    ground_truth_path: str,
    output_dir: str,
    models: List[str],
    api_key: Optional[str] = None,
    device: str = "auto",
    max_samples: Optional[int] = None,
    max_words: int = 50,
    compute_clip: bool = False,
    refinement_mode: bool = False,
    run_dual_eval: bool = False,
    eval_mmvqa: bool = True,
    eval_vqa2: bool = True,
    images_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate multiple models on ground truth.
    
    Args:
        ground_truth_path: Path to ground truth JSON (per_image_all.json or per-question)
        output_dir: Output directory for results
        models: List of model names (e.g. ["florence2", "qwen", "llava"])
        api_key: API key for API-based models
        device: Device for local models
        max_samples: Limit number of samples
        compute_clip: Whether to compute CLIP score (slow)
    
    Returns:
        Summary dict with results for all models
    """
    # Load ground truth
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    
    logger.info("="*60)
    logger.info(f"Loading ground truth: {gt_path}")
    logger.info("="*60)
    
    # Resolve images_dir: explicit arg, or auto-infer when ground truth is vqa/* dir
    img_dir = Path(images_dir) if images_dir else None
    if not img_dir and gt_path.is_dir():
        # e.g. .../vqa/qwen-3-235b -> try .../images
        parent = gt_path.parent  # vqa
        if parent.name.lower() == "vqa":
            candidate = parent.parent / "images"
            if candidate.exists():
                img_dir = candidate
                logger.info(f"Auto-detected images dir: {img_dir}")
    # Load per-image for batch mode (1 call per image)
    per_image = SimpleVQAEvaluator.load_ground_truth_per_image(gt_path, images_dir=img_dir)
    # Filter out entries with no resolvable image path
    valid = [e for e in per_image if e.get("image_path") and Path(e["image_path"]).exists()]
    skipped = len(per_image) - len(valid)
    if skipped:
        logger.warning(
            f"Skipping {skipped} images (no valid image_path). "
            "Add --images-dir to point to your images folder (e.g. --images-dir \"path/to/images\")."
        )
    per_image = valid
    if not per_image:
        raise FileNotFoundError(
            "No valid images found. Ground truth entries have no resolvable image paths. "
            "Use --images-dir to point to your images folder: --images-dir \"path/to/images\""
        )
    total_samples = sum(len(e.get("questions", [])) for e in per_image)
    logger.info(f"✓ Loaded {len(per_image)} images, {total_samples} questions [batch mode: 1 call per image]")
    
    if max_samples:
        logger.info(f"  (limiting to {max_samples} questions for testing)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = {}
    
    for model_name in models:
        logger.info("\n" + "="*60)
        logger.info(f"Evaluating: {model_name}")
        logger.info("="*60)
        
        try:
            # Create evaluator
            evaluator = SimpleVQAEvaluator(
                model_name=model_name,
                model_type=model_name,
                api_key=api_key,
                device=device,
                max_words=max_words,
            )
            
            # Evaluate (batch mode: 1 call per image, incremental per-image save)
            model_output_dir = output_path / model_name
            results = evaluator.evaluate(
                per_image,
                max_samples=max_samples,
                batch_mode=True,
                output_dir=model_output_dir,
                refinement_mode=refinement_mode,
            )
            
            # Save full results, summary, and comparison (per_image already saved incrementally)
            evaluator.save_results(results, model_output_dir)
            
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"✗ Failed to evaluate {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save comparison
    _save_comparison(all_results, output_path)
    
    # Save combined per-image results (all models' predictions for each image)
    _save_per_image_all_models(all_results, output_path)
    
    # Run dual-mode evaluation (MMVQA + VQA2) if requested
    if run_dual_eval:
        logger.info("\n" + "="*60)
        logger.info("Running dual-mode evaluation (MMVQA + VQA2)")
        logger.info("="*60)
        _run_dual_mode_evaluation(
            output_path,
            Path(ground_truth_path),
            models,
            eval_mmvqa=eval_mmvqa,
            eval_vqa2=eval_vqa2,
        )
    
    logger.info("\n" + "="*60)
    logger.info("✓ Evaluation complete!")
    logger.info(f"  Results: {output_path}")
    logger.info("="*60)
    
    return all_results


def _save_per_image_all_models(results: Dict[str, Any], output_dir: Path):
    """Save per-image results combining all models' predictions."""
    from collections import defaultdict
    # Collect predictions by image: image -> {model -> [predictions]}
    by_image: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for model_name, result in results.items():
        if "error" in result:
            continue
        predictions = result.get("predictions", [])
        for p in predictions:
            img = p.get("image", "unknown")
            by_image[img].setdefault(model_name, []).append(p)
    if not by_image:
        return
    per_image_dir = output_dir / "per_image_all_models"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    for image_name, model_preds in by_image.items():
        stem = Path(image_name).stem
        out_path = per_image_dir / f"{stem}.json"
        entry = {
            "image": image_name,
            "models": list(model_preds.keys()),
            "predictions_by_model": {
                model: preds for model, preds in model_preds.items()
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved per-image (all models): {per_image_dir} ({len(by_image)} images)")


def _save_comparison(results: Dict[str, Any], output_dir: Path):
    """Save comparison of all models."""
    # JSON comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            name: {
                "total_samples": r.get("total_samples", 0),
                "metrics": r.get("metrics", {}),
            }
            for name, r in results.items()
            if "error" not in r
        }
    }
    
    comparison_path = output_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved comparison: {comparison_path}")
    
    # CSV comparison
    csv_path = output_dir / "comparison.csv"
    lines = []
    
    metric_names = ["exact_match", "rouge1_f1", "rouge2_f1", "rouge3_f1", "rouge4_f1",
                    "rougeL_f1", "rougeLsum_f1", "bleu", "bertscore_f1", "avg_inference_time_s"]
    header = ["model", "total_samples"] + metric_names
    lines.append(",".join(header))
    
    for model_name, result in results.items():
        if "error" in result:
            lines.append(f"{model_name},ERROR")
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
    logger.info(f"✓ Saved comparison CSV: {csv_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple VQA Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate local GPU models
  python vqa/evaluation/evaluate_vqa_simple.py \\
      --ground-truth "path/to/per_image_all.json" \\
      --output-dir "path/to/results" \\
      --models florence2 qwen llava

  # Quick test (10 samples)
  python vqa/evaluation/evaluate_vqa_simple.py \\
      --ground-truth "path/to/per_image_all.json" \\
      --output-dir "path/to/results" \\
      --models florence2 \\
      --max-samples 10

  # Run models + MMVQA+VQA2 evaluation (merged pipeline)
  python -m vqa.evaluation.evaluate_vqa_simple \\
      --ground-truth "C:\\...\\vqa\\qwen-3-235b" \\
      --images-dir "C:\\...\\images" \\
      --output-dir "path/to/results" \\
      --models florence2 qwen8b \\
      --run-dual-eval
  
  # Evaluation only (use existing per_image results)
  python -m vqa.evaluation.evaluate_vqa_simple \\
      --ground-truth "C:\\...\\vqa\\qwen-3-235b" \\
      --output-dir "path/to/results" \\
      --eval-only --run-dual-eval

  # Evaluation with direct per_image path (nested structure)
  python -m vqa.evaluation.evaluate_vqa_simple \\
      --ground-truth "C:\\...\\vqa\\qwen-3-235b" \\
      --eval-only --predictions-dir "C:\\...\\results_eval\\qwen8b\\qwen8b\\per_image"
        """
    )
    
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground truth: per_image_all.json, per-question file, or vqa/qwen-3-235b per-image dir"
    )
    parser.add_argument(
        "--images-dir",
        help="Directory with images (required when ground truth has no image_path; e.g. .../images)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (optional with --eval-only --predictions-dir)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (e.g. florence2 qwen llava openrouter_qwen3_vl_8b)"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate all local GPU models (florence2, qwen, llava)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50,
        help="Max words for model response (default: 50)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for API-based models"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for local models"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--compute-clip",
        action="store_true",
        help="Compute CLIP score (slow, requires more GPU memory)"
    )
    parser.add_argument(
        "--refinement-mode",
        action="store_true",
        help="Send reference answer to model as helper (NOT used in standard VQA2 eval - model gets questions only)"
    )
    parser.add_argument(
        "--run-dual-eval",
        action="store_true",
        help="After running models, also run MMVQA + VQA2 evaluation (requires ground truth from vqa/qwen-3-235b)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip model inference and only run dual-mode evaluation on existing per_image results"
    )
    parser.add_argument(
        "--predictions-dir",
        help="[eval-only] Point directly at per_image folder (e.g. results_eval/qwen8b/qwen8b/per_image). Overrides discovery."
    )
    parser.add_argument(
        "--no-mmvqa",
        action="store_true",
        help="When --run-dual-eval: skip MMVQA evaluation"
    )
    parser.add_argument(
        "--no-vqa2",
        action="store_true",
        help="When --run-dual-eval: skip VQA2 evaluation"
    )
    
    args = parser.parse_args()
    
    if not args.eval_only or not args.predictions_dir:
        if not args.output_dir:
            parser.error("--output-dir is required (unless using --eval-only with --predictions-dir)")
    
    if args.eval_only:
        if not args.run_dual_eval:
            args.run_dual_eval = True
        if args.predictions_dir:
            # Direct path to per_image folder - save evaluation next to it
            pred_dir = Path(args.predictions_dir)
            if not pred_dir.exists():
                raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
            if not pred_dir.is_dir():
                raise ValueError(f"Predictions path must be a directory: {pred_dir}")
            from vqa.evaluation.evaluate_dual_mode import DualModeEvaluator
            gt_path = Path(args.ground_truth)
            out_dir = pred_dir.parent  # save eval results next to per_image
            if not args.no_mmvqa:
                ev = DualModeEvaluator(mode="mmvqa")
                try:
                    res = ev.evaluate(pred_dir, gt_path)
                    ev.save_results(res, out_dir)
                except Exception as e:
                    logger.error(f"MMVQA eval failed: {e}")
            if not args.no_vqa2:
                ev = DualModeEvaluator(mode="vqa2")
                try:
                    res = ev.evaluate(pred_dir, gt_path)
                    ev.save_results(res, out_dir)
                except Exception as e:
                    logger.error(f"VQA2 eval failed: {e}")
            logger.info(f"✓ Evaluation complete. Results: {out_dir}")
            return
        output_path = Path(args.output_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Output directory not found: {output_path}")
        models = args.models or _discover_models_from_output(output_path)
        if not models:
            raise ValueError(
                "No models found in output directory (missing per_image folders). "
                "Use --predictions-dir to point directly at a per_image folder."
            )
        _run_dual_mode_evaluation(
            output_path,
            Path(args.ground_truth),
            models,
            eval_mmvqa=not args.no_mmvqa,
            eval_vqa2=not args.no_vqa2,
        )
        return
    
    models = args.models
    if args.all_models:
        models = LOCAL_MODELS
    elif not models:
        models = ["florence2"]

    # Load .env and resolve API key for OpenRouter/API models
    api_key = args.api_key
    if not api_key and any(m for m in models if "openrouter" in m.lower() or m in ("gpt4o", "gpt5nano", "gpt5mini")):
        try:
            from config import load_env_file, get_openrouter_api_key, get_openai_api_key
            load_env_file()
            if any("openrouter" in m.lower() for m in models):
                api_key = get_openrouter_api_key()
            elif not api_key:
                api_key = get_openai_api_key()
        except ImportError:
            pass

    evaluate_multiple_models(
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        models=models,
        api_key=api_key,
        device=args.device,
        max_samples=args.max_samples,
        max_words=args.max_words,
        compute_clip=args.compute_clip,
        refinement_mode=args.refinement_mode,
        run_dual_eval=args.run_dual_eval,
        eval_mmvqa=not args.no_mmvqa,
        eval_vqa2=not args.no_vqa2,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()
