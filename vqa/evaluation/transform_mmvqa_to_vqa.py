"""Transform MMVQA (Multiple Choice) to Standard VQA format.

This script converts multiple-choice VQA datasets (with options A/B/C/D) to standard 
VQA format where each question has only the correct answer text (no options).

Input format (MMVQA):
{
    "id": "image_001|action_command",
    "question": "What should the person do?",
    "options": {"A": "Turn left", "B": "Go straight", "C": "Stop", "D": "Turn right"},
    "answer": "B",
    "answer_text": "Go straight"
}

Output format (Standard VQA):
{
    "id": "image_001|action_command",
    "question": "What should the person do?",
    "answer": "Go straight"
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def transform_sample_to_vqa(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single MMVQA sample to standard VQA format.
    
    Args:
        sample: MMVQA sample with id, question, options, answer, answer_text
    
    Returns:
        Standard VQA sample with id, question, answer (text only)
    """
    # Extract the correct answer text
    answer_text = sample.get("answer_text", "")
    
    # If answer_text is missing, try to get it from options
    if not answer_text:
        answer_label = sample.get("answer", "")
        options = sample.get("options", {})
        answer_text = options.get(answer_label, "")
    
    # Build standard VQA sample
    vqa_sample = {
        "id": sample.get("id", ""),
        "question_id": sample.get("question_id", ""),
        "image": sample.get("image", ""),
        "question": sample.get("question", ""),
        "answer": answer_text,
    }
    
    # Optionally include image path if available
    if "image_path" in sample:
        vqa_sample["image_path"] = sample["image_path"]
    
    # Optionally include metadata (sources, ground_truth) for reference
    if sample.get("ground_truth"):
        vqa_sample["metadata"] = {
            "ground_truth": sample["ground_truth"]
        }
    
    if sample.get("sources"):
        if "metadata" not in vqa_sample:
            vqa_sample["metadata"] = {}
        vqa_sample["metadata"]["sources"] = sample["sources"]
    
    return vqa_sample


def transform_per_question_file(
    input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Transform a per-question MMVQA file to standard VQA format.
    
    Args:
        input_path: Path to MMVQA JSON file (e.g. action_command.json)
        output_path: Path to output standard VQA JSON file
    
    Returns:
        Statistics dict
    """
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract samples
    samples = data.get("samples", [])
    
    # Transform each sample
    vqa_samples = []
    for sample in samples:
        vqa_sample = transform_sample_to_vqa(sample)
        vqa_samples.append(vqa_sample)
    
    # Build output dataset
    output_data = {
        "question_id": data.get("question_id", ""),
        "question": data.get("question", ""),
        "num_samples": len(vqa_samples),
        "samples": vqa_samples,
        "metadata": {
            "source_file": str(input_path),
            "format": "standard_vqa",
            "transformed_from": "mmvqa_multiple_choice",
        }
    }
    
    # Add original metadata if present
    if "metadata" in data:
        output_data["metadata"]["original_metadata"] = data["metadata"]
    
    # Save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transformed {len(vqa_samples)} samples: {input_path.name} → {output_path.name}")
    
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "num_samples": len(vqa_samples),
    }


def transform_per_image_all_file(
    input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Transform per_image_all.json from MMVQA to standard VQA format.
    
    Args:
        input_path: Path to MMVQA per_image_all.json
        output_path: Path to output standard VQA per_image_all.json
    
    Returns:
        Statistics dict
    """
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Data is a list of per-image entries
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {input_path}, got {type(data)}")
    
    # Transform each image entry
    output_entries = []
    total_questions = 0
    
    for entry in data:
        image = entry.get("image", "")
        questions = entry.get("questions", [])
        
        # Transform questions
        vqa_questions = []
        for q in questions:
            vqa_q = transform_sample_to_vqa(q)
            vqa_questions.append(vqa_q)
        
        total_questions += len(vqa_questions)
        
        # Build output entry
        output_entry = {
            "image": image,
            "image_path": entry.get("image_path"),
            "questions": vqa_questions,
        }
        
        # Include metadata if present
        if entry.get("annotation_json"):
            output_entry["annotation_json"] = entry["annotation_json"]
        if entry.get("sources"):
            output_entry["sources"] = entry["sources"]
        
        output_entries.append(output_entry)
    
    # Save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transformed {len(output_entries)} images ({total_questions} questions): {input_path.name} → {output_path.name}")
    
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "num_images": len(output_entries),
        "num_questions": total_questions,
    }


def transform_directory(
    input_dir: str,
    output_dir: str,
    per_question_only: bool = False,
) -> Dict[str, Any]:
    """
    Transform an entire directory of MMVQA files to standard VQA format.
    
    Args:
        input_dir: Directory with MMVQA JSON files
        output_dir: Directory to save standard VQA files
        per_question_only: If True, only transform per-question files (skip per_image_all.json)
    
    Returns:
        Summary dict with statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Transforming MMVQA → Standard VQA")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {output_path}")
    
    stats = []
    
    # Find per-question files (e.g. action_command.json, main_obstacle.json, etc.)
    per_question_files = [
        "action_command.json",
        "main_obstacle.json",
        "closest_obstacle.json",
        "risk_assessment.json",
        "spatial_clock.json",
        "action_suggestion.json",
    ]
    
    for filename in per_question_files:
        input_file = input_path / filename
        if not input_file.exists():
            logger.warning(f"Per-question file not found: {filename}")
            continue
        
        output_file = output_path / filename
        stat = transform_per_question_file(input_file, output_file)
        stats.append(stat)
    
    # Transform per_image_all.json if present and not skipped
    if not per_question_only:
        per_image_all = input_path / "per_image_all.json"
        if per_image_all.exists():
            output_per_image_all = output_path / "per_image_all.json"
            stat = transform_per_image_all_file(per_image_all, output_per_image_all)
            stats.append(stat)
    
    # Save summary
    summary = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "transformed_files": len(stats),
        "files": stats,
    }
    
    summary_path = output_path / "transformation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Transformation complete!")
    logger.info(f"  Transformed {len(stats)} files")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"{'='*60}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Transform MMVQA (multiple choice) to standard VQA format"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with MMVQA JSON files (per-question or per_image_all.json)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save standard VQA JSON files"
    )
    parser.add_argument(
        "--per-question-only",
        action="store_true",
        help="Only transform per-question files (skip per_image_all.json)"
    )
    
    args = parser.parse_args()
    
    transform_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        per_question_only=args.per_question_only,
    )
