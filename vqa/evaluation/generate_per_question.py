"""Generate VQA datasets grouped by question type (one file per question).

Instead of one JSON per image, this outputs one file per question type containing
all images' Q&A for that question. For 7k images and 6 question types, you get 6 files.

Output files:
- main_obstacle.json (Question: What is the main obstacle?)
- closest_obstacle.json (Question: Which object is closest?)
- risk_assessment.json (Question: How safe is this scene?)
- spatial_clock.json (Question: Locate the X based on clock direction)
- action_suggestion.json (Question: What action do you suggest?)
- action_command.json (Question: What is the recommended navigation action?)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from annotation.nav_parse import parse_navigation_response, ParsedNavigation
from annotation.schemas import AnnotationRecord

from .ground_truth_model import apply_ground_truth_model
from .navigation_core import build_navigation_core_samples
from .action_command import build_action_command_mcq
from .schemas import VQAMultipleChoiceSample

logger = logging.getLogger(__name__)


@dataclass
class PerQuestionConfig:
    """Configuration for per-question VQA generation."""

    annotations_dir: str
    output_dir: str
    images_dir: Optional[str] = None
    max_samples: Optional[int] = None
    seed: int = 1337
    verbose: bool = False
    ground_truth_model: Optional[str] = None  # e.g. openrouter_qwen3_vl_235b; run VLM to set reference answers
    per_image: bool = False  # If true, write per-image JSON files with all questions
    per_image_dir: Optional[str] = None  # Optional override for per-image output directory
    skip_existing_per_image: bool = False  # If true, reuse existing per-image JSONs and do not overwrite


class PerQuestionVQAGenerator:
    """Generate VQA datasets organized by question type."""
    
    def __init__(self, config: PerQuestionConfig):
        self.config = config
        
        # Storage for questions grouped by question_id
        self.questions_by_type: Dict[str, List[Dict[str, Any]]] = {
            "main_obstacle": [],
            "closest_obstacle": [],
            "risk_assessment": [],
            "spatial_clock": [],
            "action_suggestion": [],
            "action_command": [],
        }
        self.per_image_records: List[Dict[str, Any]] = []
    
    def _read_annotation(self, path: Path) -> Optional[AnnotationRecord]:
        """Read and validate annotation JSON."""
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return AnnotationRecord.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
            return None
    
    def _parse_navigation(self, rec: AnnotationRecord) -> Optional[ParsedNavigation]:
        """Extract navigation data from annotation."""
        try:
            if not rec.accessibility:
                return None
            
            # Build navigation text from accessibility fields
            navigation_text = self._build_navigation_text(rec)
            parsed = parse_navigation_response(navigation_text)
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse navigation for {rec.image}: {e}")
            return None
    
    def _build_navigation_text(self, rec: AnnotationRecord) -> str:
        """Build navigation text from structured annotation."""
        parts = []
        
        if rec.accessibility:
            acc = rec.accessibility
            
            # Scene
            if acc.scene_description:
                parts.append(f"Scene: {acc.scene_description}")
            elif acc.ground_text:
                parts.append(f"Scene: {acc.ground_text}")
            
            # Risk
            if acc.risk_assessment:
                risk = acc.risk_assessment
                level = risk.level or "Medium"
                reason = risk.reason or ""
                parts.append(f"Risk: {level}")
                if reason:
                    parts.append(f"  {reason}")
            
            # Obstacles
            if acc.spatial_objects:
                parts.append("Obstacles:")
                for obj in acc.spatial_objects:
                    parts.append(f"  - {obj}")
            elif acc.risk_assessment and acc.risk_assessment.obstacles:
                parts.append("Obstacles:")
                for obs in acc.risk_assessment.obstacles:
                    obj_name = obs.get("object", "unknown")
                    position = obs.get("position", "")
                    distance = obs.get("distance", "")
                    parts.append(f"  - {obj_name} ({position}, {distance})")
            
            # Guidance
            if acc.guidance:
                parts.append(f"Guidance: {acc.guidance}")
        
        return "\n".join(parts)
    
    def _sample_to_dict(self, sample: VQAMultipleChoiceSample) -> Dict[str, Any]:
        """Convert sample to a flat dict for JSON output."""
        sample_dict = {
            "id": sample.id,
            "image": sample.image,
            "question_id": sample.question_id,
            "question": sample.question,
            "options": {str(k): v for k, v in sample.options.items()},
            "answer": str(sample.answer),
            "answer_text": sample.answer_text,
            "ground_truth": sample.ground_truth,
        }

        # Add image path if available
        if sample.image_abs_path:
            sample_dict["image_path"] = sample.image_abs_path

        # Add feedback if present
        if sample.feedback:
            sample_dict["feedback"] = {str(k): v for k, v in sample.feedback.items()}

        return sample_dict

    def _add_sample(self, sample: VQAMultipleChoiceSample):
        """Add sample to appropriate question type list."""
        qid = sample.question_id
        if qid in self.questions_by_type:
            self.questions_by_type[qid].append(self._sample_to_dict(sample))
    
    @staticmethod
    def _resolve_annotations_dir(p: Path) -> Path:
        """Accept either the Phase-5 output dir or its `annotations/` child."""
        if p.is_dir() and (p / "annotations").exists():
            return p / "annotations"
        return p

    def generate(self) -> Dict[str, Any]:
        """Generate VQA datasets grouped by question type."""
        annotations_dir = Path(self.config.annotations_dir)
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
        annotations_dir = self._resolve_annotations_dir(annotations_dir)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        per_image_dir: Optional[Path] = None
        if self.config.per_image:
            per_image_dir = Path(self.config.per_image_dir) if self.config.per_image_dir else (output_dir / "per_image")
            per_image_dir.mkdir(parents=True, exist_ok=True)

        images_dir = Path(self.config.images_dir) if self.config.images_dir else None
        if images_dir and not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            images_dir = None

        # Find all annotation files
        annotation_files = sorted(annotations_dir.glob("*.json"))
        if not annotation_files:
            raise RuntimeError(f"No JSON files found in {annotations_dir}")
        
        if self.config.max_samples:
            annotation_files = annotation_files[:self.config.max_samples]
        
        logger.info(f"Processing {len(annotation_files)} annotations...")
        
        processed = 0
        failed = 0
        
        for ann_path in tqdm(annotation_files, desc="Generating VQA"):
            try:
                # Read annotation
                rec = self._read_annotation(ann_path)
                if not rec:
                    failed += 1
                    continue

                # Fast path: reuse existing per-image VQA JSON (no overwrite, no new model calls)
                if self.config.per_image and self.config.skip_existing_per_image and per_image_dir is not None:
                    image_stem = Path(rec.image).stem
                    existing_path = per_image_dir / f"{image_stem}.json"
                    if existing_path.exists():
                        try:
                            with existing_path.open("r", encoding="utf-8") as f:
                                existing_entry = json.load(f)
                            # Re-add questions to per-question buckets
                            for q in existing_entry.get("questions", []):
                                qid = q.get("question_id")
                                if qid in self.questions_by_type:
                                    self.questions_by_type[qid].append(q)
                            # Re-add per-image entry for all-in-one file
                            self.per_image_records.append(existing_entry)
                            processed += 1
                            continue
                        except Exception as e:
                            logger.warning(f"Failed to reuse per-image VQA for {rec.image}: {e}")
                            # fall through to normal processing
                
                # Parse navigation
                parsed = self._parse_navigation(rec)
                if not parsed:
                    failed += 1
                    continue
                
                # Get image path if images_dir provided
                image_abs_path = None
                if images_dir:
                    image_name = rec.image
                    image_path = images_dir / image_name
                    if image_path.exists():
                        image_abs_path = str(image_path)
                
                # Build sources dict
                sources = dict(rec.sources) if rec.sources else {}
                
                # Generate 5 core navigation questions
                nav_samples = build_navigation_core_samples(
                    image=rec.image,
                    parsed=parsed,
                    sources=sources,
                    image_abs_path=image_abs_path,
                    base_seed=self.config.seed,
                )
                all_samples_for_image = list(nav_samples)

                # Generate action-command question
                action_sample = build_action_command_mcq(
                    image=rec.image,
                    parsed=parsed,
                    sources=sources,
                    image_abs_path=image_abs_path,
                    base_seed=self.config.seed,
                )
                all_samples_for_image.append(action_sample)

                if self.config.ground_truth_model:
                    api_key = None
                    if "openrouter" in (self.config.ground_truth_model or "").lower():
                        try:
                            from config import get_openrouter_api_key
                            api_key = get_openrouter_api_key()
                        except Exception:
                            pass
                    apply_ground_truth_model(
                        all_samples_for_image,
                        self.config.ground_truth_model,
                        images_dir=str(images_dir) if images_dir else None,
                        api_key=api_key,
                    )

                for sample in all_samples_for_image:
                    self._add_sample(sample)

                if self.config.per_image:
                    image_entry = {
                        "image": rec.image,
                        "image_path": image_abs_path,
                        "annotation_json": str(ann_path),
                        "sources": sources,
                        "questions": [self._sample_to_dict(s) for s in all_samples_for_image],
                    }
                    self.per_image_records.append(image_entry)
                    if per_image_dir is not None:
                        image_name = Path(rec.image).stem
                        image_path = per_image_dir / f"{image_name}.json"
                        with image_path.open("w", encoding="utf-8") as f:
                            json.dump(image_entry, f, indent=2)
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {ann_path.name}: {e}")
                failed += 1
        
        # Save per-question files
        logger.info("Saving per-question datasets...")
        saved_files = {}
        
        for question_id, samples in self.questions_by_type.items():
            if not samples:
                continue
            
            output_file = output_dir / f"{question_id}.json"
            
            # Build dataset
            dataset = {
                "question_id": question_id,
                "question": samples[0]["question"] if samples else "",
                "num_samples": len(samples),
                "samples": samples,
                "metadata": {
                    "generated_from": str(annotations_dir),
                    "total_images": processed,
                    "seed": self.config.seed,
                }
            }
            
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
            
            saved_files[question_id] = str(output_file)
            logger.info(f"  {question_id}: {len(samples)} samples → {output_file.name}")
        
        # Save summary
        summary = {
            "processed": processed,
            "failed": failed,
            "total_samples_per_question": {k: len(v) for k, v in self.questions_by_type.items()},
            "output_files": saved_files,
            "config": {
                "annotations_dir": str(annotations_dir),
                "output_dir": str(output_dir),
                "images_dir": str(images_dir) if images_dir else None,
                "seed": self.config.seed,
            }
        }

        if self.config.per_image:
            all_in_one_path = output_dir / "per_image_all.json"
            with all_in_one_path.open("w", encoding="utf-8") as f:
                json.dump(self.per_image_records, f, indent=2)
            summary["per_image_dir"] = str(per_image_dir) if per_image_dir else None
            summary["per_image_all"] = str(all_in_one_path)
        
        summary_path = output_dir / "vqa_generation_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"VQA generation complete!")
        logger.info(f"  Processed: {processed} annotations")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*60}")
        
        return summary


def generate_per_question_vqa(
    annotations_dir: str,
    output_dir: str,
    images_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 1337,
    verbose: bool = False,
    ground_truth_model: Optional[str] = None,
    per_image: bool = False,
    per_image_dir: Optional[str] = None,
     skip_existing_per_image: bool = False,
) -> Dict[str, Any]:
    """Convenience function to generate per-question VQA datasets.

    Args:
        annotations_dir: Directory with annotation JSON files
        output_dir: Where to save question-type files
        images_dir: Optional directory with images (for absolute paths; required if ground_truth_model is set)
        max_samples: Limit number of annotations to process
        seed: Random seed for deterministic generation
        verbose: Verbose logging
        ground_truth_model: Optional VLM (e.g. openrouter_qwen3_vl_235b) to set reference answers
        per_image: If true, write per-image JSON files (one per image)
        per_image_dir: Optional override for per-image output directory
        skip_existing_per_image: If true, reuse existing per-image JSONs and do not overwrite them

    Returns:
        Summary dict with stats and output paths
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = PerQuestionConfig(
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        images_dir=images_dir,
        max_samples=max_samples,
        seed=seed,
        verbose=verbose,
        ground_truth_model=ground_truth_model,
        per_image=per_image,
        per_image_dir=per_image_dir,
        skip_existing_per_image=skip_existing_per_image,
    )
    
    generator = PerQuestionVQAGenerator(config)
    return generator.generate()
