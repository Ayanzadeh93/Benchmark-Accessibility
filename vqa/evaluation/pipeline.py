"""VQA evaluation dataset generation pipeline.

Consumes Phase-5 annotation outputs and generates evaluation datasets.

Current scope:
- Generate 5 core navigation MCQs (A/B/C/D) per annotation.
- Emit both tuning/evaluation splits for downstream usage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation.nav_parse import parse_navigation_response
from annotation.schemas import AnnotationRecord

from .ground_truth_model import apply_ground_truth_model
from .navigation_core import build_navigation_core_samples
from .schemas import VQADataset, VQAMultipleChoiceSample

logger = logging.getLogger(__name__)


@dataclass
class VQAEvalRunConfig:
    """Configuration for VQA evaluation dataset generation."""

    annotations_dir: str
    output_dir: Optional[str] = None  # If None, inferred as <run>/VQA/evaluation
    images_dir: Optional[str] = None  # Optional; used to construct absolute image paths (required for ground_truth_model)
    max_samples: Optional[int] = None
    seed: int = 1337
    tuning_split: float = 0.2
    ground_truth_model: Optional[str] = None  # e.g. openrouter_qwen3_vl_235b; run VLM to set reference answers


class VQAEvaluationPipeline:
    """Generate VQA evaluation datasets from annotation records."""

    def __init__(self, cfg: VQAEvalRunConfig):
        self.cfg = cfg

    @staticmethod
    def _resolve_annotations_dir(p: Path) -> Path:
        """Accept either the Phase-5 output dir or its `annotations/` child."""
        if p.is_dir() and (p / "annotations").exists():
            return p / "annotations"
        return p

    @staticmethod
    def _infer_output_dir(phase5_out_dir: Path) -> Path:
        """Infer output directory as sibling: <run>/VQA/evaluation."""
        # If user passed <run>/Annotation..., then parent is <run>.
        run_dir = phase5_out_dir.parent
        return run_dir / "VQA" / "evaluation"

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def process(self) -> Tuple[Path, Dict[str, Any]]:
        """Generate the dataset and return (output_dir, summary)."""
        ann_in = Path(self.cfg.annotations_dir)
        if not ann_in.exists():
            raise FileNotFoundError(f"annotations-dir not found: {ann_in}")

        annotations_dir = self._resolve_annotations_dir(ann_in)
        if not annotations_dir.exists():
            raise FileNotFoundError(f"annotations directory not found: {annotations_dir}")

        # Determine Phase-5 output dir (parent of annotations/)
        phase5_out_dir = annotations_dir.parent if annotations_dir.name == "annotations" else annotations_dir

        out_dir = Path(self.cfg.output_dir) if self.cfg.output_dir else self._infer_output_dir(phase5_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images_dir = Path(self.cfg.images_dir) if self.cfg.images_dir else None
        if images_dir is not None and not images_dir.exists():
            logger.warning(f"images-dir does not exist: {images_dir} (will omit image_abs_path)")
            images_dir = None

        if not (0.0 <= float(self.cfg.tuning_split) <= 1.0):
            raise ValueError("tuning_split must be between 0 and 1")

        files = sorted([p for p in annotations_dir.glob("*.json") if p.is_file()])
        if not files:
            raise RuntimeError(f"No annotation JSON files found in: {annotations_dir}")

        samples: List[VQAMultipleChoiceSample] = []
        failures: List[Dict[str, Any]] = []

        for meta_path in files:
            try:
                payload = self._read_json(meta_path)
                rec = AnnotationRecord.model_validate(payload)
                if str(rec.task).lower().strip() != "navigation":
                    continue

                parsed = parse_navigation_response(rec.text)
                if parsed is None:
                    raise ValueError("Failed to parse navigation text (Scene/Risk/Obstacles/Guidance).")

                image_abs_path = None
                if images_dir is not None:
                    try:
                        image_abs_path = str((images_dir / rec.image).resolve())
                    except Exception:
                        image_abs_path = None

                new_samples = build_navigation_core_samples(
                    image=str(rec.image),
                    parsed=parsed,
                    sources={**(rec.sources or {}), "annotation_json": str(meta_path)},
                    image_abs_path=image_abs_path,
                    base_seed=int(self.cfg.seed),
                )
                samples.extend(new_samples)
            except Exception as e:
                failures.append({"annotation_json": str(meta_path), "error": str(e)})

            if self.cfg.max_samples is not None and len(samples) >= int(self.cfg.max_samples):
                break

        if self.cfg.ground_truth_model and samples:
            api_key = None
            if "openrouter" in (self.cfg.ground_truth_model or "").lower():
                try:
                    from config import get_openrouter_api_key
                    api_key = get_openrouter_api_key()
                except Exception:
                    pass
            apply_ground_truth_model(
                samples,
                self.cfg.ground_truth_model,
                images_dir=str(images_dir) if images_dir else None,
                api_key=api_key,
            )

        dataset = VQADataset(
            created_at=datetime.now().isoformat(),
            question_set="navigation_core_mcq",
            samples=samples,
            failures=failures,
        )

        dataset_path = out_dir / "navigation_core_mcq_all.json"
        dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

        tune, eval_ = _split_samples(samples, split_ratio=self.cfg.tuning_split, seed=int(self.cfg.seed))
        tuning_dataset = VQADataset(
            created_at=datetime.now().isoformat(),
            question_set="navigation_core_mcq",
            split="tuning",
            samples=tune,
            failures=failures,
        )
        eval_dataset = VQADataset(
            created_at=datetime.now().isoformat(),
            question_set="navigation_core_mcq",
            split="evaluation",
            samples=eval_,
            failures=failures,
        )

        tuning_path = out_dir / "navigation_core_mcq_tuning.json"
        eval_path = out_dir / "navigation_core_mcq_evaluation.json"
        tuning_path.write_text(tuning_dataset.model_dump_json(indent=2), encoding="utf-8")
        eval_path.write_text(eval_dataset.model_dump_json(indent=2), encoding="utf-8")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "annotations_dir": str(annotations_dir),
            "phase5_out_dir": str(phase5_out_dir),
            "output_dir": str(out_dir),
            "dataset_path": str(dataset_path),
            "tuning_path": str(tuning_path),
            "evaluation_path": str(eval_path),
            "num_files": len(files),
            "generated": len(samples),
            "failed": len(failures),
            "failures": failures[:50],
        }

        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out_dir, summary


def _split_samples(
    samples: List[VQAMultipleChoiceSample], *, split_ratio: float, seed: int
) -> Tuple[List[VQAMultipleChoiceSample], List[VQAMultipleChoiceSample]]:
    if split_ratio <= 0:
        return [], list(samples)
    if split_ratio >= 1:
        return list(samples), []

    tune: List[VQAMultipleChoiceSample] = []
    eval_: List[VQAMultipleChoiceSample] = []
    for s in samples:
        h = hashlib.sha256()
        h.update(str(seed).encode("utf-8"))
        h.update(b"|")
        h.update(str(s.id).encode("utf-8", errors="ignore"))
        bucket = int(h.hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < split_ratio:
            tune.append(s)
        else:
            eval_.append(s)
    return tune, eval_

