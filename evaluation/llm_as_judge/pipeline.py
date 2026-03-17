"""Main evaluation pipeline for LLM-as-judge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .schemas import EvaluationConfig, EvaluationResult
from .judge_models import create_judge, BaseLLMJudge
from .visualization import generate_all_plots

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main pipeline for running LLM-as-judge evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation pipeline.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.judges: List[BaseLLMJudge] = []
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize judge models
        self._initialize_judges()
    
    def _initialize_judges(self) -> None:
        """Initialize all judge models from config."""
        for model_name in self.config.judge_models:
            try:
                judge = create_judge(
                    model_name=model_name,
                    openai_api_key=self.config.openai_api_key,
                    anthropic_api_key=self.config.anthropic_api_key,
                    openrouter_api_key=self.config.openrouter_api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                self.judges.append(judge)
                logger.info(f"Initialized judge: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize judge {model_name}: {e}")
                raise
    
    def _load_annotations(self) -> List[dict]:
        """Load annotation files from annotations directory.
        
        Returns:
            List of annotation dictionaries with 'image_name' and 'content' keys
        """
        annotations_dir = Path(self.config.annotations_dir)
        
        # Look for JSON files
        json_files = sorted(annotations_dir.glob("**/*_annotation.json"))
        
        if not json_files:
            # Try finding any JSON files
            json_files = sorted(annotations_dir.glob("**/*.json"))
        
        if not json_files:
            raise ValueError(f"No annotation JSON files found in {annotations_dir}")
        
        annotations = []
        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract annotation content
                annotation_text = self._extract_annotation_text(data)
                
                if annotation_text:
                    annotations.append({
                        "image_name": json_file.stem.replace("_annotation", ""),
                        "file_path": str(json_file),
                        "content": annotation_text,
                    })
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        if self.config.max_annotations:
            annotations = annotations[:self.config.max_annotations]
        
        logger.info(f"Loaded {len(annotations)} annotations from {annotations_dir}")
        return annotations
    
    def _extract_annotation_text(self, annotation_data: dict) -> Optional[str]:
        """Extract annotation text from JSON structure.
        
        Args:
            annotation_data: Loaded annotation JSON
            
        Returns:
            Extracted annotation text or None
        """
        # Try different common keys
        possible_keys = [
            "accessibility_description",
            "scene_description",
            "description",
            "annotation",
            "caption",
            "navigation",
            "navigation_instruction",
        ]
        
        for key in possible_keys:
            if key in annotation_data and annotation_data[key]:
                return str(annotation_data[key])
        
        # If navigation object exists
        if "navigation" in annotation_data and isinstance(annotation_data["navigation"], dict):
            nav = annotation_data["navigation"]
            parts = []
            if "action" in nav:
                parts.append(f"Action: {nav['action']}")
            if "reasoning" in nav:
                parts.append(f"Reasoning: {nav['reasoning']}")
            if "obstacles" in nav:
                parts.append(f"Obstacles: {nav['obstacles']}")
            if parts:
                return "\n".join(parts)
        
        # Fallback: use entire JSON as string
        return json.dumps(annotation_data, indent=2)
    
    def _should_skip(self, image_name: str, model_name: str) -> bool:
        """Check if evaluation should be skipped (already exists).
        
        Args:
            image_name: Name of image being evaluated
            model_name: Name of judge model
            
        Returns:
            True if should skip, False otherwise
        """
        if not self.config.skip_existing:
            return False
        
        result_file = self.results_dir / f"{image_name}_{model_name.replace('/', '_')}.json"
        return result_file.exists()
    
    def _save_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to JSON file.
        
        Args:
            result: Evaluation result to save
        """
        if not self.config.save_json:
            return
        
        filename = f"{result.image_name}_{result.model_name.replace('/', '_')}.json"
        output_path = self.results_dir / filename
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2)
    
    def run(self) -> pd.DataFrame:
        """Run complete evaluation pipeline.
        
        Returns:
            DataFrame with all evaluation results
        """
        logger.info("Starting LLM-as-judge evaluation pipeline")
        
        # Load annotations
        annotations = self._load_annotations()
        
        if not annotations:
            raise ValueError("No annotations found to evaluate")
        
        # Run evaluation
        all_results: List[EvaluationResult] = []
        skipped = 0
        failed = 0
        
        total_evaluations = len(annotations) * len(self.judges)
        
        with tqdm(total=total_evaluations, desc="Evaluating annotations") as pbar:
            for annotation in annotations:
                image_name = annotation["image_name"]
                content = annotation["content"]
                
                for judge in self.judges:
                    if self._should_skip(image_name, judge.model_name):
                        skipped += 1
                        pbar.update(1)
                        continue
                    
                    try:
                        result = judge.evaluate(image_name, content)
                        self._save_result(result)
                        all_results.append(result)
                        
                        if self.config.verbose:
                            logger.info(
                                f"[{judge.model_name}] {image_name}: "
                                f"Overall={result.overall_score:.2f}"
                            )
                    
                    except Exception as e:
                        failed += 1
                        logger.error(f"Evaluation failed for {image_name} with {judge.model_name}: {e}")
                    
                    finally:
                        pbar.update(1)
        
        logger.info(
            f"Evaluation complete: {len(all_results)} successful, "
            f"{skipped} skipped, {failed} failed"
        )
        
        if not all_results:
            raise ValueError("No successful evaluations")
        
        # Convert to DataFrame
        results_df = self._results_to_dataframe(all_results)
        
        # Save results
        self._save_outputs(results_df)
        
        # Generate visualizations
        if self.config.generate_plots:
            try:
                generate_all_plots(results_df, self.plots_dir)
            except Exception as e:
                logger.error(f"Failed to generate plots: {e}", exc_info=True)
        
        return results_df
    
    def _results_to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert evaluation results to pandas DataFrame.
        
        Args:
            results: List of evaluation results
            
        Returns:
            DataFrame with flattened results
        """
        flat_results = [result.to_flat_dict() for result in results]
        df = pd.DataFrame(flat_results)
        return df
    
    def _save_outputs(self, df: pd.DataFrame) -> None:
        """Save evaluation results to CSV and Excel files.
        
        Args:
            df: Results DataFrame
        """
        # Save CSV
        if self.config.save_csv:
            csv_path = self.output_dir / "evaluation_results.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved CSV results to {csv_path}")
        
        # Save Excel
        if self.config.save_excel:
            try:
                excel_path = self.output_dir / "evaluation_results.xlsx"
                
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    # Main results
                    df.to_excel(writer, sheet_name="Results", index=False)
                    
                    # Summary statistics
                    summary_df = self._create_summary_stats(df)
                    summary_df.to_excel(writer, sheet_name="Summary", index=True)
                    
                    # Per-model statistics
                    if "model" in df.columns and df["model"].nunique() > 1:
                        model_stats = df.groupby("model").agg({
                            "clarity": ["mean", "std"],
                            "completeness": ["mean", "std"],
                            "robustness": ["mean", "std"],
                            "user_friendliness": ["mean", "std"],
                            "accuracy": ["mean", "std"],
                            "overall_score": ["mean", "std"],
                        }).round(3)
                        model_stats.to_excel(writer, sheet_name="Model Comparison")
                
                logger.info(f"Saved Excel results to {excel_path}")
            except Exception as e:
                logger.error(f"Failed to save Excel file: {e}")
    
    def _create_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics DataFrame.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
        available = [c for c in criteria if c in df.columns]
        
        stats = df[available].agg(["mean", "std", "min", "max", "median"]).round(3)
        return stats


def run_evaluation(
    annotations_dir: str,
    output_dir: str,
    judge_models: List[str],
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    max_annotations: Optional[int] = None,
    skip_existing: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convenience function to run evaluation pipeline.
    
    Args:
        annotations_dir: Directory containing annotation JSON files
        output_dir: Output directory for results
        judge_models: List of judge model names
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        openrouter_api_key: OpenRouter API key
        max_annotations: Limit number of annotations
        skip_existing: Skip already-evaluated annotations
        temperature: LLM sampling temperature
        max_tokens: Max tokens for LLM response
        verbose: Verbose logging
        
    Returns:
        DataFrame with evaluation results
    """
    config = EvaluationConfig(
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        judge_models=judge_models,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        openrouter_api_key=openrouter_api_key,
        max_annotations=max_annotations,
        skip_existing=skip_existing,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )
    
    pipeline = EvaluationPipeline(config)
    return pipeline.run()
