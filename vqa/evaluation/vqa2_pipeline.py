"""VQA2 Pipeline: Transform MMVQA to Standard VQA and Evaluate Models.

This is the main entry point for:
1. Transforming MMVQA (multiple choice) to standard VQA format
2. Evaluating multiple VLM models on the standard VQA dataset

Usage:
    # Full pipeline: transform + evaluate
    python vqa2_pipeline.py \\
        --mmvqa-dir path/to/mmvqa \\
        --vqa-output-dir path/to/standard_vqa \\
        --images-dir path/to/images \\
        --eval-output-dir path/to/results \\
        --models florence2 qwen llava gpt4o
    
    # Transform only
    python vqa2_pipeline.py \\
        --mmvqa-dir path/to/mmvqa \\
        --vqa-output-dir path/to/standard_vqa \\
        --transform-only
    
    # Evaluate only (assumes VQA dataset already exists)
    python vqa2_pipeline.py \\
        --vqa-dataset path/to/standard_vqa/action_command.json \\
        --images-dir path/to/images \\
        --eval-output-dir path/to/results \\
        --models florence2 qwen
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_transformation(
    mmvqa_dir: str,
    vqa_output_dir: str,
    per_question_only: bool = False,
) -> Dict[str, Any]:
    """
    Transform MMVQA to standard VQA format.
    
    Args:
        mmvqa_dir: Directory with MMVQA JSON files
        vqa_output_dir: Directory to save standard VQA files
        per_question_only: Only transform per-question files
    
    Returns:
        Transformation summary
    """
    logger.info("="*60)
    logger.info("STEP 1: Transform MMVQA → Standard VQA")
    logger.info("="*60)
    
    from .transform_mmvqa_to_vqa import transform_directory
    
    summary = transform_directory(
        input_dir=mmvqa_dir,
        output_dir=vqa_output_dir,
        per_question_only=per_question_only,
    )
    
    return summary


def run_evaluation(
    vqa_dataset_path: str,
    images_dir: str,
    output_dir: str,
    models: List[str],
    api_key: Optional[str] = None,
    device: str = "auto",
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate multiple models on standard VQA dataset.
    
    Args:
        vqa_dataset_path: Path to VQA JSON dataset
        images_dir: Directory with images
        output_dir: Output directory for results
        models: List of model names (e.g. ["florence2", "qwen", "llava"])
        api_key: API key for API-based models
        device: Device for local models
        max_samples: Limit number of samples
        verbose: Verbose logging
    
    Returns:
        Evaluation results for all models
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Evaluate Models on Standard VQA")
    logger.info("="*60)
    
    from .vqa_standard_evaluation import run_multi_model_evaluation
    
    # Build model configs
    model_configs = []
    for model_name in models:
        config = {
            "name": model_name,
            "type": model_name,
            "device": device,
        }
        
        # Add API key for API-based models
        if "gpt" in model_name.lower() or "openrouter" in model_name.lower():
            if api_key:
                config["api_key"] = api_key
            else:
                logger.warning(f"No API key provided for {model_name}, skipping...")
                continue
        
        model_configs.append(config)
    
    if not model_configs:
        raise ValueError("No valid model configs. Please provide models and API keys if needed.")
    
    results = run_multi_model_evaluation(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_base_dir=output_dir,
        models=model_configs,
        max_samples=max_samples,
        save_predictions=True,
        verbose=verbose,
    )
    
    return results


def run_full_pipeline(
    mmvqa_dir: str,
    vqa_output_dir: str,
    images_dir: str,
    eval_output_dir: str,
    models: List[str],
    api_key: Optional[str] = None,
    device: str = "auto",
    max_samples: Optional[int] = None,
    per_question_only: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run full pipeline: transform + evaluate.
    
    Args:
        mmvqa_dir: Directory with MMVQA JSON files
        vqa_output_dir: Directory to save standard VQA files
        images_dir: Directory with images
        eval_output_dir: Output directory for evaluation results
        models: List of model names
        api_key: API key for API-based models
        device: Device for local models
        max_samples: Limit number of samples
        per_question_only: Only transform per-question files
        verbose: Verbose logging
    
    Returns:
        Full pipeline results
    """
    # Step 1: Transform
    transform_summary = run_transformation(
        mmvqa_dir=mmvqa_dir,
        vqa_output_dir=vqa_output_dir,
        per_question_only=per_question_only,
    )
    
    # Step 2: Evaluate on per_image_all.json (or first per-question file)
    vqa_path = Path(vqa_output_dir)
    
    # Prefer per_image_all.json for batch evaluation
    if (vqa_path / "per_image_all.json").exists():
        vqa_dataset_path = str(vqa_path / "per_image_all.json")
    else:
        # Fall back to first per-question file
        per_question_files = [
            "action_command.json",
            "main_obstacle.json",
            "closest_obstacle.json",
            "risk_assessment.json",
            "spatial_clock.json",
            "action_suggestion.json",
        ]
        vqa_dataset_path = None
        for filename in per_question_files:
            if (vqa_path / filename).exists():
                vqa_dataset_path = str(vqa_path / filename)
                break
        
        if not vqa_dataset_path:
            raise FileNotFoundError(f"No VQA dataset found in {vqa_output_dir}")
    
    logger.info(f"\nUsing VQA dataset: {vqa_dataset_path}")
    
    eval_results = run_evaluation(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_dir=eval_output_dir,
        models=models,
        api_key=api_key,
        device=device,
        max_samples=max_samples,
        verbose=verbose,
    )
    
    return {
        "transform_summary": transform_summary,
        "evaluation_results": eval_results,
    }


def get_default_models() -> List[str]:
    """Get list of default models to evaluate."""
    return [
        "florence2",      # Fastest, local, GPU
        "qwen",          # Fast, local, GPU
        "llava",         # Medium, local, GPU
        # API-based models (require API key):
        # "gpt4o",
        # "gpt5nano",
        # "openrouter_qwen3_vl_235b",
    ]


def get_all_available_models() -> List[str]:
    """Get list of all available models."""
    return [
        # Local models (GPU)
        "florence2",
        "qwen",
        "llava",
        # API models (OpenAI)
        "gpt4o",
        "gpt5nano",
        "gpt5mini",
        # API models (OpenRouter)
        "openrouter_trinity",
        "openrouter_llama32_11b_vision",
        "openrouter_llama4_maverick",
        "openrouter_molmo_8b",
        "openrouter_ministral_3b",
        "openrouter_qwen3_vl_235b",
        "openrouter_qwen3_vl_8b",
        "openrouter_qwen_vl_plus",
    ]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VQA2 Pipeline: Transform MMVQA to Standard VQA and Evaluate Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (transform + evaluate with local models)
  python vqa2_pipeline.py \\
      --mmvqa-dir ./vqa_mmvqa \\
      --vqa-output-dir ./vqa_standard \\
      --images-dir ./images \\
      --eval-output-dir ./vqa_results \\
      --models florence2 qwen llava

  # Full pipeline (with API models)
  python vqa2_pipeline.py \\
      --mmvqa-dir ./vqa_mmvqa \\
      --vqa-output-dir ./vqa_standard \\
      --images-dir ./images \\
      --eval-output-dir ./vqa_results \\
      --models florence2 gpt4o \\
      --api-key sk-...

  # Transform only
  python vqa2_pipeline.py \\
      --mmvqa-dir ./vqa_mmvqa \\
      --vqa-output-dir ./vqa_standard \\
      --transform-only

  # Evaluate only (VQA dataset already exists)
  python vqa2_pipeline.py \\
      --vqa-dataset ./vqa_standard/per_image_all.json \\
      --images-dir ./images \\
      --eval-output-dir ./vqa_results \\
      --models florence2 qwen

  # List available models
  python vqa2_pipeline.py --list-models
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--transform-only",
        action="store_true",
        help="Only transform MMVQA to standard VQA (skip evaluation)"
    )
    mode_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate models (skip transformation, requires --vqa-dataset)"
    )
    mode_group.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    # Transformation args
    parser.add_argument(
        "--mmvqa-dir",
        help="Directory with MMVQA JSON files (required for transformation)"
    )
    parser.add_argument(
        "--vqa-output-dir",
        help="Directory to save standard VQA files (required for transformation)"
    )
    parser.add_argument(
        "--per-question-only",
        action="store_true",
        help="Only transform per-question files (skip per_image_all.json)"
    )
    
    # Evaluation args
    parser.add_argument(
        "--vqa-dataset",
        help="Path to VQA JSON dataset (required for eval-only mode)"
    )
    parser.add_argument(
        "--images-dir",
        help="Directory with images (required for evaluation)"
    )
    parser.add_argument(
        "--eval-output-dir",
        help="Output directory for evaluation results (required for evaluation)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (e.g. florence2 qwen llava gpt4o). Use --list-models to see all."
    )
    
    # Model args
    parser.add_argument(
        "--api-key",
        help="API key for API-based models (GPT-4o, OpenRouter, etc.)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for local models (default: auto)"
    )
    
    # Other args
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # List models
    if args.list_models:
        print("\n" + "="*60)
        print("Available Models")
        print("="*60)
        print("\nLocal Models (GPU, FREE):")
        for model in ["florence2", "qwen", "llava"]:
            print(f"  - {model}")
        print("\nAPI Models (OpenAI, PAID):")
        for model in ["gpt4o", "gpt5nano", "gpt5mini"]:
            print(f"  - {model}")
        print("\nAPI Models (OpenRouter, PAID/FREE):")
        for model in get_all_available_models():
            if model.startswith("openrouter_"):
                print(f"  - {model}")
        print("\nDefault models (--models not specified):")
        for model in get_default_models():
            print(f"  - {model}")
        print()
        return 0
    
    # Validate args based on mode
    if args.transform_only:
        if not args.mmvqa_dir or not args.vqa_output_dir:
            parser.error("--transform-only requires --mmvqa-dir and --vqa-output-dir")
        
        run_transformation(
            mmvqa_dir=args.mmvqa_dir,
            vqa_output_dir=args.vqa_output_dir,
            per_question_only=args.per_question_only,
        )
        
        logger.info("\n" + "="*60)
        logger.info("Transformation complete!")
        logger.info(f"Output: {args.vqa_output_dir}")
        logger.info("="*60)
        return 0
    
    elif args.eval_only:
        if not args.vqa_dataset or not args.images_dir or not args.eval_output_dir:
            parser.error("--eval-only requires --vqa-dataset, --images-dir, and --eval-output-dir")
        
        models = args.models or get_default_models()
        
        run_evaluation(
            vqa_dataset_path=args.vqa_dataset,
            images_dir=args.images_dir,
            output_dir=args.eval_output_dir,
            models=models,
            api_key=args.api_key,
            device=args.device,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation complete!")
        logger.info(f"Results: {args.eval_output_dir}")
        logger.info("="*60)
        return 0
    
    else:
        # Full pipeline
        if not args.mmvqa_dir or not args.vqa_output_dir or not args.images_dir or not args.eval_output_dir:
            parser.error("Full pipeline requires --mmvqa-dir, --vqa-output-dir, --images-dir, and --eval-output-dir")
        
        models = args.models or get_default_models()
        
        run_full_pipeline(
            mmvqa_dir=args.mmvqa_dir,
            vqa_output_dir=args.vqa_output_dir,
            images_dir=args.images_dir,
            eval_output_dir=args.eval_output_dir,
            models=models,
            api_key=args.api_key,
            device=args.device,
            max_samples=args.max_samples,
            per_question_only=args.per_question_only,
            verbose=args.verbose,
        )
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline complete!")
        logger.info(f"VQA Dataset: {args.vqa_output_dir}")
        logger.info(f"Results: {args.eval_output_dir}")
        logger.info("="*60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
