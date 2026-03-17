"""Example usage of VQA2 pipeline.

This script demonstrates how to:
1. Transform MMVQA to Standard VQA
2. Evaluate multiple models
3. Compare results
"""

from __future__ import annotations

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def example_transform_only():
    """Example: Transform MMVQA to Standard VQA only."""
    from vqa.evaluation.transform_mmvqa_to_vqa import transform_directory
    
    logger.info("="*60)
    logger.info("Example 1: Transform MMVQA to Standard VQA")
    logger.info("="*60)
    
    # Paths (adjust to your setup)
    mmvqa_dir = "./vqa_mmvqa"  # Directory with MMVQA files
    vqa_output_dir = "./vqa_standard"  # Output directory
    
    summary = transform_directory(
        input_dir=mmvqa_dir,
        output_dir=vqa_output_dir,
        per_question_only=False,
    )
    
    logger.info("\nTransformation complete!")
    logger.info(f"Transformed {summary['transformed_files']} files")
    logger.info(f"Output: {summary['output_dir']}")


def example_evaluate_local_models():
    """Example: Evaluate local GPU models."""
    from vqa.evaluation.vqa_standard_evaluation import run_multi_model_evaluation
    
    logger.info("\n" + "="*60)
    logger.info("Example 2: Evaluate Local GPU Models")
    logger.info("="*60)
    
    # Paths (adjust to your setup)
    vqa_dataset_path = "./vqa_standard/per_image_all.json"
    images_dir = "./images"
    output_dir = "./vqa_results_local"
    
    # Models to evaluate
    models = [
        {"name": "florence2", "type": "florence2", "device": "cuda"},
        {"name": "qwen", "type": "qwen", "device": "cuda"},
        # {"name": "llava", "type": "llava", "device": "cuda"},  # Uncomment if you have enough VRAM
    ]
    
    results = run_multi_model_evaluation(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_base_dir=output_dir,
        models=models,
        max_samples=10,  # Test with 10 samples first
        save_predictions=True,
        verbose=False,
    )
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Results: {output_dir}")
    
    # Print summary
    logger.info("\nModel Comparison:")
    for model_name, result in results.items():
        if "error" in result:
            logger.info(f"  {model_name}: ERROR - {result['error']}")
        else:
            metrics = result.get("metrics", {})
            exact_match = metrics.get("exact_match", 0.0)
            rouge1 = metrics.get("rouge1_f1", 0.0)
            logger.info(f"  {model_name}:")
            logger.info(f"    Exact Match: {exact_match:.4f}")
            logger.info(f"    ROUGE-1 F1:  {rouge1:.4f}")


def example_evaluate_api_models():
    """Example: Evaluate API models (GPT-4o, OpenRouter)."""
    from vqa.evaluation.vqa_standard_evaluation import run_multi_model_evaluation
    import os
    
    logger.info("\n" + "="*60)
    logger.info("Example 3: Evaluate API Models")
    logger.info("="*60)
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping API model evaluation")
        return
    
    # Paths (adjust to your setup)
    vqa_dataset_path = "./vqa_standard/per_image_all.json"
    images_dir = "./images"
    output_dir = "./vqa_results_api"
    
    # Models to evaluate
    models = [
        {"name": "gpt4o", "type": "gpt4o", "api_key": api_key},
        {"name": "gpt5nano", "type": "gpt5nano", "api_key": api_key},
    ]
    
    results = run_multi_model_evaluation(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_base_dir=output_dir,
        models=models,
        max_samples=5,  # API calls are expensive, test with 5 samples
        save_predictions=True,
        verbose=False,
    )
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Results: {output_dir}")


def example_full_pipeline():
    """Example: Full pipeline (transform + evaluate)."""
    from vqa.evaluation.vqa2_pipeline import run_full_pipeline
    
    logger.info("\n" + "="*60)
    logger.info("Example 4: Full Pipeline (Transform + Evaluate)")
    logger.info("="*60)
    
    # Paths (adjust to your setup)
    mmvqa_dir = "./vqa_mmvqa"
    vqa_output_dir = "./vqa_standard"
    images_dir = "./images"
    eval_output_dir = "./vqa_results_full"
    
    # Models to evaluate
    models = ["florence2", "qwen"]
    
    results = run_full_pipeline(
        mmvqa_dir=mmvqa_dir,
        vqa_output_dir=vqa_output_dir,
        images_dir=images_dir,
        eval_output_dir=eval_output_dir,
        models=models,
        api_key=None,
        device="cuda",
        max_samples=10,  # Test with 10 samples
        per_question_only=False,
        verbose=False,
    )
    
    logger.info("\nFull pipeline complete!")
    logger.info(f"VQA Dataset: {vqa_output_dir}")
    logger.info(f"Results: {eval_output_dir}")


def example_single_model():
    """Example: Evaluate a single model."""
    from vqa.evaluation.vqa_standard_evaluation import StandardVQAEvaluator
    
    logger.info("\n" + "="*60)
    logger.info("Example 5: Evaluate Single Model")
    logger.info("="*60)
    
    # Paths (adjust to your setup)
    vqa_dataset_path = "./vqa_standard/action_command.json"  # Single question type
    images_dir = "./images"
    output_dir = "./vqa_results_single"
    
    # Create evaluator
    evaluator = StandardVQAEvaluator(
        model_name="florence2",
        model_type="florence2",
        device="cuda",
    )
    
    # Run evaluation
    result = evaluator.evaluate(
        vqa_dataset_path=vqa_dataset_path,
        images_dir=images_dir,
        output_dir=output_dir,
        max_samples=10,
        save_predictions=True,
    )
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Model: {result['model_name']}")
    logger.info(f"Samples: {result['total_samples']}")
    
    metrics = result.get("metrics", {})
    logger.info("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")


def main():
    """Run examples."""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "transform":
            example_transform_only()
        elif example == "local":
            example_evaluate_local_models()
        elif example == "api":
            example_evaluate_api_models()
        elif example == "full":
            example_full_pipeline()
        elif example == "single":
            example_single_model()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_vqa2.py [transform|local|api|full|single]")
            sys.exit(1)
    else:
        # Run all examples
        print("\nRunning all examples...")
        print("Note: Adjust paths in the script to match your setup!")
        print()
        
        print("Available examples:")
        print("  python example_vqa2.py transform  # Transform MMVQA to Standard VQA")
        print("  python example_vqa2.py local      # Evaluate local GPU models")
        print("  python example_vqa2.py api        # Evaluate API models")
        print("  python example_vqa2.py full       # Full pipeline")
        print("  python example_vqa2.py single     # Single model evaluation")


if __name__ == "__main__":
    main()
