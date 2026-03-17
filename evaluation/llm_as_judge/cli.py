"""Command-line interface for LLM-as-judge evaluation."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import run_evaluation


def main():
    """Main CLI entry point for LLM-as-judge evaluation."""
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Evaluation Platform for Accessibility Annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with GPT-4o
  python -m evaluation.llm_as_judge.cli \\
    --annotations-dir "C:/path/to/annotations" \\
    --output-dir "./eval_output" \\
    --judge-models gpt-4o
  
  # Evaluate with multiple models (run sequentially)
  python -m evaluation.llm_as_judge.cli \\
    --annotations-dir "./annotations" \\
    --output-dir "./eval_output" \\
    --judge-models gpt-4o claude-sonnet-4-20250514 \\
    --openai-api-key $OPENAI_API_KEY \\
    --anthropic-api-key $ANTHROPIC_API_KEY
  
  # Evaluate with OpenRouter models
  python -m evaluation.llm_as_judge.cli \\
    --annotations-dir "./annotations" \\
    --output-dir "./eval_output" \\
    --judge-models "openrouter:qwen/qwen3-vl-235b-a22b-instruct" "openrouter:openai/gpt-4o" \\
    --openrouter-api-key $OPENROUTER_API_KEY
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Directory containing annotation JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--judge-models",
        type=str,
        nargs="+",
        required=True,
        help="Judge model(s) to use (e.g., gpt-4o, claude-sonnet-4-20250514, openrouter:qwen/qwen3-vl)"
    )
    
    # API keys
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Limit number of annotations to evaluate (for testing)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip already-evaluated annotations (re-evaluate all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Max tokens for LLM response (default: 1500)"
    )
    
    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get API keys from environment if not provided
    import os
    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    
    # Validate directories
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        print(f"ERROR: Annotations directory not found: {annotations_dir}", file=sys.stderr)
        return 1
    
    try:
        # Run evaluation
        results_df = run_evaluation(
            annotations_dir=str(annotations_dir),
            output_dir=args.output_dir,
            judge_models=args.judge_models,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            openrouter_api_key=openrouter_key,
            max_annotations=args.max_annotations,
            skip_existing=not args.no_skip_existing,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total evaluations: {len(results_df)}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nResults saved to:")
        print(f"  - CSV: {args.output_dir}/evaluation_results.csv")
        print(f"  - Excel: {args.output_dir}/evaluation_results.xlsx")
        print(f"  - Plots: {args.output_dir}/plots/")
        print(f"  - Individual results: {args.output_dir}/results/")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
        for criterion in criteria:
            if criterion in results_df.columns:
                mean = results_df[criterion].mean()
                std = results_df[criterion].std()
                print(f"{criterion.replace('_', ' ').title():25s}: {mean:5.2f} ± {std:4.2f}")
        
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
