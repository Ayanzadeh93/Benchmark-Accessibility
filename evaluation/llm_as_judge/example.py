"""Example usage of LLM-as-judge evaluation platform."""

import os
from pathlib import Path
from evaluation.llm_as_judge.pipeline import run_evaluation


def example_basic_evaluation():
    """Basic example: evaluate with a single judge model."""
    
    # Set API key
    openai_key = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
    
    # Run evaluation
    results_df = run_evaluation(
        annotations_dir="./annotations",  # Directory with annotation JSON files
        output_dir="./eval_output",       # Where to save results
        judge_models=["gpt-4o"],         # Judge model to use
        openai_api_key=openai_key,
        max_annotations=None,             # Evaluate all (set to 5 for testing)
        skip_existing=True,               # Skip already-evaluated
        temperature=0.2,
        max_tokens=1500,
        verbose=True,
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"Total evaluations: {len(results_df)}")
    print(f"\nAverage scores:")
    print(results_df[["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]].mean())


def example_multi_model_evaluation():
    """Advanced example: evaluate with multiple judge models."""
    
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Multiple judge models
    judge_models = [
        "gpt-4o",
        "claude-sonnet-4-20250514",
        "openrouter:qwen/qwen3-vl-235b-a22b-instruct",
    ]
    
    # Run evaluation
    results_df = run_evaluation(
        annotations_dir="./annotations",
        output_dir="./eval_output_multi",
        judge_models=judge_models,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        openrouter_api_key=openrouter_key,
        max_annotations=10,  # Limit for testing
        skip_existing=True,
        temperature=0.2,
        max_tokens=1500,
        verbose=True,
    )
    
    print(f"\n✅ Multi-model evaluation complete!")
    print(f"Total evaluations: {len(results_df)}")
    
    # Compare models
    print("\n📊 Model comparison (average overall scores):")
    model_comparison = results_df.groupby("model")["overall_score"].mean()
    print(model_comparison)


def example_openrouter_only():
    """Example: use OpenRouter for cost-effective evaluation."""
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-key-here")
    
    # Run evaluation with OpenRouter models
    results_df = run_evaluation(
        annotations_dir="./annotations",
        output_dir="./eval_output_openrouter",
        judge_models=[
            "openrouter:qwen/qwen3-vl-235b-a22b-instruct",
            "openrouter:openai/gpt-4o",
        ],
        openrouter_api_key=openrouter_key,
        max_annotations=5,
        skip_existing=True,
        temperature=0.2,
        max_tokens=1500,
        verbose=True,
    )
    
    print(f"\n✅ OpenRouter evaluation complete!")
    print(f"Results saved to: ./eval_output_openrouter/")


if __name__ == "__main__":
    print("LLM-as-Judge Evaluation Examples")
    print("=" * 60)
    
    # Choose which example to run
    print("\nAvailable examples:")
    print("1. Basic evaluation (single model)")
    print("2. Multi-model evaluation")
    print("3. OpenRouter only")
    
    choice = input("\nSelect example (1-3): ").strip()
    
    if choice == "1":
        example_basic_evaluation()
    elif choice == "2":
        example_multi_model_evaluation()
    elif choice == "3":
        example_openrouter_only()
    else:
        print("Invalid choice. Running basic example...")
        example_basic_evaluation()
