# LLM-as-Judge Evaluation Platform

Comprehensive evaluation platform for assessing accessibility annotations using LLM judges.

## Overview

This platform enables systematic evaluation of accessibility annotations using multiple Language Model judges. Each annotation is scored across multiple criteria critical for blind user navigation:

- **Clarity**: Language clarity, conciseness, and understandability
- **Completeness**: Coverage of necessary information for safe navigation
- **Robustness**: Handling of edge cases and uncertainties
- **User-Friendliness**: Practical utility for blind users
- **Accuracy**: Correctness of descriptions and spatial relationships

Each criterion is scored 1-10, with an overall score calculated as the average.

## Features

- ✅ Multiple LLM judge support (OpenAI, Anthropic, OpenRouter)
- ✅ Automated scoring across 5 key criteria
- ✅ CSV and Excel export with detailed results
- ✅ Comprehensive visualizations (distributions, heatmaps, correlations)
- ✅ Individual JSON results for detailed analysis
- ✅ Skip existing evaluations for incremental processing
- ✅ Model comparison support

## Installation

Required packages:
```bash
pip install pandas openpyxl matplotlib seaborn tqdm openai anthropic
```

## Usage

### Basic Usage

Evaluate annotations using a single judge model:

```bash
python main.py eval \
  --annotations-dir "C:/path/to/annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o
```

### Multiple Judge Models

Run evaluation with multiple models sequentially:

```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o claude-sonnet-4-20250514 \
  --openai-api-key $OPENAI_API_KEY \
  --anthropic-api-key $ANTHROPIC_API_KEY
```

### OpenRouter Models

Use OpenRouter for access to many models:

```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_output" \
  --judge-models "openrouter:qwen/qwen3-vl-235b-a22b-instruct" "openrouter:openai/gpt-4o" \
  --openrouter-api-key $OPENROUTER_API_KEY
```

### Advanced Options

```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o \
  --max-annotations 10 \
  --temperature 0.2 \
  --max-tokens 1500 \
  --no-skip-existing \
  --verbose
```

## Command-Line Arguments

### Required Arguments

- `--annotations-dir`: Directory containing annotation JSON files
- `--output-dir`: Output directory for evaluation results
- `--judge-models`: One or more judge model names

### API Keys

- `--openai-api-key`: OpenAI API key (or set `OPENAI_API_KEY` env var)
- `--anthropic-api-key`: Anthropic API key (or set `ANTHROPIC_API_KEY` env var)
- `--openrouter-api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` env var)

### Evaluation Settings

- `--max-annotations`: Limit number of annotations to evaluate (for testing)
- `--no-skip-existing`: Re-evaluate all annotations (default: skip existing)
- `--temperature`: LLM sampling temperature (default: 0.2)
- `--max-tokens`: Max tokens for LLM response (default: 1500)

### Other

- `-v, --verbose`: Enable verbose logging

## Supported Judge Models

### OpenAI Models
- `gpt-4o` (recommended)
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

### Anthropic Models
- `claude-sonnet-4-20250514` (Claude 4.5 Sonnet - recommended)
- `claude-3-7-sonnet-20250219` (Claude 3.7 Sonnet)
- `claude-3-5-sonnet-20241022` (Claude 3.5 Sonnet)
- `claude-3-opus-20240229` (Claude 3 Opus)

### OpenRouter Models
Prefix with `openrouter:` to use any OpenRouter model:
- `openrouter:qwen/qwen3-vl-235b-a22b-instruct`
- `openrouter:openai/gpt-4o`
- `openrouter:anthropic/claude-sonnet-4`
- And many more available at [OpenRouter](https://openrouter.ai/models)

## Output Structure

```
eval_output/
├── evaluation_results.csv         # Main results in CSV format
├── evaluation_results.xlsx        # Excel file with multiple sheets
├── plots/                          # Visualization plots
│   ├── score_distribution.png     # Score distributions by criterion
│   ├── criteria_comparison.png    # Average scores across criteria
│   ├── model_comparison.png       # Compare different judge models
│   ├── score_heatmap.png          # Heatmap of scores (first 30 images)
│   ├── overall_score_histogram.png # Overall score distribution
│   └── criteria_correlation.png   # Correlation matrix between criteria
└── results/                       # Individual JSON results
    ├── image1_gpt-4o.json
    ├── image1_claude-sonnet-4.json
    └── ...
```

## CSV/Excel Output Format

Each row contains:

| Column | Description |
|--------|-------------|
| `image_name` | Name of the evaluated image |
| `model` | Judge model used |
| `clarity` | Clarity score (1-10) |
| `clarity_reasoning` | Reasoning for clarity score |
| `completeness` | Completeness score (1-10) |
| `completeness_reasoning` | Reasoning for completeness score |
| `robustness` | Robustness score (1-10) |
| `robustness_reasoning` | Reasoning for robustness score |
| `user_friendliness` | User-friendliness score (1-10) |
| `user_friendliness_reasoning` | Reasoning for user-friendliness score |
| `accuracy` | Accuracy score (1-10) |
| `accuracy_reasoning` | Reasoning for accuracy score |
| `overall_score` | Average of all scores (1-10) |
| `feedback` | Concise overall feedback (1-2 sentences) |

## Evaluation Criteria Details

### 1. Clarity (1-10)
- Is the language clear, concise, and easy to understand?
- Are technical terms avoided or explained?
- Is the description free of ambiguity?

### 2. Completeness (1-10)
- Does it provide all necessary information for safe navigation?
- Are obstacles, distances, and spatial relationships adequately described?
- Is critical safety information included?

### 3. Robustness (1-10)
- Does it handle edge cases and uncertainties?
- Is it reliable across different scenarios?
- Does it account for potential variations?

### 4. User-Friendliness for Blind Users (1-10)
- Is it practical and actionable for a blind person?
- Does it prioritize safety and ease of navigation?
- Are distances and spatial descriptions intuitive?
- Is information presented in a logical order?

### 5. Accuracy (1-10)
- Are objects and obstacles correctly identified?
- Are spatial relationships accurately described?
- Are distances and measurements reasonable?

## Visualization Plots

### Score Distribution
Histograms showing the distribution of scores for each criterion, with mean values highlighted.

### Criteria Comparison
Bar chart comparing average scores across all criteria with standard deviation error bars.

### Model Comparison
Grouped bar chart comparing performance of different judge models across all criteria.

### Score Heatmap
Color-coded heatmap showing scores for individual images (first 30) across all criteria.

### Overall Score Histogram
Distribution of overall scores with mean and median markers.

### Criteria Correlation
Heatmap showing correlation coefficients between different evaluation criteria.

## Programmatic Usage

You can also use the evaluation platform programmatically:

```python
from evaluation.llm_as_judge.pipeline import run_evaluation

results_df = run_evaluation(
    annotations_dir="./annotations",
    output_dir="./eval_output",
    judge_models=["gpt-4o", "claude-sonnet-4-20250514"],
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key",
    max_annotations=None,
    skip_existing=True,
    temperature=0.2,
    max_tokens=1500,
    verbose=False,
)

# Access results as pandas DataFrame
print(results_df.head())
print(results_df.describe())
```

## Tips & Best Practices

1. **Start Small**: Test with `--max-annotations 5` first to validate setup
2. **Cost Management**: Be aware of API costs when using OpenAI/Anthropic models
3. **Model Selection**: 
   - GPT-4o: Fast, consistent, good value
   - Claude 4.5 Sonnet: High quality, nuanced evaluations
   - OpenRouter: Cost-effective alternatives
4. **Multiple Judges**: Use 2-3 different models for comparison and validation
5. **Temperature**: Keep low (0.2) for consistent, deterministic scoring
6. **Incremental Processing**: Use default `--skip-existing` for large batches

## Troubleshooting

### Missing API Keys
Ensure API keys are set in environment variables or passed as arguments:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```

### No Annotations Found
Check that annotation files:
- Are in JSON format
- Contain expected fields (e.g., `accessibility_description`, `navigation`, etc.)
- Are in the correct directory

### Excel Export Fails
Install openpyxl:
```bash
pip install openpyxl
```

### Plot Generation Fails
Install visualization dependencies:
```bash
pip install matplotlib seaborn
```

## License

Part of the Nature VQA project.
