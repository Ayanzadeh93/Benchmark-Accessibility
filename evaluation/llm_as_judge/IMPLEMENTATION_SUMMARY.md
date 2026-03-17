# LLM-as-Judge Evaluation Platform - Implementation Summary

## 📋 Overview

A complete LLM-as-judge evaluation platform has been implemented in `evaluation/llm_as_judge/` to systematically assess accessibility annotations for blind users. The platform supports multiple LLM judges, provides detailed scoring across 5 key criteria, generates comprehensive visualizations, and exports results to CSV/Excel formats.

## 🗂️ Project Structure

```
evaluation/llm_as_judge/
├── __init__.py              # Package initialization
├── prompts.py               # Evaluation prompts and templates
├── schemas.py               # Pydantic data models
├── judge_models.py          # LLM judge implementations
├── visualization.py         # Plotting and visualization
├── pipeline.py              # Main evaluation pipeline
├── cli.py                   # Command-line interface
├── example.py               # Usage examples
├── README.md                # Full documentation
└── QUICKSTART.md            # Quick start guide
```

## ✨ Key Features

### 1. Multi-Model Support
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 4.5 Sonnet, Claude 3.7/3.5 Sonnet, Claude 3 Opus
- **OpenRouter**: Access to 200+ models including Qwen, LLaMA, etc.

### 2. Comprehensive Evaluation Criteria
Each annotation is scored 1-10 on:
- **Clarity**: Language clarity and understandability
- **Completeness**: Coverage of necessary navigation information
- **Robustness**: Handling of edge cases and uncertainties
- **User-Friendliness**: Practical utility for blind users
- **Accuracy**: Correctness of descriptions and spatial relationships
- **Overall Score**: Average of all criteria

### 3. Rich Output Formats
- **CSV**: Simple tabular format for spreadsheet analysis
- **Excel**: Multi-sheet workbook with Results, Summary, and Model Comparison
- **JSON**: Individual detailed results for each evaluation
- **Plots**: 6 comprehensive visualizations

### 4. Visualization Suite
1. **Score Distribution**: Histograms for each criterion
2. **Criteria Comparison**: Bar chart with averages and std dev
3. **Model Comparison**: Compare multiple judge models
4. **Score Heatmap**: Visual scores for first 30 images
5. **Overall Score Histogram**: Distribution with statistics
6. **Criteria Correlation**: Correlation matrix between criteria

### 5. Production-Ready Features
- ✅ Skip existing evaluations for incremental processing
- ✅ Configurable temperature and max tokens
- ✅ Verbose logging for debugging
- ✅ Error handling and recovery
- ✅ API key management (env vars or CLI args)
- ✅ Limit annotations for testing
- ✅ Detailed progress tracking with tqdm

## 🚀 Usage

### Command-Line Interface (via main.py)

```bash
# Basic evaluation
python main.py eval \
  --annotations-dir "C:/path/to/annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o

# Multiple models
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o claude-sonnet-4-20250514 \
  --openai-api-key $OPENAI_API_KEY \
  --anthropic-api-key $ANTHROPIC_API_KEY

# OpenRouter models
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_output" \
  --judge-models "openrouter:qwen/qwen3-vl-235b-a22b-instruct" \
  --openrouter-api-key $OPENROUTER_API_KEY

# Advanced options
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

### Programmatic Usage

```python
from evaluation.llm_as_judge.pipeline import run_evaluation

results_df = run_evaluation(
    annotations_dir="./annotations",
    output_dir="./eval_output",
    judge_models=["gpt-4o", "claude-sonnet-4-20250514"],
    openai_api_key="your-key",
    anthropic_api_key="your-key",
    max_annotations=None,
    skip_existing=True,
    temperature=0.2,
    max_tokens=1500,
    verbose=False,
)

# Results as pandas DataFrame
print(results_df.head())
print(results_df.describe())
```

## 📊 Output Structure

```
eval_output/
├── evaluation_results.csv              # Main results (flat format)
├── evaluation_results.xlsx             # Excel with multiple sheets
│   ├── Results                         # Full evaluation data
│   ├── Summary                         # Statistics (mean, std, min, max)
│   └── Model Comparison                # Compare judge models (if multiple)
├── plots/                              # Visualizations
│   ├── score_distribution.png          # Score histograms by criterion
│   ├── criteria_comparison.png         # Average scores with error bars
│   ├── model_comparison.png            # Multi-model comparison
│   ├── score_heatmap.png               # Heatmap for first 30 images
│   ├── overall_score_histogram.png     # Overall distribution
│   └── criteria_correlation.png        # Correlation matrix
└── results/                            # Individual JSON results
    ├── image1_gpt-4o.json
    ├── image1_claude-sonnet-4.json
    └── ...
```

## 📝 CSV/Excel Columns

| Column | Description |
|--------|-------------|
| `image_name` | Name of evaluated image |
| `model` | Judge model used |
| `clarity` | Clarity score (1-10) |
| `clarity_reasoning` | Explanation for clarity score |
| `completeness` | Completeness score (1-10) |
| `completeness_reasoning` | Explanation for completeness score |
| `robustness` | Robustness score (1-10) |
| `robustness_reasoning` | Explanation for robustness score |
| `user_friendliness` | User-friendliness score (1-10) |
| `user_friendliness_reasoning` | Explanation for user-friendliness score |
| `accuracy` | Accuracy score (1-10) |
| `accuracy_reasoning` | Explanation for accuracy score |
| `overall_score` | Average of all scores (1-10) |
| `feedback` | Concise overall feedback (1-2 sentences) |

## 🔧 Command-Line Arguments

### Required
- `--annotations-dir`: Directory with annotation JSON files
- `--output-dir`: Output directory for results
- `--judge-models`: One or more judge model names

### API Keys
- `--openai-api-key`: OpenAI API key (or `OPENAI_API_KEY` env var)
- `--anthropic-api-key`: Anthropic API key (or `ANTHROPIC_API_KEY` env var)
- `--openrouter-api-key`: OpenRouter API key (or `OPENROUTER_API_KEY` env var)

### Evaluation Settings
- `--max-annotations`: Limit evaluations for testing
- `--no-skip-existing`: Re-evaluate all (default: skip existing)
- `--temperature`: LLM temperature (default: 0.2)
- `--max-tokens`: Max response tokens (default: 1500)

### Other
- `-v, --verbose`: Verbose logging

## 🎯 Evaluation Prompt Design

The evaluation prompt is designed to:
1. Provide clear context about the task (accessibility for blind users)
2. Define each criterion with specific questions
3. Request structured JSON output with scores and reasoning
4. Ensure consistency across different judge models

System prompt emphasizes:
- Objective, consistent evaluation
- Focus on practical utility for blind users
- Real-world navigation scenarios

User prompt includes:
- Image name and annotation text
- Detailed criteria descriptions
- JSON response format specification
- Example of expected output

## 🏗️ Architecture

### Core Components

1. **Schemas (`schemas.py`)**
   - `CriterionScore`: Score + reasoning for single criterion
   - `EvaluationResult`: Complete evaluation with all criteria
   - `EvaluationConfig`: Pipeline configuration

2. **Judge Models (`judge_models.py`)**
   - `BaseLLMJudge`: Abstract base class
   - `OpenAIJudge`: GPT models implementation
   - `AnthropicJudge`: Claude models implementation
   - `OpenRouterJudge`: OpenRouter models implementation
   - `create_judge()`: Factory function for model creation

3. **Pipeline (`pipeline.py`)**
   - `EvaluationPipeline`: Main orchestration
   - `run_evaluation()`: Convenience function
   - Handles annotation loading, evaluation, and result saving

4. **Visualization (`visualization.py`)**
   - 6 comprehensive plot functions
   - Consistent styling with seaborn
   - High-resolution output (300 DPI)

5. **CLI (`cli.py`)**
   - Standalone command-line interface
   - Argument parsing and validation
   - Integration with environment variables

## 🔄 Integration with main.py

The evaluation platform is fully integrated into the project's main.py:

```python
# New command added
python main.py eval [options]

# Handler function
def _run_llm_judge_eval(args) -> int:
    # Loads API keys from config.py or env
    # Calls evaluation pipeline
    # Displays results summary
```

Routing in `main()`:
```python
if args.cmd == "eval":
    return _run_llm_judge_eval(args)
```

## 📦 Dependencies Added

Updated `requirements.txt`:
```
openai       # OpenAI API client
pandas       # Data manipulation
openpyxl     # Excel export
```

Existing dependencies used:
- `anthropic` (already present)
- `matplotlib` (already present)
- `seaborn` (already present)
- `tqdm` (already present)
- `pydantic` (already present)

## 🧪 Testing & Validation

### Quick Test
```bash
# 1. Create test annotation
mkdir test_annotations
echo '{"accessibility_description": "Test description"}' > test_annotations/test.json

# 2. Run evaluation
python main.py eval \
  --annotations-dir test_annotations \
  --output-dir test_output \
  --judge-models gpt-4o \
  --max-annotations 1 \
  --verbose
```

### Example Script
Run `evaluation/llm_as_judge/example.py` for interactive examples.

## 💡 Best Practices

1. **Start Small**: Use `--max-annotations 10` for initial testing
2. **Cost Management**: Monitor API usage with verbose logging
3. **Model Selection**:
   - GPT-4o: Fast, consistent, good value
   - Claude 4.5 Sonnet: High quality, nuanced
   - OpenRouter: Cost-effective alternatives
4. **Multiple Judges**: Use 2-3 models for validation
5. **Low Temperature**: Keep at 0.2 for consistent scoring
6. **Incremental Processing**: Default skip_existing saves time/cost

## 📚 Documentation

- **README.md**: Comprehensive documentation with all features
- **QUICKSTART.md**: 5-minute setup guide
- **example.py**: Practical usage examples
- **Inline docstrings**: All functions well-documented

## 🔮 Future Enhancements

Potential improvements:
1. Batch processing for faster evaluation
2. Inter-annotator agreement metrics (when multiple judges)
3. Fine-grained scoring (sub-criteria)
4. Custom criteria support
5. Comparative evaluation (A/B testing)
6. Cost tracking and reporting
7. Retry logic for API failures
8. Caching of expensive evaluations

## ✅ Deliverables Checklist

- ✅ Complete evaluation pipeline implementation
- ✅ Multi-model support (OpenAI, Anthropic, OpenRouter)
- ✅ 5 evaluation criteria with scoring
- ✅ CSV and Excel export with detailed results
- ✅ 6 comprehensive visualization plots
- ✅ Command-line interface via main.py
- ✅ Programmatic API
- ✅ Comprehensive documentation
- ✅ Quick start guide
- ✅ Usage examples
- ✅ Integration with existing project structure
- ✅ API key management
- ✅ Error handling and logging
- ✅ Progress tracking
- ✅ Skip existing functionality
- ✅ Requirements.txt updated

## 🎉 Summary

The LLM-as-judge evaluation platform is now fully implemented and ready for use! You can:

1. **Evaluate annotations** with multiple LLM judges
2. **Export results** to CSV/Excel for analysis
3. **Generate visualizations** to understand score distributions
4. **Compare models** to validate consistency
5. **Process incrementally** with skip-existing functionality

Get started with:
```bash
python main.py eval \
  --annotations-dir "C:/Tim/Database/InVLM/My Data/Scene_summary.v8i.yolov11/test/Phases/Phase 4/annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o \
  --verbose
```

For more details, see `evaluation/llm_as_judge/README.md` and `QUICKSTART.md`.
