# 🎯 LLM-as-Judge Evaluation Platform - Complete Package

## What Has Been Created

A production-ready LLM-as-judge evaluation platform for assessing accessibility annotations. The system evaluates annotations across 5 key criteria using multiple AI judges, generates comprehensive visualizations, and exports results to CSV/Excel formats.

## 📁 Complete File Structure

```
evaluation/llm_as_judge/
├── __init__.py                      # Package initialization
├── prompts.py                       # Evaluation prompts and templates
├── schemas.py                       # Pydantic data models (EvaluationResult, etc.)
├── judge_models.py                  # LLM judge implementations (OpenAI, Anthropic, OpenRouter)
├── visualization.py                 # 6 comprehensive plotting functions
├── pipeline.py                      # Main evaluation pipeline orchestration
├── cli.py                           # Standalone command-line interface
├── example.py                       # Interactive usage examples
├── test_installation.py             # Installation verification script
├── README.md                        # Full documentation (detailed)
├── QUICKSTART.md                    # 5-minute quick start guide
└── IMPLEMENTATION_SUMMARY.md        # This implementation overview
```

## 🎯 Five Evaluation Criteria (1-10 scale)

1. **Clarity**: Clear, concise language without jargon
2. **Completeness**: All necessary navigation information included
3. **Robustness**: Handles edge cases and uncertainties
4. **User-Friendliness**: Practical and actionable for blind users
5. **Accuracy**: Correct objects, distances, spatial relationships

**Overall Score**: Average of all 5 criteria

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install openai anthropic pandas openpyxl matplotlib seaborn
```

### 2. Set API Key
```bash
# Windows
$env:OPENAI_API_KEY="sk-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

### 3. Run Evaluation
```bash
python main.py eval \
  --annotations-dir "C:/path/to/annotations" \
  --output-dir "./eval_output" \
  --judge-models gpt-4o
```

## 💻 Usage Examples

### Single Model
```bash
python main.py eval \
  --annotations-dir "./Phase4/annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o
```

### Multiple Models (Sequential)
```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o claude-sonnet-4-20250514 \
  --openai-api-key $OPENAI_API_KEY \
  --anthropic-api-key $ANTHROPIC_API_KEY
```

### OpenRouter (Cost-Effective)
```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_results" \
  --judge-models "openrouter:qwen/qwen3-vl-235b-a22b-instruct" \
  --openrouter-api-key $OPENROUTER_API_KEY
```

### Test with Limited Annotations
```bash
python main.py eval \
  --annotations-dir "./annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o \
  --max-annotations 10 \
  --verbose
```

## 📊 Output Files

```
eval_output/
├── evaluation_results.csv              # Flat table format
├── evaluation_results.xlsx             # Multi-sheet Excel
│   ├── Results                         # Full data
│   ├── Summary                         # Statistics
│   └── Model Comparison                # Compare judges
├── plots/                              # 6 visualization plots
│   ├── score_distribution.png
│   ├── criteria_comparison.png
│   ├── model_comparison.png
│   ├── score_heatmap.png
│   ├── overall_score_histogram.png
│   └── criteria_correlation.png
└── results/                            # Individual JSON files
    └── {image}_{model}.json
```

## 🎨 Visualizations

1. **Score Distribution**: Histograms for each criterion with mean markers
2. **Criteria Comparison**: Bar chart with averages and error bars
3. **Model Comparison**: Compare different judge models side-by-side
4. **Score Heatmap**: Color-coded scores for first 30 images
5. **Overall Score Histogram**: Distribution with statistics
6. **Criteria Correlation**: Correlation matrix heatmap

## 🤖 Supported Models

### OpenAI
- `gpt-4o` (recommended)
- `gpt-4-turbo`
- `gpt-4`

### Anthropic
- `claude-sonnet-4-20250514` (Claude 4.5 Sonnet - recommended)
- `claude-3-7-sonnet-20250219`
- `claude-3-5-sonnet-20241022`

### OpenRouter (200+ models)
- `openrouter:qwen/qwen3-vl-235b-a22b-instruct` (good value)
- `openrouter:openai/gpt-4o`
- `openrouter:anthropic/claude-sonnet-4`
- And many more...

## 🔧 Command-Line Arguments

### Required
- `--annotations-dir`: Annotation JSON directory
- `--output-dir`: Output directory
- `--judge-models`: One or more model names

### API Keys (optional if set in environment)
- `--openai-api-key`
- `--anthropic-api-key`
- `--openrouter-api-key`

### Settings
- `--max-annotations`: Limit evaluations (for testing)
- `--no-skip-existing`: Re-evaluate all
- `--temperature`: LLM temperature (default: 0.2)
- `--max-tokens`: Max response tokens (default: 1500)
- `-v, --verbose`: Verbose logging

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive documentation with all features |
| `QUICKSTART.md` | 5-minute setup guide |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `example.py` | Interactive usage examples |
| `test_installation.py` | Verify installation |

## ✅ Verification

Test your installation:
```bash
cd evaluation/llm_as_judge
python test_installation.py
```

Expected output:
```
✅ pandas
✅ openpyxl
✅ matplotlib
✅ seaborn
✅ All tests passed!
```

## 💡 Tips

1. **Start Small**: Use `--max-annotations 10` for testing
2. **Monitor Costs**: Use verbose mode to track API usage
3. **Multiple Judges**: Compare 2-3 models for validation
4. **Low Temperature**: Keep at 0.2 for consistent scoring
5. **Skip Existing**: Default behavior saves time and money

## 🔍 CSV Output Format

Each row contains:
```
image_name, model, 
clarity, clarity_reasoning,
completeness, completeness_reasoning,
robustness, robustness_reasoning,
user_friendliness, user_friendliness_reasoning,
accuracy, accuracy_reasoning,
overall_score, feedback
```

## 🎓 Programmatic API

```python
from evaluation.llm_as_judge.pipeline import run_evaluation

results_df = run_evaluation(
    annotations_dir="./annotations",
    output_dir="./eval_output",
    judge_models=["gpt-4o"],
    openai_api_key="your-key",
    max_annotations=None,
    skip_existing=True,
    temperature=0.2,
    max_tokens=1500,
    verbose=False,
)

# Analyze results
print(results_df["overall_score"].mean())
print(results_df.groupby("model")["overall_score"].mean())
```

## 🚦 Getting Started Checklist

- [ ] Install dependencies: `pip install openai anthropic pandas openpyxl matplotlib seaborn`
- [ ] Set API key: `export OPENAI_API_KEY="sk-..."`
- [ ] Verify installation: `python evaluation/llm_as_judge/test_installation.py`
- [ ] Test with sample: `python main.py eval --annotations-dir ./test --output-dir ./test_out --judge-models gpt-4o --max-annotations 1`
- [ ] Run full evaluation: `python main.py eval --annotations-dir ./annotations --output-dir ./results --judge-models gpt-4o`
- [ ] Review outputs: Check `results/evaluation_results.csv` and `results/plots/`

## 📞 Support & Documentation

- **Quick Start**: `evaluation/llm_as_judge/QUICKSTART.md`
- **Full Docs**: `evaluation/llm_as_judge/README.md`
- **Examples**: `evaluation/llm_as_judge/example.py`
- **Implementation**: `evaluation/llm_as_judge/IMPLEMENTATION_SUMMARY.md`
- **Help**: `python main.py eval --help`

## 🎉 Summary

The LLM-as-judge evaluation platform is **production-ready** with:

✅ Multi-model support (OpenAI, Anthropic, OpenRouter)  
✅ 5 comprehensive evaluation criteria  
✅ CSV/Excel export with detailed results  
✅ 6 professional visualization plots  
✅ Incremental processing with skip-existing  
✅ Command-line and programmatic APIs  
✅ Complete documentation and examples  
✅ Error handling and logging  
✅ Integration with main.py  

**Ready to use immediately!**

---

**Next Steps:**
1. Run `python evaluation/llm_as_judge/test_installation.py` to verify setup
2. See `QUICKSTART.md` for a 5-minute tutorial
3. Review `example.py` for usage patterns
4. Start evaluating with `python main.py eval --help`
