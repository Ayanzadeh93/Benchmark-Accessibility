# LLM-as-Judge Quick Start Guide

## Installation

```bash
# Install required packages
pip install openai anthropic pandas openpyxl matplotlib seaborn
```

## Quick Test (5-Minute Setup)

### 1. Create test annotation
```bash
mkdir -p test_annotations
```

Create a file `test_annotations/test_image_annotation.json`:
```json
{
  "accessibility_description": "A wide hallway with overhead lighting. Two exit doors are visible ahead at approximately 4 meters. The path is clear with no obstacles. Doors have illuminated exit signs above them.",
  "navigation": {
    "action": "proceed_forward",
    "reasoning": "Clear path ahead with visible exits",
    "obstacles": []
  }
}
```

### 2. Set API key
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-key-here"
```

### 3. Run evaluation
```bash
python main.py eval \
  --annotations-dir test_annotations \
  --output-dir test_eval_output \
  --judge-models gpt-4o \
  --max-annotations 1
```

### 4. Check results
```
test_eval_output/
├── evaluation_results.csv    # Open in Excel/Google Sheets
├── evaluation_results.xlsx   # Multi-sheet Excel file
├── plots/                     # Visualization plots
└── results/                   # Individual JSON results
```

## Common Commands

### Single Model Evaluation
```bash
python main.py eval \
  --annotations-dir "./Phase4/annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o
```

### Multiple Models (Sequential)
```bash
python main.py eval \
  --annotations-dir "./Phase4/annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o claude-sonnet-4-20250514 \
  --openai-api-key $OPENAI_API_KEY \
  --anthropic-api-key $ANTHROPIC_API_KEY
```

### OpenRouter (Cost-Effective)
```bash
python main.py eval \
  --annotations-dir "./Phase4/annotations" \
  --output-dir "./eval_results" \
  --judge-models "openrouter:qwen/qwen3-vl-235b-a22b-instruct" \
  --openrouter-api-key $OPENROUTER_API_KEY
```

### Test First 10 Annotations
```bash
python main.py eval \
  --annotations-dir "./Phase4/annotations" \
  --output-dir "./eval_results" \
  --judge-models gpt-4o \
  --max-annotations 10 \
  --verbose
```

## Model Recommendations

### Best Quality
- `gpt-4o` - Fast, consistent, high quality
- `claude-sonnet-4-20250514` - Excellent reasoning, nuanced feedback

### Cost-Effective
- `openrouter:qwen/qwen3-vl-235b-a22b-instruct` - Great value
- `openrouter:openai/gpt-4o` - Same as GPT-4o but potentially cheaper

### For Comparison
Use 2-3 models to validate consistency:
```bash
--judge-models gpt-4o claude-sonnet-4-20250514 "openrouter:qwen/qwen3-vl-235b-a22b-instruct"
```

## Output Files

### CSV (evaluation_results.csv)
- Open in Excel, Google Sheets, or Python
- Each row = one evaluation
- Columns: image_name, model, scores, reasoning, feedback

### Excel (evaluation_results.xlsx)
- **Results**: Full evaluation data
- **Summary**: Statistics across all annotations
- **Model Comparison**: Compare different judges (if multiple models used)

### Plots (plots/ directory)
- `score_distribution.png` - How scores are distributed
- `criteria_comparison.png` - Average score by criterion
- `model_comparison.png` - Compare different judge models
- `score_heatmap.png` - Visual scores for first 30 images
- `overall_score_histogram.png` - Overall score distribution
- `criteria_correlation.png` - How criteria relate to each other

## Scoring Criteria (1-10)

| Criterion | What It Measures |
|-----------|------------------|
| **Clarity** | Clear language, no jargon, easy to understand |
| **Completeness** | All necessary info for safe navigation |
| **Robustness** | Handles edge cases, uncertainties, variations |
| **User-Friendliness** | Practical for blind users, intuitive, safe |
| **Accuracy** | Correct objects, distances, spatial relationships |
| **Overall Score** | Average of all 5 criteria |

## Troubleshooting

### "No annotation JSON files found"
- Check that files are in correct directory
- Ensure files end with `.json`
- Files should contain accessibility/navigation data

### "API key required"
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or pass as argument
python main.py eval ... --openai-api-key "sk-..."
```

### Slow evaluation
- Use `--max-annotations 10` for testing
- OpenRouter models are typically faster/cheaper
- Evaluations run sequentially (1 model at a time)

### Import errors
```bash
pip install openai anthropic pandas openpyxl matplotlib seaborn
```

## Tips

1. **Start Small**: Test with 5-10 annotations first
2. **Check Costs**: Monitor API usage, especially with proprietary models
3. **Skip Existing**: Default behavior skips already-evaluated annotations
4. **Multiple Judges**: Use 2-3 models for validation
5. **Save Results**: Keep evaluation outputs for comparison over time

## Support

See full documentation: `evaluation/llm_as_judge/README.md`

For issues, check:
- API keys are set correctly
- Annotation files are valid JSON
- Required packages are installed
