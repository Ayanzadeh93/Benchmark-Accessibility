# How to Use VQA2 System

This guide shows you **exactly** how to transform your MMVQA dataset to Standard VQA and evaluate models.

## What You Have Now

✅ **Complete VQA2 Implementation**
- Transformation tool (MMVQA → Standard VQA)
- Evaluation tool (Standard VQA with text metrics)
- CLI pipeline (combines both)
- Full documentation

✅ **Integration Tests Passed**
```
Test 1: MMVQA → Standard VQA Transformation    ✓
Test 2: File-based Transformation              ✓
Test 3: Evaluation Metrics                     ✓
Test 4: VQA Evaluator Initialization           ✓
Test 5: Pipeline CLI                           ✓
```

## Quick Start (3 Commands)

### 1. Transform Your MMVQA Dataset

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --transform-only
```

**What this does:**
- Reads your MMVQA files (with A/B/C/D options)
- Extracts only the correct answer text
- Saves as Standard VQA format (no multiple choice)

### 2. Test with 10 Samples

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_test \
    --models florence2 \
    --max-samples 10 \
    --eval-only
```

**What this does:**
- Loads 10 images from your VQA dataset
- Runs Florence2 model (fastest, GPU)
- Computes text similarity metrics
- Saves results to `./vqa_results_test/`

### 3. Full Evaluation (All Models)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 qwen llava \
    --eval-only
```

**What this does:**
- Evaluates Florence2, Qwen, and LLaVA models
- Runs on all images in your dataset
- Compares models using ROUGE, BLEU, BERTScore, CLIP
- Saves comparison report to `./vqa_results/comparison_summary.csv`

## View Results

```bash
# View comparison of all models
cat vqa_results/comparison_summary.csv

# View detailed results for one model
cat vqa_results/florence2/florence2_summary.csv
```

**Example Output:**
```csv
model,total_samples,exact_match,rouge1_f1,rouge2_f1,rougeL_f1,bleu,bertscore_f1
florence2,300,0.4500,0.6234,0.4521,0.5987,0.3821,0.7234
qwen,300,0.5200,0.6987,0.5234,0.6543,0.4521,0.7891
llava,300,0.5800,0.7234,0.5987,0.6987,0.5123,0.8234
```

## File Locations

All files are in `vqa/evaluation/`:

| File | Purpose |
|------|---------|
| `vqa2_pipeline.py` | **Main CLI** (use this!) |
| `transform_mmvqa_to_vqa.py` | Transformation logic |
| `vqa_standard_evaluation.py` | Evaluation logic |
| `README_VQA2.md` | Full documentation |
| `QUICKSTART_VQA2.md` | Quick reference |
| `example_vqa2.py` | Code examples |
| `test_vqa2_integration.py` | Integration tests |

## Before You Start

### Check Your MMVQA Dataset

```bash
ls vqa_mmvqa/
```

You should see:
```
action_command.json
main_obstacle.json
closest_obstacle.json
risk_assessment.json
spatial_clock.json
action_suggestion.json
per_image_all.json  ← This is ideal for evaluation
```

**Don't have this?** Generate it first:
```bash
python vqa/evaluation/generate_per_question.py \
    --annotations-dir ./annotations \
    --output-dir ./vqa_mmvqa \
    --images-dir ./images \
    --per-image
```

### Check Your Images

```bash
ls images/ | head -5
```

You should see your image files (e.g., `image_001.jpg`, `image_002.jpg`, etc.)

## Common Scenarios

### Scenario 1: I want to test quickly (10 samples, 1 model)

```bash
# Full pipeline (transform + evaluate)
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_test \
    --models florence2 \
    --max-samples 10
```

### Scenario 2: I want to compare all local GPU models

```bash
# Transform once
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --transform-only

# Evaluate all models
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 qwen llava \
    --eval-only
```

### Scenario 3: I want to include API models (GPT-4o)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_api \
    --models florence2 qwen gpt4o \
    --api-key sk-... \
    --eval-only
```

### Scenario 4: I only have CPU (no GPU)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_cpu \
    --models florence2 \
    --device cpu \
    --max-samples 10 \
    --eval-only
```

### Scenario 5: I want to evaluate one specific question type

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/action_command.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_action \
    --models florence2 qwen \
    --eval-only
```

## Understanding the Output

After running evaluation, you'll get:

```
vqa_results/
├── florence2/
│   ├── florence2_evaluation.json   ← Full results with all predictions
│   ├── florence2_summary.json      ← Compact summary
│   └── florence2_summary.csv       ← Easy-to-read CSV
├── qwen/
│   ├── qwen_evaluation.json
│   ├── qwen_summary.json
│   └── qwen_summary.csv
├── comparison_summary.json         ← Compare all models (JSON)
└── comparison_summary.csv          ← Compare all models (CSV) ⭐ Look here!
```

### Key Files

1. **`comparison_summary.csv`** - Compare all models side-by-side
2. **`{model}_summary.csv`** - Per-model results
3. **`{model}_evaluation.json`** - Full results with individual predictions

## Metrics Explained

- **exact_match**: Perfect string match (0.0-1.0, higher = better)
- **rouge1_f1**: Unigram overlap (0.0-1.0, higher = better)
- **rouge2_f1**: Bigram overlap (0.0-1.0, higher = better)
- **rougeL_f1**: Longest common subsequence (0.0-1.0, higher = better)
- **bleu**: Precision-based n-gram metric (0.0-1.0, higher = better)
- **bertscore_f1**: Semantic similarity using BERT (0.0-1.0, higher = better)

**Which metric to focus on?**
- For exact correctness: `exact_match`
- For semantic similarity: `bertscore_f1`
- For partial correctness: `rouge1_f1` or `rougeL_f1`

## Available Models

### Local GPU Models (FREE)
```bash
--models florence2      # Fastest (1-2s/image), 230M params
--models qwen          # Fast (2-3s/image), 2B params
--models llava         # Medium (3-5s/image), 7B params
```

### API Models (PAID)
```bash
--models gpt4o                        # High quality, ~$0.005/image
--models gpt5nano                     # Good quality, ~$0.0005/image
--models openrouter_qwen3_vl_235b    # Best quality, OpenRouter
```

### List All Models
```bash
python vqa/evaluation/vqa2_pipeline.py --list-models
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'vlm_factory'"

Make sure you're running from the project root:
```bash
cd C:/Tim/nature
python vqa/evaluation/vqa2_pipeline.py --list-models
```

### "CUDA out of memory"

Try a smaller model or CPU:
```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 \
    --device cpu \
    --eval-only
```

### "Image not found"

Check that `--images-dir` points to the correct directory:
```bash
ls ./images/ | head -5
```

### "No such file or directory: vqa_mmvqa"

You need to generate MMVQA dataset first:
```bash
python vqa/evaluation/generate_per_question.py \
    --annotations-dir ./annotations \
    --output-dir ./vqa_mmvqa \
    --images-dir ./images \
    --per-image
```

## Get Help

```bash
# Main CLI help
python vqa/evaluation/vqa2_pipeline.py --help

# List available models
python vqa/evaluation/vqa2_pipeline.py --list-models

# Read full documentation
cat vqa/evaluation/README_VQA2.md

# Read quick reference
cat vqa/evaluation/QUICKSTART_VQA2.md

# Run integration tests
python vqa/evaluation/test_vqa2_integration.py
```

## Next Steps

1. **Transform your MMVQA dataset** (1 command)
2. **Test with 10 samples** (verify it works)
3. **Run full evaluation** (all models)
4. **View comparison results** (CSV file)

That's it! 🎉

## Summary

**What you created:**
- ✅ Transform MMVQA → Standard VQA
- ✅ Evaluate multiple models
- ✅ Compute text similarity metrics
- ✅ Generate comparison reports

**What you need to do:**
1. Run transformation command
2. Run evaluation command
3. View results in CSV

**Time estimate:**
- Transformation: ~1-5 minutes (depends on dataset size)
- Evaluation (10 samples, 1 model): ~30 seconds
- Evaluation (full dataset, 3 models): ~1-3 hours (depends on dataset size)

Good luck! 🚀
