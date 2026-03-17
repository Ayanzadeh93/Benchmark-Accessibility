# Evaluate Your VQA Ground Truth

This guide shows you how to evaluate models on **your existing ground truth** files.

## Your Ground Truth Files

You have:
```
C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\
├── per_image_all.json          ← Use this (all questions in one file)
├── action_command.json         ← Or use individual question files
├── closest_obstacle.json
├── main_obstacle.json
├── risk_assessment.json
└── spatial_clock.json
```

## Quick Start (3 Commands)

### 1. Quick Test (10 samples, 1 model)

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results_test" \
    --models florence2 \
    --max-samples 10
```

**Time:** ~30 seconds  
**Output:** `vqa_results_test/florence2/`

### 2. Evaluate All Local Models

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results" \
    --models florence2 qwen llava
```

**Time:** ~1-3 hours (depends on dataset size)  
**Output:** `vqa_results/comparison.csv`

### 3. View Results

```bash
# View comparison of all models
cat vqa_results/comparison.csv

# Or open in Excel
# File: vqa_results/comparison.csv
```

## What This Does

```
┌─────────────────────────────────────────────────────────┐
│ Your Ground Truth (MMVQA format)                        │
│ {                                                        │
│   "question": "What should the person do?",             │
│   "options": {"A": "...", "B": "...", "C": "...", ...}, │
│   "answer": "C",                                         │
│   "answer_text": "Move forward slowly..."  ← This!     │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Script extracts correct answer                          │
│ Standard VQA format:                                     │
│ {                                                        │
│   "question": "What should the person do?",             │
│   "answer": "Move forward slowly..."  ← Ground truth   │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ For each model (florence2, qwen, llava):                │
│   1. Ask question: "What should the person do?"         │
│   2. Model generates: "Go straight ahead"               │
│   3. Compare with ground truth: "Move forward slowly..."│
│   4. Compute metrics:                                    │
│      - Exact Match: 0.0 (not exact)                     │
│      - ROUGE-1 F1: 0.5 (50% word overlap)               │
│      - BERTScore: 0.85 (semantically similar)           │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Results saved:                                           │
│ vqa_results/                                             │
│ ├── florence2/                                           │
│ │   ├── florence2_results.json  (full predictions)      │
│ │   └── florence2_summary.csv   (metrics)               │
│ ├── qwen/                                                │
│ │   ├── qwen_results.json                               │
│ │   └── qwen_summary.csv                                │
│ ├── comparison.json                                      │
│ └── comparison.csv  ⭐ LOOK HERE!                        │
└─────────────────────────────────────────────────────────┘
```

## Metrics Computed

✅ **Exact Match** - Perfect string match  
✅ **ROUGE-1 F1** - Unigram overlap  
✅ **ROUGE-2 F1** - Bigram overlap  
✅ **ROUGE-L F1** - Longest common subsequence  
✅ **BLEU** - Precision-based n-gram metric  
✅ **BERTScore F1/Precision/Recall** - Semantic similarity  
⚠️ **CLIP Score** - Optional (use `--compute-clip` flag, slow)

## Available Models

### Local GPU Models (FREE)
```bash
--models florence2      # Fastest (1-2s/image), 230M params
--models qwen          # Fast (2-3s/image), 2B params
--models llava         # Medium (3-5s/image), 7B params
```

### API Models (PAID)
```bash
--models gpt4o         # High quality, ~$0.005/image
--models gpt5nano      # Good quality, ~$0.0005/image
```

## Examples

### Example 1: Quick Test (10 samples)

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results_test" \
    --models florence2 \
    --max-samples 10
```

### Example 2: Evaluate Specific Question Type

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\action_command.json" \
    --output-dir "C:\Tim\nature\vqa_results_action" \
    --models florence2 qwen
```

### Example 3: Compare All Local Models

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results_full" \
    --models florence2 qwen llava
```

### Example 4: Include API Models

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results_api" \
    --models florence2 qwen gpt4o \
    --api-key sk-...
```

### Example 5: CPU Only (No GPU)

```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" \
    --output-dir "C:\Tim\nature\vqa_results_cpu" \
    --models florence2 \
    --device cpu \
    --max-samples 10
```

## Output Structure

After running evaluation:

```
vqa_results/
├── florence2/
│   ├── florence2_results.json      ← Full results with all predictions
│   └── florence2_summary.csv       ← Metrics summary
├── qwen/
│   ├── qwen_results.json
│   └── qwen_summary.csv
├── llava/
│   ├── llava_results.json
│   └── llava_summary.csv
├── comparison.json                 ← Compare all models (JSON)
└── comparison.csv                  ← Compare all models (CSV) ⭐
```

### Example: comparison.csv

```csv
model,total_samples,exact_match,rouge1_f1,rouge2_f1,rougeL_f1,bleu,bertscore_f1,avg_inference_time_s
florence2,6977,0.4234,0.6123,0.4521,0.5876,0.3654,0.7234,1.234
qwen,6977,0.5123,0.6987,0.5432,0.6543,0.4321,0.7891,2.345
llava,6977,0.5876,0.7456,0.6123,0.7012,0.5123,0.8234,3.456
```

### Example: florence2_summary.csv

```csv
metric,value
model,florence2
total_samples,6977
total_failed,0
exact_match,0.4234
rouge1_f1,0.6123
rouge2_f1,0.4521
rougeL_f1,0.5876
bleu,0.3654
bertscore_f1,0.7234
bertscore_precision,0.7123
bertscore_recall,0.7345
avg_inference_time_s,1.234

question_type,accuracy,correct,total
action_command,0.4500,3140,6977
main_obstacle,0.4100,2860,6977
closest_obstacle,0.3900,2721,6977
risk_assessment,0.4300,3000,6977
spatial_clock,0.3800,2651,6977
```

## Understanding Results

### What's Good?

- **exact_match > 0.40**: Good (40%+ exactly correct)
- **rouge1_f1 > 0.60**: Good (60%+ word overlap)
- **bertscore_f1 > 0.70**: Good (semantically similar)

### Which Metric to Focus On?

- **For exact correctness**: `exact_match`
- **For semantic similarity**: `bertscore_f1`
- **For partial correctness**: `rouge1_f1` or `rougeL_f1`

### Example Interpretation

```
Model: florence2
- exact_match: 0.42 → Gets 42% exactly right
- rouge1_f1: 0.61 → 61% word overlap on average
- bertscore_f1: 0.72 → Semantically similar in 72% of cases
```

## Troubleshooting

### "No module named 'vlm_factory'"

Run from project root:
```bash
cd C:\Tim\nature
python vqa/evaluation/evaluate_vqa_simple.py --help
```

### "CUDA out of memory"

Use smaller model or CPU:
```bash
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth "..." \
    --output-dir "..." \
    --models florence2 \
    --device cpu \
    --max-samples 10
```

### "Image not found"

The script uses `image_path` from your ground truth JSON. Make sure the paths are correct.

### Evaluation is slow

- Start with `--max-samples 10` to test
- Use `florence2` (fastest model)
- Use `--device cuda` if you have GPU

## Command Reference

```bash
# Basic usage
python vqa/evaluation/evaluate_vqa_simple.py \
    --ground-truth PATH \
    --output-dir PATH \
    --models MODEL1 MODEL2 ...

# Options
--ground-truth PATH       # Your per_image_all.json or per-question file
--output-dir PATH         # Where to save results
--models MODEL1 MODEL2    # Models to evaluate
--api-key KEY            # API key for gpt4o, etc.
--device auto|cuda|cpu   # Device for local models
--max-samples N          # Limit samples (for testing)
--compute-clip           # Compute CLIP score (slow)
```

## Next Steps

1. **Quick test** (10 samples, 1 model) - verify it works
2. **Full evaluation** (all samples, all models) - get results
3. **View comparison** - open `comparison.csv` in Excel
4. **Analyze per-question** - check which questions are hard/easy
5. **Select best model** - based on metrics

## Summary

**What you need:**
- ✅ Your ground truth JSON (you have it!)
- ✅ Images directory (paths in your JSON)
- ✅ Models to evaluate (florence2, qwen, llava, etc.)

**What you get:**
- ✅ Predictions for each model
- ✅ Metrics: ROUGE, BLEU, BERTScore
- ✅ Comparison CSV showing which model is best
- ✅ Per-question breakdown

**Time:**
- Quick test (10 samples): ~30 seconds
- Full evaluation (6977 samples, 3 models): ~2-4 hours

Good luck! 🚀
