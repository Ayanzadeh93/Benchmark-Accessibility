# VQA2 Quick Start Guide

Transform MMVQA (multiple choice) to Standard VQA and evaluate models.

## Prerequisites

```bash
# Install dependencies (if not already installed)
pip install torch transformers pillow tqdm pydantic
pip install rouge-score sacrebleu bert-score  # For metrics
```

## Step 1: Find Your MMVQA Dataset

Your MMVQA dataset should have been generated using `generate_per_question.py`. Look for:

```
vqa_mmvqa/
├── action_command.json
├── main_obstacle.json
├── closest_obstacle.json
├── risk_assessment.json
├── spatial_clock.json
├── action_suggestion.json
└── per_image_all.json  # ← This file is ideal for batch evaluation
```

## Step 2: Transform MMVQA → Standard VQA

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --transform-only
```

**Output:**
```
vqa_standard/
├── action_command.json           # Same questions, but only correct answer (no A/B/C/D)
├── main_obstacle.json
├── closest_obstacle.json
├── risk_assessment.json
├── spatial_clock.json
├── action_suggestion.json
├── per_image_all.json            # All questions grouped by image
└── transformation_summary.json
```

## Step 3: Evaluate Models

### Option A: Local GPU Models (FREE)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 qwen llava \
    --device cuda \
    --eval-only
```

**Speed:**
- Florence2: ~1-2 seconds per image (fastest)
- Qwen: ~2-3 seconds per image
- LLaVA: ~3-5 seconds per image

### Option B: API Models (PAID)

```bash
# OpenAI (GPT-4o, GPT-5)
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_api \
    --models gpt4o gpt5nano \
    --api-key sk-... \
    --eval-only

# OpenRouter
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_openrouter \
    --models openrouter_qwen3_vl_235b openrouter_llama4_maverick \
    --api-key sk-or-... \
    --eval-only
```

### Option C: Full Pipeline (Transform + Evaluate)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 qwen
```

## Step 4: View Results

### Results Structure

```
vqa_results/
├── florence2/
│   ├── florence2_evaluation.json   # Full results + predictions
│   ├── florence2_summary.json      # Compact summary
│   └── florence2_summary.csv       # Easy-to-read CSV
├── qwen/
│   ├── qwen_evaluation.json
│   ├── qwen_summary.json
│   └── qwen_summary.csv
├── comparison_summary.json         # Compare all models
└── comparison_summary.csv          # CSV comparison
```

### View CSV Summary

```bash
# Single model
cat vqa_results/florence2/florence2_summary.csv

# All models comparison
cat vqa_results/comparison_summary.csv
```

**Example CSV Output:**
```
model,total_samples,exact_match,rouge1_f1,rouge2_f1,rougeL_f1,bleu,bertscore_f1
florence2,300,0.4500,0.6234,0.4521,0.5987,0.3821,0.7234
qwen,300,0.5200,0.6987,0.5234,0.6543,0.4521,0.7891
```

## Common Use Cases

### Test with Small Dataset (10 samples)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_test \
    --models florence2 \
    --max-samples 10 \
    --eval-only
```

### Compare Local vs API Models

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_comparison \
    --models florence2 qwen gpt4o \
    --api-key sk-... \
    --eval-only
```

### Evaluate on Specific Question Type

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/action_command.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_action \
    --models florence2 qwen \
    --eval-only
```

### Use CPU (if no GPU)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_cpu \
    --models florence2 \
    --device cpu \
    --eval-only
```

## List Available Models

```bash
python vqa/evaluation/vqa2_pipeline.py --list-models
```

**Output:**
```
Available Models
================================================================

Local Models (GPU, FREE):
  - florence2
  - qwen
  - llava

API Models (OpenAI, PAID):
  - gpt4o
  - gpt5nano
  - gpt5mini

API Models (OpenRouter, PAID/FREE):
  - openrouter_trinity
  - openrouter_llama32_11b_vision
  - openrouter_llama4_maverick
  - openrouter_molmo_8b
  - openrouter_ministral_3b
  - openrouter_qwen3_vl_235b
  - openrouter_qwen3_vl_8b
  - openrouter_qwen_vl_plus
```

## Understanding Metrics

### Exact Match
- Perfect string match (case-insensitive)
- Range: 0.0 - 1.0 (higher is better)
- Example: "turn left" == "turn left" ✓, "turn left" == "go left" ✗

### ROUGE-1/2/L F1
- N-gram overlap between prediction and reference
- Range: 0.0 - 1.0 (higher is better)
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence

### BLEU
- Precision-based n-gram metric
- Range: 0.0 - 1.0 (higher is better)
- Commonly used in machine translation

### BERTScore
- Semantic similarity using BERT embeddings
- Range: 0.0 - 1.0 (higher is better)
- F1, Precision, Recall variants

### CLIP Score
- Vision-language similarity using CLIP
- Range: 0.0 - 1.0 (higher is better)
- Measures semantic similarity in multimodal space

## Troubleshooting

### "No module named 'vlm_factory'"

```bash
# Make sure you're running from the project root
cd /path/to/nature
python vqa/evaluation/vqa2_pipeline.py --list-models
```

### "CUDA out of memory"

```bash
# Option 1: Use smaller model
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 \
    --eval-only

# Option 2: Use CPU
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 \
    --device cpu \
    --eval-only
```

### "Image not found"

Make sure `--images-dir` points to the directory containing your images:

```bash
ls ./images/
# Should show: image_001.jpg, image_002.jpg, etc.
```

### "API key not provided"

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Or pass directly
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models gpt4o \
    --api-key sk-... \
    --eval-only
```

## Next Steps

1. **Generate MMVQA dataset** (if you haven't already):
   ```bash
   python vqa/evaluation/generate_per_question.py \
       --annotations-dir ./annotations \
       --output-dir ./vqa_mmvqa \
       --images-dir ./images \
       --per-image
   ```

2. **Transform to Standard VQA**:
   ```bash
   python vqa/evaluation/vqa2_pipeline.py \
       --mmvqa-dir ./vqa_mmvqa \
       --vqa-output-dir ./vqa_standard \
       --transform-only
   ```

3. **Evaluate models**:
   ```bash
   python vqa/evaluation/vqa2_pipeline.py \
       --vqa-dataset ./vqa_standard/per_image_all.json \
       --images-dir ./images \
       --eval-output-dir ./vqa_results \
       --models florence2 qwen \
       --eval-only
   ```

4. **View results**:
   ```bash
   cat vqa_results/comparison_summary.csv
   ```

## Full Documentation

See `README_VQA2.md` for complete documentation.
