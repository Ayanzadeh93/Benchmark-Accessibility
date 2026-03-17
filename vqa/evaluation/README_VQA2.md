# VQA2: Standard VQA Evaluation System

This system transforms Multiple Choice VQA (MMVQA) datasets to standard VQA format and evaluates multiple VLM models on the resulting dataset.

## Overview

**What's the difference?**

- **MMVQA (Multiple Choice VQA)**: Models choose from options A/B/C/D
- **Standard VQA (VQA2)**: Models generate free-form text answers, evaluated using text similarity metrics

**Pipeline:**
1. Transform MMVQA → Standard VQA (extract correct answer text, remove choices)
2. Evaluate multiple VLM models on Standard VQA
3. Compare models using text similarity metrics (ROUGE, BLEU, BERTScore, CLIP)

## Quick Start

### 1. Transform MMVQA to Standard VQA

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir path/to/mmvqa \
    --vqa-output-dir path/to/standard_vqa \
    --transform-only
```

**Input format (MMVQA):**
```json
{
  "id": "image_001|action_command",
  "question": "What should the person do?",
  "options": {
    "A": "Turn left",
    "B": "Go straight",
    "C": "Stop",
    "D": "Turn right"
  },
  "answer": "B",
  "answer_text": "Go straight"
}
```

**Output format (Standard VQA):**
```json
{
  "id": "image_001|action_command",
  "question": "What should the person do?",
  "answer": "Go straight"
}
```

### 2. Evaluate Models on Standard VQA

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/standard_vqa/per_image_all.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 qwen llava \
    --eval-only
```

### 3. Full Pipeline (Transform + Evaluate)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir path/to/mmvqa \
    --vqa-output-dir path/to/standard_vqa \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 qwen llava gpt4o \
    --api-key sk-...
```

## Available Models

### Local Models (GPU, FREE)
- **florence2**: Fastest (1-2s per image), 230M params
- **qwen**: Fast (2-3s per image), 2B params
- **llava**: Medium (3-5s per image), 7B params

### API Models (OpenAI, PAID)
- **gpt4o**: High quality, ~$0.005 per image
- **gpt5nano**: Good quality, ~$0.0005 per image
- **gpt5mini**: Budget option, cheapest

### API Models (OpenRouter, PAID/FREE)
- **openrouter_trinity**: Free
- **openrouter_llama32_11b_vision**: Paid
- **openrouter_llama4_maverick**: Paid
- **openrouter_molmo_8b**: Free
- **openrouter_ministral_3b**: Paid
- **openrouter_qwen3_vl_235b**: Paid, best quality
- **openrouter_qwen3_vl_8b**: Paid
- **openrouter_qwen_vl_plus**: Paid

### List All Models

```bash
python vqa/evaluation/vqa2_pipeline.py --list-models
```

## Advanced Usage

### Evaluate Specific Models

```bash
# Local GPU models only
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 qwen \
    --device cuda \
    --eval-only

# API models (requires API key)
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models gpt4o openrouter_qwen3_vl_235b \
    --api-key sk-... \
    --eval-only
```

### Limit Samples (Testing)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 qwen \
    --max-samples 10 \
    --eval-only
```

### Transform Per-Question Files Only

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir path/to/mmvqa \
    --vqa-output-dir path/to/standard_vqa \
    --per-question-only \
    --transform-only
```

## Output Structure

### After Transformation

```
standard_vqa/
├── action_command.json          # Per-question VQA datasets
├── main_obstacle.json
├── closest_obstacle.json
├── risk_assessment.json
├── spatial_clock.json
├── action_suggestion.json
├── per_image_all.json           # All questions grouped by image
└── transformation_summary.json  # Transformation stats
```

### After Evaluation

```
results/
├── florence2/
│   ├── florence2_evaluation.json   # Full results with predictions
│   ├── florence2_summary.json      # Compact summary
│   └── florence2_summary.csv       # CSV for easy viewing
├── qwen/
│   ├── qwen_evaluation.json
│   ├── qwen_summary.json
│   └── qwen_summary.csv
├── comparison_summary.json         # Compare all models
└── comparison_summary.csv          # CSV comparison
```

## Evaluation Metrics

Standard VQA uses text similarity metrics:

1. **Exact Match**: Exact string match (case-insensitive)
2. **ROUGE-1/2/L F1**: N-gram overlap (1-gram, 2-gram, longest common subsequence)
3. **BLEU**: Precision-based n-gram metric
4. **BERTScore F1/Precision/Recall**: Semantic similarity using BERT embeddings
5. **CLIP Score**: Vision-language similarity using CLIP embeddings

### Per-Question Metrics

Results are broken down by question type:
- action_command
- main_obstacle
- closest_obstacle
- risk_assessment
- spatial_clock
- action_suggestion

## API Keys

### OpenAI (GPT-4o, GPT-5)

```bash
export OPENAI_API_KEY="sk-..."

python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models gpt4o gpt5nano \
    --api-key $OPENAI_API_KEY \
    --eval-only
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."

python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models openrouter_qwen3_vl_235b \
    --api-key $OPENROUTER_API_KEY \
    --eval-only
```

## Programmatic Usage

### Transform Only

```python
from vqa.evaluation.transform_mmvqa_to_vqa import transform_directory

summary = transform_directory(
    input_dir="path/to/mmvqa",
    output_dir="path/to/standard_vqa",
    per_question_only=False,
)
```

### Evaluate Single Model

```python
from vqa.evaluation.vqa_standard_evaluation import StandardVQAEvaluator

evaluator = StandardVQAEvaluator(
    model_name="florence2",
    model_type="florence2",
    device="cuda",
)

result = evaluator.evaluate(
    vqa_dataset_path="path/to/vqa.json",
    images_dir="path/to/images",
    output_dir="path/to/results",
    max_samples=None,
    save_predictions=True,
)
```

### Evaluate Multiple Models

```python
from vqa.evaluation.vqa_standard_evaluation import run_multi_model_evaluation

models = [
    {"name": "florence2", "type": "florence2", "device": "cuda"},
    {"name": "qwen", "type": "qwen", "device": "cuda"},
    {"name": "gpt4o", "type": "gpt4o", "api_key": "sk-..."},
]

results = run_multi_model_evaluation(
    vqa_dataset_path="path/to/vqa.json",
    images_dir="path/to/images",
    output_base_dir="path/to/results",
    models=models,
    max_samples=None,
    save_predictions=True,
    verbose=False,
)
```

## Comparison with MMVQA Evaluation

| Feature | MMVQA (Multiple Choice) | VQA2 (Standard VQA) |
|---------|------------------------|---------------------|
| Answer format | A/B/C/D label | Free-form text |
| Model task | Classification (4 choices) | Generation (open-ended) |
| Evaluation | Exact match on label | Text similarity metrics |
| Metrics | Accuracy (%) | ROUGE, BLEU, BERTScore, CLIP |
| Speed | Fast (1 API call = 1 answer) | Slower (model generates text) |
| Cost (API) | Low (simple classification) | Higher (text generation) |
| Realism | Less realistic (hints from options) | More realistic (no hints) |

## Troubleshooting

### GPU Out of Memory

If you get OOM errors with local models:

```bash
# Use CPU instead
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 \
    --device cpu \
    --eval-only

# Or use a smaller model
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models florence2 \
    --eval-only  # florence2 is smallest (230M params)
```

### API Rate Limits

If you hit rate limits with API models, reduce batch size or add delays:

```bash
# Evaluate fewer samples
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset path/to/vqa.json \
    --images-dir path/to/images \
    --eval-output-dir path/to/results \
    --models gpt4o \
    --max-samples 100 \
    --eval-only
```

### Missing Metrics

If you get errors computing metrics (BERTScore, CLIP), install dependencies:

```bash
pip install bert-score rouge-score sacrebleu transformers clip-by-openai
```

## Examples

### Example 1: Quick Test (10 samples, local model)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_test \
    --models florence2 \
    --max-samples 10
```

### Example 2: Full Evaluation (all local models)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_local \
    --models florence2 qwen llava
```

### Example 3: Compare Local vs API Models

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_comparison \
    --models florence2 qwen gpt4o \
    --api-key sk-...
```

### Example 4: API Models Only (OpenRouter)

```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_openrouter \
    --models openrouter_qwen3_vl_235b openrouter_llama4_maverick \
    --api-key sk-or-... \
    --eval-only
```

## Architecture

```
vqa/evaluation/
├── transform_mmvqa_to_vqa.py       # Transform MMVQA → Standard VQA
├── vqa_standard_evaluation.py     # Evaluate models on Standard VQA
├── vqa2_pipeline.py                # Main CLI (combines both)
├── metrics.py                       # Evaluation metrics
├── eval_schemas.py                  # Pydantic schemas for results
└── README_VQA2.md                  # This file
```

## Related Files

- `generate_per_question.py`: Generate MMVQA datasets from annotations
- `model_evaluation.py`: Evaluate models on MMVQA (multiple choice)
- `ground_truth_model.py`: Use VLM to generate ground truth answers

## References

- VQA2 Dataset: https://visualqa.org/
- ROUGE Metric: https://github.com/google-research/google-research/tree/master/rouge
- BERTScore: https://github.com/Tiiiger/bert_score
- CLIP: https://github.com/openai/CLIP
