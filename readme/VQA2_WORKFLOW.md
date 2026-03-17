# VQA2 Workflow Diagram

## Complete Pipeline Overview

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         VQA2 COMPLETE PIPELINE                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 0: Prerequisites (You should already have these)                   │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐         ┌──────────────────┐
    │   annotations/   │         │     images/      │
    │  ├─ image_001.   │         │  ├─ image_001.   │
    │  │  json         │         │  │  jpg          │
    │  ├─ image_002.   │         │  ├─ image_002.   │
    │  │  json         │         │  │  jpg          │
    │  └─ ...          │         │  └─ ...          │
    └──────────────────┘         └──────────────────┘
            │                            │
            └────────────┬───────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │ generate_per_question.py   │  ← Run this if you haven't
            └────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │      vqa_mmvqa/            │
            │  ├─ action_command.json    │
            │  ├─ main_obstacle.json     │
            │  ├─ closest_obstacle.json  │
            │  ├─ risk_assessment.json   │
            │  ├─ spatial_clock.json     │
            │  ├─ action_suggestion.json │
            │  └─ per_image_all.json     │ ← Multiple Choice VQA
            └────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Transform MMVQA → Standard VQA                                  │
│ Command: python vqa/evaluation/vqa2_pipeline.py \                       │
│              --mmvqa-dir ./vqa_mmvqa \                                   │
│              --vqa-output-dir ./vqa_standard \                           │
│              --transform-only                                            │
└─────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────┐
    │      vqa_mmvqa/            │
    │  ├─ action_command.json    │  ← Input: Multiple Choice
    │  │  {                       │
    │  │    "question": "...",    │
    │  │    "options": {          │
    │  │      "A": "Turn left",   │
    │  │      "B": "Go straight", │
    │  │      "C": "Stop",        │
    │  │      "D": "Turn right"   │
    │  │    },                    │
    │  │    "answer": "B"         │
    │  │  }                       │
    │  └─ ...                     │
    └────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ transform_mmvqa_to_vqa.py  │  ← Transformation
    └────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │     vqa_standard/          │
    │  ├─ action_command.json    │  ← Output: Standard VQA
    │  │  {                       │
    │  │    "question": "...",    │
    │  │    "answer": "Go straight"│ (No options!)
    │  │  }                       │
    │  └─ per_image_all.json     │ ← Use this for evaluation
    └────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Evaluate Models                                                 │
│ Command: python vqa/evaluation/vqa2_pipeline.py \                       │
│              --vqa-dataset ./vqa_standard/per_image_all.json \           │
│              --images-dir ./images \                                     │
│              --eval-output-dir ./vqa_results \                           │
│              --models florence2 qwen llava \                             │
│              --eval-only                                                 │
└─────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────┐        ┌──────────────────┐
    │  vqa_standard/             │        │    images/       │
    │  └─ per_image_all.json     │        │  ├─ image_001.   │
    │     {                       │        │  │  jpg          │
    │       "image": "...",       │        │  ├─ image_002.   │
    │       "questions": [        │        │  │  jpg          │
    │         {                   │        │  └─ ...          │
    │           "question": "...", │        └──────────────────┘
    │           "answer": "..."   │                │
    │         }                   │                │
    │       ]                     │                │
    │     }                       │                │
    └────────────────────────────┘                │
                 │                                 │
                 └────────────┬────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ vqa_standard_evaluation.py    │
              │                               │
              │ For each model:               │
              │  1. Load model                │
              │  2. For each image:           │
              │     - Ask questions           │
              │     - Get free-form answers   │
              │  3. Compute metrics           │
              └───────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
              ┌──────────┬──────────┬──────────┐
              │florence2 │  qwen    │  llava   │
              └──────────┴──────────┴──────────┘
                    │         │         │
                    └─────────┼─────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Compute Metrics│
                    │  - Exact Match  │
                    │  - ROUGE-1/2/L  │
                    │  - BLEU         │
                    │  - BERTScore    │
                    │  - CLIP Score   │
                    └─────────────────┘
                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: View Results                                                    │
│ Command: cat vqa_results/comparison_summary.csv                         │
└─────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────┐
    │                      vqa_results/                              │
    │  ├─ florence2/                                                 │
    │  │  ├─ florence2_evaluation.json    (Full results)             │
    │  │  ├─ florence2_summary.json       (Compact summary)          │
    │  │  └─ florence2_summary.csv        (CSV format)               │
    │  ├─ qwen/                                                      │
    │  │  ├─ qwen_evaluation.json                                    │
    │  │  ├─ qwen_summary.json                                       │
    │  │  └─ qwen_summary.csv                                        │
    │  ├─ llava/                                                     │
    │  │  ├─ llava_evaluation.json                                   │
    │  │  ├─ llava_summary.json                                      │
    │  │  └─ llava_summary.csv                                       │
    │  ├─ comparison_summary.json         (Compare all models)       │
    │  └─ comparison_summary.csv          ⭐ LOOK HERE FIRST!        │
    └────────────────────────────────────────────────────────────────┘

    Example: comparison_summary.csv
    ┌───────────────────────────────────────────────────────────────┐
    │ model    │ samples │ exact_match │ rouge1_f1 │ bertscore_f1  │
    │──────────┼─────────┼─────────────┼───────────┼───────────────│
    │ florence2│   300   │    0.4500   │   0.6234  │     0.7234    │
    │ qwen     │   300   │    0.5200   │   0.6987  │     0.7891    │
    │ llava    │   300   │    0.5800   │   0.7234  │     0.8234    │
    └───────────────────────────────────────────────────────────────┘
```

## Data Format Transformation

### Input: MMVQA (Multiple Choice)

```json
{
  "id": "image_001|action_command",
  "question_id": "action_command",
  "image": "image_001.jpg",
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

### Output: Standard VQA

```json
{
  "id": "image_001|action_command",
  "question_id": "action_command",
  "image": "image_001.jpg",
  "question": "What should the person do?",
  "answer": "Go straight"
}
```

### Key Difference

| MMVQA | Standard VQA |
|-------|--------------|
| Model chooses from A/B/C/D | Model generates free-form text |
| Easy (guided by options) | Hard (no hints) |
| Evaluated by exact label match | Evaluated by text similarity |
| Fast inference | Slower inference |

## Evaluation Process

```
For each image:
    For each question:
        1. Ask model: "What should the person do?"
        2. Model generates: "Go forward"          ← Free-form answer
        3. Compare with reference: "Go straight"  ← Ground truth
        4. Compute similarity:
           - Exact match: 0.0 (not exact)
           - ROUGE-1 F1: 0.5 (50% word overlap)
           - BERTScore: 0.85 (semantically similar)
```

## Metrics Explained

```
┌─────────────────────────────────────────────────────────────────┐
│ Metric         │ What it measures           │ Range │ Higher = │
│────────────────┼────────────────────────────┼───────┼──────────│
│ exact_match    │ Perfect string match       │ 0-1   │ Better   │
│ rouge1_f1      │ Word overlap (unigrams)    │ 0-1   │ Better   │
│ rouge2_f1      │ Phrase overlap (bigrams)   │ 0-1   │ Better   │
│ rougeL_f1      │ Longest common subsequence │ 0-1   │ Better   │
│ bleu           │ Precision-based overlap    │ 0-1   │ Better   │
│ bertscore_f1   │ Semantic similarity (BERT) │ 0-1   │ Better   │
│ clip_score     │ Multimodal similarity      │ 0-1   │ Better   │
└─────────────────────────────────────────────────────────────────┘
```

### Examples

```
Prediction: "Go straight"
Reference:  "Go straight"
→ exact_match: 1.0, rouge1_f1: 1.0, bertscore: 1.0

Prediction: "Go forward"
Reference:  "Go straight"
→ exact_match: 0.0, rouge1_f1: 0.5, bertscore: 0.85

Prediction: "Turn left"
Reference:  "Go straight"
→ exact_match: 0.0, rouge1_f1: 0.0, bertscore: 0.35
```

## Model Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│ Model      │ Speed       │ Quality │ Cost  │ Requirements      │
│────────────┼─────────────┼─────────┼───────┼───────────────────│
│ florence2  │ ⚡⚡⚡ Fastest│ 🟢 Good │ FREE  │ GPU (230M params) │
│ qwen       │ ⚡⚡ Fast    │ 🟢 Good │ FREE  │ GPU (2B params)   │
│ llava      │ ⚡ Medium   │ 🟢 Good │ FREE  │ GPU (7B params)   │
│ gpt4o      │ ⚡ Medium   │ 🟢 Best │ $$$   │ API key           │
│ gpt5nano   │ ⚡⚡ Fast   │ 🟢 Great│ $     │ API key           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Commands Reference

### List Models
```bash
python vqa/evaluation/vqa2_pipeline.py --list-models
```

### Transform Only
```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --transform-only
```

### Evaluate Only (Fast Test)
```bash
python vqa/evaluation/vqa2_pipeline.py \
    --vqa-dataset ./vqa_standard/per_image_all.json \
    --images-dir ./images \
    --eval-output-dir ./vqa_results_test \
    --models florence2 \
    --max-samples 10 \
    --eval-only
```

### Full Pipeline (Transform + Evaluate)
```bash
python vqa/evaluation/vqa2_pipeline.py \
    --mmvqa-dir ./vqa_mmvqa \
    --vqa-output-dir ./vqa_standard \
    --images-dir ./images \
    --eval-output-dir ./vqa_results \
    --models florence2 qwen llava
```

### View Results
```bash
cat vqa_results/comparison_summary.csv
```

## Summary

**3 Steps to Success:**

1. **Transform** MMVQA → Standard VQA (1 command, ~1 minute)
2. **Evaluate** models (1 command, ~1-3 hours depending on dataset)
3. **View** results in CSV (open `comparison_summary.csv`)

**That's it!** 🎉
