# VQA2 Evaluation - Run Commands

## Batch Mode (1 call per image)

**VQA evaluation uses batch mode by default** – all questions for each image are sent in ONE API/model call (like MMVQA). This makes evaluation much faster (seconds per image instead of seconds per question).

- **Ground truth (VQA2):** `C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json`
- **Output:** Save results to a folder of your choice

## Quick Start

### 1. Quick Test (10 samples, Florence2, 50-word limit)

```powershell
cd C:\Tim\nature

python -m vqa.evaluation.evaluate_vqa_simple `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" `
    --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results_test" `
    --models florence2 `
    --max-samples 10
```

### 2. Run Qwen 8B (OpenRouter)

```powershell
cd C:\Tim\nature

$env:OPENROUTER_API_KEY = "your-key-here"

# Use "qwen8b" (shortcut) or "openrouter_qwen3_vl_8b"
python -m vqa.evaluation.evaluate_vqa_simple `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" `
    --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results_qwen8b" `
    --models qwen8b `
    --api-key $env:OPENROUTER_API_KEY
```

### 3. Run All Local Models (Florence2, Qwen, LLaVA)

```powershell
cd C:\Tim\nature

python -m vqa.evaluation.evaluate_vqa_simple `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" `
    --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results" `
    --all-models
```

### 4. Run Multiple Models with Custom Word Limit

```powershell
cd C:\Tim\nature

python -m vqa.evaluation.evaluate_vqa_simple `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" `
    --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results" `
    --models florence2 qwen llava openrouter_qwen3_vl_8b `
    --max-words 50 `
    --api-key $env:OPENROUTER_API_KEY
```

## What Happens

1. **Load** VQA2 ground truth (question + correct answer per sample)
2. **Ask** each model the question (with "Answer in 50 words or less")
3. **Limit** model response to 50 words (or your `--max-words`)
4. **Compare** model answer vs ground truth
5. **Compute** ROUGE-1/2/L, BLEU, BERTScore
6. **Save** results per model + comparison CSV

## Output Structure

```
results/
├── florence2/
│   ├── florence2_results.json       # Full predictions (all images)
│   ├── florence2_summary.csv        # Metrics
│   └── per_image/                   # Per-image results for this model
│       ├── image_001.json
│       ├── image_002.json
│       └── ...
├── qwen8b/
│   ├── qwen8b_results.json
│   ├── qwen8b_summary.csv
│   └── per_image/
│       ├── image_001.json
│       └── ...
├── per_image_all_models/            # Combined: all models' predictions per image
│   ├── image_001.json               # Contains florence2 + qwen8b + ... for this image
│   ├── image_002.json
│   └── ...
├── comparison.json
└── comparison.csv   ⭐ Compare all models
```

## Available Models

### Local GPU (no API key)
- `florence2` - Fastest, 230M params
- `qwen` - Qwen3-VL-2B (local)
- `llava` - LLaVA 1.5 7B

### OpenRouter (need OPENROUTER_API_KEY)
- `qwen8b` or `openrouter_qwen3_vl_8b` - **Qwen 8B**
- `openrouter_qwen3_vl_235b` - Qwen 235B
- `openrouter_llama4_maverick`
- `openrouter_molmo_8b` - Free
- `openrouter_trinity` - Free

### OpenAI (need OPENAI_API_KEY)
- `gpt4o`
- `gpt5nano`
- `gpt5mini`

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ground-truth` | required | Path to per_image_all.json |
| `--output-dir` | required | Where to save results |
| `--models` | florence2 | Models to evaluate |
| `--all-models` | - | Use florence2, qwen, llava |
| `--max-words` | 50 | Limit model response length |
| `--max-samples` | - | Limit samples (for testing) |
| `--api-key` | - | For API models |
| `--device` | auto | cuda, cpu, or auto |

## Single-Line Commands

**Test (10 samples):**
```
python -m vqa.evaluation.evaluate_vqa_simple --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results_test" --models florence2 --max-samples 10
```

**Full evaluation (all local models):**
```
python -m vqa.evaluation.evaluate_vqa_simple --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results" --all-models
```

**Qwen 8B only:**
```
python -m vqa.evaluation.evaluate_vqa_simple --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\per_image_all.json" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA2\results_qwen8b" --models openrouter_qwen3_vl_8b --api-key YOUR_OPENROUTER_KEY
```
