# VQA Model Evaluation

Comprehensive evaluation system for testing Vision-Language Models (VLMs) on VQA navigation tasks.

## Overview

This pipeline evaluates models on VQA datasets by:
1. Loading VQA dataset (per-question or per-image JSONs)
2. Running model inference on each (image, question, options) sample
3. Computing metrics: Accuracy, ROUGE, BLEU
4. Saving detailed results in organized trial folders (T1, T2, T3, etc.)

### 🚀 Batch Mode (Default)

**Batch mode is enabled by default** and makes evaluation **6x cheaper and faster**!

- **Without batch mode**: 6 API calls per image (one per question) = 7,546 images × 6 = 45,276 API calls
- **With batch mode**: 1 API call per image (all 6 questions at once) = 7,546 API calls

**Cost savings**: ~$450 → ~$75 for 7,546 images (Llama 4 Maverick pricing)

## Metrics Computed

### Core Metrics
- **Accuracy**: Exact match for predicted vs ground truth answer (A/B/C/D)
- **Per-Question Accuracy**: Breakdown by question type (main_obstacle, closest_obstacle, risk_assessment, spatial_clock, action_suggestion, action_command)

### Text Similarity Metrics
- **ROUGE-1/2/L F1**: Measures n-gram overlap between predicted and reference answer texts
- **BLEU**: Sentence-level BLEU score for answer text quality

### Performance Metrics
- **Avg Inference Time**: Average time per sample
- **Total Inference Time**: Total evaluation duration

## Usage

### Basic Evaluation (Batch Mode - Default)

```powershell
python main.py vqa-eval --vqa-dataset "C:\path\to\vqa\per_image_all.json" --images-dir "C:\path\to\images" --output-dir "C:\path\to\vqa\T1" --model-name openrouter_llama4_maverick -v
```

**Batch mode ON by default** - asks all 6 questions per image in one API call. Response format: "Q1: A, Q2: B, Q3: C, Q4: D, Q5: A, Q6: B"

### Disable Batch Mode (Sequential)

```powershell
python main.py vqa-eval --vqa-dataset "C:\path\to\vqa\per_image_all.json" --images-dir "C:\path\to\images" --output-dir "C:\path\to\vqa\T1" --model-name openrouter_llama4_maverick --no-batch-mode -v
```

Only use `--no-batch-mode` if batch mode fails or for debugging.

### Organized Trial Structure

Use T1, T2, T3, T4, T5 folders for different evaluation runs:

```powershell
# Trial 1: Llama 4 Maverick
python main.py vqa-eval --vqa-dataset "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\T1" --model-name openrouter_llama4_maverick -v

# Trial 2: Qwen 3 VL 235B
python main.py vqa-eval --vqa-dataset "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\T2" --model-name openrouter_qwen3_vl_235b -v

# Trial 3: Trinity (free)
python main.py vqa-eval --vqa-dataset "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\T3" --model-name openrouter_trinity -v
```

### Evaluate on Per-Question Dataset

You can also use individual per-question files:

```powershell
python main.py vqa-eval --vqa-dataset "C:\path\to\vqa\main_obstacle.json" --images-dir "C:\path\to\images" --output-dir "C:\path\to\vqa\T1" --model-name openrouter_llama4_maverick -v
```

### Test on Limited Samples

```powershell
python main.py vqa-eval --vqa-dataset "C:\path\to\vqa\per_image_all.json" --images-dir "C:\path\to\images" --output-dir "C:\path\to\vqa\T1_test" --model-name openrouter_llama4_maverick --max-samples 50 -v
```

## Output Structure

Each evaluation run (e.g. in `T1/`) produces:

### 1. Full Results JSON
`{model_name}_evaluation.json`
- All predictions with details
- Individual sample results
- Complete metrics

### 2. Summary JSON
`{model_name}_summary.json`
- Overall accuracy
- Per-question breakdown
- Key metrics only (no predictions)

### 3. CSV Summary
`{model_name}_summary.csv`
- Easy-to-read format
- Overall + per-question metrics
- Import into Excel/Google Sheets

## Example Output

```
============================================================
VQA MODEL EVALUATION COMPLETE
============================================================
Model: openrouter_llama4_maverick
Overall Accuracy: 0.7842
Total Samples: 7546
Correct: 5916
Failed: 23
ROUGE-1 F1: 0.8123
BLEU: 0.7456
Avg Inference Time: 6.234s

Per-Question Accuracy:
  main_obstacle: 0.8234 (6214/7546)
  closest_obstacle: 0.7645 (5769/7546)
  risk_assessment: 0.8012 (6045/7546)
  spatial_clock: 0.7123 (5375/7546)
  action_suggestion: 0.8345 (6299/7546)
  action_command: 0.8234 (6214/7546)

Results saved to: C:\path\to\vqa\T1
============================================================
```

## Supported Models

### OpenRouter Models
- `openrouter_llama4_maverick` - Meta Llama 4 Maverick
- `openrouter_qwen3_vl_235b` - Qwen 3 VL 235B (powerful)
- `openrouter_qwen3_vl_8b` - Qwen 3 VL 8B Instruct
- `openrouter_qwen_vl_plus` - Qwen VL Plus
- `openrouter_trinity` - Trinity Large Preview (free)
- `openrouter_molmo_8b` - Molmo 2 8B (free)
- `openrouter_llama32_11b_vision` - Llama 3.2 11B Vision
- `openrouter_ministral_3b` - Ministral 3B
- `openrouter_gpt_oss_safeguard_20b` - GPT OSS Safeguard 20B

### Requirements
- OpenRouter API key in `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`
- Images must be accessible at paths specified in VQA dataset

## Dependencies

```bash
pip install rouge-score nltk
```

Optional: Download NLTK data for BLEU:
```python
import nltk
nltk.download('punkt')
```

## Trial Folder Organization

Use T1–T5 (or more) to organize different evaluation experiments:

- **T1**: Baseline model (e.g. Llama 4 Maverick)
- **T2**: Powerful model (e.g. Qwen 235B)
- **T3**: Free alternative (e.g. Trinity)
- **T4**: Different prompt/config variations
- **T5**: Ensemble or hybrid approach

This structure allows easy comparison across trials without overwriting results.

## Comparison Across Models

After running multiple evaluations (T1, T2, T3), compare using the summary files:

```python
import json

trials = ["T1", "T2", "T3"]
for trial in trials:
    path = f"C:/path/to/vqa/{trial}/openrouter_*_summary.json"
    # Load and compare overall_accuracy, per_question_accuracy
```

Or manually compare CSV files in Excel.

## Notes

- **Ground Truth**: The VQA dataset's `answer` field is used as ground truth (either annotation-derived or from a ground-truth model like Qwen 235B)
- **API Costs**: 
  - **Batch mode (default)**: 1 API call per image = 7,546 calls for 7,546 images (~$75 for Llama 4 Maverick)
  - **Sequential mode**: 6 API calls per image = 45,276 calls for 7,546 images (~$450 for Llama 4 Maverick)
- **Resume Support**: Not yet implemented; re-running overwrites results
- **Batch Mode**: Asks all questions for an image in one prompt, expects short responses like "Q1: A, Q2: B"

## Related Commands

- `vqa`: Generate VQA dataset (tuning/eval splits)
- `vqa-grouped`: Generate VQA grouped by question type
- `annotate`: Phase 5 annotation (creates the source data for VQA)
