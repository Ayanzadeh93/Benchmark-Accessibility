# VQA Evaluation Example Commands

Quick reference for common VQA generation and evaluation workflows.

## Variables (adjust to your paths)

```powershell
$ANNOTATIONS = "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\ammotation"
$VQA_OUT = "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa"
$IMAGES = "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images"
```

## Phase 6: Generate VQA Datasets

### Generate with Annotation Ground Truth (no model)

```powershell
python main.py vqa-grouped --annotations-dir $ANNOTATIONS --output-dir $VQA_OUT --images-dir $IMAGES --seed 1337 -v --per-image
```

**Outputs:**
- Per-question: `{VQA_OUT}\*.json` (main_obstacle, closest_obstacle, etc.)
- Per-image: `{VQA_OUT}\per_image\*.json`
- Combined: `{VQA_OUT}\per_image_all.json`

### Generate with LLM Ground Truth (e.g. Qwen 235B)

```powershell
python main.py vqa-grouped --annotations-dir $ANNOTATIONS --output-dir $VQA_OUT --images-dir $IMAGES --ground-truth-model openrouter_qwen3_vl_235b --seed 1337 -v --per-image
```

**Note:** Requires OpenRouter API key and credits for Qwen models.

### Resume/Skip Already Processed Images

```powershell
python main.py vqa-grouped --annotations-dir $ANNOTATIONS --output-dir $VQA_OUT --images-dir $IMAGES --ground-truth-model openrouter_qwen3_vl_235b --seed 1337 -v --per-image --skip-existing-per-image
```

Reuses existing per-image JSONs; only processes new images.

## VQA Model Evaluation

**Note: Batch mode is ON by default** (1 API call per image, 6x cheaper than sequential mode)

### Trial 1: Llama 4 Maverick (Batch Mode)

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T1" `
  --model-name openrouter_llama4_maverick `
  -v
```

**Batch mode**: Asks all 6 questions per image in one call. Model responds with "Q1: A, Q2: B, Q3: C, Q4: D, Q5: A, Q6: B"

### Trial 2: Qwen 3 VL 235B

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T2" `
  --model-name openrouter_qwen3_vl_235b `
  -v
```

### Trial 3: Trinity (Free)

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T3" `
  --model-name openrouter_trinity `
  -v
```

### Trial 4: Qwen VL Plus

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T4" `
  --model-name openrouter_qwen_vl_plus `
  -v
```

### Trial 5: Molmo 8B (Free)

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T5" `
  --model-name openrouter_molmo_8b `
  --max-samples 100 `
  -v
```

## Quick Test (Limited Samples)

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T_test" `
  --model-name openrouter_llama4_maverick `
  --max-samples 50 `
  -v
```

## Full Pipeline Example

```powershell
# 1. Generate VQA dataset with Qwen ground truth
python main.py vqa-grouped `
  --annotations-dir $ANNOTATIONS `
  --output-dir $VQA_OUT `
  --images-dir $IMAGES `
  --ground-truth-model openrouter_qwen3_vl_235b `
  --seed 1337 -v `
  --per-image

# 2. Evaluate Llama 4 Maverick (T1)
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T1" `
  --model-name openrouter_llama4_maverick `
  -v

# 3. Evaluate Trinity (T2, free)
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T2" `
  --model-name openrouter_trinity `
  -v

# 4. Compare results
# Check T1\openrouter_llama4_maverick_summary.csv
# Check T2\openrouter_trinity_summary.csv
```

## Per-Question Evaluation

Evaluate on a specific question type only:

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\main_obstacle.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T1_main_obstacle_only" `
  --model-name openrouter_llama4_maverick `
  -v
```

## Disable Batch Mode (Sequential)

If batch mode fails or for debugging:

```powershell
python main.py vqa-eval `
  --vqa-dataset "$VQA_OUT\per_image_all.json" `
  --images-dir $IMAGES `
  --output-dir "$VQA_OUT\T1_sequential" `
  --model-name openrouter_llama4_maverick `
  --no-batch-mode `
  -v
```

## Tips

1. **Batch mode is DEFAULT** and 6x cheaper - only disable if it fails
2. **Use T1-T5 folders** to organize trials without overwriting
3. **Test with --max-samples 10** before full runs (tests 10 images = 60 questions in batch mode)
4. **Free models** for initial testing: `openrouter_trinity`, `openrouter_molmo_8b`
5. **CSV files** are easiest for quick comparison in Excel
6. **Estimated time (batch mode)**: 
   - ~6-7s per image × 7546 images ≈ 14 hours
   - Sequential mode: ~42-49 hours (6x slower)
