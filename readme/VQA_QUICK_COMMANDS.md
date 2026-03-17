# VQA Evaluation - Quick Commands

## Your Setup

**Ground Truth:** `C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json`

**Images:** Paths are in your ground truth JSON (image_path field)

## Commands (Copy & Paste)

### 1. Quick Test (10 samples, Florence2)

```powershell
cd C:\Tim\nature

python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" `
    --output-dir "C:\Tim\nature\vqa_results_test" `
    --models florence2 `
    --max-samples 10
```

**Time:** ~30 seconds  
**Check:** `cat vqa_results_test\florence2\florence2_summary.csv`

---

### 2. Full Evaluation (All Local Models)

```powershell
cd C:\Tim\nature

python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" `
    --output-dir "C:\Tim\nature\vqa_results" `
    --models florence2 qwen llava
```

**Time:** ~2-4 hours  
**Check:** `cat vqa_results\comparison.csv`

---

### 3. Evaluate Specific Question Type

```powershell
cd C:\Tim\nature

python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\action_command.json" `
    --output-dir "C:\Tim\nature\vqa_results_action" `
    --models florence2 qwen
```

**Time:** ~30-60 minutes per model  
**Check:** `cat vqa_results_action\comparison.csv`

---

### 4. CPU Only (No GPU)

```powershell
cd C:\Tim\nature

python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" `
    --output-dir "C:\Tim\nature\vqa_results_cpu" `
    --models florence2 `
    --device cpu `
    --max-samples 10
```

---

### 5. With API Models (GPT-4o)

```powershell
cd C:\Tim\nature

$env:OPENAI_API_KEY="sk-..."

python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" `
    --output-dir "C:\Tim\nature\vqa_results_api" `
    --models florence2 gpt4o `
    --api-key $env:OPENAI_API_KEY
```

---

## View Results

```powershell
# View comparison of all models
cat vqa_results\comparison.csv

# View specific model results
cat vqa_results\florence2\florence2_summary.csv

# Open in Excel
start vqa_results\comparison.csv
```

---

## Expected Output

### comparison.csv
```
model,total_samples,exact_match,rouge1_f1,rouge2_f1,rougeL_f1,bleu,bertscore_f1,avg_inference_time_s
florence2,6977,0.4234,0.6123,0.4521,0.5876,0.3654,0.7234,1.234
qwen,6977,0.5123,0.6987,0.5432,0.6543,0.4321,0.7891,2.345
llava,6977,0.5876,0.7456,0.6123,0.7012,0.5123,0.8234,3.456
```

### florence2_summary.csv
```
metric,value
model,florence2
total_samples,6977
exact_match,0.4234
rouge1_f1,0.6123
rouge2_f1,0.4521
rougeL_f1,0.5876
bleu,0.3654
bertscore_f1,0.7234
avg_inference_time_s,1.234

question_type,accuracy,correct,total
action_command,0.4500,3140,6977
main_obstacle,0.4100,2860,6977
...
```

---

## Metrics Explained

| Metric | What it measures | Good value |
|--------|------------------|------------|
| exact_match | Perfect string match | > 0.40 |
| rouge1_f1 | Word overlap (unigrams) | > 0.60 |
| rouge2_f1 | Phrase overlap (bigrams) | > 0.45 |
| rougeL_f1 | Longest common subsequence | > 0.55 |
| bleu | Precision-based overlap | > 0.35 |
| bertscore_f1 | Semantic similarity (BERT) | > 0.70 |

---

## Available Models

### Local GPU (FREE)
- `florence2` - Fastest (1-2s/image), 230M params
- `qwen` - Fast (2-3s/image), 2B params
- `llava` - Medium (3-5s/image), 7B params

### API (PAID)
- `gpt4o` - High quality, ~$0.005/image
- `gpt5nano` - Good quality, ~$0.0005/image

---

## Troubleshooting

### "No module named 'vlm_factory'"
```powershell
cd C:\Tim\nature
python vqa/evaluation/evaluate_vqa_simple.py --help
```

### "CUDA out of memory"
```powershell
# Use smaller model
--models florence2

# Or use CPU
--device cpu --max-samples 10
```

### Check if it's working
```powershell
# Run quick test first
python vqa/evaluation/evaluate_vqa_simple.py `
    --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" `
    --output-dir "C:\Tim\nature\vqa_results_test" `
    --models florence2 `
    --max-samples 10
```

---

## Summary

**1 command to test:**
```powershell
python vqa/evaluation/evaluate_vqa_simple.py --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" --output-dir "C:\Tim\nature\vqa_results_test" --models florence2 --max-samples 10
```

**1 command for full evaluation:**
```powershell
python vqa/evaluation/evaluate_vqa_simple.py --ground-truth "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\vqa\per_image_all.json" --output-dir "C:\Tim\nature\vqa_results" --models florence2 qwen llava
```

**1 command to view results:**
```powershell
cat vqa_results\comparison.csv
```

That's it! 🚀
