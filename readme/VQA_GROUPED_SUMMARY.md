# VQA Grouped Generation - Complete Implementation

## What Was Built

A new VQA generation system that creates **one file per question type** instead of one file per image.

For **7,721 images** × **6 question types** = **6 output files** (instead of 46,326 files)

---

## 📁 Files Created

1. **`vqa/evaluation/generate_per_question.py`** - Main generator
2. **`vqa/evaluation/README_PER_QUESTION.md`** - Full documentation
3. **CLI integration in `main.py`** - New `vqa-grouped` command

---

## 🚀 Quick Start

### Your Exact Command

```powershell
python main.py vqa-grouped --annotations-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\ammotation\annotations" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA_grouped" --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images"
```

### Test with 10 Images First

```powershell
python main.py vqa-grouped --annotations-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\ammotation\annotations" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA_test" --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images" --max-samples 10 --verbose
```

---

## 📊 Output Structure

```
VQA_grouped/
├── main_obstacle.json          (7,721 samples)
├── closest_obstacle.json        (7,721 samples)
├── risk_assessment.json         (7,721 samples)
├── spatial_clock.json           (7,721 samples)
├── action_suggestion.json       (7,721 samples)
├── action_command.json          (7,721 samples)
└── vqa_generation_summary.json  (statistics)
```

---

## 📝 6 Question Types

| # | Question ID | Question Text |
|---|-------------|---------------|
| 1 | `main_obstacle` | What is the main obstacle or barrier in this scene? |
| 2 | `closest_obstacle` | Which object is closest to the user? |
| 3 | `risk_assessment` | How safe is this scene for blind users? |
| 4 | `spatial_clock` | Locate the {object} based on clock direction. |
| 5 | `action_suggestion` | What action do you suggest to the user in this scene? |
| 6 | `action_command` | What is the recommended navigation action for a blind user? |

All are **multiple-choice (A/B/C/D)**.

---

## 📄 Example Output File (`main_obstacle.json`)

```json
{
  "question_id": "main_obstacle",
  "question": "What is the main obstacle or barrier in this scene?",
  "num_samples": 7721,
  "samples": [
    {
      "id": "00d343d4a829121c_jpg|main_obstacle",
      "image": "00d343d4a829121c_jpg.rf.2d4285585f5ddea08381cc2ecee4b7af.jpg",
      "question_id": "main_obstacle",
      "question": "What is the main obstacle or barrier in this scene?",
      "options": {
        "A": "Large window",
        "B": "Brick wall",
        "C": "Wooden cabinets",
        "D": "No obstacle"
      },
      "answer": "A",
      "ground_truth": {
        "main_obstacle": "large window"
      },
      "image_path": "C:\\Tim\\Database\\...\\images\\00d343d4a829121c_jpg.rf.2d4285585f5ddea08381cc2ecee4b7af.jpg"
    },
    ...
  ],
  "metadata": {
    "generated_from": "...",
    "total_images": 7721,
    "seed": 1337
  }
}
```

---

## ✅ What This Does

1. **Reads** all annotation JSONs from your annotations directory
2. **Extracts** navigation data (scene, risk, obstacles, guidance)
3. **Generates** 6 VQA questions per image
4. **Groups** all images' Q&A by question type
5. **Saves** 6 JSON files (one per question) + 1 summary

---

## 🎯 Benefits

| Before (per-image) | After (per-question) |
|--------------------|---------------------|
| 46,326 files | **6 files** |
| Hard to analyze by question | **Easy per-question analysis** |
| Large disk usage | **Compact** |
| Requires aggregation | **Ready to use** |

---

## 🔍 Use Cases

1. ✅ **Evaluate models** on specific question types
2. ✅ **Analyze answer distribution** per question
3. ✅ **Debug** specific question generation
4. ✅ **Train models** on individual question types
5. ✅ **Benchmark** performance by question category

---

## 📌 Key Features

- ✅ Deterministic generation (seed-based)
- ✅ Handles missing/incomplete data gracefully
- ✅ Progress bar with tqdm
- ✅ Verbose logging option
- ✅ Absolute image paths (if `--images-dir` provided)
- ✅ Error handling and failure tracking
- ✅ Summary statistics
- ✅ Supports max-samples for testing

---

## 🛠️ Command Options

```
Required:
  --annotations-dir   Directory with annotation JSON files
  --output-dir        Where to save question-type files

Optional:
  --images-dir        Images directory (adds absolute paths)
  --max-samples N     Process only N annotations (testing)
  --seed N            Random seed (default: 1337)
  -v, --verbose       Verbose logging
```

---

## 📚 Documentation

- **Full docs:** `vqa/evaluation/README_PER_QUESTION.md`
- **Code:** `vqa/evaluation/generate_per_question.py`
- **CLI:** `main.py` (command: `vqa-grouped`)

---

## 🎉 Ready to Use!

Run the command above to generate your VQA dataset grouped by question type.

For 7,721 images, expect:
- **Processing time:** ~2-5 minutes
- **Output:** 6 JSON files + 1 summary
- **Total size:** ~50-100 MB (depending on annotation size)

---

## 🔗 Your Paths

| What | Path |
|------|------|
| **Annotations** | `C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\ammotation\annotations` |
| **Images** | `C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images` |
| **Output** | `C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA_grouped` |

---

**Next step:** Run the command and you'll get your 6 question-type files ready for analysis! 🚀
