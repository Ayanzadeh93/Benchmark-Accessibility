# VQA Generation - Grouped by Question Type

## Overview

This module generates VQA (Visual Question Answering) datasets **grouped by question type** instead of by image. For 7,000 images with 6 question types, you get **6 output files** (one per question) instead of 7,000 files (one per image).

## Output Structure

Each question type gets its own JSON file containing all images' Q&A for that question:

```
output_dir/
├── main_obstacle.json          # "What is the main obstacle?"
├── closest_obstacle.json        # "Which object is closest?"
├── risk_assessment.json         # "How safe is this scene?"
├── spatial_clock.json           # "Locate the X based on clock direction"
├── action_suggestion.json       # "What action do you suggest?"
├── action_command.json          # "What is the recommended navigation action?"
└── vqa_generation_summary.json  # Generation statistics
```

## Question Types

| Question ID | Question | Options |
|-------------|----------|---------|
| `main_obstacle` | What is the main obstacle or barrier in this scene? | A/B/C/D (objects + distractors) |
| `closest_obstacle` | Which object is closest to the user? | A/B/C/D (objects by distance) |
| `risk_assessment` | How safe is this scene for blind users? | A: Low risk / B: Medium risk / C: High risk / D: Extreme risk |
| `spatial_clock` | Locate the {object} based on clock direction. | A/B/C/D (clock positions) |
| `action_suggestion` | What action do you suggest to the user in this scene? | A/B/C/D (navigation actions) |
| `action_command` | What is the recommended navigation action for a blind user in this situation? | A/B/C/D (guidance + distractors) |

## Usage

### Command Line

```bash
python main.py vqa-grouped \
  --annotations-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\ammotation\annotations" \
  --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\VQA_grouped" \
  --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\images"
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--annotations-dir` | Yes | Directory with annotation JSON files |
| `--output-dir` | Yes | Where to save per-question files |
| `--images-dir` | No | Images directory (adds absolute paths) |
| `--max-samples` | No | Limit annotations to process (for testing) |
| `--seed` | No | Random seed (default: 1337) |
| `-v, --verbose` | No | Verbose logging |

### Examples

**Full run:**
```bash
python main.py vqa-grouped \
  --annotations-dir "./annotations" \
  --output-dir "./VQA_by_question" \
  --images-dir "./images"
```

**Test with 10 images:**
```bash
python main.py vqa-grouped \
  --annotations-dir "./annotations" \
  --output-dir "./VQA_test" \
  --max-samples 10 \
  --verbose
```

## Output Format

Each question file (e.g., `main_obstacle.json`) has this structure:

```json
{
  "question_id": "main_obstacle",
  "question": "What is the main obstacle or barrier in this scene?",
  "num_samples": 7000,
  "samples": [
    {
      "id": "image001|main_obstacle",
      "image": "image001.jpg",
      "question_id": "main_obstacle",
      "question": "What is the main obstacle or barrier in this scene?",
      "options": {
        "A": "Wall",
        "B": "Door",
        "C": "Chair",
        "D": "No obstacle"
      },
      "answer": "B",
      "ground_truth": {
        "main_obstacle": "door"
      },
      "image_path": "/absolute/path/to/image001.jpg"
    },
    ...
  ],
  "metadata": {
    "generated_from": "/path/to/annotations",
    "total_images": 7000,
    "seed": 1337
  }
}
```

### Fields per Sample

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID: `{image_stem}\|{question_id}` |
| `image` | string | Image filename |
| `question_id` | string | Question type identifier |
| `question` | string | Question text |
| `options` | dict | A/B/C/D options |
| `answer` | string | Correct answer letter (A/B/C/D) |
| `ground_truth` | dict | Ground truth data used to generate answer |
| `image_path` | string | Absolute path to image (if `--images-dir` provided) |

## Annotation Requirements

The script expects structured annotations with these fields:

```json
{
  "image": "image.jpg",
  "accessibility": {
    "scene_description": "...",
    "ground_text": "...",
    "spatial_objects": [
      "Obstacle in your front: wall at 12 o'clock, 3m away",
      ...
    ],
    "guidance": "Proceed forward, path clear.",
    "risk_assessment": {
      "level": "Medium",
      "reason": "...",
      "obstacles": [
        {
          "object": "wall",
          "position": "front",
          "distance": "medium",
          ...
        }
      ]
    }
  }
}
```

## Programmatic Usage

```python
from vqa.evaluation.generate_per_question import generate_per_question_vqa

summary = generate_per_question_vqa(
    annotations_dir="./annotations",
    output_dir="./VQA_grouped",
    images_dir="./images",
    max_samples=None,
    seed=1337,
    verbose=True,
)

print(f"Processed: {summary['processed']}")
print(f"Output files: {summary['output_files']}")
```

## Benefits Over Per-Image Output

| Per-Image (old) | Per-Question (new) |
|-----------------|-------------------|
| 7,000 files (1 per image × 6 questions) | 6 files (1 per question) |
| Hard to analyze specific question types | Easy to analyze per question |
| Large disk usage | Compact storage |
| Requires aggregation for evaluation | Ready for evaluation by question |

## Use Cases

1. **Question-specific evaluation:** Evaluate model performance on each question type separately
2. **Dataset analysis:** Analyze distribution of answers per question
3. **Debugging:** Check if specific question types are working correctly
4. **Model training:** Train models on specific question types
5. **Benchmarking:** Compare models across different question categories

## Summary File

`vqa_generation_summary.json` contains:

```json
{
  "processed": 7000,
  "failed": 5,
  "total_samples_per_question": {
    "main_obstacle": 7000,
    "closest_obstacle": 7000,
    "risk_assessment": 7000,
    "spatial_clock": 7000,
    "action_suggestion": 7000,
    "action_command": 7000
  },
  "output_files": {
    "main_obstacle": "output_dir/main_obstacle.json",
    ...
  },
  "config": {
    "annotations_dir": "...",
    "output_dir": "...",
    "seed": 1337
  }
}
```

## Code Location

- **Generator:** `vqa/evaluation/generate_per_question.py`
- **CLI integration:** `main.py` (`vqa-grouped` command)
- **Navigation questions:** `vqa/evaluation/navigation_core.py`
- **Action command:** `vqa/evaluation/action_command.py`

## Related Commands

- `python main.py vqa` - Original VQA generation (per-image files with train/eval splits)
- `python main.py vqa-grouped` - New VQA generation (per-question files)

Choose based on your use case:
- Use `vqa` for training/evaluation splits
- Use `vqa-grouped` for analysis and question-specific evaluation
