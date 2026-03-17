# Inwalk_Benchmark

A comprehensive Video Quality Assessment (VQA) and Object Detection pipeline for indoor accessibility benchmarking, designed for NeurIPS-style publication.

## Overview

This repository implements a two-phase pipeline for video quality assessment and open-vocabulary object detection:

- **Phase 1**: Keyframe extraction with multi-metric quality assessment and VLM-assisted analysis
- **Phase 2**: Open-vocabulary object detection using ensemble methods (YOLO-World + GroundingDINO)

## Features

### Phase 1: Keyframe Extraction & Quality Assessment
- **Multi-metric quality assessment**: Sharpness, noise, brightness, contrast, BRISQUE/NIQE proxies, temporal stability
- **CLIP-based keyframe selection**: Semantic embedding-based selection with batched processing
- **VLM-assisted analysis**: Qwen3-VL or GPT-4o for object extraction and artifact classification
- **VLM-routed expert fusion**: Dynamic weight adjustment based on semantic content (Q-Router Tier-1 inspired)
- **Hysteresis artifact selection**: Stable artifact frame detection (Q-Router Tier-2 inspired)
- **Score calibration**: Wider quality score distribution for better benchmarking

### Phase 2: Object Detection
- **Ensemble detection**: Combines YOLO-World and GroundingDINO for maximum accuracy
- **VLM-driven vocabulary**: Uses Phase 1 VLM outputs to build per-image detection vocabularies
- **Open-vocabulary**: Detects any object class specified by VLM, not limited to pre-trained classes
- **YOLO format output**: Standardized bounding box and segmentation annotations
- **Professional visualization**: PIL-based bounding box rendering with class-specific colors

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Setup

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

2. Download Qwen3-VL model (optional, will auto-download):
```bash
python setup/download_qwen_model.py
```

## Usage

### Phase 1: Keyframe Extraction & Quality Assessment

```bash
python main.py phase1 --video "path/to/video.mp4" --output ./output --enable-vlm --vlm-model qwen
```

Or use the direct script:
```bash
python keyfram_analysis.py --video "path/to/video.mp4" --output ./output --enable-vlm
```

### Phase 2: Object Detection

```bash
python main.py phase2 \
    --images-dir ./output/VideoName/Keyframes \
    --vlm-json-dir ./output/VideoName/VLM_analysis \
    --output-dir ./output/VideoName/Detection \
    --detector ensemble \
    --conf 0.15
```

### Both Phases

```bash
python main.py both "path/to/video.mp4" --output ./output --vlm-model qwen
```

## Project Structure

```
.
├── keyframe_extraction/      # Phase 1: CLIP-based keyframe selection
│   ├── clip_selector.py      # Batched CLIP embeddings
│   └── vlm_analyzer.py       # VLM integration wrapper
├── detection/                # Phase 2: Object detection
│   ├── pipeline.py           # Main detection pipeline
│   ├── grounding_dino.py     # GroundingDINO detector
│   ├── yolo_world.py         # YOLO-World detector
│   ├── ensemble.py           # Ensemble detector
│   ├── vocab.py              # VLM vocabulary building
│   └── utils.py              # NMS, IoU, coordinate conversion
├── experiments/              # Benchmarking suite
│   ├── run_benchmark.py      # Main benchmark runner
│   ├── metrics.py            # Evaluation metrics
│   └── schemas.py            # Pydantic data validation
├── keyfram_analysis.py       # Phase 1 main pipeline
├── main.py                   # Unified entry point
├── vlm_qwen.py               # Qwen3-VL implementation
├── vlm_gpt4o.py              # GPT-4o implementation
└── requirements.txt          # Dependencies
```

## Configuration

Key parameters can be adjusted via command-line arguments:

- `--segment-duration`: Segment length in seconds (default: 6)
- `--fps-target`: Frames per second to extract (default: 1)
- `--score-stretch`: Quality score distribution stretch factor (default: 1.60)
- `--router-alpha`: VLM-routed quality blending factor (default: 0.35)
- `--detection-conf`: Detection confidence threshold (default: 0.15)
- `--detector`: Detector choice (grounding_dino, yolo_world, ensemble)

## Output Format

### Phase 1 Outputs
- `Keyframes/`: Selected keyframes (one per segment)
- `VLM_analysis/`: VLM JSON outputs (objects, artifacts)
- `Quality_reports/`: Quality assessment reports
- `Visualizations/`: Quality plots and heatmaps

### Phase 2 Outputs
- `labels/`: YOLO format annotations (.txt)
- `metadata/`: Full detection metadata (.json)
- `visualizations/`: Annotated images with bounding boxes




REM Qwen3-VL 235B
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_qwen3_235b" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_qwen3_vl_235b --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Qwen3-VL 8B
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_qwen3_8b" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_qwen3_vl_8b --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Qwen-VL Plus
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_qwen_vl_plus" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_qwen_vl_plus --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Llama 4 Maverick
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_llama4_maverick" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_llama4_maverick --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Llama 3.2 11B Vision
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_llama32_11b_vision" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_llama32_11b_vision --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Trinity (free)
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_trinity" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_trinity --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Molmo 8B (free)
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_molmo_8b" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_molmo_8b --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM Ministral 3B
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_ministral_3b" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_ministral_3b --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

REM GPT OSS Safeguard 20B
python main.py annotate --images-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Fianl Version\images" --output-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\Annotations_accessibility_gpt_oss_safeguard_20b" --vlm-json-dir "C:\Tim\Database\InVLM\My Data\Scene_summary.v8i.yolov11\test\Phases\Fianl Version\openrouter\VLM_analysis" --caption-model openrouter --openrouter-model openrouter_gpt_oss_safeguard_20b --task accessibility --export-formats all --annotation-version comprehensive --nav-backend openrouter -v

## Citation

If you use this code in your research, please cite:

```bibtex
@software{inwalk_benchmark,
  title={Inwalk_Benchmark: Video Quality Assessment and Object Detection Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Inwalk_Benchmark}
}
```

## License

[Specify your license here]

## Acknowledgments

- Q-Router framework inspiration
- CLIP (OpenAI)
- GroundingDINO (IDEA-Research)
- YOLO-World (Ultralytics)
- Qwen3-VL (Alibaba Cloud)
