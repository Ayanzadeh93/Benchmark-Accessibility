# Quick Start: Florence-2 for Fast Object Extraction

## 🚀 Installation

```bash
pip install transformers torch pillow
```

## ⚡ Usage Examples

### 1. Command Line (Recommended for Batch Processing)

```bash
# Process existing images with Florence-2 (FASTEST)
python main.py phase1 \
  --images-dir "C:\path\to\images" \
  --output-dir ./output \
  --vlm-model florence2 \
  --vlm-device cuda

# Run detection pipeline with Florence-2
python main.py phase2 \
  --images-dir output/Keyframes \
  --output-dir output/Detection \
  --vlm-model florence2 \
  --vlm-device cuda

# Full pipeline (video → keyframes → detection)
python main.py both "video.MOV" \
  --output ./output \
  --vlm-model florence2 \
  --vlm-device cuda
```

### 2. Python API (Direct Use)

```python
from vlm_florence2 import Florence2Extractor

# Initialize (loads once)
extractor = Florence2Extractor(device="cuda", model_size="base")

# Extract objects from single image (1-2 seconds)
result = extractor.extract_objects_from_keyframe("image.jpg")

print(f"Found {result['num_objects']} objects:")
for obj in result['objects']:
    print(f"  - {obj}")

# Generate detailed caption
caption = extractor.generate_freeform_text(
    "image.jpg", 
    prompt="Describe this image in detail"
)
print(f"Caption: {caption}")
```

### 3. Factory Pattern (Switch Models Easily)

```python
from vlm_factory import VLMFactory

# Florence-2 (FASTEST)
extractor = VLMFactory.create_extractor("florence2", device="cuda")

# Or Qwen (more detailed)
# extractor = VLMFactory.create_extractor("qwen", device="cuda")

# Or GPT-4o (API)
# extractor = VLMFactory.create_extractor("gpt4o", api_key="sk-...")

# Same API for all models
result = extractor.extract_objects_from_keyframe("image.jpg")
```

### 4. SimpleVLMIntegration (High-Level)

```python
from simple_vlm_integration import SimpleVLMIntegration

# Initialize
vlm = SimpleVLMIntegration(model_type="florence2", device="cuda")

# Analyze single image
result = vlm.analyze_keyframe("image.jpg")
objects = result['objects']['objects']
has_artifacts = result['artifacts']['has_artifacts']

# Batch analyze multiple images
keyframes = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = vlm.batch_analyze_keyframes(keyframes, output_dir="./output")
vocabulary = results['vocabulary']['unique_objects']
```

## 🧪 Test Speed

```bash
# Test Florence-2 speed on your RTX 5070
python test_florence2.py path/to/image.jpg

# Expected output:
# Average: 1.004s per image
# 🎉 SUCCESS! 1.004s is UNDER 2 seconds target!
```

## 🔄 Switching Between Models

All models use the same API, so switching is just one argument:

```bash
# Florence-2 (fastest)
python main.py phase1 --images-dir ./images --vlm-model florence2

# Qwen (more detailed descriptions)
python main.py phase1 --images-dir ./images --vlm-model qwen

# GPT-4o (highest accuracy, requires API key)
python main.py phase1 --images-dir ./images --vlm-model gpt4o
```

## 📊 Expected Performance (RTX 5070)

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| florence2 | **1-2s** | ⭐⭐⭐⭐ | FREE |
| qwen | 2-3s | ⭐⭐⭐⭐⭐ | FREE |
| gpt4o | ~1s | ⭐⭐⭐⭐⭐ | $0.005/img |

## 🎯 Which Model Should I Use?

- **Production/Fast**: `florence2` (1-2s, optimized for objects)
- **Detailed Descriptions**: `qwen` (2-3s, better scene understanding)
- **Maximum Accuracy**: `gpt4o` (API, highest quality)
- **Budget API**: `gpt5nano` (10x cheaper than GPT-4o)

## 💡 Tips

1. **First run is slower**: Model loads and compiles CUDA kernels (one-time cost)
2. **Use batch processing**: Amortize model load time across many images
3. **RTX 5070 has 16GB VRAM**: Can run Florence-2 + other models simultaneously
4. **bf16 is automatic**: RTX 5070 supports bfloat16 for faster inference

## ❓ Troubleshooting

**Q: "No module named 'transformers'"**
```bash
pip install transformers torch pillow
```

**Q: "CUDA out of memory"**
```bash
# Use CPU fallback
python main.py phase1 --images-dir ./images --vlm-model florence2 --vlm-device cpu
```

**Q: "Model loads slowly"**
- First run compiles CUDA kernels (normal)
- Subsequent runs are much faster (1-2s)

## 📚 More Information

See `MODELS_COMPARISON.md` for detailed comparison of all models.
