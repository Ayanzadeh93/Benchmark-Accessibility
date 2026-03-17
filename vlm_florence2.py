#!/usr/bin/env python3
"""
Florence-2 Object Extractor (Local GPU - FASTEST)
Uses Microsoft's Florence-2 for ultra-fast object detection/extraction

Requirements:
    pip install transformers torch pillow
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from vlm_base import BaseVLMExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Florence2Extractor(BaseVLMExtractor):
    """
    Florence-2 Vision-Language Model for object extraction.
    Ultra-fast local inference on GPU (230M-770M params).
    Optimized for object detection, grounding, and captioning.
    """
    
    def __init__(self, model_path: str = None, device: str = "auto", model_size: str = "base"):
        """
        Initialize Florence-2 extractor.
        
        Args:
            model_path: Local model path or HuggingFace model path (default: auto)
            device: "auto", "cuda", or "cpu"
            model_size: "base" (230M) or "large" (770M) - base is MUCH faster
        """
        super().__init__(model_name=f"Florence-2-{model_size}")
        
        import torch
        
        # Auto device: prefer CUDA if available
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device='cuda' but CUDA is not available; falling back to CPU.")
            device = "cpu"
        
        self.device = device
        self._cuda_disabled = False
        
        # Choose model size
        if model_path is None:
            if model_size.lower() == "large":
                model_path = "microsoft/Florence-2-large"
            else:
                model_path = "microsoft/Florence-2-base"  # Faster, recommended
        
        # CRITICAL: Use fixed revision to avoid remote code bugs
        # Use None or "main" for latest, or specific commit for stability
        model_revision = None  # Let transformers handle it
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Performance optimizations
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            
            logger.info(f"Loading Florence-2-{model_size} on {device}...")
            
            # Choose precision based on device
            if device == "cuda":
                try:
                    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True,
                revision=model_revision
            )
            
            # Load model with attn_implementation="eager" to bypass _supports_sdpa check
            # This is the official workaround for transformers >= 4.50.0 compatibility
            # Use dtype instead of torch_dtype to avoid deprecation warning
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,  # Use dtype instead of torch_dtype
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # CRITICAL: Bypasses _supports_sdpa check
                revision=model_revision,
                # Force clean reload to avoid cached buggy remote code
                force_download=False,  # Don't force, just use latest cached
            ).to(device)
            
            self.model.eval()
            
            self.enabled = True
            logger.info(f"[OK] Florence-2-{model_size} initialized on {device} (actual: {next(self.model.parameters()).device})")
            
        except ImportError as e:
            logger.error(f"transformers not installed: pip install transformers torch")
            logger.error(f"Import error: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Florence-2 initialization failed: {e}")
            self.enabled = False
    
    def extract_objects_from_keyframe(
        self, 
        image_path: str,
        focus_areas: List[str] = None,
        include_accessibility: bool = True
    ) -> Dict[str, Any]:
        """Extract objects using Florence-2 (ultra-fast)."""
        if not self.enabled:
            return self._empty_result()
        
        try:
            import torch
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Florence-2 task: <MORE_DETAILED_CAPTION> gives comprehensive object list
            # Other tasks: <CAPTION>, <DETAILED_CAPTION>, <OD> (object detection with boxes)
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            # Prepare inputs
            inputs = self.processor(
                text=task_prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # Move to device and match model dtype
            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            model_dtype = next(self.model.parameters()).dtype
            inputs = {
                k: v.to(target_device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
                else v.to(target_device) if isinstance(v, torch.Tensor) 
                else v 
                for k, v in inputs.items()
            }
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)
            
            # Generate (very fast)
            # CRITICAL: use_cache=False prevents past_key_values NoneType error
            with torch.inference_mode():
                try:
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256,
                        num_beams=3,
                        use_cache=False,  # CRITICAL: Prevents past_key_values error
                    )
                    generated_text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=False
                    )[0]
                except (RuntimeError, AssertionError) as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "assert" in error_str:
                        logger.warning(f"CUDA error, switching to CPU fallback: {e}")
                        # Fallback to CPU
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        generated_ids = model_cpu.generate(
                            input_ids=inputs_cpu["input_ids"],
                            pixel_values=inputs_cpu["pixel_values"],
                            max_new_tokens=256,
                            num_beams=3,
                            use_cache=False,
                        )
                        generated_text = self.processor.batch_decode(
                            generated_ids, 
                            skip_special_tokens=False
                        )[0]
                        self._cuda_disabled = True
                        self.device = "cpu"
                    else:
                        raise
            
            # Post-process Florence-2 output
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            # Extract description from Florence-2 output
            description = parsed_answer.get(task_prompt, "")
            
            # Parse objects from description
            parsed = self._parse_florence2_description(description, include_accessibility)
            parsed['model'] = self.model_name
            parsed['timestamp'] = datetime.now().isoformat()
            
            if parsed.get('error', False):
                logger.info("Florence-2 object extraction returned empty result; marking as error.")
            else:
                logger.info(f"Florence-2: Extracted {parsed['num_objects']} objects")
            return parsed
            
        except Exception as e:
            logger.error(f"Florence-2 extraction error: {e}", exc_info=True)
            return self._empty_result()
    
    def generate_freeform_text(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate free-form text from an image.
        
        Florence-2 works with predefined tasks, so we use <MORE_DETAILED_CAPTION> for general description.
        """
        if not self.enabled:
            return ""
        
        try:
            import torch
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Use detailed caption task
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = self.processor(
                text=task_prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            model_dtype = next(self.model.parameters()).dtype
            inputs = {
                k: v.to(target_device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
                else v.to(target_device) if isinstance(v, torch.Tensor) 
                else v 
                for k, v in inputs.items()
            }
            
            # Ensure model is on the same device
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)
            
            with torch.inference_mode():
                try:
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                        use_cache=False,
                    )
                    generated_text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=False
                    )[0]
                except (RuntimeError, AssertionError) as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "assert" in error_str:
                        logger.warning(f"CUDA error, switching to CPU fallback: {e}")
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        generated_ids = model_cpu.generate(
                            input_ids=inputs_cpu["input_ids"],
                            pixel_values=inputs_cpu["pixel_values"],
                            max_new_tokens=max_new_tokens,
                            num_beams=3,
                            use_cache=False,
                        )
                        generated_text = self.processor.batch_decode(
                            generated_ids, 
                            skip_special_tokens=False
                        )[0]
                        self._cuda_disabled = True
                        self.device = "cpu"
                    else:
                        raise
            
            # Post-process
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(Image.open(image_path).size)
            )
            
            return parsed_answer.get(task_prompt, "").strip()
        
        except Exception as e:
            logger.error(f"Florence-2 freeform generation error: {e}", exc_info=True)
            return ""

    def generate_freeform_text_batch(
        self,
        image_path: str,
        questions: List[Dict[str, Any]],
        max_words_per_answer: int = 50,
        include_reference_in_prompt: bool = False,
    ) -> Dict[str, str]:
        """
        Answer multiple VQA questions - Florence2 uses sequential calls
        (no native batch). Returns Dict mapping question id -> answer text.
        """
        result = {}
        for q in questions:
            qid = q.get("id", "")
            question = q.get("question", "")
            if include_reference_in_prompt:
                from vlm_refinement_prompts import REFINEMENT_SYSTEM_INSTRUCTIONS
                prompt = (
                    f"{REFINEMENT_SYSTEM_INSTRUCTIONS}\n\n"
                    f"Question: {question}\n\n"
                    f"Your answer in {max_words_per_answer} words or less:"
                )
            else:
                prompt = f"{question}\n\nAnswer in {max_words_per_answer} words or less."
            ans = self.generate_freeform_text(image_path, prompt, max_new_tokens=80)
            if ans:
                words = ans.split()
                if len(words) > max_words_per_answer:
                    ans = " ".join(words[:max_words_per_answer])
            result[qid] = (ans or "").strip()
        return result
    
    def classify_frame_artifacts(self, image_path: str) -> Dict[str, Any]:
        """Classify frame artifacts using Florence-2."""
        if not self.enabled:
            return self._empty_artifact_result()
        
        try:
            import torch
            from PIL import Image
            
            # Use detailed caption to detect quality issues
            image = Image.open(image_path).convert("RGB")
            task_prompt = "<DETAILED_CAPTION>"
            
            inputs = self.processor(
                text=task_prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            model_dtype = next(self.model.parameters()).dtype
            inputs = {
                k: v.to(target_device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
                else v.to(target_device) if isinstance(v, torch.Tensor) 
                else v 
                for k, v in inputs.items()
            }
            
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)
            
            with torch.inference_mode():
                try:
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=128,
                        num_beams=3,
                        use_cache=False,
                    )
                    generated_text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=False
                    )[0]
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuda" in str(e).lower():
                        logger.warning(f"CUDA error, switching to CPU fallback: {e}")
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        generated_ids = model_cpu.generate(
                            input_ids=inputs_cpu["input_ids"],
                            pixel_values=inputs_cpu["pixel_values"],
                            max_new_tokens=128,
                            num_beams=3,
                            use_cache=False,
                        )
                        generated_text = self.processor.batch_decode(
                            generated_ids, 
                            skip_special_tokens=False
                        )[0]
                        self._cuda_disabled = True
                        self.device = "cpu"
                    else:
                        raise
            
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            description = parsed_answer.get(task_prompt, "").lower()
            
            # Heuristic artifact detection from description
            has_artifacts = any(
                word in description 
                for word in ["blurry", "blur", "noise", "grain", "artifact", "distorted", "pixelated"]
            )
            
            artifact_type = "none"
            severity = "low"
            confidence = 0.3  # Low confidence for heuristic
            
            if "blur" in description or "blurry" in description:
                artifact_type = "1"  # Motion blur
                severity = "medium"
                confidence = 0.6
            elif "noise" in description or "grain" in description:
                artifact_type = "2"  # Noise
                severity = "low"
                confidence = 0.5
            
            return {
                'has_artifacts': has_artifacts,
                'artifact_type': artifact_type,
                'confidence': confidence,
                'description': description[:200],
                'severity': severity,
                'affected_regions': [],
                'model': self.model_name,
                'error': False
            }
            
        except Exception as e:
            logger.error(f"Florence-2 artifact error: {e}")
            return self._empty_artifact_result()
    
    def _parse_florence2_description(self, description: str, include_accessibility: bool) -> Dict[str, Any]:
        """Parse Florence-2 description into structured object list."""
        import re
        
        # Florence-2 generates natural language descriptions like:
        # "A room with a table, chairs, a laptop, and a person standing near the window."
        
        # Extract nouns (simple heuristic)
        # Common patterns: "a X", "the X", "an X", "X and Y", "X, Y, and Z"
        
        # Split by common separators
        parts = re.split(r'[,;]|\s+and\s+|\s+with\s+', description.lower())
        
        objects = []
        for part in parts:
            # Remove articles and prepositions
            part = re.sub(r'\b(a|an|the|in|on|at|by|near|next to|beside|behind|in front of)\b', '', part)
            # Extract meaningful words (nouns)
            words = [w.strip() for w in part.split() if len(w.strip()) > 2]
            objects.extend(words)
        
        # Normalize
        objects = self._normalize_objects(objects)
        
        # Categorize (heuristic)
        categories = {
            "signs": [],
            "accessibility": [],
            "people": [],
            "furniture": [],
            "technology": [],
            "other": []
        }
        
        def _categorize(obj: str) -> str:
            o = obj.lower()
            if any(k in o for k in ["sign", "exit", "restroom", "elevator", "emergency"]):
                return "signs"
            if any(k in o for k in ["wheelchair", "braille", "ramp", "accessible"]):
                return "accessibility"
            if o == "person" or "people" in o or "man" in o or "woman" in o:
                return "people"
            if any(k in o for k in ["chair", "table", "couch", "sofa", "desk", "bed", "bench", "cabinet"]):
                return "furniture"
            if any(k in o for k in ["phone", "laptop", "computer", "screen", "monitor", "tv", "camera", "keyboard"]):
                return "technology"
            return "other"
        
        for obj in objects:
            cat = _categorize(obj)
            categories[cat].append(obj)
        
        return {
            "objects": objects[:200],  # Limit to 200
            "categories": categories,
            "scene_description": description[:500],
            "primary_focus": objects[0] if objects else "",
            "num_objects": len(objects),
            "error": len(objects) == 0,
        }


def test_florence2_extractor():
    """Test Florence-2 extractor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vlm_florence2.py <image_path>")
        return
    
    print("\n" + "="*60)
    print("Testing Florence-2 Extractor (FASTEST)")
    print("="*60)
    
    extractor = Florence2Extractor(model_size="base")
    
    if not extractor.enabled:
        print("❌ Florence-2 not available (check installation)")
        return
    
    import time
    start = time.time()
    result = extractor.extract_objects_from_keyframe(sys.argv[1])
    elapsed = time.time() - start
    
    print(f"\n✅ Objects: {result['num_objects']} (extracted in {elapsed:.2f}s)")
    for obj in result['objects'][:15]:
        print(f"  - {obj}")
    
    print(f"\n📝 Scene: {result['scene_description'][:200]}")
    print("\n" + "="*60)


if __name__ == "__main__":
    test_florence2_extractor()
