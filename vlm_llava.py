#!/usr/bin/env python3
"""
LLaVA 1.5 7B Object Extractor (Local GPU)
Uses LLaVA-v1.5-7B for local inference

Requirements:
    pip install transformers accelerate
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from vlm_base import BaseVLMExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLaVAExtractor(BaseVLMExtractor):
    """
    LLaVA 1.5 7B Vision-Language Model for object extraction and VQA.
    Runs locally on GPU (free, requires GPU).
    """
    
    @staticmethod
    def _enable_max_gpu_performance() -> None:
        """Best-effort GPU speed knobs for PyTorch."""
        try:
            import torch
        except Exception:
            return

        if not torch.cuda.is_available():
            return

        # Fast matmul / convolution paths
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "auto"):
        """
        Initialize LLaVA extractor.
        
        Args:
            model_path: HuggingFace model path (default: llava-hf/llava-1.5-7b-hf)
            device: "auto", "cuda", or "cpu"
        """
        super().__init__(model_name="LLaVA-1.5-7B")

        import torch
        # Auto device: prefer CUDA if available
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device='cuda' but CUDA is not available; falling back to CPU.")
            device = "cpu"

        self.device = device
        self._cuda_disabled = False

        # Enable GPU performance optimizations
        self._enable_max_gpu_performance()

        # Load model and processor
        self._load_model(model_path)
        
        logger.info(f"[OK] LLaVA-1.5-7B loaded on {self.device}")

    def _load_model(self, model_path: str):
        """Load LLaVA model and processor."""
        try:
            import torch
            try:
                from transformers import AutoProcessor, LlavaForConditionalGeneration
                model_cls = LlavaForConditionalGeneration
            except Exception:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq
                logger.warning("LlavaForConditionalGeneration not available; using AutoModelForVision2Seq")
            
            logger.info(f"Loading LLaVA from {model_path}...")
            
            # LLaVA HF models expose an AutoProcessor and LlavaForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise

    def extract_objects(
        self,
        image_path: str,
        focus_areas: Optional[List[str]] = None,
        include_accessibility: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract objects from image using LLaVA.
        
        Args:
            image_path: Path to image file
            focus_areas: Optional list of focus areas (not used by LLaVA)
            include_accessibility: Include accessibility features
        
        Returns:
            Dict with objects list and metadata
        """
        if self._cuda_disabled and self.device == "cuda":
            logger.warning("CUDA disabled due to previous error; skipping")
            return self._empty_result()

        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # Build prompt for object detection
            prompt = "USER: <image>\nList all objects visible in this image. Be specific and include their locations.\nASSISTANT:"
            
            # Process image and text together via processor
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
            # Decode
            output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract assistant response (after "ASSISTANT:")
            if "ASSISTANT:" in output_text:
                response = output_text.split("ASSISTANT:")[-1].strip()
            else:
                response = output_text.strip()
            
            # Parse objects from response (simple split by comma/newline)
            objects = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('USER:'):
                    # Remove numbering like "1.", "2.", etc.
                    import re
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if line:
                        objects.append(line)
            
            return {
                "objects": objects,
                "raw_response": response,
                "model": "LLaVA-1.5-7B",
                "timestamp": datetime.now().isoformat(),
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e).lower():
                logger.error(f"CUDA error in LLaVA: {e}; disabling CUDA for this run")
                self._cuda_disabled = True
                return self._empty_result()
            raise
        except Exception as e:
            logger.error(f"LLaVA extraction error: {e}")
            return self._empty_result()

    def answer_multiple_choice(
        self,
        image_path: str,
        question: str,
        options: Union[Dict[str, str], Any],
    ) -> Optional[str]:
        """
        Answer a multiple-choice VQA question using LLaVA.
        
        Args:
            image_path: Path to image
            question: Question text
            options: Dict with keys A, B, C, D
        
        Returns:
            One of "A", "B", "C", "D" or None
        """
        if self._cuda_disabled and self.device == "cuda":
            logger.warning("CUDA disabled; skipping VQA")
            return None

        try:
            from PIL import Image
            import torch
            
            opts = {k: str(v).strip() for k, v in (options or {}).items() if k in ("A", "B", "C", "D")}
            if len(opts) != 4:
                return None
            
            image = Image.open(image_path).convert("RGB")
            
            # Build prompt
            prompt = (
                f"USER: <image>\n{question}\n\n"
                f"A: {opts['A']}\n"
                f"B: {opts['B']}\n"
                f"C: {opts['C']}\n"
                f"D: {opts['D']}\n\n"
                "Answer with only the letter (A, B, C, or D).\nASSISTANT:"
            )
            
            # Process image and text together via processor
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            
            # Decode
            output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract answer
            if "ASSISTANT:" in output_text:
                response = output_text.split("ASSISTANT:")[-1].strip().upper()
            else:
                response = output_text.strip().upper()
            
            # Find first A/B/C/D
            for char in response:
                if char in "ABCD":
                    return char
            
            logger.warning(f"LLaVA VQA: no A/B/C/D in response: {response[:100]}")
            return None
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e).lower():
                logger.error(f"CUDA error in LLaVA VQA: {e}")
                self._cuda_disabled = True
                return None
            raise
        except Exception as e:
            logger.error(f"LLaVA VQA error: {e}")
            return None

    def generate_freeform_text(self, image_path: str, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate free-form text from image + prompt (for VQA, captions, etc.).
        
        Args:
            image_path: Path to image file
            prompt: Question or instruction text
            max_new_tokens: Max tokens to generate (default 100, ~75 words)
        
        Returns:
            Generated text, or empty string on failure
        """
        if self._cuda_disabled and self.device == "cuda":
            return ""
        try:
            from PIL import Image
            import torch
            image = Image.open(image_path).convert("RGB")
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                )
            output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            if "ASSISTANT:" in output_text:
                return output_text.split("ASSISTANT:")[-1].strip()
            return output_text.strip()
        except Exception as e:
            logger.error(f"LLaVA generate_freeform_text error: {e}")
            return ""

    def generate_freeform_text_batch(
        self,
        image_path: str,
        questions: List[Dict[str, Any]],
        max_words_per_answer: int = 50,
        include_reference_in_prompt: bool = False,
    ) -> Dict[str, str]:
        """
        Answer multiple free-form VQA questions for one image in ONE call.
        Returns Dict mapping question id -> answer text.
        """
        if self._cuda_disabled and self.device == "cuda":
            return {q.get("id", ""): "" for q in questions}
        if not questions:
            return {}
        try:
            import re
            if include_reference_in_prompt:
                from vlm_refinement_prompts import REFINEMENT_SYSTEM_INSTRUCTIONS
                parts = [
                    REFINEMENT_SYSTEM_INSTRUCTIONS,
                    f"Keep each answer to {max_words_per_answer} words or less. Respond in exactly this format: Q1: [answer]\nQ2: [answer]\n...\n\n"
                ]
                for i, q in enumerate(questions, 1):
                    parts.append(f"Q{i}: {q.get('question', '')}")
                parts.append("\nYour answers:")
            else:
                parts = [
                    f"Answer these questions about the image. Keep each answer to {max_words_per_answer} words or less. "
                    "Respond in exactly this format:\nQ1: [answer]\nQ2: [answer]\n...\n\n"
                ]
                for i, q in enumerate(questions, 1):
                    parts.append(f"Q{i}: {q.get('question', '')}")
                parts.append("\nYour answers:")
            prompt = "\n".join(parts)
            max_tokens = min(600, 50 + len(questions) * (max_words_per_answer * 2))
            text = self.generate_freeform_text(image_path, prompt, max_new_tokens=max_tokens)
            result = {}
            for i, q in enumerate(questions, 1):
                qid = q.get("id", "")
                match = re.search(rf"Q{i}\s*:\s*(.+?)(?=Q{i+1}\s*:|$)", text, re.DOTALL | re.IGNORECASE)
                if match:
                    ans = match.group(1).strip()
                    words = ans.split()
                    if len(words) > max_words_per_answer:
                        ans = " ".join(words[:max_words_per_answer])
                    result[qid] = ans
                else:
                    result[qid] = ""
            return result
        except Exception as e:
            logger.error("LLaVA generate_freeform_text_batch error: %s", e)
            return {q.get("id", ""): "" for q in questions}

    def extract_objects_from_keyframe(
        self,
        image_path: str,
        focus_areas: Optional[List[str]] = None,
        include_accessibility: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract objects from keyframe (wrapper for extract_objects).
        
        Args:
            image_path: Path to keyframe image
            focus_areas: Optional focus areas
            include_accessibility: Include accessibility features
        
        Returns:
            Dictionary with extracted objects and metadata
        """
        return self.extract_objects(image_path, focus_areas, include_accessibility)

    def classify_frame_artifacts(self, image_path: str) -> Dict[str, Any]:
        """
        Classify frame artifacts (not implemented for LLaVA).
        
        Args:
            image_path: Path to frame image
        
        Returns:
            Dictionary with artifact classification (always returns no artifacts)
        """
        return {
            "has_artifacts": False,
            "artifact_type": "none",
            "confidence": 0.0,
            "description": "Artifact detection not implemented for LLaVA",
            "severity": "low",
            "model": "LLaVA-1.5-7B",
            "timestamp": datetime.now().isoformat(),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result on error."""
        return {
            "objects": [],
            "model": "LLaVA-1.5-7B",
            "timestamp": datetime.now().isoformat(),
            "error": True,
        }
