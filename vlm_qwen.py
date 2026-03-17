#!/usr/bin/env python3
"""
Qwen3-VL Object Extractor (Local GPU)
Uses Qwen3-VL-2B-Instruct for local inference

Requirements:
    pip install transformers qwen-vl-utils accelerate
"""

# SAM3 backend - masks only
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from vlm_base import BaseVLMExtractor

# Import qwen_vl_utils at module level
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    process_vision_info = None
    logging.warning("qwen_vl_utils not available - install with: pip install qwen-vl-utils")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3VLExtractor(BaseVLMExtractor):
    """
    Qwen3-VL Vision-Language Model for object extraction.
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

        # Reduced-precision reductions (speed > tiny numeric differences)
        try:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        except Exception:
            pass

        # Prefer faster SDP/flash attention kernels when available
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize Qwen3-VL extractor.
        
        Args:
            model_path: Local model path or HuggingFace model path (default: auto-detect local)
            device: "auto", "cuda", or "cpu"
        """
        super().__init__(model_name="Qwen3-VL-2B")

        import torch
        # Auto device: prefer CUDA if available (we still fall back to CPU on CUDA errors)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device='cuda' but CUDA is not available; falling back to CPU.")
            device = "cpu"

        self.device = device
        # If a CUDA device-side assert happens once, CUDA can become unstable for the rest of the run.
        # We disable CUDA after the first such error to keep batch runs reliable.
        self._cuda_disabled = False
        self._compiled = False

        # Max GPU performance knobs (best-effort, safe fallbacks)
        if self.device == "cuda":
            self._enable_max_gpu_performance()
        
        # Check for local model first (deterministic snapshot selection)
        if model_path is None:
            # Repo-local snapshot (downloaded into this project)
            local_model = Path(__file__).parent / "models" / "Qwen3-VL-2B-Instruct"
            if local_model.exists():
                # Check for HuggingFace cache structure (snapshots subdirectory)
                snapshots = local_model / "snapshots"
                if snapshots.exists() and snapshots.is_dir():
                    # Sort snapshot directories for deterministic selection (use most recent)
                    snapshot_dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()], 
                                         key=lambda x: x.stat().st_mtime, reverse=True)
                    if snapshot_dirs:
                        model_path = str(snapshot_dirs[0])
                        logger.info(f"Using local model snapshot: {model_path}")
                    else:
                        model_path = "Qwen/Qwen3-VL-2B-Instruct"
                        logger.info("No snapshots found in local model, using HuggingFace")
                else:
                    # Direct model directory (not in HuggingFace cache format)
                    model_path = str(local_model)
                    logger.info(f"Using local model directory: {model_path}")
            else:
                model_path = "Qwen/Qwen3-VL-2B-Instruct"
                logger.info("Local model not found, using HuggingFace")
        
        try:
            import torch
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoProcessor, AutoModelForVision2Seq as AutoModelForImageTextToText

            # Performance knobs (safe defaults)
            # - TF32 speeds up matmul on Ampere+ without changing model weights
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            
            # Get HuggingFace token (only needed if using HuggingFace)
            hf_token = None
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                try:
                    from config import get_huggingface_token
                    hf_token = get_huggingface_token()
                except ImportError:
                    import os
                    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
            
            logger.info(f"Loading Qwen3-VL model on {device}...")
            # NOTE: avoid `device_map="cuda"` sharding for stability; load then `.to(device)`.
            # Prefer bf16 on newer GPUs for speed/stability; fallback to fp16.
            if device == "cuda":
                try:
                    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    model_dtype = torch.float16
            else:
                model_dtype = torch.float32

            # Optional: faster attention kernels (only if your environment supports it)
            # We keep this best-effort: if unsupported, transformers will raise and we fall back.
            # Priority: SDPA (built-in, Windows-friendly) > FlashAttention2 (requires flash-attn)
            attn_impl: Optional[str] = None
            if device == "cuda":
                # Try SDPA first (PyTorch 2.0+ built-in, works on Windows without extra DLLs)
                # SDPA is faster than eager and doesn't require flash-attn
                try:
                    import torch.nn.functional as F
                    if hasattr(F, "scaled_dot_product_attention"):
                        attn_impl = "sdpa"  # Built-in optimized attention (Windows-friendly, no DLL issues)
                        logger.info("[OK] Using SDPA (Scaled Dot Product Attention) - built-in optimized attention")
                    else:
                        # PyTorch < 2.0: try flash_attention_2 if available
                        try:
                            import flash_attn
                            attn_impl = "flash_attention_2"
                            logger.info("[OK] Using FlashAttention2 (requires flash-attn package)")
                        except ImportError:
                            attn_impl = None
                            logger.info("Using default attention (SDPA/FlashAttention not available)")
                except Exception as e:
                    logger.debug(f"SDPA check failed: {e}, using default attention")
                    attn_impl = None
            
            if model_path_obj.exists():
                # Local model - no token needed
                logger.info(f"Loading from local path: {model_path}")
                self.processor = AutoProcessor.from_pretrained(model_path)
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_impl,
                    )
                except TypeError:
                    # Older transformers: no attn_implementation param
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                    )
                except (ImportError, ValueError, RuntimeError) as e:
                    # Attention implementation failed (flash-attn missing, SDPA unsupported, etc). Retry without it.
                    if attn_impl is not None:
                        error_str = str(e).lower()
                        if "flash" in error_str or "attn" in error_str or "attention" in error_str or "dll" in error_str:
                            logger.warning(f"Attention optimization ({attn_impl}) not available; retrying with default attention. Error: {e}")
                            attn_impl = None
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_path,
                                dtype=model_dtype,
                                low_cpu_mem_usage=True,
                            )
                        else:
                            raise
                    else:
                        raise
            elif hf_token:
                # HuggingFace with token
                logger.info("Using HuggingFace token for authentication")
                self.processor = AutoProcessor.from_pretrained(model_path, token=hf_token)
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        token=hf_token,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_impl,
                    )
                except TypeError:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        token=hf_token,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                    )
                except (ImportError, ValueError, RuntimeError) as e:
                    if attn_impl is not None:
                        error_str = str(e).lower()
                        if "flash" in error_str or "attn" in error_str or "attention" in error_str or "dll" in error_str:
                            logger.warning(f"Attention optimization ({attn_impl}) not available; retrying with default attention. Error: {e}")
                            attn_impl = None
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_path,
                                token=hf_token,
                                dtype=model_dtype,
                                low_cpu_mem_usage=True,
                            )
                        else:
                            raise
                    else:
                        raise
            else:
                # HuggingFace without token
                logger.warning("No HuggingFace token found, using public access")
                self.processor = AutoProcessor.from_pretrained(model_path)
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_impl,
                    )
                except TypeError:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        dtype=model_dtype,
                        low_cpu_mem_usage=True,
                    )
                except (ImportError, ValueError, RuntimeError) as e:
                    if attn_impl is not None:
                        error_str = str(e).lower()
                        if "flash" in error_str or "attn" in error_str or "attention" in error_str or "dll" in error_str:
                            logger.warning(f"Attention optimization ({attn_impl}) not available; retrying with default attention. Error: {e}")
                            attn_impl = None
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_path,
                                dtype=model_dtype,
                                low_cpu_mem_usage=True,
                            )
                        else:
                            raise
                    else:
                        raise

            # Explicitly move model to the requested device
            self.model = self.model.to(device)
            self.model.eval()

            # Ensure a sane pad token for generation (helps avoid warnings/edge-cases)
            try:
                tok = getattr(self.processor, "tokenizer", None)
                if tok is not None and getattr(self.model.generation_config, "pad_token_id", None) is None:
                    self.model.generation_config.pad_token_id = tok.eos_token_id
            except Exception:
                pass

            # Reduce noisy warnings when we run greedy decoding (do_sample=False)
            try:
                gc = getattr(self.model, "generation_config", None)
                if gc is not None:
                    if hasattr(gc, "temperature"):
                        gc.temperature = 1.0
                    if hasattr(gc, "top_p"):
                        gc.top_p = 1.0
                    if hasattr(gc, "top_k"):
                        gc.top_k = 50
                    if hasattr(gc, "use_cache"):
                        gc.use_cache = True
            except Exception:
                pass

            # Best-effort compile for faster inference (PyTorch 2.x)
            if device == "cuda":
                try:
                    if hasattr(torch, "compile"):
                        self.model = torch.compile(self.model, mode="max-autotune", fullgraph=False)
                        self._compiled = True
                        logger.info("[OK] Qwen3-VL torch.compile enabled (max-autotune)")
                except Exception as e:
                    logger.warning(f"torch.compile unavailable/failed, continuing without compile: {e}")
            
            self.enabled = True
            logger.info(f"[OK] Qwen3-VL initialized on {device} (actual device: {next(self.model.parameters()).device})")
        except ImportError as e:
            logger.error(f"transformers/qwen-vl-utils not installed: pip install transformers qwen-vl-utils accelerate")
            logger.error(f"Import error: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Qwen3-VL initialization failed: {e}")
            self.enabled = False

        # Default generation limits (smaller = faster). These are tuned for "object list" tasks.
        # You can still override by calling generate_freeform_text(..., max_new_tokens=...)
        # User request: cap to 100 for speed
        self._default_max_new_tokens_objects = 100
        self._default_max_new_tokens_freeform = 192
    
    def extract_objects_from_keyframe(
        self, 
        image_path: str,
        focus_areas: List[str] = None,
        include_accessibility: bool = True
    ) -> Dict[str, Any]:
        """Extract objects using Qwen3-VL."""
        if not self.enabled:
            return self._empty_result()
        
        try:
            if not QWEN_VL_UTILS_AVAILABLE:
                logger.error("qwen_vl_utils not available")
                return self._empty_result()
            
            import torch
            from PIL import Image
            
            # Build prompt
            prompt = self._build_extraction_prompt(focus_areas, include_accessibility)
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs with validation
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device and validate
            # Force CPU if device is "cpu" OR if we've disabled CUDA due to prior device-side assert
            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)
            
            # Validate input_ids
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                if input_ids.numel() == 0:
                    logger.error(f"Empty input_ids: shape={input_ids.shape}")
                    return self._empty_result()
                # Validate against actual embedding size (tokenizer.vocab_size may exclude added/special tokens)
                try:
                    vocab_limit = None
                    if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
                        vocab_limit = int(self.model.get_input_embeddings().weight.shape[0])
                    else:
                        tok = getattr(self.processor, "tokenizer", None)
                        vocab_limit = int(len(tok)) if tok is not None else None
                    if vocab_limit and int(input_ids.max()) >= vocab_limit:
                        logger.error(f"Invalid input_ids: shape={input_ids.shape}, max={int(input_ids.max())}, vocab_limit={vocab_limit}")
                        return self._empty_result()
                except AttributeError:
                    # Skip validation if vocab_size not available
                    pass
            
            # Generate with error handling
            # Final safety check: ensure model and inputs are on same device
            model_device = next(self.model.parameters()).device
            input_device = next(iter(inputs.values())).device if any(isinstance(v, torch.Tensor) for v in inputs.values()) else model_device
            if model_device != input_device:
                logger.warning(f"Device mismatch: model={model_device}, inputs={input_device}, fixing...")
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Inference optimizations:
            # - inference_mode reduces overhead and can speed up CUDA kernels
            # - autocast enables fp16/bf16 math for attention/MLP
            autocast_dtype = None
            if target_device != "cpu":
                try:
                    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    autocast_dtype = torch.float16

            with torch.inference_mode():
                try:
                    tok = getattr(self.processor, "tokenizer", None)
                    pad_token_id = getattr(tok, "eos_token_id", None)
                    if target_device != "cpu" and autocast_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=int(self._default_max_new_tokens_objects),
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=pad_token_id,
                                use_cache=True,  # KV cache for efficiency
                                repetition_penalty=1.1,
                            )
                    else:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=int(self._default_max_new_tokens_objects),
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            use_cache=True,
                            repetition_penalty=1.1,
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                except (RuntimeError, AssertionError) as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "assert" in error_str:
                        logger.warning(f"CUDA/Assertion error, switching to CPU fallback: {e}")
                        # Fallback to CPU
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        tok = getattr(self.processor, "tokenizer", None)
                        pad_token_id = getattr(tok, "eos_token_id", None)
                        generated_ids = model_cpu.generate(
                            **inputs_cpu,
                            max_new_tokens=256,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            use_cache=True,  # Enable KV cache for efficiency
                            repetition_penalty=1.1  # Slight penalty to avoid repetition
                        )
                        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cpu['input_ids'], generated_ids)]
                        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        # Disable CUDA for subsequent calls to keep batch processing stable
                        self._cuda_disabled = True
                        self.device = "cpu"
                    else:
                        raise
            
            # Parse response
            parsed = self._parse_qwen_response(output_text)
            parsed['model'] = self.model_name
            parsed['timestamp'] = datetime.now().isoformat()
            
            if parsed.get('error', False):
                logger.info("Qwen3-VL object extraction returned empty/invalid JSON; marking as error.")
            else:
                logger.info(f"Qwen3-VL: Extracted {parsed['num_objects']} objects")
            return parsed
            
        except Exception as e:
            logger.error(f"Qwen3-VL extraction error: {e}", exc_info=True)
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return self._empty_result()

    def generate_freeform_text(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate free-form text from an image given a prompt.

        This is used by the dataset annotator to produce a single clean caption/annotation.

        Args:
            image_path: Path to the image file.
            prompt: Instruction/prompt text (should include any formatting rules).
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text (may be empty on failure).
        """
        if not self.enabled:
            return ""
        if not QWEN_VL_UTILS_AVAILABLE or process_vision_info is None:
            logger.error("qwen_vl_utils not available")
            return ""

        try:
            import torch
            from PIL import Image

            # Load image
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": str(prompt)},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )

            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Ensure model is on the same device
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)

            # Final device match safeguard
            model_device = next(self.model.parameters()).device
            if model_device != inputs["input_ids"].device:
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            autocast_dtype = None
            if target_device != "cpu":
                try:
                    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    autocast_dtype = torch.float16

            with torch.inference_mode():
                try:
                    tok = getattr(self.processor, "tokenizer", None)
                    pad_token_id = getattr(tok, "eos_token_id", None)
                    effective_max = int(max_new_tokens) if max_new_tokens is not None else int(self._default_max_new_tokens_freeform)
                    if target_device != "cpu" and autocast_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=effective_max,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=pad_token_id,
                                use_cache=True,
                                repetition_penalty=1.1,
                            )
                    else:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=effective_max,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            use_cache=True,
                            repetition_penalty=1.1,
                        )
                except (RuntimeError, AssertionError) as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "assert" in error_str:
                        logger.warning(f"CUDA/Assertion error, switching to CPU fallback: {e}")
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        tok = getattr(self.processor, "tokenizer", None)
                        pad_token_id = getattr(tok, "eos_token_id", None)
                        generated_ids = model_cpu.generate(
                            **inputs_cpu,
                            max_new_tokens=int(max_new_tokens),
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            use_cache=True,
                            repetition_penalty=1.1,
                        )
                        self._cuda_disabled = True
                        self.device = "cpu"
                        inputs = inputs_cpu
                    else:
                        raise

            # Trim the prompt tokens
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return str(output_text).strip()

        except Exception as e:
            logger.error(f"Qwen3-VL freeform generation error: {e}", exc_info=True)
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
        if not self.enabled or not questions:
            return {q.get("id", ""): "" for q in questions}
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
            logger.error("Qwen generate_freeform_text_batch error: %s", e)
            return {q.get("id", ""): "" for q in questions}
    
    def classify_frame_artifacts(self, image_path: str) -> Dict[str, Any]:
        """Classify frame artifacts using Qwen3-VL."""
        if not self.enabled:
            return self._empty_artifact_result()
        
        try:
            if not QWEN_VL_UTILS_AVAILABLE or process_vision_info is None:
                logger.error("qwen_vl_utils not available")
                return self._empty_artifact_result()
            
            import torch
            from PIL import Image
            
            prompt = """Examine this image for quality issues. Respond with JSON:
{
    "has_artifacts": true/false,
    "artifact_type": "1" or "2" or "3" or "none",
    "confidence": 0.0-1.0,
    "description": "brief description",
    "severity": "low" or "medium" or "high"
}"""
            
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device (force CPU if CUDA has been disabled)
            target_device = "cpu" if (self.device == "cpu" or self._cuda_disabled) else self.device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            if target_device == "cpu" and model_device.type != "cpu":
                self.model = self.model.cpu()
            elif target_device != "cpu" and model_device.type == "cpu":
                self.model = self.model.to(target_device)
            
            # Generate with CUDA error handling
            # Final safety check: ensure model and inputs are on same device
            model_device = next(self.model.parameters()).device
            input_device = next(iter(inputs.values())).device if any(isinstance(v, torch.Tensor) for v in inputs.values()) else model_device
            if model_device != input_device:
                logger.warning(f"Device mismatch: model={model_device}, inputs={input_device}, fixing...")
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    tok = getattr(self.processor, "tokenizer", None)
                    pad_token_id = getattr(tok, "eos_token_id", None)
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=pad_token_id,
                        use_cache=True,  # Enable KV cache for efficiency
                        repetition_penalty=1.1  # Slight penalty to avoid repetition
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuda" in str(e).lower() or "assert" in str(e).lower():
                        logger.warning(f"CUDA error, switching to CPU fallback: {e}")
                        # Fallback to CPU
                        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        tok = getattr(self.processor, "tokenizer", None)
                        pad_token_id = getattr(tok, "eos_token_id", None)
                        generated_ids = model_cpu.generate(
                            **inputs_cpu,
                            max_new_tokens=256,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=pad_token_id,
                            use_cache=True,  # Enable KV cache for efficiency
                            repetition_penalty=1.1  # Slight penalty to avoid repetition
                        )
                        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cpu['input_ids'], generated_ids)]
                        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        self._cuda_disabled = True
                        self.device = "cpu"
                    else:
                        raise
            
            parsed = self._parse_artifact_response(output_text)
            parsed['model'] = self.model_name
            parsed['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Qwen3-VL: Artifact type={parsed.get('artifact_type', 'none')}")
            return parsed
            
        except Exception as e:
            logger.error(f"Qwen3-VL artifact error: {e}")
            return self._empty_artifact_result()
    
    def _build_extraction_prompt(
        self, 
        focus_areas: Optional[List[str]], 
        include_accessibility: bool
    ) -> str:
        """Build a speed-optimized extraction prompt.

        Goal: minimize generation length while keeping output machine-parsable.
        """
        priority = ", ".join(focus_areas) if focus_areas else ""
        access = "YES" if include_accessibility else "NO"

        # Strict, minimal schema to keep decoding short/fast.
        prompt = f"""Return ONLY valid JSON. No markdown. No extra text.

Schema (exactly):
{{"objects":["..."]}}

Rules:
- Output MUST be exactly one JSON object with exactly one key: "objects"
- "objects" is a list of 1-50 short singular nouns (no adjectives, no colors, no materials)
- Use "person" for any human
- Prefer shortest common name (e.g., "bike" not "bicycle")
- No duplicates, no synonyms, no explanations
- Accessibility mode: {access}
"""

        if include_accessibility:
            prompt += """- Include hazards/obstacles first (ground-level): cable, cord, cart, bag, box, trash can, cone, wet floor sign, step, stairs, curb, pillar, pole, stanchion, barrier
- Include signs as compound nouns: exit sign, restroom sign, elevator sign, emergency sign
"""

        if priority:
            prompt += f"- Priority objects: {priority}\n"

        return prompt
    
    def _parse_qwen_response(self, response: str) -> Dict[str, Any]:
        """Parse Qwen response."""
        import json
        import re
        try:
            def _strip_code_fences(text: str) -> str:
                text = text.strip()
                # Remove common fenced wrappers
                text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
                text = re.sub(r"\s*```$", "", text)
                text = text.replace("```json", "").replace("```", "")
                return text.strip()

            def _extract_balanced_json(text: str) -> Optional[str]:
                """Extract first balanced JSON object by brace counting (robust to nesting)."""
                start = text.find("{")
                if start == -1:
                    return None
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                return text[start:i + 1]
                return None

            cleaned = _strip_code_fences(response)
            json_str = _extract_balanced_json(cleaned)
            if json_str:
                for attempt in range(2):
                    try:
                        data = json.loads(json_str)
                        break
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}, trying to fix...")
                        # Fix common JSON issues
                        json_str = json_str.replace("'", '"')
                        json_str = re.sub(r",\s*}", "}", json_str)
                        json_str = re.sub(r",\s*]", "]", json_str)
                else:
                    data = None

                if isinstance(data, dict):
                    objects = self._normalize_objects(data.get("objects", []) or [])
                    categories_in = data.get("categories", {}) or {}
                    categories: Dict[str, List[str]] = {}
                    for k in ["signs", "accessibility", "people", "furniture", "technology", "other"]:
                        v = categories_in.get(k, [])
                        if not isinstance(v, list):
                            v = []
                        categories[k] = self._normalize_objects([str(x) for x in v])

                    # Merge category objects into the global list to guarantee consistency
                    merged = objects[:]
                    for items in categories.values():
                        merged.extend(items)
                    objects = self._normalize_objects(merged)[:200]

                    # If the model didn't populate categories, apply a lightweight heuristic categorizer
                    if all(len(v) == 0 for v in categories.values()) and objects:
                        def _cat(obj: str) -> str:
                            o = obj.lower()
                            if any(k in o for k in ["sign", "exit", "restroom", "elevator", "emergency"]):
                                return "signs"
                            if any(k in o for k in ["wheelchair", "braille", "ramp", "accessible"]):
                                return "accessibility"
                            # All people-related terms normalized to "person" by _normalize_objects, so check for "person"
                            if o == "person":
                                return "people"
                            if any(k in o for k in ["chair", "table", "couch", "sofa", "desk", "bed", "bench", "cabinet"]):
                                return "furniture"
                            if any(k in o for k in ["phone", "laptop", "computer", "screen", "monitor", "tv", "camera", "keyboard"]):
                                return "technology"
                            return "other"

                        auto = {k: [] for k in categories.keys()}
                        for obj in objects:
                            auto[_cat(obj)].append(obj)
                        categories = {k: self._normalize_objects(v) for k, v in auto.items()}

                    return {
                        "objects": objects,
                        "categories": categories,
                        "scene_description": str(data.get("scene_description", "") or "")[:500],
                        "primary_focus": str(data.get("primary_focus", "") or "")[:200],
                        "num_objects": len(objects),
                        "error": False,
                    }
            
            # If no JSON found, try to extract objects from plain text (quiet for batch runs)
            logger.debug(f"No JSON found in response, attempting text extraction. Response: {response[:300]}...")
            # Look for list-like patterns
            objects_list = re.findall(r'["\']([^"\']+)["\']', response)
            if objects_list:
                objects = self._normalize_objects(objects_list)[:200]
                # Heuristic categorization for consistent downstream use
                def _cat(obj: str) -> str:
                    o = obj.lower()
                    if any(k in o for k in ["sign", "exit", "restroom", "elevator", "emergency"]):
                        return "signs"
                    if any(k in o for k in ["wheelchair", "braille", "ramp", "accessible"]):
                        return "accessibility"
                    # All people-related terms normalized to "person" by _normalize_objects, so check for "person"
                    if o == "person":
                        return "people"
                    if any(k in o for k in ["chair", "table", "couch", "sofa", "desk", "bed", "bench", "cabinet"]):
                        return "furniture"
                    if any(k in o for k in ["phone", "laptop", "computer", "screen", "monitor", "tv", "camera", "keyboard"]):
                        return "technology"
                    return "other"

                auto = {
                    "signs": [],
                    "accessibility": [],
                    "people": [],
                    "furniture": [],
                    "technology": [],
                    "other": [],
                }
                for obj in objects:
                    auto[_cat(obj)].append(obj)
                categories = {k: self._normalize_objects(v) for k, v in auto.items()}
                return {
                    'objects': objects,
                    'categories': categories,
                    # Avoid returning truncated JSON/code fences as "scene description"
                    'scene_description': '',
                    'primary_focus': '',
                    'num_objects': len(objects),
                    'error': False
                }
        except Exception as e:
            logger.warning(f"Failed to parse Qwen3-VL response: {e}, response: {response[:200]}...")
        return self._empty_result()
    
    def _parse_artifact_response(self, response: str) -> Dict[str, Any]:
        """Parse artifact response."""
        import json
        import re
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'has_artifacts': data.get('has_artifacts', False),
                    'artifact_type': data.get('artifact_type', 'none'),
                    'confidence': float(data.get('confidence', 0.0)),
                    'description': data.get('description', ''),
                    'severity': data.get('severity', 'low'),
                    'affected_regions': [],
                    'error': False
                }
        except Exception as e:
            logger.warning(f"Failed to parse Qwen3-VL artifact JSON: {e}")
        return self._empty_artifact_result()


def test_qwen_extractor():
    """Test Qwen3-VL extractor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vlm_qwen.py <image_path>")
        return
    
    print("\n" + "="*60)
    print("Testing Qwen3-VL Extractor")
    print("="*60)
    
    extractor = Qwen3VLExtractor()
    
    if not extractor.enabled:
        print("❌ Qwen3-VL not available (check installation)")
        return
    
    result = extractor.extract_objects_from_keyframe(sys.argv[1])
    
    print(f"\n✅ Objects: {result['num_objects']}")
    for obj in result['objects'][:10]:
        print(f"  - {obj}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_qwen_extractor()
