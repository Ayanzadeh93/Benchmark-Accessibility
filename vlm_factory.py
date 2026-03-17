#!/usr/bin/env python3
"""
VLM Factory - Easy switching between GPT-4o and Qwen3-VL
"""

import logging
from typing import Optional
from vlm_base import BaseVLMExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMFactory:
    """Factory for creating VLM extractors."""
    
    @staticmethod
    def create_extractor(
        model_type: str = "qwen",
        api_key: Optional[str] = None,
        device: str = "auto"
    ) -> BaseVLMExtractor:
        """
        Create VLM extractor.
        
        Args:
            model_type: "qwen", "florence2", "gpt4o", "gpt5nano"
            api_key: API key for GPT models
            device: "auto", "cuda", or "cpu"
            
        Returns:
            VLM extractor instance
            
        Examples:
            # Use Florence-2 (FASTEST, free, local GPU)
            extractor = VLMFactory.create_extractor("florence2", device="cuda")
            
            # Use Qwen (free, local GPU)
            extractor = VLMFactory.create_extractor("qwen", device="cuda")
            
            # Use GPT-4o (API)
            extractor = VLMFactory.create_extractor("gpt4o", api_key="sk-...")
        """
        model_type = model_type.lower().strip()
        
        if model_type in ["florence2", "florence", "florence-2"]:
            logger.info("Creating Florence-2 extractor (FASTEST, GPU-accelerated)")
            from vlm_florence2 import Florence2Extractor
            return Florence2Extractor(
                device=device,
                model_size="base"  # base is faster than large
            )
        
        elif model_type in ["qwen", "qwen3", "qwen3vl", "qwen-vl"]:
            logger.info("Creating Qwen3-VL extractor (GPU-accelerated)")
            from vlm_qwen import Qwen3VLExtractor
            return Qwen3VLExtractor(
                device=device
            )
        
        elif model_type in ["llava", "llava15", "llava-1.5", "llava1.5"]:
            logger.info("Creating LLaVA-1.5-7B extractor (GPU-accelerated)")
            from vlm_llava import LLaVAExtractor
            return LLaVAExtractor(
                device=device
            )
        
        elif model_type in ["gpt4o", "gpt-4o", "openai"]:
            logger.info("Creating GPT-4o extractor (API)")
            from vlm_gpt4o import GPT4oExtractor
            return GPT4oExtractor(api_key=api_key, model_name="gpt-4o")
        
        elif model_type in ["gpt5nano", "gpt-5-nano", "gpt5-nano", "gptnano"]:
            logger.info("Creating GPT-5 Nano extractor (API)")
            from vlm_gpt4o import GPT4oExtractor
            return GPT4oExtractor(api_key=api_key, model_name="gpt-5-nano-2025-08-07")
        
        elif model_type in ["gpt5mini", "gpt-5-mini", "gpt5-mini", "gptmini"]:
            logger.info("Creating GPT-5 Mini extractor (API, CHEAPEST)")
            from vlm_gpt4o import GPT4oExtractor
            return GPT4oExtractor(api_key=api_key, model_name="gpt-5-mini-2025-08-07")

        elif model_type in [
            "openrouter_trinity",
            "openrouter_llama32_11b_vision",
            "openrouter_llama4_maverick",
            "openrouter_molmo_8b",
            "openrouter_ministral_3b",
            "openrouter_gpt_oss_safeguard_20b",
            "openrouter_qwen3_vl_235b",
            "openrouter_qwen3_vl_8b",
            "openrouter_qwen_vl_plus",
            "qwen8b",  # Alias for openrouter_qwen3_vl_8b
            "openrouter",
        ]:
            from vlm_openrouter import OpenRouterExtractor

            # Map friendly names to OpenRouter model ids
            model_map = {
                "openrouter_trinity": "arcee-ai/trinity-large-preview:free",
                "openrouter_llama32_11b_vision": "meta-llama/llama-3.2-11b-vision-instruct",
                "openrouter_llama4_maverick": "meta-llama/llama-4-maverick",
                "openrouter_molmo_8b": "allenai/molmo-2-8b:free",
                "openrouter_ministral_3b": "mistralai/ministral-3b-2512",
                "openrouter_gpt_oss_safeguard_20b": "openai/gpt-oss-safeguard-20b",
                "openrouter_qwen3_vl_235b": "qwen/qwen3-vl-235b-a22b-instruct",
                "openrouter_qwen3_vl_8b": "qwen/qwen3-vl-8b-instruct",
                "qwen8b": "qwen/qwen3-vl-8b-instruct",
                "openrouter_qwen_vl_plus": "qwen/qwen-vl-plus",
            }
            # qwen8b -> use openrouter_qwen3_vl_8b model id
            model_id = model_map.get(model_type, model_type)
            logger.info(f"Creating OpenRouter extractor (model: {model_id})")
            return OpenRouterExtractor(api_key=api_key, model_id=model_id)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'florence2', 'qwen', 'llava', 'gpt4o', 'gpt5nano', or 'gpt5mini'")
    
    @staticmethod
    def list_available_models():
        """List available VLM models."""
        print("\n" + "="*60)
        print("Available VLM Models")
        print("="*60)
        
        print("\n1. Florence-2 (florence2) ⭐ FASTEST")
        print("   - Local GPU (FREE)")
        print("   - Speed: 1-2 seconds per image on RTX GPU")
        print("   - Size: 230M params (tiny!)")
        print("   - Optimized for: Object detection, grounding, captioning")
        
        print("\n2. LLaVA-1.5-7B (llava)")
        print("   - Local GPU (FREE)")
        print("   - Speed: 3-5 seconds per image on RTX GPU")
        print("   - Size: 7B params")
        print("   - Optimized for: VQA, detailed image understanding")
        print("   - Usage: VLMFactory.create_extractor('florence2', device='cuda')")
        
        print("\n2. Qwen3-VL-2B (qwen)")
        print("   - Local GPU (FREE)")
        print("   - Speed: 2-3 seconds per image")
        print("   - Size: 2B params")
        print("   - Usage: VLMFactory.create_extractor('qwen', device='cuda')")

        print("\n3. GPT-4o (gpt4o)")
        print("   - API-based (paid)")
        print("   - Requires: OpenAI API key")
        print("   - Cost: ~$0.005 per image")
        print("   - Usage: VLMFactory.create_extractor('gpt4o', api_key='sk-...')")
        
        print("\n4. GPT-5 Nano (gpt5nano)")
        print("   - API-based (paid, cheaper than GPT-4o)")
        print("   - Requires: OpenAI API key")
        print("   - Cost: ~$0.00005 per image (input), ~$0.0004 per image (output)")
        print("   - Usage: VLMFactory.create_extractor('gpt5nano', api_key='sk-...')")
        
        print("\n" + "="*60)


# Convenience functions
def create_qwen_extractor(device: str = "auto") -> BaseVLMExtractor:
    """Create Qwen3-VL extractor (shortcut)."""
    return VLMFactory.create_extractor("qwen", device=device)


def create_gpt4o_extractor(api_key: Optional[str] = None) -> BaseVLMExtractor:
    """Create GPT-4o extractor (shortcut)."""
    return VLMFactory.create_extractor("gpt4o", api_key=api_key)


if __name__ == "__main__":
    VLMFactory.list_available_models()