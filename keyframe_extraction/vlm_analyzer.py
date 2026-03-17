"""VLM-based keyframe analysis integration."""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

try:
    from simple_vlm_integration import SimpleVLMIntegration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logging.warning("VLM integration not available")

logger = logging.getLogger(__name__)


class VLMKeyframeAnalyzer:
    """Wrapper for VLM keyframe analysis."""
    
    def __init__(
        self,
        model_type: str = "qwen",
        api_key: Optional[str] = None,
        device: str = "auto",
        enabled: bool = True
    ):
        self.enabled = enabled and VLM_AVAILABLE
        self.vlm = None
        
        if self.enabled:
            try:
                self.vlm = SimpleVLMIntegration(
                    model_type=model_type,
                    api_key=api_key,
                    device=device
                )
                logger.info(f"VLM analyzer ready ({model_type})")
            except Exception as e:
                logger.warning(f"VLM initialization failed: {e}")
                self.enabled = False
    
    def analyze_keyframe(self, keyframe_path: str) -> Dict[str, Any]:
        """Analyze single keyframe with VLM."""
        if not self.enabled or not self.vlm:
            return {'enabled': False}
        
        try:
            result = self.vlm.analyze_keyframe(keyframe_path)
            result['enabled'] = True
            return result
        except (RuntimeError, AssertionError) as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "assert" in error_str:
                logger.warning(f"VLM CUDA error, skipping analysis for this keyframe: {e}")
                return {'enabled': False, 'error': 'CUDA assertion error', 'skipped': True}
            else:
                logger.error(f"VLM analysis error: {e}")
                return {'enabled': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"VLM analysis error: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def batch_analyze(self, keyframe_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Batch analyze keyframes."""
        if not self.enabled or not self.vlm:
            return {'enabled': False}
        
        try:
            return self.vlm.batch_analyze_keyframes(keyframe_paths, output_dir)
        except Exception as e:
            logger.error(f"VLM batch analysis error: {e}")
            return {'enabled': False, 'error': str(e)}
