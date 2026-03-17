#!/usr/bin/env python3
"""
Simple VLM Integration for Keyframe Analysis
Works with both Qwen3-VL and GPT-4o
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from vlm_factory import VLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleVLMIntegration:
    """
    Simple VLM integration for keyframe analysis.
    
    Usage:
        # With Florence-2 (FASTEST, free, GPU)
        vlm = SimpleVLMIntegration(model_type="florence2")
        
        # With Qwen (free, GPU)
        vlm = SimpleVLMIntegration(model_type="qwen")
        
        # With GPT-4o (API)
        vlm = SimpleVLMIntegration(model_type="gpt4o", api_key="sk-...")
        
        # With GPT-5 Nano (API, cheaper)
        vlm = SimpleVLMIntegration(model_type="gpt5nano", api_key="sk-...")
        
        # Analyze keyframe
        result = vlm.analyze_keyframe("path/to/keyframe.jpg")
    """
    
    def __init__(
        self,
        model_type: str = "qwen",
        api_key: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize VLM integration.
        
        Args:
            model_type: "florence2" (fastest), "qwen", "gpt4o", or "gpt5nano"
            api_key: OpenAI API key (for GPT models)
            device: "auto", "cuda", or "cpu" (for Florence-2/Qwen)
        """
        self.model_type = model_type
        self.extractor = VLMFactory.create_extractor(
            model_type=model_type,
            api_key=api_key,
            device=device
        )
        
        if not self.extractor.enabled:
            raise RuntimeError(f"{model_type} extractor failed to initialize")
        
        logger.info(f"VLM Integration ready ({model_type})")
    
    def analyze_keyframe(
        self,
        keyframe_path: str,
        include_artifacts: bool = True,
        include_accessibility: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze single keyframe.
        
        Args:
            keyframe_path: Path to keyframe image
            include_artifacts: Include artifact detection
            include_accessibility: Include accessibility features
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing: {Path(keyframe_path).name}")
        
        result = {
            'keyframe_path': keyframe_path,
            'model': self.model_type
        }
        
        # Extract objects
        objects_result = self.extractor.extract_objects_from_keyframe(
            keyframe_path,
            include_accessibility=include_accessibility
        )
        result['objects'] = objects_result
        
        # Detect artifacts
        if include_artifacts:
            artifacts_result = self.extractor.classify_frame_artifacts(
                keyframe_path
            )
            result['artifacts'] = artifacts_result
        
        return result
    
    def batch_analyze_keyframes(
        self,
        keyframe_paths: List[str],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple keyframes and create vocabulary.
        
        Args:
            keyframe_paths: List of keyframe paths
            output_dir: Optional output directory for results
            
        Returns:
            Dictionary with batch results and vocabulary
        """
        logger.info(f"Batch analyzing {len(keyframe_paths)} keyframes...")
        
        all_objects = []
        all_results = []
        artifact_count = 0
        
        for i, kf_path in enumerate(keyframe_paths, 1):
            logger.info(f"[{i}/{len(keyframe_paths)}] {Path(kf_path).name}")
            
            result = self.analyze_keyframe(kf_path)
            all_results.append(result)
            
            # Collect objects
            if 'objects' in result and not result['objects'].get('error'):
                all_objects.extend(result['objects'].get('objects', []))
            
            # Count artifacts
            if result.get('artifacts', {}).get('has_artifacts'):
                artifact_count += 1
        
        # Create vocabulary
        vocabulary = self._create_vocabulary(all_objects, all_results)
        vocabulary['artifact_rate'] = artifact_count / len(keyframe_paths) if keyframe_paths else 0
        vocabulary['model'] = self.model_type
        
        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save vocabulary JSON
            vocab_json = output_path / "vocabulary.json"
            with open(vocab_json, 'w') as f:
                json.dump(vocabulary, f, indent=2)
            logger.info(f"Saved: {vocab_json}")
            
            # Save object list (for detectors)
            object_list = output_path / "object_list.txt"
            with open(object_list, 'w') as f:
                for obj in vocabulary['unique_objects']:
                    f.write(f"{obj}\n")
            logger.info(f"Saved: {object_list}")
            
            # Save full results
            results_json = output_path / "full_results.json"
            with open(results_json, 'w') as f:
                json.dump({
                    'keyframes': all_results,
                    'vocabulary': vocabulary
                }, f, indent=2)
            logger.info(f"Saved: {results_json}")
        
        return {
            'keyframes': all_results,
            'vocabulary': vocabulary
        }
    
    def _create_vocabulary(
        self,
        all_objects: List[str],
        all_results: List[Dict]
    ) -> Dict[str, Any]:
        """Create object vocabulary from results."""
        # Count frequencies
        counter = Counter(all_objects)
        
        # Unique objects (sorted)
        unique_objects = sorted(set(all_objects))
        
        # Top objects
        top_objects = dict(counter.most_common(50))
        
        # Aggregate categories
        categories = {}
        for result in all_results:
            if 'objects' in result and 'categories' in result['objects']:
                for cat, items in result['objects']['categories'].items():
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].extend(items)
        
        # Deduplicate categories
        for cat in categories:
            categories[cat] = sorted(set(categories[cat]))
        
        return {
            'unique_objects': unique_objects,
            'num_unique': len(unique_objects),
            'top_50': top_objects,
            'categories': categories,
            'total_detections': len(all_objects)
        }


# Example usage
def example_usage():
    """Example of how to use SimpleVLMIntegration."""
    print("\n" + "="*70)
    print("EXAMPLE: Simple VLM Integration")
    print("="*70)
    
    print("\n# Option 1: Use Florence-2 (FASTEST, FREE, GPU)")
    print("vlm = SimpleVLMIntegration(model_type='florence2', device='cuda')")
    print("result = vlm.analyze_keyframe('keyframe.jpg')")
    
    print("\n# Option 2: Use Qwen (FREE, GPU)")
    print("vlm = SimpleVLMIntegration(model_type='qwen', device='cuda')")
    print("result = vlm.analyze_keyframe('keyframe.jpg')")
    
    print("\n# Option 3: Use GPT-4o (API)")
    print("vlm = SimpleVLMIntegration(model_type='gpt4o', api_key='sk-...')")
    print("result = vlm.analyze_keyframe('keyframe.jpg')")
    
    print("\n# Batch processing")
    print("keyframes = ['kf1.jpg', 'kf2.jpg', 'kf3.jpg']")
    print("results = vlm.batch_analyze_keyframes(keyframes, output_dir='./output')")
    
    print("\n# Access results")
    print("objects = result['objects']['objects']")
    print("has_artifacts = result['artifacts']['has_artifacts']")
    print("vocabulary = results['vocabulary']['unique_objects']")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with image
        model = sys.argv[2] if len(sys.argv) > 2 else "qwen"
        
        print(f"\nTesting with {model.upper()}...")
        vlm = SimpleVLMIntegration(model_type=model)
        result = vlm.analyze_keyframe(sys.argv[1])
        
        print(f"\nObjects: {result['objects']['num_objects']}")
        for obj in result['objects']['objects'][:10]:
            print(f"  - {obj}")
    else:
        example_usage()