#!/usr/bin/env python3
"""
Base VLM Extractor Interface
Abstract base class for all VLM implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseVLMExtractor(ABC):
    """
    Abstract base class for VLM extractors.
    All VLM implementations should inherit from this.
    """
    
    def __init__(self, model_name: str = "base"):
        """Initialize VLM extractor."""
        self.model_name = model_name
        self.enabled = True
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def extract_objects_from_keyframe(
        self, 
        image_path: str,
        focus_areas: List[str] = None,
        include_accessibility: bool = True
    ) -> Dict[str, Any]:
        """
        Extract objects from keyframe.
        
        Args:
            image_path: Path to keyframe image
            focus_areas: Optional focus areas
            include_accessibility: Include accessibility features
            
        Returns:
            Dictionary with extracted objects and metadata
        """
        pass
    
    @abstractmethod
    def classify_frame_artifacts(
        self, 
        image_path: str
    ) -> Dict[str, Any]:
        """
        Classify frame artifacts (Q-Router style).
        
        Args:
            image_path: Path to frame image
            
        Returns:
            Dictionary with artifact classification
        """
        pass
    
    def _normalize_objects(self, objects: List[str]) -> List[str]:
        """Normalize object names."""
        normalized: List[str] = []
        seen = set()
        for obj in objects:
            if not isinstance(obj, str):
                continue
            obj = obj.strip().lower()
            if not obj or len(obj) <= 1 or len(obj) >= 50:
                continue
            # Filter common prompt/format artifacts
            if obj in {"object", "objects", "item", "items"}:
                continue
            if obj in seen:
                continue
            seen.add(obj)
            normalized.append(obj)
        return normalized
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'objects': [],
            'categories': {},
            'scene_description': '',
            'primary_focus': '',
            'num_objects': 0,
            'model': self.model_name,
            'error': True
        }
    
    def _empty_artifact_result(self) -> Dict[str, Any]:
        """Return empty artifact result."""
        return {
            'has_artifacts': False,
            'artifact_type': 'none',
            'confidence': 0.0,
            'description': '',
            'severity': 'low',
            'affected_regions': [],
            'model': self.model_name,
            'error': True
        }