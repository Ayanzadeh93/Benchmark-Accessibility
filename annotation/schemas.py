"""Pydantic schemas for annotation inputs/outputs.

We validate JSON metadata produced by Phase 2/3/4 pipelines so the annotator
can safely reason over object lists and relative spatial information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DetectionItem(BaseModel):
    """Single detection entry (bbox-only or bbox+segmentation)."""

    class_id: int = Field(default=0)
    class_name: str = Field(default="object")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Normalized bbox (0-1). Present in Phase 2 and Phase 3 fallbacks.
    x_center: Optional[float] = Field(default=None)
    y_center: Optional[float] = Field(default=None)
    width: Optional[float] = Field(default=None)
    height: Optional[float] = Field(default=None)

    # Pixel bbox (xyxy). Present in Phase 2 and SAM-based segmentation.
    bbox_xyxy: Optional[List[float]] = Field(default=None)

    # Segmentation polygon (normalized) if available.
    has_segmentation: Optional[bool] = Field(default=None)
    segmentation: Optional[List[float]] = Field(default=None)

    @field_validator("class_name", mode="before")
    @classmethod
    def _normalize_class_name(cls, v: Any) -> str:
        if v is None:
            return "object"
        s = str(v).strip()
        return s if s else "object"


class DetectionMetadata(BaseModel):
    """Metadata produced by `detection.pipeline`."""

    image: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    vocab: List[str] = Field(default_factory=list)
    detections: List[DetectionItem] = Field(default_factory=list)
    vlm_result: Optional[Dict[str, Any]] = None


class SegmentationMetadata(BaseModel):
    """Metadata produced by segmentation pipelines (YOLOv8-seg or SAM)."""

    image: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    detections: List[DetectionItem] = Field(default_factory=list)


class DepthMetadata(BaseModel):
    """Metadata produced by `depth_estimation.pipeline`."""

    image: Optional[str] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_mean: Optional[float] = None
    depth_std: Optional[float] = None
    raw_depth_path: Optional[str] = None


class RiskObstacle(BaseModel):
    """Structured obstacle entry for risk assessment."""
    type: str = Field(default="static", description="static | dynamic | none")
    object: str = Field(default="object", description="Object name")
    position: str = Field(default="none", description="front | left | right | none")
    distance: str = Field(default="medium", description="very near | near | medium | far")
    motion: str = Field(default="stationary", description="moving | stationary | none")
    fall_risk: str = Field(default="no", description="yes | no")


class RiskAssessment(BaseModel):
    """Risk assessment for navigation safety.

    Levels:
    - Low: Path clear, safe to proceed.
    - Medium: Potential hazard; obstacles present but not immediately dangerous.
    - High: Can be hazardous in a few seconds if person doesn't get aligned.
    - Extreme: Falling risk, can hit in <1 sec; dangerous - stop immediately.
    """
    level: str = Field(default="Medium", description="Low, Medium, High, or Extreme")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk score from 0.0 (safe) to 1.0 (dangerous)")
    reason: Optional[str] = Field(default=None, description="Short safety-focused explanation")
    scene_summary: Optional[str] = Field(default=None, description="One-sentence scene summary")
    obstacles: List[RiskObstacle] = Field(default_factory=list, description="Structured obstacle list")


class AccessibilityData(BaseModel):
    """Accessibility-focused data for blind user assistance."""
    
    location: str = Field(default="Unknown", description="Inferred location type (Airport, Store, Street, etc.)")
    time: str = Field(default="Unknown", description="Day, Night, or Indoor based on lighting")
    scene_description: str = Field(default="", description="Detailed scene description with spatial positions")
    ground_text: str = Field(default="", description="Very short (1 sentence) summary of the scene")
    spatial_objects: List[str] = Field(default_factory=list, description="Objects with clock positions and distances")
    highlight: List[str] = Field(default_factory=list, description="Accessibility alerts (ramps, stairs, elevators, etc.)")
    guidance: str = Field(default="", description="Brief navigation instruction")
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment, description="Scene risk assessment")


class AnnotationRecord(BaseModel):
    """Final per-image annotation record saved by the annotator."""

    image: str
    text: str
    timestamp: str
    task: str = Field(default="caption", description="caption, navigation, scene, or accessibility")
    sources: Dict[str, Optional[str]] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional structured fields (non-sensitive).")
    
    # Accessibility data (populated for accessibility task)
    accessibility: Optional[AccessibilityData] = Field(default=None, description="Accessibility-focused data for blind users")

