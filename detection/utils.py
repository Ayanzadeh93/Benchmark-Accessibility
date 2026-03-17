"""Detection utilities (NMS, IoU, coordinate conversions)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

Detection = Dict[str, Any]


def compute_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] pixel coordinates."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    a1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    a2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def nms_detections(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> List[Detection]:
    """Apply (class-aware) NMS over detections.
    
    Enhanced to filter overlapping objects with same name more aggressively.

    Args:
        detections: list of dicts containing at least "bbox_xyxy" and "confidence"
        iou_threshold: IoU threshold for suppression
        class_aware: if True, suppress only within the same class_name (or class_id)
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept: List[Detection] = []

    while dets:
        best = dets.pop(0)
        kept.append(best)

        best_box = best.get("bbox_xyxy")
        if not best_box:
            continue

        best_class = best.get("class_name", None) if class_aware else None
        best_id = best.get("class_id", None) if class_aware else None

        remaining: List[Detection] = []
        for det in dets:
            if class_aware:
                same = False
                if best_class is not None and det.get("class_name") is not None:
                    # Normalize class names for comparison (case-insensitive, strip whitespace)
                    best_class_norm = str(best_class).lower().strip()
                    det_class_norm = str(det.get("class_name")).lower().strip()
                    same = best_class_norm == det_class_norm
                elif best_id is not None and det.get("class_id") is not None:
                    same = int(best.get("class_id")) == int(det.get("class_id"))
                
                # If same class, check overlap more strictly
                if same:
                    iou = compute_iou_xyxy(best_box, det.get("bbox_xyxy", best_box))
                    # For same class, use stricter threshold to filter duplicates
                    # Also check if one box is mostly contained in the other
                    if iou >= float(iou_threshold):
                        # Suppress overlapping detection with same name
                        continue
                    # Additional check: if one box is >80% contained in the other, suppress the smaller one
                    det_box = det.get("bbox_xyxy", best_box)
                    best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    if best_area > 0 and det_area > 0:
                        # Compute intersection area
                        x1 = max(best_box[0], det_box[0])
                        y1 = max(best_box[1], det_box[1])
                        x2 = min(best_box[2], det_box[2])
                        y2 = min(best_box[3], det_box[3])
                        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                        # If smaller box is >90% contained in larger, suppress it (less aggressive)
                        if inter_area > 0:
                            smaller_area = min(best_area, det_area)
                            containment = inter_area / smaller_area
                            if containment > 0.9:  # Changed from 0.8 to 0.9 to be less aggressive
                                continue
                
                if not same:
                    remaining.append(det)
                    continue
            else:
                # Not class-aware: check all overlaps
                iou = compute_iou_xyxy(best_box, det.get("bbox_xyxy", best_box))
                if iou < float(iou_threshold):
                    remaining.append(det)

        dets = remaining

    return kept


def xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert xyxy pixel coords to YOLO normalized (x_center,y_center,w,h)."""
    x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
    w = max(0.0, x2f - x1f)
    h = max(0.0, y2f - y1f)
    xc = x1f + w / 2.0
    yc = y1f + h / 2.0
    return (
        float(xc / float(img_w)),
        float(yc / float(img_h)),
        float(w / float(img_w)),
        float(h / float(img_h)),
    )


def clip01(x: float) -> float:
    """Clamp to [0, 1]."""
    return float(np.clip(float(x), 0.0, 1.0))

