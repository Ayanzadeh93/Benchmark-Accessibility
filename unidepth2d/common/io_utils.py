"""Config loading, image listing, and frozen-detection parsing (two supported layouts)."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(images_dir: str, exts: List[str]) -> List[str]:
    exts_l = {e.lower() for e in exts}
    out = []
    for p in glob.glob(os.path.join(images_dir, "*")):
        if os.path.splitext(p)[1].lower() in exts_l:
            out.append(p)
    return sorted(out)


class Box:
    """A frozen detection box in ORIGINAL-image pixel coordinates."""

    __slots__ = ("x1", "y1", "x2", "y2", "class_id", "class_name", "confidence")

    def __init__(self, x1, y1, x2, y2, class_id=None, class_name=None, confidence=None):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _detection_json_path(metadata_dir: str, image_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cand = os.path.join(metadata_dir, stem + ".json")
    return cand if os.path.exists(cand) else None


def parse_detection_ensemble(
    metadata_dir: str, image_path: str, min_conf: float = 0.0
) -> Tuple[List[Box], Optional[Tuple[int, int]]]:
    """Parse one Detection_ensemble metadata JSON.

    Structure (per the dataset): {image_width, image_height, detections: [{class_id, class_name,
    confidence, bbox_xyxy:[x1,y1,x2,y2] (pixels), x_center,y_center,width,height (normalized)}]}.
    Prefers pixel bbox_xyxy; falls back to normalized center/size * (W,H). Returns (boxes, (W,H)).
    """
    jp = _detection_json_path(metadata_dir, image_path)
    if jp is None:
        return [], None
    with open(jp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    W = meta.get("image_width")
    H = meta.get("image_height")
    wh = (int(W), int(H)) if (W and H) else None
    boxes: List[Box] = []
    for det in meta.get("detections", []):
        conf = det.get("confidence")
        if conf is not None and conf < min_conf:
            continue
        xyxy = det.get("bbox_xyxy")
        if xyxy and len(xyxy) == 4:
            x1, y1, x2, y2 = xyxy
        elif wh is not None and all(k in det for k in ("x_center", "y_center", "width", "height")):
            cxn, cyn, wn, hn = det["x_center"], det["y_center"], det["width"], det["height"]
            x1 = (cxn - wn / 2) * wh[0]
            y1 = (cyn - hn / 2) * wh[1]
            x2 = (cxn + wn / 2) * wh[0]
            y2 = (cyn + hn / 2) * wh[1]
        else:
            continue
        boxes.append(
            Box(x1, y1, x2, y2, det.get("class_id"), det.get("class_name"), conf)
        )
    return boxes, wh


def _load_yolo_classes(classes_file: str) -> Optional[List[str]]:
    if not classes_file or not os.path.exists(classes_file):
        return None
    with open(classes_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_yolo_txt(
    labels_dir: str,
    image_path: str,
    img_wh: Tuple[int, int],
    classes: Optional[List[str]] = None,
    min_conf: float = 0.0,
) -> List[Box]:
    """Parse a YOLO txt (``class cx cy w h`` normalized) into pixel-xyxy boxes. img_wh required."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    tp = os.path.join(labels_dir, stem + ".txt")
    if not os.path.exists(tp):
        return []
    W, H = img_wh
    boxes: List[Box] = []
    with open(tp, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cxn, cyn, wn, hn = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = (cxn - wn / 2) * W
            y1 = (cyn - hn / 2) * H
            x2 = (cxn + wn / 2) * W
            y2 = (cyn + hn / 2) * H
            cname = classes[cid] if (classes and 0 <= cid < len(classes)) else None
            boxes.append(Box(x1, y1, x2, y2, cid, cname, None))
    return boxes


def get_detections(
    image_path: str, cfg: dict, img_wh: Optional[Tuple[int, int]] = None
) -> Tuple[List[Box], Optional[Tuple[int, int]]]:
    """Dispatch on cfg['data']['detection_source']; returns (boxes, (W,H) or None)."""
    dcfg = cfg["data"]
    src = dcfg.get("detection_source", "detection_ensemble")
    min_conf = float(cfg["teacher"].get("min_box_conf", 0.0))
    if src == "detection_ensemble":
        return parse_detection_ensemble(dcfg["detection_ensemble_metadata"], image_path, min_conf)
    elif src == "yolo_txt":
        if img_wh is None:
            raise ValueError("yolo_txt detection source requires img_wh (image width/height).")
        classes = _load_yolo_classes(dcfg.get("yolo_classes_file", ""))
        return parse_yolo_txt(dcfg["yolo_labels_dir"], image_path, img_wh, classes, min_conf), img_wh
    raise ValueError(f"Unknown detection_source: {src}")
