"""CLIP-based similarity evaluation between images and VLM object lists."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from pydantic import BaseModel, Field, validator

try:
    import clip  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    clip = None

try:
    from detection.vocab import normalize_detection_label, _STOP_OBJECTS  # type: ignore
except Exception:  # pragma: no cover - fallback if import fails
    def normalize_detection_label(label: str) -> str:
        return " ".join(str(label).strip().lower().split())

    _STOP_OBJECTS = set()

logger = logging.getLogger(__name__)


_PART_TOKENS = {
    # clothing parts
    "button",
    "zipper",
    "cuff",
    "collar",
    "sleeve",
    "pocket",
    "hood",
    # body parts (keep "person" instead)
    "beard",
    "hair",
    # car/vehicle parts
    "bumper",
    "tire",
    "rim",
    "hubcap",
    "fender",
    "headlight",
    "taillight",
    "license plate",
    # building/fixture parts
    "windowpane",
    "trim",
    "handle",
    "knob",
    "hinge",
    "panel",
    "vent",
    "grille",
}


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _is_part(label: str) -> bool:
    l = _norm(label)
    if l in _PART_TOKENS:
        return True
    for token in _PART_TOKENS:
        if token in l:
            return True
    return False


class SimilarityRunConfig(BaseModel):
    """Configuration for CLIP similarity evaluation."""

    images_dir: Path
    vlm_json_dir: Path
    output_dir: Path
    clip_model: str = Field(default="ViT-B/32")
    device: str = Field(default="auto")
    min_similarity: float = Field(default=0.18, ge=-1.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    max_objects: int = Field(default=50, ge=1)
    keep_parts: bool = Field(default=False)
    skip_existing: bool = Field(default=True)
    max_images: Optional[int] = Field(default=None, ge=1)

    @validator("device")
    def _check_device(cls, value: str) -> str:
        v = str(value).lower().strip()
        if v not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        return v

    class Config:
        arbitrary_types_allowed = True


class SimilarityObject(BaseModel):
    """Similarity score for a single object."""

    name: str
    score: float


class SimilarityResult(BaseModel):
    """Per-image similarity evaluation result."""

    image: str
    vlm_json: Optional[str]
    clip_model: str
    num_objects_in: int
    num_objects_out: int
    objects: List[SimilarityObject]
    purified_objects: List[str]


def _extract_objects(vlm_result: Dict[str, Any]) -> List[str]:
    if not isinstance(vlm_result, dict):
        return []
    if isinstance(vlm_result.get("objects"), list):
        return [str(x) for x in vlm_result.get("objects") or []]
    if isinstance(vlm_result.get("objects"), dict):
        obj = vlm_result.get("objects") or {}
        if isinstance(obj.get("objects"), list):
            return [str(x) for x in obj.get("objects") or []]
    return []


def _purify_objects(objects: Iterable[str], keep_parts: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for obj in objects:
        norm = normalize_detection_label(obj)
        if not norm:
            continue
        if norm in _STOP_OBJECTS:
            continue
        if not keep_parts and _is_part(norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class SimilarityPipeline:
    """Compute CLIP similarity between images and object lists."""

    def __init__(self, cfg: SimilarityRunConfig) -> None:
        self.cfg = cfg
        if clip is None:
            raise RuntimeError("CLIP is not available. Install via: pip install git+https://github.com/openai/CLIP.git")
        self.device = _select_device(cfg.device)
        self.model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.model.eval()
        logger.info(f"[OK] CLIP loaded: {cfg.clip_model} on {self.device}")

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_image(self, image_path: Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_image(img_input)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _score_objects(self, image_path: Path, objects: List[str]) -> List[SimilarityObject]:
        if not objects:
            return []
        image_feats = self._encode_image(image_path)
        text_feats = self._encode_text(objects)
        with torch.inference_mode():
            sims = (image_feats @ text_feats.T).squeeze(0).detach().cpu().numpy()
        return [
            SimilarityObject(name=obj, score=float(sims[i]))
            for i, obj in enumerate(objects)
        ]

    def process_dir(self) -> Dict[str, Any]:
        images_dir = self.cfg.images_dir
        vlm_json_dir = self.cfg.vlm_json_dir
        out_dir = self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
        if self.cfg.max_images is not None:
            images = images[: int(self.cfg.max_images)]

        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        skipped = 0

        from tqdm.auto import tqdm

        for idx, img_path in enumerate(tqdm(images, desc="Similarity"), 1):
            try:
                stem = img_path.stem.replace("_keyframe", "")
                out_path = out_dir / f"{stem}_similarity.json"
                if self.cfg.skip_existing and out_path.exists():
                    skipped += 1
                    continue

                vlm_json = vlm_json_dir / f"{stem}_vlm.json"
                vlm_result = _load_json(vlm_json) if vlm_json.exists() else None
                if vlm_result is None:
                    failures.append({"image": str(img_path), "error": "Missing VLM JSON"})
                    continue

                raw_objects = _extract_objects(vlm_result)
                purified = _purify_objects(raw_objects, keep_parts=self.cfg.keep_parts)
                purified = purified[: self.cfg.max_objects]

                scored = self._score_objects(img_path, purified)
                scored.sort(key=lambda x: x.score, reverse=True)

                if self.cfg.min_similarity is not None:
                    scored = [s for s in scored if s.score >= float(self.cfg.min_similarity)]
                if self.cfg.top_k is not None:
                    scored = scored[: int(self.cfg.top_k)]

                result = SimilarityResult(
                    image=str(img_path),
                    vlm_json=str(vlm_json) if vlm_json.exists() else None,
                    clip_model=self.cfg.clip_model,
                    num_objects_in=len(raw_objects),
                    num_objects_out=len(scored),
                    objects=scored,
                    purified_objects=[s.name for s in scored],
                )

                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result.dict(), f, indent=2)

                results.append(result.dict())
                logger.info(f"[{idx}/{len(images)}] {img_path.name}: {len(scored)} objects kept")
            except Exception as e:
                failures.append({"image": str(img_path), "error": str(e)})

        summary = {
            "images_dir": str(images_dir),
            "vlm_json_dir": str(vlm_json_dir),
            "output_dir": str(out_dir),
            "processed": len(results),
            "skipped": skipped,
            "failed": len(failures),
            "failures": failures[:50],
        }
        with (out_dir / "similarity_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
