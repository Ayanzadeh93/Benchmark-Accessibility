"""V2 (CoDeQ) training / evaluation data (docs/V2_PLAN.md sections 5.1-5.3; amendments A1, A2, A7, A21).

Contract
--------
``V2Dataset(roles, split, cfg)``: ``roles`` is the path of configs/v2/roles.yaml (or an already
loaded dict); ``split`` one of :data:`SPLITS`; ``cfg`` the ``data.dataset`` (train) or
``eval.dataset`` (eval) mapping of a rung config. Items::

    {"image" (3,H,W) float32 ImageNet-normalised RGB, "target_log" (1,H,W) float32 ln(metres)
     (0 where invalid), "valid" (1,H,W) bool, "is_gt" bool, "dmax" float metres,
     "source" str, "id" str}  (+ "intrinsics" (4,) float32 [fx/W, fy/H, cx/W, cy/H] when
     cfg["intrinsics"] is true)

Attributes used by the trainer: ``ids``, ``sources`` (one per row), ``split``, ``roles_sha256``.

``make_sampler(ds, draw_weights, n_draws, seed)`` -> list of ``n_draws`` row indices: a group is
drawn with probability proportional to its weight (``"own+x_indoor"`` = union of those sources),
then a row uniformly inside it. Deterministic in ``seed``. A positive-weight group naming a source
with no rows raises (e.g. K1 asking for X-indoor before the pool is labelled).

Splits (all ids come from roles.yaml; nothing is re-derived here)
-----------------------------------------------------------------
* ``train``: ``train.own`` + ``train.nyu`` (+ every ``train.pool.by_pool.<name>`` when the cover
  pipeline is used; pool rows exist only in the cover cache).
* ``val``: legacy 3 % carve (``val.own`` + ``val.nyu``); never used for selection.
* ``dev``: D_SUN (``dev.d_sun``) - every ladder decision.
* ``frozen_own``, ``video_own`` (``test.*``), ``calibration`` (``calibration.own``), ``sf6_own``
  (``sf6_temporal.own_ids``): evaluation-only splits; ``augment`` is refused on them.

Before anything is read the training ids are checked against every held-out list (dev, tests,
SF6 footage, NYU family, the leak-blocker exclusions); any overlap raises :class:`LeakError`.

Pipelines (``cfg["pipeline"]``)
--------------------------------
* ``"squash"`` (R0-R4d): rows of ``paths.manifest`` (outputs/data_gt), already 640x480. Training
  adds the legacy S5 zoom crop (``zoom_crop: {p, min_scale}``; scale U(min_scale, .9), crop,
  resize back) exactly as distill/dataset.py MultiSourceDataset.
* ``"cover"`` (R5a onward): own and pool rows from ``paths.cover_cache/<source>/manifest.csv``
  (aspect kept, scale max(640/W, 480/H)); a 480x640 window is cropped at random for training and
  centred for evaluation. NYU and SUN are 4:3, for which the cover transform is the identity, so
  they are read from ``paths.manifest``. A7 split: R5a-cover keeps ``zoom_crop``, R5a-nozoom sets
  ``zoom_crop.p: 0``.

Training augmentation order (legacy): window crop -> zoom crop -> resize to ``hw`` -> h-flip
(``flip_p``) -> legacy jitter (``jitter: {p, gain, bias}``) -> optional T2-05 strong photometric
perturbation (``photometric: strong``, pseudo-labelled rows only; colour jitter .4/.4/.4/.1 at
p .8, Gaussian blur sigma .1-2 at p .5, grayscale p .1). Depth is resized with INTER_NEAREST only.
Intrinsics follow every geometric step (crop rescales f and shifts c; flip mirrors cx).

Randomness uses ``numpy.random`` (seeded per worker by the trainer), as the legacy loader did.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

cv2.setNumThreads(0)

SPLITS = ("train", "val", "dev", "frozen_own", "video_own", "calibration", "sf6_own")
EVAL_ONLY_SPLITS = ("dev", "frozen_own", "video_own", "calibration", "sf6_own")
PIPELINES = ("squash", "cover")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INTR = (0.7, 0.9, 0.5, 0.5)       # distill/dataset.py DEFAULT_INTR (last-resort fallback)

#: Every key ``cfg`` may hold, with its default (unknown keys raise: one-factor checks compare
#: resolved configs, so a typo must not silently become a no-op).
DEFAULTS: Dict[str, object] = {
    "pipeline": "squash",
    "hw": [480, 640],
    "augment": None,                        # None -> True for train, False otherwise
    "intrinsics": False,
    "zoom_crop": {"p": 0.5, "min_scale": 0.5},   # S16 recipe (crop_context p .5, min_scale .5)
    "flip_p": 0.5,
    "jitter": {"p": 0.5, "gain": 0.15, "bias": 10.0},
    "photometric": None,                    # None | "strong" (T2-05)
    "cutmix": None,                         # T2-06: not implemented here
    "sources": None,                        # optional subset of sources to keep
    "limit": None,                          # first N rows per source after sorting (smoke / probes)
}


class LeakError(RuntimeError):
    """A training id is also in a held-out role."""


# ═════════════════════════════════════════════════════════════════════════════════════════
# roles
# ═════════════════════════════════════════════════════════════════════════════════════════

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_roles(roles) -> Tuple[dict, Optional[str], Optional[str]]:
    """(roles dict, path or None, sha256 or None)."""
    if isinstance(roles, dict):
        return roles, None, None
    path = str(roles)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"roles file not found: {path} (run python -m scripts.v2.build_roles)")
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as f:
        r = yaml.load(f, Loader=loader) or {}
    if not isinstance(r, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return r, path, _sha256_file(path)


def _get(d: dict, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ids(d: dict, dotted: str) -> List[str]:
    v = _get(d, dotted, None)
    if v is None:
        return []
    if isinstance(v, dict):
        v = v.get("ids", [])
    return [str(x) for x in v]


def pool_ids(roles: dict) -> Dict[str, List[str]]:
    by_pool = _get(roles, "train.pool.by_pool", {}) or {}
    out = {}
    for name, v in by_pool.items():
        ids = v.get("ids", []) if isinstance(v, dict) else v
        out[str(name)] = [str(x) for x in ids]
    return out


def held_out_ids(roles: dict) -> Dict[str, set]:
    """Every list a training row must never appear in."""
    out = {
        "dev.d_sun": set(_ids(roles, "dev.d_sun")),
        "test.frozen_own": set(_ids(roles, "test.frozen_own")),
        "test.video_own": set(_ids(roles, "test.video_own")),
        "sf6_temporal.own_ids": set(_ids(roles, "sf6_temporal.own_ids")),
        "nyu_family_own": set(_ids(roles, "nyu_family_own")),
        "blacklist.sun_8x": set(_ids(roles, "blacklist.sun_8x")),
        "val.own": set(_ids(roles, "val.own")),
        "val.nyu": set(_ids(roles, "val.nyu")),
    }
    exc = _get(roles, "excluded", {}) or {}
    for src in ("own", "nyu"):
        v = exc.get(src) or {}
        out[f"excluded.{src}"] = set(map(str, v.keys() if isinstance(v, dict) else v))
    return out


def check_train_disjoint(roles: dict) -> None:
    train = set(_ids(roles, "train.own")) | set(_ids(roles, "train.nyu"))
    for ids in pool_ids(roles).values():
        train |= set(ids)
    bad = {k: sorted(train & v)[:5] for k, v in held_out_ids(roles).items() if train & v}
    if bad:
        raise LeakError(f"training ids overlap held-out roles (first ids shown): {bad}")


def split_ids(roles: dict, split: str, with_pools: bool) -> List[Tuple[str, str]]:
    """[(id, source)] of a split, in roles.yaml order."""
    if split == "train":
        rows = [(i, "own") for i in _ids(roles, "train.own")] + [(i, "nyu") for i in _ids(roles, "train.nyu")]
        if with_pools:
            for name, ids in sorted(pool_ids(roles).items()):
                rows += [(i, name) for i in ids]
        return rows
    if split == "val":
        return [(i, "own") for i in _ids(roles, "val.own")] + [(i, "nyu") for i in _ids(roles, "val.nyu")]
    key = {"dev": ("dev.d_sun", "sun"), "frozen_own": ("test.frozen_own", "own"),
           "video_own": ("test.video_own", "own"), "calibration": ("calibration.own", "own"),
           "sf6_own": ("sf6_temporal.own_ids", "own")}[split]
    return [(i, key[1]) for i in _ids(roles, key[0])]


# ═════════════════════════════════════════════════════════════════════════════════════════
# config
# ═════════════════════════════════════════════════════════════════════════════════════════

def resolve_config(cfg: Optional[dict], split: str) -> dict:
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown data.dataset keys: {sorted(unknown)} (known: {sorted(DEFAULTS)})")
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in cfg.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            extra = set(v) - set(out[k])
            if extra:
                raise ValueError(f"unknown data.dataset.{k} keys: {sorted(extra)}")
            out[k].update(v)
        else:
            out[k] = v
    if out["pipeline"] not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}, got {out['pipeline']!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if out["augment"] is None:
        out["augment"] = split == "train"
    if out["augment"] and split in EVAL_ONLY_SPLITS:
        raise ValueError(f"augment is not allowed on the evaluation split {split!r}")
    if out["photometric"] not in (None, "strong"):
        raise ValueError(f"photometric must be null or 'strong', got {out['photometric']!r}")
    if out["cutmix"]:
        raise NotImplementedError("T2-06 CutMix is not implemented in distill/v2/data.py yet")
    zc = out["zoom_crop"]
    if not (0.0 <= float(zc["p"]) <= 1.0) or not (0.0 < float(zc["min_scale"]) <= 0.9):
        raise ValueError(f"zoom_crop needs 0 <= p <= 1 and 0 < min_scale <= 0.9, got {zc}")
    hw = out["hw"]
    if len(hw) != 2 or int(hw[0]) < 8 or int(hw[1]) < 8:
        raise ValueError(f"hw must be [H, W], got {hw}")
    out["hw"] = [int(hw[0]), int(hw[1])]
    return out


# ═════════════════════════════════════════════════════════════════════════════════════════
# manifests
# ═════════════════════════════════════════════════════════════════════════════════════════

def _norm(p: str) -> str:
    return p.replace("\\", "/") if p else p


def read_manifest(path: str) -> Dict[str, dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _float(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ═════════════════════════════════════════════════════════════════════════════════════════
# image ops (all numpy, HxW[xC])
# ═════════════════════════════════════════════════════════════════════════════════════════

def crop_intrinsics(k: Sequence[float], x0: int, y0: int, cw: int, ch: int, w: int, h: int) -> List[float]:
    """Normalised K of the window (x0, y0, cw, ch) of a (w, h) image."""
    fx, fy, cx, cy = k
    nx0, ny0, nw, nh = x0 / w, y0 / h, cw / w, ch / h
    return [fx / nw, fy / nh, (cx - nx0) / nw, (cy - ny0) / nh]


def window(h: int, w: int, th: int, tw: int, random: bool) -> Tuple[int, int]:
    """Top-left (y0, x0) of a th x tw window: uniform when random, centred otherwise."""
    if th > h or tw > w:
        raise ValueError(f"window {th}x{tw} larger than image {h}x{w}")
    if random:
        return int(np.random.randint(0, h - th + 1)), int(np.random.randint(0, w - tw + 1))
    return (h - th) // 2, (w - tw) // 2


def _gauss_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101)


def strong_photometric(rgb: np.ndarray) -> np.ndarray:
    """T2-05 on a uint8 RGB image: colour jitter (brightness .4, contrast .4, saturation .4,
    hue .1) at p .8 in random order, Gaussian blur sigma U(.1, 2) at p .5, grayscale at p .1."""
    x = rgb.astype(np.float32) / 255.0
    if np.random.rand() < 0.8:
        for op in np.random.permutation(4):
            if op == 0:
                x = x * np.random.uniform(0.6, 1.4)
            elif op == 1:
                m = float((x @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
                x = (x - m) * np.random.uniform(0.6, 1.4) + m
            elif op == 2:
                g = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
                x = (x - g) * np.random.uniform(0.6, 1.4) + g
            else:
                hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)      # H in [0, 360)
                hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-0.1, 0.1) * 360.0) % 360.0
                x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            x = np.clip(x, 0.0, 1.0)
    if np.random.rand() < 0.5:
        x = _gauss_blur(x, np.random.uniform(0.1, 2.0))
    if np.random.rand() < 0.1:
        g = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = np.repeat(g[..., None], 3, axis=2)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════════════════
# dataset
# ═════════════════════════════════════════════════════════════════════════════════════════

class V2Dataset(torch.utils.data.Dataset):
    """See the module docstring."""

    def __init__(self, roles, split: str, cfg: Optional[dict] = None):
        self.cfg = resolve_config(cfg, split)
        self.split = split
        self.roles, self.roles_path, self.roles_sha256 = load_roles(roles)
        if self.roles.get("frozen") is not True:
            raise ValueError("roles.yaml is not frozen (frozen: true); rebuild with scripts/v2/build_roles.py")
        check_train_disjoint(self.roles)
        cover = self.cfg["pipeline"] == "cover"
        paths = self.roles.get("paths") or {}
        manifest_path = paths.get("manifest")
        if not manifest_path:
            raise ValueError("roles.yaml paths.manifest is missing")
        self.data_root = os.path.dirname(manifest_path)
        base = read_manifest(manifest_path)
        cover_root = paths.get("cover_cache")
        cover_rows: Dict[str, Dict[str, dict]] = {}

        def cover_manifest(src: str) -> Dict[str, dict]:
            if src not in cover_rows:
                if not cover_root:
                    raise ValueError("roles.yaml paths.cover_cache is missing")
                p = os.path.join(cover_root, src, "manifest.csv")
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"cover cache for {src!r} not built: {p} "
                                            f"(python -m scripts.v2.build_cover_cache --source {src})")
                cover_rows[src] = read_manifest(p)
            return cover_rows[src]

        wanted = split_ids(self.roles, split, with_pools=cover)
        keep_src = set(self.cfg["sources"]) if self.cfg["sources"] else None
        rows, missing = [], []
        for rid, src in wanted:
            if keep_src is not None and src not in keep_src:
                continue
            is_pool = src not in ("own", "nyu", "sun")
            if cover and (src == "own" or is_pool):
                r = cover_manifest(src).get(rid)
                b = base.get(rid)
                if r is not None and b is not None:
                    # the cover manifest carries no K; normalised K is invariant to the squash
                    # resize, so the base row's K is the original image's normalised K
                    r = dict(r, **{c: b.get(c, "") for c in ("fx", "fy", "cx", "cy")})
            else:
                r = base.get(rid)
            if r is None:
                missing.append((rid, src))
                continue
            rows.append(dict(r, id=rid, source=src))
        if missing:
            raise FileNotFoundError(f"{len(missing)} {split} ids have no manifest row "
                                    f"(first: {missing[:3]}); rebuild the cache / manifest")
        if self.cfg["limit"]:
            n = int(self.cfg["limit"])
            by: Dict[str, List[dict]] = {}
            for r in rows:
                by.setdefault(r["source"], []).append(r)
            rows = [r for src in sorted(by) for r in sorted(by[src], key=lambda r: r["id"])[:n]]
        if not rows:
            raise ValueError(f"split {split!r} has no rows (sources={keep_src})")
        self.rows = rows
        self.ids = [r["id"] for r in rows]
        self.sources = [r["source"] for r in rows]
        self.fallback_intr: Dict[str, Dict[str, list]] = {}
        if self.cfg["intrinsics"]:
            for src in sorted(set(self.sources)):
                p = os.path.join(self.data_root, f"{src}_intrinsics.json")
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        self.fallback_intr[src] = json.load(f)

    def __len__(self) -> int:
        return len(self.rows)

    def describe(self) -> dict:
        from collections import Counter
        return {"split": self.split, "n": len(self), "by_source": dict(Counter(self.sources)),
                "cfg": self.cfg, "roles_path": self.roles_path, "roles_sha256": self.roles_sha256}

    def _intrinsics(self, r: dict) -> List[float]:
        k = [_float(r.get(c)) for c in ("fx", "fy", "cx", "cy")]
        if all(v is not None for v in k):
            return k
        v = self.fallback_intr.get(r["source"], {}).get(r["id"])
        return [float(x) for x in v] if v else list(DEFAULT_INTR)

    def load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """(uint8 RGB HxWx3, float32 depth metres HxW, row) before any transform."""
        r = self.rows[idx]
        bgr = cv2.imread(_norm(r["img"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read image {r['img']} (id {r['id']})")
        d16 = cv2.imread(_norm(r["depth"]), cv2.IMREAD_UNCHANGED)
        if d16 is None:
            raise IOError(f"cannot read depth {r['depth']} (id {r['id']})")
        if d16.ndim != 2:
            raise ValueError(f"depth {r['depth']} must be single-channel 16-bit, got shape {d16.shape}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = d16.astype(np.float32) / 1000.0
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"id {r['id']}: image {rgb.shape[:2]} and depth {depth.shape} differ")
        return rgb, depth, r

    def __getitem__(self, idx: int) -> dict:
        c = self.cfg
        rgb, depth, r = self.load_raw(idx)
        th, tw = c["hw"]
        aug = bool(c["augment"])
        k = self._intrinsics(r) if c["intrinsics"] else None
        h, w = depth.shape
        is_gt = str(r.get("is_gt", "0")).strip() in ("1", "True", "true")

        # 1. cover window (cover pipeline: images are >= 480x640 with the original aspect)
        if c["pipeline"] == "cover" and (h, w) != (480, 640):
            if h < 480 or w < 640:
                raise ValueError(f"id {r['id']}: cover image {h}x{w} smaller than 480x640")
            y0, x0 = window(h, w, 480, 640, random=aug)
            if k is not None:
                k = crop_intrinsics(k, x0, y0, 640, 480, w, h)
            rgb, depth = rgb[y0:y0 + 480, x0:x0 + 640], depth[y0:y0 + 480, x0:x0 + 640]
            h, w = 480, 640

        # 2. S5 zoom crop (training only)
        zc = c["zoom_crop"]
        if aug and float(zc["p"]) > 0 and np.random.rand() < float(zc["p"]):
            s = np.random.uniform(float(zc["min_scale"]), 0.9)
            ch, cw = int(round(h * s)), int(round(w * s))
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            if k is not None:
                k = crop_intrinsics(k, x0, y0, cw, ch, w, h)
            rgb, depth = rgb[y0:y0 + ch, x0:x0 + cw], depth[y0:y0 + ch, x0:x0 + cw]

        # 3. to the network size (normalised K is resize-invariant)
        if depth.shape != (th, tw):
            depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)
        if rgb.shape[:2] != (th, tw):
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

        # 4. flip and photometric (training only)
        if aug:
            if np.random.rand() < float(c["flip_p"]):
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])
                if k is not None:
                    k = [k[0], k[1], 1.0 - k[2], k[3]]
            j = c["jitter"]
            if float(j["p"]) > 0 and np.random.rand() < float(j["p"]):
                gain = 1.0 + np.random.uniform(-float(j["gain"]), float(j["gain"]))
                bias = np.random.uniform(-float(j["bias"]), float(j["bias"]))
                rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if c["photometric"] == "strong" and not is_gt:
                rgb = strong_photometric(rgb)

        dmin = _float(r.get("dmin")) or 0.0
        dmax = _float(r.get("dmax")) or 30.0
        valid = np.isfinite(depth) & (depth > 0) & (depth >= dmin) & (depth <= dmax)
        log_depth = np.zeros_like(depth, dtype=np.float32)
        log_depth[valid] = np.log(depth[valid])
        img = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            "target_log": torch.from_numpy(log_depth[None]),
            "valid": torch.from_numpy(valid[None]),
            "is_gt": bool(is_gt),
            "dmax": float(dmax),
            "source": r["source"],
            "id": r["id"],
        }
        if k is not None:
            item["intrinsics"] = torch.tensor(k, dtype=torch.float32)
        return item


# ═════════════════════════════════════════════════════════════════════════════════════════
# sampler
# ═════════════════════════════════════════════════════════════════════════════════════════

def group_members(sources: Sequence[str], key: str) -> List[int]:
    names = {s.strip() for s in str(key).split("+") if s.strip()}
    return [i for i, s in enumerate(sources) if s in names]


def make_sampler(ds, draw_weights: Dict[str, float], n_draws: int, seed: int) -> List[int]:
    """``n_draws`` row indices (see the module docstring). Raises on an empty source."""
    present = set(ds.sources)
    groups = []
    for key, wt in (draw_weights or {}).items():
        wt = float(wt)
        if wt < 0:
            raise ValueError(f"negative draw weight {key}: {wt}")
        if wt == 0:
            continue
        names = [s.strip() for s in str(key).split("+") if s.strip()]
        absent = [s for s in names if s not in present]
        if absent:
            raise ValueError(f"draw_weights group {key!r} names sources with no rows in split "
                             f"{getattr(ds, 'split', '?')!r}: {absent} (present: {sorted(present)})")
        groups.append((key, wt, group_members(ds.sources, key)))
    if not groups:
        raise ValueError(f"no positive draw weights in {draw_weights}")
    p = np.array([g[1] for g in groups], dtype=np.float64)
    p /= p.sum()
    rng = np.random.default_rng(int(seed))
    gi = rng.choice(len(groups), size=int(n_draws), p=p)
    out = np.empty(int(n_draws), dtype=np.int64)
    for g, (_, _, members) in enumerate(groups):
        sel = np.flatnonzero(gi == g)
        if sel.size:
            m = np.asarray(members, dtype=np.int64)
            out[sel] = m[rng.integers(0, m.size, size=sel.size)]
    return [int(i) for i in out]
