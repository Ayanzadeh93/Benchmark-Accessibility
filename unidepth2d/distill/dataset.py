"""Pairs each split image with its cached UniDepth v2 teacher depth (+ confidence) as the target.

Targets are METRIC metres. The student learns log-depth, so we hand it log-target + a validity mask
(depth within [depth_min_m, depth_max_m]) + normalized teacher confidence for optional weighting.

Two dataset classes:
  DistillDataset      — legacy single-source (own images + teacher depth via splits.json).
  MultiSourceDataset  — manifest-driven mix of real-GT sources (NYU, SUN RGB-D) + own
                        teacher-pseudo-labeled rows (scripts/build_gt_dataset.py).
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DistillDataset(Dataset):
    def __init__(self, cfg: dict, split: str, augment: bool = False):
        self.cfg = cfg
        self.split = split
        self.augment = augment
        with open(cfg["splits"]["file"], "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.stems: List[str] = splits[split]

        self.images_dir = cfg["data"]["images_dir"]
        self.depth_dir = cfg["output"]["depth_dir"]
        self.conf_dir = cfg["output"]["conf_dir"]
        self.exts = cfg["data"]["image_exts"]

        s = cfg["student"]
        self.th, self.tw = int(s["train_size"][0]), int(s["train_size"][1])
        self.depth_min = float(s["depth_min_m"])
        self.depth_max = float(s["depth_max_m"])

        self.use_intrinsics = bool(s.get("use_intrinsics", False))
        self.intrinsics = {}
        self.default_intr = [0.7, 0.9, 0.5, 0.5]  # rough dataset-typical fallback
        if self.use_intrinsics:
            ipath = os.path.join(cfg["output"]["root"], "intrinsics.json")
            if os.path.exists(ipath):
                with open(ipath, "r", encoding="utf-8") as f:
                    self.intrinsics = json.load(f)

    def __len__(self) -> int:
        return len(self.stems)

    def _find_image(self, stem: str) -> str:
        for e in self.exts:
            p = os.path.join(self.images_dir, stem + e)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"image for stem {stem} not found in {self.images_dir}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]
        img_path = self._find_image(stem)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_png = cv2.imread(os.path.join(self.depth_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
        depth_m = depth_png.astype(np.float32) / 1000.0

        conf_png = cv2.imread(os.path.join(self.conf_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
        conf = (conf_png.astype(np.float32) / 255.0) if conf_png is not None else np.ones_like(depth_m)

        # Resize to train size. Image: area; depth/conf: nearest (avoid mixing depths across edges).
        rgb = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA)
        depth_m = cv2.resize(depth_m, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)

        flipped = False
        if self.augment:
            if np.random.rand() < 0.5:  # horizontal flip (geometry-preserving for depth)
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                depth_m = np.ascontiguousarray(depth_m[:, ::-1])
                conf = np.ascontiguousarray(conf[:, ::-1])
                flipped = True
            if np.random.rand() < 0.5:  # mild photometric jitter (image only)
                gain = 1.0 + np.random.uniform(-0.15, 0.15)
                bias = np.random.uniform(-10, 10)
                rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        valid = np.isfinite(depth_m) & (depth_m >= self.depth_min) & (depth_m <= self.depth_max)
        log_depth = np.zeros_like(depth_m)
        log_depth[valid] = np.log(depth_m[valid])

        img = rgb.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "log_depth": torch.from_numpy(log_depth[None]).float(),
            "depth_m": torch.from_numpy(depth_m[None]).float(),
            "conf": torch.from_numpy(conf[None]).float(),
            "valid": torch.from_numpy(valid[None]).bool(),
            "stem": stem,
        }
        if self.use_intrinsics:
            fx, fy, cx, cy = self.intrinsics.get(stem, self.default_intr)
            if flipped:
                cx = 1.0 - cx      # horizontal flip mirrors the principal point
            out["intrinsics"] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        return out


DEFAULT_INTR = [0.7, 0.9, 0.5, 0.5]


class MultiSourceDataset(Dataset):
    """Manifest-driven corpus: real-GT rows (is_gt=1, conf=1.0) + teacher-pseudo-label rows.

    Rows come pre-resized to 640x480 by scripts/build_gt_dataset.py, so IO is cheap. A small
    per-source val slice is carved deterministically (cfg data_gt.val_frac / seed). Per-row
    valid-depth range (dmin/dmax) masks the loss. Missing per-row intrinsics fall back to
    <data_gt.root>/<source>_intrinsics.json (teacher-predicted), then to DEFAULT_INTR.
    """

    def __init__(self, cfg: dict, split: str, augment: bool = False):
        assert split in ("train", "val")
        self.cfg = cfg
        self.augment = augment
        dg = cfg["data_gt"]
        self.root = dg["root"]

        # frozen own-domain test stems are excluded from BOTH train and val so every model
        # trained from v3 on can be honestly evaluated on them (teacher-agreement protocol)
        frozen: set = set()
        fp = os.path.join(os.path.dirname(self.root), "frozen_test.json")
        if os.path.exists(fp):
            frozen = set(json.load(open(fp, encoding="utf-8")).get("frozen_test_own", []))

        rows: List[dict] = []
        with open(dg.get("manifest", os.path.join(self.root, "manifest.csv")),
                  newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source"] == "own" and r["id"] in frozen:
                    continue
                rows.append(r)

        # deterministic per-source val slice
        val_frac = float(dg.get("val_frac", 0.03))
        rng = random.Random(int(dg.get("seed", 1234)))
        by_src: Dict[str, List[dict]] = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        picked: List[dict] = []
        for src, rs in sorted(by_src.items()):
            rs = sorted(rs, key=lambda r: r["id"])
            rng.shuffle(rs)
            n_val = max(1, int(round(len(rs) * val_frac)))
            picked.extend(rs[:n_val] if split == "val" else rs[n_val:])
        self.rows = picked
        self.sources = [r["source"] for r in self.rows]

        # fallback intrinsics per source (e.g. sun_intrinsics.json)
        self.fallback_intr: Dict[str, Dict[str, list]] = {}
        for src in by_src:
            p = os.path.join(self.root, f"{src}_intrinsics.json")
            if os.path.exists(p):
                self.fallback_intr[src] = json.load(open(p, encoding="utf-8"))

        s = cfg["student"]
        self.th, self.tw = int(s["train_size"][0]), int(s["train_size"][1])
        self.use_intrinsics = bool(s.get("use_intrinsics", False))

        dc = cfg["distill"]
        cc = dc.get("crop_context") or {}
        self.crop_p = float(cc.get("p", 0.0)) if augment else 0.0        # S5-crop
        self.crop_min_scale = float(cc.get("min_scale", 0.5))
        self.conf_mode = str(dc.get("conf_mode", "none"))                 # S6: "rank"
        self.conf_floor = float(dc.get("conf_floor", 0.3))
        self.use_taskw = bool(dc.get("task_weight_maps", False))          # S7
        self.range_alpha = float(dc.get("range_weight_alpha", 0.0))       # range-first weighting
        self.use_assistant = bool(dc.get("assistant", False))             # S8
        self.assist_tau = float(dc.get("assistant_tau", 0.2))
        _lc = dc.get("loss", {})
        # emit per-object boxes for the object-distance term (S7) OR the anchor-consistency loss
        self.emit_boxes = (float(_lc.get("obj_dist_weight", 0.0)) > 0
                           or float(_lc.get("anchor_weight", 0.0)) > 0)
        self.max_boxes = 8

        # CSD (Consensus-Scale Distillation, P-series): per-image consensus correction of the
        # pseudo-label's global log-scale + the teacher's own N-view scale disagreement (sigma).
        # Built offline by scripts/build_consensus_scale.py; refinement applies to pseudo-label
        # rows only (is_gt rows are real sensor GT — nothing to refine).
        self.csd: Dict[str, dict] = {}
        if bool(dc.get("csd", False)):
            for src in by_src:
                p = os.path.join(self.root, f"{src}_consensus_scale.json")
                if os.path.exists(p):
                    self.csd.update(json.load(open(p, encoding="utf-8")))
            if not self.csd:
                raise SystemExit("recipe tag 'csd' set but no <src>_consensus_scale.json found "
                                 "— run scripts/build_consensus_scale.py first")

        # ── b20 C3b fov: zoom-out uncrop sharing ONE u draw with the s5 crop ──
        fv = dc.get("fov_uncrop") or {}
        self.fov_p = float(fv.get("p", 0.0)) if augment else 0.0
        self.fov_zmin = float(fv.get("zmin", 0.6))
        self.fov_zmax = float(fv.get("zmax", 0.95))

        # ── b20 C3a ownar: aspect-true own rows from the v2 cache (crop-or-pad, no stretch) ──
        self.own_source = str(dc.get("own_source", "data_gt"))
        self.v2cache: Dict[str, dict] = {}
        self.own_intr_json: Dict[str, list] = {}
        if self.own_source == "v2cache":
            if self.use_taskw or self.use_assistant:
                raise SystemExit("ownar is incompatible with s7/s8: their maps are stored on the "
                                 "stretched 640x480 grid")
            mpath = dc.get("v2cache_manifest") or os.path.join(
                os.path.dirname(self.root), "v2", "cache", "own", "manifest.csv")
            with open(mpath, newline="", encoding="utf-8") as f:
                for cr in csv.DictReader(f):
                    self.v2cache[cr["id"]] = cr
            # intrinsics.json: per-id (fx/W, fy/H, cx/W, cy/H) NORMALISED by the original image
            # size (verified: fx_px/fy_px median 0.99, cx median 0.503). Normalised K is invariant
            # to the aspect-preserving cover resize, so K_px(cache) = K_norm * (W_c, H_c).
            ipath = dc.get("v2cache_intrinsics") or os.path.join(cfg["output"]["root"], "intrinsics.json")
            if os.path.exists(ipath):
                with open(ipath, "r", encoding="utf-8") as f:
                    self.own_intr_json = json.load(f)
            own_ids = [r["id"] for r in self.rows if r["source"] == "own"]
            self.n_own_cache = sum(1 for i in own_ids if i in self.v2cache)
            self.n_own_fallback = len(own_ids) - self.n_own_cache
            # K sanity (spec C3a): cx_px / W_c in [0.35, 0.65] for >= 99% of cached own rows
            cxr = []
            for i in own_ids:
                cr = self.v2cache.get(i)
                if cr is None:
                    continue
                k = self._own_k_norm(i, None)
                if k is not None:
                    cxr.append(k[2])            # cx_px / W_c == normalised cx
            if cxr:
                frac_ok = float(np.mean([(0.35 <= c <= 0.65) for c in cxr]))
                assert frac_ok >= 0.99, f"ownar K check failed: only {frac_ok:.3f} of cx in [.35,.65]"
        self.emit_pad = self.own_source == "v2cache" or bool(dc.get("fov_uncrop"))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _rank_normalize(conf: np.ndarray, floor: float) -> np.ndarray:
        """Per-image rank transform -> [floor, 1]. Robust to UniDepth's skewed conf distribution
        (plain min-max normalization drove ~all pixels to ~0 and collapsed training)."""
        flat = conf.reshape(-1)
        order = np.argsort(flat, kind="stable")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(flat.size, dtype=np.float32)
        r = (ranks / max(1, flat.size - 1)).reshape(conf.shape)
        return floor + (1.0 - floor) * r

    def _row_intrinsics(self, r: dict) -> list:
        if r.get("fx") not in ("", None):
            try:
                return [float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"])]
            except (TypeError, ValueError):
                pass
        fb = self.fallback_intr.get(r["source"], {})
        v = fb.get(r["id"])
        return list(v) if v else list(DEFAULT_INTR)

    # ── b20 helpers ──────────────────────────────────────────────────────────────────────────
    def _own_k_norm(self, rid: str, r: dict = None):
        """Normalised (fx/W, fy/H, cx/W, cy/H) of an own image: intrinsics.json, else the row."""
        v = self.own_intr_json.get(rid)
        if v is not None:
            return [float(x) for x in v]
        return self._row_intrinsics(r) if r is not None else None

    def _draw_geom_mode(self):
        """Geometric augmentation choice. Without fov this is EXACTLY the legacy s5 draw
        (one np.random.rand() only when crop_p > 0). With fov, one shared u ~ U(0,1):
        u < crop_p -> 's5'; crop_p <= u < crop_p + fov_p -> 'fov'; else None. So the s5 rate
        stays crop_p (0.5) and the uncrop rate is fov_p (0.25)."""
        if self.fov_p > 0:
            u = np.random.rand()
            if u < self.crop_p:
                return "s5"
            if u < self.crop_p + self.fov_p:
                return "fov"
            return None
        if self.crop_p > 0 and np.random.rand() < self.crop_p:
            return "s5"
        return None

    @staticmethod
    def _noise_rgb(h: int, w: int) -> np.ndarray:
        return np.clip(np.random.normal(127.0, 40.0, (h, w, 3)), 0, 255).astype(np.uint8)

    @staticmethod
    def _affine_boxes(boxes, box_dists, box_sizes, n_boxes, sx, tx, sy, ty):
        """x' = x*sx + tx, y' = y*sy + ty on normalised xyxy; clip, drop degenerate. -> n_boxes."""
        if not n_boxes:
            return 0
        b = boxes[:n_boxes].copy()
        b[:, [0, 2]] = b[:, [0, 2]] * sx + tx
        b[:, [1, 3]] = b[:, [1, 3]] * sy + ty
        b = np.clip(b, 0.0, 1.0)
        keep = ((b[:, 2] - b[:, 0]) > 0.02) & ((b[:, 3] - b[:, 1]) > 0.02)
        kept, kd, ks = b[keep], box_dists[:n_boxes][keep], box_sizes[:n_boxes][keep]
        boxes[:] = 0.0
        box_dists[:] = 0.0
        box_sizes[:] = 0.0
        n = len(kept)
        boxes[:n], box_dists[:n], box_sizes[:n] = kept, kd, ks
        return n

    def _crop_or_pad_to(self, H, W, rgb, depth, weight, pad, K_px, train):
        """C3a: bring a cache image to (H, W) WITHOUT resizing. Each axis independently:
        larger -> crop (random offset when training, centred otherwise; c -= offset);
        smaller -> paste into a N(127,40^2) noise canvas (random/centred offset; c += offset),
        depth = weight = 0 and pad = 1 outside. Returns (rgb, depth, weight, pad, K_px, geo)
        with geo = (sx0, dx0, sy0, dy0) for box remapping."""
        h, w = depth.shape
        fx, fy, cx, cy = K_px

        def axis(n, N):
            if n >= N:
                o = np.random.randint(0, n - N + 1) if train else (n - N) // 2
                return o, o + N, 0          # src [o, o+N) -> dst [0, N)
            o = np.random.randint(0, N - n + 1) if train else (N - n) // 2
            return 0, n, o                  # src [0, n) -> dst [o, o+n)

        sx0, sx1, dx0 = axis(w, W)
        sy0, sy1, dy0 = axis(h, H)
        if w >= W and h >= H:
            rgb = rgb[sy0:sy1, sx0:sx1]
            depth = depth[sy0:sy1, sx0:sx1]
            weight = weight[sy0:sy1, sx0:sx1]
            pad = pad[sy0:sy1, sx0:sx1]
        else:
            ch, cw = sy1 - sy0, sx1 - sx0
            c_rgb = self._noise_rgb(H, W)
            c_dep = np.zeros((H, W), np.float32)
            c_w = np.zeros((H, W), np.float32)
            c_pad = np.ones((H, W), np.uint8)
            c_rgb[dy0:dy0 + ch, dx0:dx0 + cw] = rgb[sy0:sy1, sx0:sx1]
            c_dep[dy0:dy0 + ch, dx0:dx0 + cw] = depth[sy0:sy1, sx0:sx1]
            c_w[dy0:dy0 + ch, dx0:dx0 + cw] = weight[sy0:sy1, sx0:sx1]
            c_pad[dy0:dy0 + ch, dx0:dx0 + cw] = pad[sy0:sy1, sx0:sx1]
            rgb, depth, weight, pad = c_rgb, c_dep, c_w, c_pad
        K = (fx, fy, cx - sx0 + dx0, cy - sy0 + dy0)
        return (np.ascontiguousarray(rgb), np.ascontiguousarray(depth),
                np.ascontiguousarray(weight), np.ascontiguousarray(pad), K, (sx0, dx0, sy0, dy0))

    def _fov_uncrop(self, rgb, depth, weight, pad, K_norm):
        """C3b: zoom-out uncrop. z ~ U(zmin, zmax); shrink the view (image AREA, labels NEAREST)
        and paste it at a random offset into a noise canvas of the SAME size. Depth values are
        untouched (metric labels stay exact); K in pixels: f' = f z_axis, c' = c z_axis + o.
        Returns (rgb, depth, weight, pad, K_norm, (sx, tx, sy, ty)) with the normalised-coord map."""
        h, w = depth.shape
        z = np.random.uniform(self.fov_zmin, self.fov_zmax)
        nh, nw = int(round(z * h)), int(round(z * w))
        oy = np.random.randint(0, h - nh + 1)
        ox = np.random.randint(0, w - nw + 1)
        c_rgb = self._noise_rgb(h, w)
        c_dep = np.zeros((h, w), np.float32)
        c_w = np.zeros((h, w), np.float32)
        c_pad = np.ones((h, w), np.uint8)
        c_rgb[oy:oy + nh, ox:ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        c_dep[oy:oy + nh, ox:ox + nw] = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c_w[oy:oy + nh, ox:ox + nw] = cv2.resize(weight, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c_pad[oy:oy + nh, ox:ox + nw] = cv2.resize(pad, (nw, nh), interpolation=cv2.INTER_NEAREST)
        sx, sy = nw / w, nh / h
        tx, ty = ox / w, oy / h
        fx, fy, cx, cy = K_norm
        K = (fx * sx, fy * sy, cx * sx + tx, cy * sy + ty)
        return c_rgb, c_dep, c_w, c_pad, K, (sx, tx, sy, ty)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        # b20 ownar: own rows read the aspect-true v2 cache (fallback: the stretched copy)
        cache = self.v2cache.get(r["id"]) if (self.v2cache and r["source"] == "own") else None
        bgr = cv2.imread(cache["img"] if cache else r["img"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        d16 = cv2.imread(cache["depth"] if cache else r["depth"], cv2.IMREAD_UNCHANGED)
        depth_m = d16.astype(np.float32) / 1000.0
        pad = np.zeros(depth_m.shape, np.uint8) if self.emit_pad else None

        is_gt = str(r.get("is_gt", "0")) in ("1", "True", "true")
        is_own = r["source"] == "own"

        # CSD refinement: correct the cached single-pass pseudo-label's global scale by the
        # teacher's multi-view consensus (pure scalar in log space; structure untouched).
        scale_sigma = 0.0
        if self.csd and not is_gt:
            e = self.csd.get(r["id"])
            if e is not None:
                depth_m = depth_m * float(np.exp(e["delta"]))
                scale_sigma = float(e["sigma"])

        # per-pixel weight channel, composed from up to three factors
        weight = np.ones_like(depth_m)
        if is_own and self.conf_mode == "rank" and r.get("conf"):           # S6
            c8 = cv2.imread(cache["conf"] if cache else r["conf"], cv2.IMREAD_UNCHANGED)
            if c8 is not None:
                weight *= self._rank_normalize(c8.astype(np.float32), self.conf_floor)
        if is_own and self.use_taskw:                                        # S7 region weighting
            tw_p = os.path.join(self.root, "own", "taskw", r["id"] + ".png")
            t8 = cv2.imread(tw_p, cv2.IMREAD_UNCHANGED)
            if t8 is not None:
                weight *= t8.astype(np.float32) / 100.0                      # 100 = 1.0
        if self.range_alpha > 0:                                             # range-first weighting
            weight *= 1.0 + self.range_alpha * np.exp(-np.clip(depth_m, 0, 60) / 3.0)
        if self.use_assistant:                                               # S8 agreement
            a_p = os.path.join(self.root, r["source"], "assist_mm", r["id"] + ".png")
            a16 = cv2.imread(a_p, cv2.IMREAD_UNCHANGED)
            if a16 is not None and not is_gt:                                # GT rows unaffected
                assist_m = a16.astype(np.float32) / 1000.0
                both = (depth_m > 1e-3) & (assist_m > 1e-3)
                agree = np.ones_like(depth_m)
                agree[both] = np.exp(-np.abs(np.log(depth_m[both]) - np.log(assist_m[both]))
                                     / self.assist_tau)
                weight *= agree

        # boxes for the object-distance term (own rows only; normalized xyxy). box_sizes = per-object
        # known physical HEIGHT in metres (0 = unknown class) for the anchor-consistency loss; the
        # npz "sizes" key is optional, so older box caches stay valid (sizes default to 0 = self-gated).
        boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        box_dists = np.zeros((self.max_boxes,), dtype=np.float32)
        box_sizes = np.zeros((self.max_boxes,), dtype=np.float32)
        n_boxes = 0
        if self.emit_boxes and is_own:
            b_p = os.path.join(self.root, "own", "boxes", r["id"] + ".npz")
            if os.path.exists(b_p):
                z = np.load(b_p)
                bb, dd = z["boxes"].astype(np.float32), z["dists"].astype(np.float32)
                ss = z["sizes"].astype(np.float32) if "sizes" in z.files else np.zeros(len(bb), np.float32)
                n_boxes = min(len(bb), self.max_boxes)
                boxes[:n_boxes] = bb[:n_boxes]
                box_dists[:n_boxes] = dd[:n_boxes]
                box_sizes[:n_boxes] = ss[:n_boxes]

        # ── S5: cross-context crop (shared-context distillation, metric-preserving) ──
        fx, fy, cx, cy = self._row_intrinsics(r)
        if cache is not None:                                                # b20 C3a ownar
            hc, wc = depth_m.shape
            kn = self._own_k_norm(r["id"], r)
            K_px = (kn[0] * wc, kn[1] * hc, kn[2] * wc, kn[3] * hc)
            rgb, depth_m, weight, pad, K_px, (sx0, dx0, sy0, dy0) = self._crop_or_pad_to(
                self.th, self.tw, rgb, depth_m, weight, pad, K_px, train=self.augment)
            fx, fy = K_px[0] / self.tw, K_px[1] / self.th
            cx, cy = K_px[2] / self.tw, K_px[3] / self.th
            n_boxes = self._affine_boxes(boxes, box_dists, box_sizes, n_boxes,
                                         wc / self.tw, (dx0 - sx0) / self.tw,
                                         hc / self.th, (dy0 - sy0) / self.th)
        geom = self._draw_geom_mode()
        if geom == "fov":                                                    # b20 C3b uncrop
            rgb, depth_m, weight, pad, (fx, fy, cx, cy), (sx, tx, sy, ty) = self._fov_uncrop(
                rgb, depth_m, weight, pad, (fx, fy, cx, cy))
            n_boxes = self._affine_boxes(boxes, box_dists, box_sizes, n_boxes, sx, tx, sy, ty)
        if geom == "s5":
            s = np.random.uniform(self.crop_min_scale, 0.9)                  # linear scale
            h, w = depth_m.shape
            ch, cw = int(round(h * s)), int(round(w * s))
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            rgb = rgb[y0:y0 + ch, x0:x0 + cw]
            depth_m = depth_m[y0:y0 + ch, x0:x0 + cw]
            weight = weight[y0:y0 + ch, x0:x0 + cw]
            if pad is not None:
                pad = pad[y0:y0 + ch, x0:x0 + cw]
            # metric depth values are crop-invariant; intrinsics change with the FOV:
            nx0, ny0, nw, nh = x0 / w, y0 / h, cw / w, ch / h
            fx, fy = fx / nw, fy / nh
            cx, cy = (cx - nx0) / nw, (cy - ny0) / nh
            if n_boxes:
                b = boxes[:n_boxes]
                b[:, [0, 2]] = (b[:, [0, 2]] - nx0) / nw
                b[:, [1, 3]] = (b[:, [1, 3]] - ny0) / nh
                b = np.clip(b, 0.0, 1.0)
                keep = ((b[:, 2] - b[:, 0]) > 0.02) & ((b[:, 3] - b[:, 1]) > 0.02)
                kept = b[keep]
                kd = box_dists[:n_boxes][keep]
                ks = box_sizes[:n_boxes][keep]
                boxes[:] = 0.0
                box_dists[:] = 0.0
                box_sizes[:] = 0.0
                n_boxes = len(kept)
                boxes[:n_boxes] = kept
                box_dists[:n_boxes] = kd
                box_sizes[:n_boxes] = ks

        if depth_m.shape != (self.th, self.tw):
            rgb = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(depth_m, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)
            weight = cv2.resize(weight, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)
            if pad is not None:
                pad = cv2.resize(pad, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)
        elif rgb.shape[:2] != (self.th, self.tw):
            rgb = cv2.resize(rgb, (self.tw, self.th), interpolation=cv2.INTER_AREA)

        flipped = False
        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                depth_m = np.ascontiguousarray(depth_m[:, ::-1])
                weight = np.ascontiguousarray(weight[:, ::-1])
                if pad is not None:
                    pad = np.ascontiguousarray(pad[:, ::-1])
                flipped = True
                if n_boxes:
                    b = boxes[:n_boxes].copy()
                    boxes[:n_boxes, 0] = 1.0 - b[:, 2]
                    boxes[:n_boxes, 2] = 1.0 - b[:, 0]
            if np.random.rand() < 0.5:
                gain = 1.0 + np.random.uniform(-0.15, 0.15)
                bias = np.random.uniform(-10, 10)
                rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        dmin, dmax = float(r["dmin"]), float(r["dmax"])
        valid = np.isfinite(depth_m) & (depth_m >= dmin) & (depth_m <= dmax)
        if pad is not None:
            valid &= pad == 0                       # synthetic pad is never supervised
        log_depth = np.zeros_like(depth_m)
        log_depth[valid] = np.log(depth_m[valid])

        img = rgb.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "log_depth": torch.from_numpy(log_depth[None]).float(),
            "depth_m": torch.from_numpy(depth_m[None]).float(),
            "conf": torch.from_numpy(np.ones_like(depth_m)[None]).float(),   # legacy key
            "weight": torch.from_numpy(weight[None]).float(),
            "valid": torch.from_numpy(valid[None]).bool(),
            "stem": r["id"],
            "source": r["source"],
            "is_gt": int(is_gt),
            "scale_sigma": float(scale_sigma),
            "boxes": torch.from_numpy(boxes).float(),
            "box_dists": torch.from_numpy(box_dists).float(),
            "box_sizes": torch.from_numpy(box_sizes).float(),
            "box_count": int(n_boxes),
        }
        if pad is not None:                         # b20 only (absent otherwise: b10 parity)
            out["pad"] = torch.from_numpy(np.ascontiguousarray(pad)[None]).to(torch.uint8)
        if self.use_intrinsics:
            if flipped:
                cx = 1.0 - cx
            out["intrinsics"] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        return out

    def sampler_weights(self, mix: Dict[str, float]) -> List[float]:
        """Per-index weights so each epoch draws sources in the requested proportion."""
        from collections import Counter
        counts = Counter(self.sources)
        total_mix = sum(mix.get(s, 0.0) for s in counts) or 1.0
        return [mix.get(s, 0.0) / total_mix / counts[s] for s in self.sources]
