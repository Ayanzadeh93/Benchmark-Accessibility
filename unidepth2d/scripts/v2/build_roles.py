"""Freeze the V2 (CoDeQ) dataset roles and run the leak blocker -> configs/v2/roles.yaml.

Contract
--------
CLI (env ``depth``; CPU only; reads Z: only for benchmark hashing)::

    python -m scripts.v2.build_roles [--manifest outputs/data_gt/manifest.csv]
        [--out configs/v2/roles.yaml] [--evidence outputs/v2/roles] [--workers 16]
        [--benchmarks auto|require|skip] [--pool-manifest outputs/v2/pool/manifest.csv]
        [--strict-local-nyu] [--recompute]
    python -m scripts.v2.build_roles --check-benchmark NAME --bench-dir DIR [--roles configs/v2/roles.yaml]

The first form writes ``configs/v2/roles.yaml`` (the single frozen source of truth read by
distill/v2/data.py, distill/v2/train.py, benchmark/policy_check.py and the int8 calibration) and
evidence under ``outputs/v2/roles/`` (feature caches, match CSVs, contact sheets).  The second
form leak-checks a benchmark that became reachable AFTER the roles were frozen: it never edits
roles.yaml; it writes ``outputs/v2/roles/bench_<NAME>_drop.json`` listing the benchmark images
that match training rows, which the evaluator drops and discloses (amendment A3).

Roles (plan section 5.1 as overridden by amendments A1-A4, A21)
---------------------------------------------------------------
* ``dev.d_sun``: local SUN rows minus the rows whose stored depth is 8x too large
  (:func:`sun_8x_stats`), minus NYU-derived frames = A3 union of (a) crop-aligned 64-bit pHash
  Hamming <= 12 (SUN raw vs NYU cropped to rows 45-472, cols 40-601) and (b) 40x30 thumbnail
  NCC >= 0.85, against the local NYU training frames AND the 1,449 official NYU labelled frames.
* ``dev.d_sun_clusters``: near-duplicate clusters for the cluster bootstrap (only clusters of size
  >= 2 are listed; ids not listed are singletons).
* ``test.frozen_own``: legacy 397 frozen own ids minus every frame of a video group (inverted
  video rule, A2) minus NYU-derived rows (``test.frozen_own_removed`` records why).
* ``test.video_own``: whole held-out small video families (group-disjoint video test, A2).
* ``sf6_temporal``: Dollar_Tree and Airport_BWI videos (A21) and every own video group that is
  footage of them (found by group closure); never trained on.
* ``train.own`` / ``train.nyu``: everything else after the legacy 3 % validation carve, minus
  leak-blocked rows (reasons per id in ``excluded``).
* ``calibration.own``: 300 ``train.own`` ids, none of the legacy 397 frozen ids, >= 30 low-light.
* ``report_only``: zero-shot sets, on-disk status, leak-check status, the rule for their use.

Leak-blocker rule (measured; see ``params.leak_rule``)
------------------------------------------------------
``a3`` hit = aligned pHash Hamming <= 12 OR thumbnail NCC >= 0.85 (amendment A3 verbatim);
alignments: raw-raw, raw-vs-NYU-crop, NYU-crop-vs-raw (``MATCH_MODES``).
``confirmed`` hit = aligned pHash <= 8 OR NCC >= 0.90.
* Evaluation-side references (D_SUN, NYU labelled 1449 incl. the 654 Eigen test frames, IBims-1,
  own frozen and video tests, SF6 footage): every training row with an ``a3`` hit is removed.
* Training-side reference (the local NYU training frames; matters only for the NYU-free branch):
  ``confirmed`` hits are removed (``--strict-local-nyu`` applies ``a3`` instead).  Measured
  reason: against 15,862 references the Hamming-12 tail is almost all false positives (24/24
  random own hits at Hamming 12 were different scenes), while every NYU-derived own row is in the
  ``<n>_png`` Roboflow family that the NYU-labelled check removes at full ``a3`` strength.
* Group closure: a video group with >= 3 ``confirmed`` hits to a reference and an ``a3`` hit on
  >= 20 % of its rows is treated as derived from that reference as a whole.
pHash note: the 64 bits always hold 32 ones (DC + 31 AC above the AC median), so distances are even.

Nothing here trains, calibrates, selects or gates on benchmark-domain images: benchmark images
are only hashed so that training rows matching them can be removed before training.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS = os.path.join(REPO, "outputs")
DEFAULT_MANIFEST = os.path.join(OUTPUTS, "data_gt", "manifest.csv")
DEFAULT_FROZEN = os.path.join(OUTPUTS, "frozen_test.json")
DEFAULT_SPLITS = os.path.join(OUTPUTS, "splits.json")
DEFAULT_OUT = os.path.join(REPO, "configs", "v2", "roles.yaml")
DEFAULT_EVIDENCE = os.path.join(OUTPUTS, "v2", "roles")
OWN_ORIGINALS = "C:/Tim/nature/database/Fianl Version/images"
SF6_VIDEOS = {
    "Dollar_Tree": "C:/Tim/nature/output/Dollar_Tree/Dollar_Tree.MOV",
    "Airport_BWI": "C:/Tim/nature/output/Airport_BWI/Airport_BWI.MOV",
}
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "Z:/Datasets")

# ── frozen parameters (recorded in roles.yaml) ──────────────────────────────────────────────
NYU_CROP = (45, 472, 40, 601)      # rows r0:r1, cols c0:c1 of a 480x640 NYU frame (A3)
HAM_DUP = 4                        # near-duplicate (plan 5.3): raw pHash Hamming <= 4
HAM_CROSS = 12                     # A3: aligned pHash Hamming <= 12
NCC_CROSS = 0.85                   # A3: thumbnail NCC >= 0.85
HAM_CONFIRM = 8                    # confirmed: aligned pHash <= 8 (A3-measured 97.5 % recall)
NCC_CONFIRM = 0.90                 # confirmed: NCC >= 0.90 (the A3 'truth' threshold)
HAM_CLUSTER = 8                    # D_SUN bootstrap clusters (plan 5.1) ...
NCC_CLUSTER = 0.85                 # ... unioned with NCC >= 0.85 (superset partition)
GROUP_MIN_CONFIRMED = 3            # group closure
GROUP_MIN_FRAC = 0.20
SUN8X_FRAC = 0.90                  # SUN 8x rule: > 90 % of valid pixels on a multiple-of-8 grid
LOWLIGHT_GRAY = 60.0               # calibration low-light: mean gray < 60
CALIB_N = 300
CALIB_LOWLIGHT_MIN = 30
CALIB_MAX_PER_GROUP = 10
SEED = 2027
LEGACY_VAL_FRAC = 0.03             # distill/dataset.py MultiSourceDataset carve
LEGACY_VAL_SEED = 1234
# own rows that are Roboflow exports of NYU labelled frames ('<index>_png.rf.<hash>'): measured
# 123 rows, all within aligned pHash 12 of an NYU labelled frame (0 of the other 7,808 own rows
# within 8). Verified on every build when the NYU labelled features are available.
NYU_FAMILY_RE = re.compile(r"^\d+_png\.rf\.[0-9a-fA-F]+$")
# Held-out video families (A2 example groups, closed over location families so the video test
# stays group-disjoint: MP_f1_m3 joins MP_f1_1, lab_ITE_Building joins ITE_corridor).
VIDEO_TEST_FAMILIES = {
    "ITE": ["ITE_corridor", "lab_ITE_Building"],
    "MP_f1": ["MP_f1_1", "MP_f1_m3"],
    "vegas_Sphere": ["vegas_Sphere"],
    "Restroom_m2": ["Restroom_m2"],
}


# ════════════════════════════════════════════════════════════════════════════════════════════
# Leak-blocker primitives
# ════════════════════════════════════════════════════════════════════════════════════════════

def to_gray(img: np.ndarray) -> np.ndarray:
    """uint8 gray from a gray or 3/4-channel uint8 image (BGR weights; RGB input differs by
    < 1 gray level on natural images, which the thresholds tolerate)."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY)


def nyu_crop(img: np.ndarray) -> np.ndarray:
    """SUN RGB-D's border crop of an NYU frame: rows 45-472, cols 40-601 at 480x640, scaled
    proportionally for other sizes."""
    h, w = img.shape[:2]
    r0, r1, c0, c1 = NYU_CROP
    return img[int(round(r0 * h / 480.0)):int(round(r1 * h / 480.0)),
               int(round(c0 * w / 640.0)):int(round(c1 * w / 640.0))]


def phash64(gray: np.ndarray) -> np.ndarray:
    """64-bit DCT perceptual hash as 8 packed bytes (uint8).

    32x32 INTER_AREA resize -> cv2.dct -> top-left 8x8 block -> bit = coeff > median(AC coeffs).
    """
    a = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(a)[:8, :8].flatten()
    return np.packbits(d > np.median(d[1:]))


def thumb_vec(gray: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm 40x30 thumbnail (float32, 1200) so that dot product = NCC."""
    t = cv2.resize(gray, (40, 30), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def image_features(img: np.ndarray) -> Dict[str, np.ndarray]:
    """pHash and thumbnail of the raw frame and of its NYU-style border crop, plus mean gray."""
    g = to_gray(img)
    c = nyu_crop(g)
    return {"ph": phash64(g), "phc": phash64(c), "th": thumb_vec(g), "thc": thumb_vec(c),
            "luma": np.float32(g.mean())}


if hasattr(np, "bitwise_count"):
    def _popcount(x: np.ndarray) -> np.ndarray:
        return np.bitwise_count(x)
else:  # pragma: no cover - numpy < 2.0
    _POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(x: np.ndarray) -> np.ndarray:
        return _POP[x]


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance between packed hashes (broadcasting over leading dims)."""
    return _popcount(np.bitwise_xor(a, b)).sum(-1).astype(np.int32)


def min_hamming(q: np.ndarray, ref: np.ndarray, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """For each query hash (n,8) the minimum Hamming distance to ``ref`` (m,8) and its index
    (first index on ties). Empty inputs give distance 65 and index -1."""
    n = len(q)
    dist = np.full(n, 65, np.int32)
    idx = np.full(n, -1, np.int64)
    if n == 0 or len(ref) == 0:
        return dist, idx
    for i in range(0, n, chunk):
        d = hamming(q[i:i + chunk, None, :], ref[None, :, :])
        idx[i:i + chunk] = d.argmin(1)
        dist[i:i + chunk] = d[np.arange(len(d)), idx[i:i + chunk]]
    return dist, idx


def max_ncc(q: np.ndarray, ref: np.ndarray, chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """For each query thumbnail (n,1200) the max NCC against ``ref`` (m,1200) and its index."""
    n = len(q)
    best = np.full(n, -1.0, np.float32)
    idx = np.full(n, -1, np.int64)
    if n == 0 or len(ref) == 0:
        return best, idx
    rt = np.ascontiguousarray(ref.astype(np.float32).T)
    for i in range(0, n, chunk):
        s = q[i:i + chunk].astype(np.float32) @ rt
        idx[i:i + chunk] = s.argmax(1)
        best[i:i + chunk] = s[np.arange(len(s)), idx[i:i + chunk]]
    return best, idx


@dataclass
class FeatureSet:
    """Per-image leak features of one image collection, aligned with ``ids``."""
    ids: List[str]
    ph: np.ndarray                      # (n,8) uint8 raw pHash
    phc: np.ndarray                     # (n,8) uint8 pHash of the NYU-style crop
    th: np.ndarray                      # (n,1200) float32 raw thumbnail
    thc: np.ndarray                     # (n,1200) float32 crop thumbnail
    luma: np.ndarray                    # (n,) float32 mean gray
    sha1: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, keep: Iterable[str]) -> "FeatureSet":
        """Rows whose id is in ``keep``, in this set's order."""
        keep = set(keep)
        m = np.array([i in keep for i in self.ids], bool)
        if not len(m):
            m = np.zeros(0, bool)
        return FeatureSet([i for i, k in zip(self.ids, m) if k], self.ph[m], self.phc[m],
                          self.th[m], self.thc[m], self.luma[m],
                          [s for s, k in zip(self.sha1, m) if k] if self.sha1 else [])

    @staticmethod
    def concat(sets: Sequence["FeatureSet"]) -> "FeatureSet":
        sets = [s for s in sets if s is not None]
        if not sets:
            return FeatureSet.from_feature_dicts([], [])
        sha = sum((s.sha1 for s in sets), []) if all(s.sha1 for s in sets) else []
        return FeatureSet(sum((s.ids for s in sets), []), np.concatenate([s.ph for s in sets]),
                          np.concatenate([s.phc for s in sets]), np.concatenate([s.th for s in sets]),
                          np.concatenate([s.thc for s in sets]), np.concatenate([s.luma for s in sets]),
                          sha)

    @staticmethod
    def from_images(ids: Sequence[str], imgs: Iterable[np.ndarray]) -> "FeatureSet":
        feats = [image_features(im) for im in imgs]
        return FeatureSet.from_feature_dicts(ids, feats)

    @staticmethod
    def from_feature_dicts(ids: Sequence[str], feats: Sequence[dict],
                           sha1: Optional[Sequence[str]] = None) -> "FeatureSet":
        if not feats:
            return FeatureSet([], np.zeros((0, 8), np.uint8), np.zeros((0, 8), np.uint8),
                              np.zeros((0, 1200), np.float32), np.zeros((0, 1200), np.float32),
                              np.zeros(0, np.float32), list(sha1 or []))
        return FeatureSet(list(ids), np.stack([f["ph"] for f in feats]),
                          np.stack([f["phc"] for f in feats]),
                          np.stack([f["th"] for f in feats]).astype(np.float32),
                          np.stack([f["thc"] for f in feats]).astype(np.float32),
                          np.array([f["luma"] for f in feats], np.float32), list(sha1 or []))

    def save(self, path: str, meta: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, ids=np.array(self.ids), ph=self.ph, phc=self.phc,
                            th=self.th.astype(np.float16), thc=self.thc.astype(np.float16),
                            luma=self.luma, sha1=np.array(self.sha1),
                            meta=np.array(json.dumps(meta, sort_keys=True)))
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> Tuple["FeatureSet", dict]:
        z = np.load(path, allow_pickle=False)
        fs = FeatureSet([str(x) for x in z["ids"]], z["ph"], z["phc"], z["th"].astype(np.float32),
                        z["thc"].astype(np.float32), z["luma"], [str(x) for x in z["sha1"]])
        return fs, json.loads(str(z["meta"]))


# Which (query variant, reference variant) pairs a cross-collection match compares.
#  "sun_vs_nyu": query is a SUN frame (already SUN-cropped), reference a full NYU frame (A3).
#  "symmetric":  raw-raw, or either side may be the NYU-crop of the other.
#  "raw":        raw-raw only (near-duplicate check between same-geometry copies).
MATCH_MODES = {
    "sun_vs_nyu": {"ham": [("ph", "phc")], "ncc": [("th", "thc"), ("th", "th")]},
    "symmetric": {"ham": [("ph", "ph"), ("ph", "phc"), ("phc", "ph")],
                  "ncc": [("th", "th"), ("th", "thc"), ("thc", "th")]},
    "raw": {"ham": [("ph", "ph")], "ncc": [("th", "th")]},
}


def cross_match(q: FeatureSet, ref: FeatureSet, mode: str = "symmetric",
                ham_th: int = HAM_CROSS, ncc_th: float = NCC_CROSS) -> Dict[str, np.ndarray]:
    """Leak-blocker match of every query image against a reference collection.

    Returns arrays aligned with ``q.ids``: ``ham`` (min aligned Hamming), ``ham_ref`` (its ref
    index), ``ncc`` (max NCC), ``ncc_ref``, ``by_ham`` = ham <= ham_th, ``by_ncc`` = ncc >= ncc_th,
    ``hit`` = by_ham | by_ncc (the A3 union rule with the default thresholds), ``confirmed`` =
    ham <= HAM_CONFIRM | ncc >= NCC_CONFIRM, and ``ref`` = the index of the reference that made
    the hit (Hamming match first). ``ncc_th`` <= -1 disables the NCC criterion.
    """
    spec = MATCH_MODES[mode]
    n = len(q)
    ham = np.full(n, 65, np.int32)
    ham_ref = np.full(n, -1, np.int64)
    for qa, ra in spec["ham"]:
        d, i = min_hamming(getattr(q, qa), getattr(ref, ra))
        better = d < ham
        ham[better], ham_ref[better] = d[better], i[better]
    ncc = np.full(n, -1.0, np.float32)
    ncc_ref = np.full(n, -1, np.int64)
    for qa, ra in spec["ncc"]:
        s, i = max_ncc(getattr(q, qa), getattr(ref, ra))
        better = s > ncc
        ncc[better], ncc_ref[better] = s[better], i[better]
    by_ham = ham <= ham_th
    by_ncc = ncc >= ncc_th if ncc_th > -1 else np.zeros(n, bool)
    confirmed = (ham <= HAM_CONFIRM) | (ncc >= NCC_CONFIRM if ncc_th > -1 else np.zeros(n, bool))
    return {"ham": ham, "ham_ref": ham_ref, "ncc": ncc, "ncc_ref": ncc_ref,
            "by_ham": by_ham, "by_ncc": by_ncc, "hit": by_ham | by_ncc, "confirmed": confirmed,
            "ref": np.where(by_ham | (ham <= HAM_CONFIRM), ham_ref, ncc_ref)}


def sha1_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ids_hash(ids: Iterable[str]) -> str:
    """sha256 over the sorted ids joined by newlines (order-independent role fingerprint)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def connected_components(ph: np.ndarray, th: int = HAM_CLUSTER, chunk: int = 512,
                         thumbs: Optional[np.ndarray] = None,
                         ncc_th: Optional[float] = None) -> np.ndarray:
    """Union-find connected components of the graph 'pHash Hamming <= th' (OR thumbnail NCC >=
    ncc_th when ``thumbs`` is given). Returns one label (the smallest member index) per row."""
    n = len(ph)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(0, n, chunk):
        adj = hamming(ph[i:i + chunk, None, :], ph[None, :, :]) <= th
        if thumbs is not None and ncc_th is not None:
            adj |= (thumbs[i:i + chunk] @ thumbs.T) >= ncc_th
        rr, cc = np.nonzero(adj)
        for r, c in zip(rr + i, cc):
            if c > r:
                a, b = find(int(r)), find(int(c))
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def group_closure(ids: Sequence[str], groups: Dict[str, Optional[str]], m: Dict[str, np.ndarray],
                  min_confirmed: int = GROUP_MIN_CONFIRMED,
                  min_frac: float = GROUP_MIN_FRAC) -> Dict[str, dict]:
    """Video groups derived from a reference as a whole.

    ``ids`` align with the match arrays ``m`` (from :func:`cross_match`). A group qualifies when
    >= ``min_confirmed`` of its rows are ``confirmed`` hits and >= ``min_frac`` of its rows (among
    ``ids``) are ``a3`` hits. Rows with group None (stills) never form a group.
    """
    tot: Dict[str, int] = {}
    conf: Dict[str, int] = {}
    hit: Dict[str, int] = {}
    for k, i in enumerate(ids):
        g = groups.get(i)
        if not g:
            continue
        tot[g] = tot.get(g, 0) + 1
        conf[g] = conf.get(g, 0) + int(m["confirmed"][k])
        hit[g] = hit.get(g, 0) + int(m["hit"][k])
    return {g: {"rows": tot[g], "confirmed": conf[g], "a3": hit[g]} for g in sorted(tot)
            if conf[g] >= min_confirmed and hit[g] >= min_frac * tot[g]}


# ════════════════════════════════════════════════════════════════════════════════════════════
# SUN 8x-depth detection, id grouping, calibration selection, legacy val carve
# ════════════════════════════════════════════════════════════════════════════════════════════

def sun_8x_stats(depth_u16: np.ndarray) -> Tuple[float, float]:
    """(fraction of valid pixels whose stored mm value is a multiple of 8, median stored metres).

    Rows written without SUN's bit-shift decode store raw = 8 * mm (the 3 low bits hold mm >> 13,
    which is 0 below 8.19 m), so nearly all their pixels sit on the multiple-of-8 grid, while
    correctly decoded depth is 8-aligned only at chance level (measured median 0.125).
    """
    v = depth_u16[depth_u16 > 0]
    if v.size == 0:
        return 0.0, 0.0
    return float(np.mean((v % 8) == 0)), float(np.median(v)) / 1000.0


VIDEO_PATTERNS = [
    re.compile(r"^(?P<g>.+?)[_-]frame[_-]?\d+"),                                    # X_frame_000123
    re.compile(r"^(?P<g>.+?)[_-](?:MOV|mov|mp4|MP4|avi|AVI|mkv|MKV|m4v)[-_]\d+$"),  # Roboflow video export
    re.compile(r"^(?P<g>frame)_\d{6,}$"),                                           # frame_0000123 (unnamed)
]
_PREHASH = re.compile(r"^(?P<s>.+?)_(?:jpg|jpeg|png|webp|bmp|JPG|JPEG|PNG)\.rf\.[0-9a-fA-F]+$")


def prehash_stem(own_id: str) -> str:
    """Roboflow export id 'X_jpg.rf.<hash>' -> 'X' (the original file stem); otherwise the id."""
    m = _PREHASH.match(own_id)
    return m.group("s") if m else own_id


def video_group(own_id: str) -> Optional[str]:
    """Video name for frames exported from a video, else None (stills)."""
    s = prehash_stem(own_id)
    for p in VIDEO_PATTERNS:
        m = p.match(s)
        if m:
            g = m.group("g")
            return "frame_unnamed" if g == "frame" else g
    return None


def legacy_val_carve(rows: List[dict], frozen: set, val_frac: float = LEGACY_VAL_FRAC,
                     seed: int = LEGACY_VAL_SEED) -> Dict[str, List[str]]:
    """Exact replica of distill/dataset.py MultiSourceDataset's per-source val slice (one RNG
    shared across sources in sorted source order; frozen own ids removed first)."""
    rng = random.Random(seed)
    by_src: Dict[str, List[dict]] = {}
    for r in rows:
        if r["source"] == "own" and r["id"] in frozen:
            continue
        by_src.setdefault(r["source"], []).append(r)
    val: Dict[str, List[str]] = {}
    for src, rs in sorted(by_src.items()):
        rs = sorted(rs, key=lambda r: r["id"])
        rng.shuffle(rs)
        n_val = max(1, int(round(len(rs) * val_frac)))
        val[src] = sorted(r["id"] for r in rs[:n_val])
    return val


def select_calibration(ids: Sequence[str], luma: Dict[str, float], groups: Dict[str, str],
                       n: int = CALIB_N, n_low: int = CALIB_LOWLIGHT_MIN,
                       luma_th: float = LOWLIGHT_GRAY, max_per_group: int = CALIB_MAX_PER_GROUP,
                       seed: int = SEED) -> List[str]:
    """Seeded calibration pick: ``n_low`` low-light ids first, then the rest, at most
    ``max_per_group`` per group (video name or pre-hash stem). Raises if quotas are infeasible."""
    rng = random.Random(seed)
    pool = sorted(ids)
    rng.shuffle(pool)
    per_group: Dict[str, int] = {}
    chosen: List[str] = []
    chosen_set: set = set()

    def take(cands: List[str], k: int) -> None:
        for i in cands:
            if len(chosen) >= k:
                return
            g = groups.get(i, i)
            if per_group.get(g, 0) >= max_per_group or i in chosen_set:
                continue
            chosen.append(i)
            chosen_set.add(i)
            per_group[g] = per_group.get(g, 0) + 1

    low = [i for i in pool if luma[i] < luma_th]
    take(low, n_low)
    if len(chosen) < n_low:
        raise ValueError(f"only {len(chosen)} low-light calibration candidates (< {n_low})")
    take(pool, n)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} calibration candidates (< {n})")
    return sorted(chosen)


# ════════════════════════════════════════════════════════════════════════════════════════════
# Feature extraction for the local corpus, benchmarks and SF6 videos
# ════════════════════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[build_roles {_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _read_feat(path: str) -> Optional[dict]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else image_features(img)


def features_for_files(ids: Sequence[str], paths: Sequence[str], workers: int,
                       sha1_paths: Optional[Sequence[str]] = None) -> FeatureSet:
    """Features of image files (threaded). Fails loudly if any file is unreadable."""
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(workers) as ex:
        feats = list(ex.map(_read_feat, paths))
        sha = list(ex.map(sha1_file, sha1_paths)) if sha1_paths is not None else []
    bad = [p for p, f in zip(paths, feats) if f is None]
    if bad:
        raise SystemExit(f"[build_roles] {len(bad)} unreadable images, e.g. {bad[:3]}")
    return FeatureSet.from_feature_dicts(ids, feats, sha)


def cached_features(name: str, evidence: str, key: dict, build: Callable[[], FeatureSet],
                    recompute: bool) -> FeatureSet:
    path = os.path.join(evidence, "features", f"{name}.npz")
    if os.path.exists(path) and not recompute:
        fs, meta = FeatureSet.load(path)
        if meta.get("key") == key:
            _log(f"features[{name}] cached ({len(fs)})")
            return fs
    _log(f"features[{name}] computing ...")
    fs = build()
    fs.save(path, {"key": key})
    _log(f"features[{name}] done ({len(fs)})")
    return fs


def datasets_root_status() -> Tuple[bool, str]:
    ok = os.path.isdir(DATASETS_ROOT)
    return ok, ("reachable" if ok else f"UNREACHABLE: {DATASETS_ROOT} is not mounted "
                "(Z: is an unstable external drive; set DATASETS_ROOT or reconnect it)")


def nyu_labelled_features(workers: int) -> Tuple[FeatureSet, List[str], dict]:
    """All 1,449 official NYU labelled frames (ids nyu_<i:04d>, zero-based mat index, the ids of
    benchmark.datasets_tier1.iter_nyu), the 654 Eigen test ids and source-file metadata."""
    import h5py
    from scipy.io import loadmat
    root = os.path.join(DATASETS_ROOT, "NYUv2")
    mat = os.path.join(root, "nyu_depth_v2_labeled.mat")
    splits = os.path.join(root, "splits.mat")
    for p in (mat, splits):
        if not os.path.exists(p):
            raise FileNotFoundError(f"NYU benchmark file missing: {p}")
    test_idx = set(int(i) - 1 for i in loadmat(splits)["testNdxs"].ravel())
    ids, feats = [], []
    with h5py.File(mat, "r") as f:
        images = f["images"]
        for i in range(images.shape[0]):
            rgb = np.transpose(images[i], (2, 1, 0))
            ids.append(f"nyu_{i:04d}")
            feats.append(image_features(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)))
    test_ids = [f"nyu_{i:04d}" for i in sorted(test_idx)]
    meta = {"mat_bytes": os.path.getsize(mat), "splits_sha256": sha256_file(splits)}
    return FeatureSet.from_feature_dicts(ids, feats), test_ids, meta


def ibims_features() -> Tuple[FeatureSet, None, dict]:
    from scipy.io import loadmat
    root = os.path.join(DATASETS_ROOT, "IBims-1")
    base = None
    for cand in ("ibims1_core_mat", "ibims1-core-mat"):
        if os.path.isdir(os.path.join(root, cand)):
            base = os.path.join(root, cand)
    if base is None:
        raise FileNotFoundError(f"IBims-1 core mats not extracted under {root}")
    mats = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(base) for f in fs if f.endswith(".mat"))
    if not mats:
        raise FileNotFoundError(f"no IBims-1 .mat files under {base}")
    ids, feats = [], []
    for mp in mats:
        d = loadmat(mp, simplify_cells=True)
        d = d.get("data", d)
        rgb = np.asarray(d["rgb"], dtype=np.uint8)
        ids.append(os.path.splitext(os.path.basename(mp))[0])
        feats.append(image_features(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
    return FeatureSet.from_feature_dicts(ids, feats), None, {"n_mats": len(mats)}


def video_frames_1fps(path: str, fps_out: float = 1.0) -> Tuple[List[str], List[np.ndarray], dict]:
    """Decode a video and keep one frame per 1/fps_out seconds (downscaled to 640 px wide)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[build_roles] cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / fps_out)))
    ids, frames, k = [], [], 0
    name = os.path.splitext(os.path.basename(path))[0]
    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % step == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                fr = cv2.resize(fr, (640, int(round(h * 640 / w))), interpolation=cv2.INTER_AREA)
                ids.append(f"{name}_t{k / fps:07.2f}")
                frames.append(fr)
        k += 1
    cap.release()
    return ids, frames, {"fps": round(float(fps), 3), "frames": n, "duration_s": round(n / fps, 2),
                         "sampled": len(ids)}


def load_benchmarks(args, reach: bool, reach_msg: str) -> Tuple[Dict[str, FeatureSet], Dict[str, str],
                                                                  List[str], Dict[str, dict]]:
    """Benchmark leak features. Computed from DATASETS_ROOT when reachable; otherwise the feature
    cache written by an earlier build is used and disclosed (same images, hashed earlier), never
    another dataset. ``--benchmarks require`` fails when neither is available."""
    bench: Dict[str, FeatureSet] = {}
    status: Dict[str, str] = {}
    meta_out: Dict[str, dict] = {}
    test_ids: List[str] = []
    if args.benchmarks == "skip":
        return bench, {"nyu_labelled": "skipped", "ibims1": "skipped"}, test_ids, meta_out
    for name, fn in (("nyu_labelled", lambda: nyu_labelled_features(args.workers)),
                     ("ibims1", ibims_features)):
        path = os.path.join(args.evidence, "features", f"bench_{name}.npz")
        try:
            if os.path.exists(path) and not args.recompute:
                fs, meta = FeatureSet.load(path)
                src = "cache"
            elif reach:
                _log(f"benchmark features[{name}] computing from {DATASETS_ROOT} ...")
                fs, extra, info = fn()
                meta = {"test_ids": extra, "root": DATASETS_ROOT, "info": info,
                        "created": _dt.datetime.now().isoformat(timespec="seconds")}
                fs.save(path, meta)
                src = "computed"
            else:
                raise FileNotFoundError(f"{reach_msg}; no cached features at {path}")
        except (OSError, KeyError) as e:
            if args.benchmarks == "require":
                raise SystemExit(f"[build_roles] benchmark {name} failed: {e}")
            print(f"[build_roles] WARNING: benchmark {name} leak check NOT run: {e}", file=sys.stderr,
                  flush=True)
            status[name] = f"failed: {e}"
            continue
        bench[name] = fs
        if name == "nyu_labelled":
            test_ids = list(meta.get("test_ids") or [])
            if len(test_ids) != 654:
                raise SystemExit(f"[build_roles] NYU labelled cache has {len(test_ids)} test ids (!= 654)")
        cached_at = _dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
        status[name] = (f"done ({len(fs)} images; features {src}"
                        + (f", hashed {cached_at} from {meta.get('root', '?')}" if src == "cache" else "")
                        + ("" if reach else "; DATASETS_ROOT unreachable at build time") + ")")
        meta_out[name] = {"features_npz": path.replace("\\", "/"), "features_sha256": sha256_file(path),
                          "source": src, "n": len(fs)}
        _log(f"benchmark {name}: {status[name]}")
    return bench, status, test_ids, meta_out


# ════════════════════════════════════════════════════════════════════════════════════════════
# Evidence writers
# ════════════════════════════════════════════════════════════════════════════════════════════

def read_manifest(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"[build_roles] manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_match_csv(path: str, q: FeatureSet, ref: FeatureSet, m: Dict[str, np.ndarray],
                    only_hits: bool = True) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ham", "ham_ref", "ncc", "ncc_ref", "by_ham", "by_ncc", "confirmed"])
        for i, qid in enumerate(q.ids):
            if only_hits and not m["hit"][i]:
                continue
            w.writerow([qid, int(m["ham"][i]), ref.ids[m["ham_ref"][i]] if m["ham_ref"][i] >= 0 else "",
                        round(float(m["ncc"][i]), 4),
                        ref.ids[m["ncc_ref"][i]] if m["ncc_ref"][i] >= 0 else "",
                        int(m["by_ham"][i]), int(m["by_ncc"][i]), int(m["confirmed"][i])])
            n += 1
    return n


def contact_sheet(path: str, pairs: List[Tuple[str, str, str]], title: str, cols: int = 4) -> None:
    """Side-by-side thumbnails (query | reference) with a caption, for visual match inspection."""
    tiles = []
    for qp, rp, cap in pairs:
        a, b = cv2.imread(qp), cv2.imread(rp)
        if a is None or b is None:
            continue
        a = cv2.resize(a, (160, 120))
        b = cv2.resize(b, (160, 120))
        t = np.concatenate([a, b], 1)
        t = np.concatenate([np.full((16, 320, 3), 255, np.uint8), t], 0)
        cv2.putText(t, cap[:52], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    sheet = np.concatenate(rows, 0)
    head = np.full((22, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title[:120], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.concatenate([head, sheet], 0))


def match_sheet(path: str, title: str, q: FeatureSet, qpath: Dict[str, str], ref: FeatureSet,
                rpath: Dict[str, str], m: Dict[str, np.ndarray], mask: np.ndarray,
                rng: np.random.Generator, n: int = 24) -> int:
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return 0
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    pairs = []
    for i in pick:
        j = int(m["ref"][i])
        if q.ids[i] in qpath and ref.ids[j] in rpath:
            pairs.append((qpath[q.ids[i]], rpath[ref.ids[j]], f"h{m['ham'][i]} n{m['ncc'][i]:.2f} {q.ids[i][:30]}"))
    contact_sheet(path, pairs, f"{title} ({len(idx)} rows, random {len(pairs)})")
    return len(idx)


def bin_counts(m: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Hit counts by evidence bin: Hamming bins for pHash-only hits, NCC bins for NCC-only hits."""
    h, s, hit = m["ham"], m["ncc"], m["hit"]
    ncc_hit = s >= NCC_CROSS
    out = {"a3_hits": int(hit.sum()), "confirmed": int(m["confirmed"].sum()),
           "both_rules": int((m["by_ham"] & m["by_ncc"]).sum())}
    for lo, hi in ((0, 4), (6, 8), (10, 10), (12, 12)):
        out[f"phash_only_h{lo}-{hi}"] = int(((h >= lo) & (h <= hi) & ~ncc_hit).sum())
    out["ncc_only_085-090"] = int((ncc_hit & (s < NCC_CONFIRM) & (h > HAM_CROSS)).sum())
    out["ncc_only_ge090"] = int(((s >= NCC_CONFIRM) & (h > HAM_CROSS)).sum())
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# Role construction
# ════════════════════════════════════════════════════════════════════════════════════════════

def build(args) -> dict:
    os.makedirs(args.evidence, exist_ok=True)
    rows = read_manifest(args.manifest)
    by_src: Dict[str, List[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    for s in ("own", "nyu", "sun"):
        if not by_src.get(s):
            raise SystemExit(f"[build_roles] manifest has no '{s}' rows")
    frozen397 = set(json.load(open(args.frozen, encoding="utf-8"))["frozen_test_own"])
    man_sha = sha256_file(args.manifest)
    _log(f"manifest rows {len(rows)}: " + ", ".join(f"{s} {len(v)}" for s, v in sorted(by_src.items())))
    img_path = {(r["source"], r["id"]): r["img"] for r in rows}
    own_path = {r["id"]: r["img"] for r in by_src["own"]}
    nyu_path = {r["id"]: r["img"] for r in by_src["nyu"]}
    sun_path = {r["id"]: r["img"] for r in by_src["sun"]}
    rng = np.random.default_rng(SEED)

    # ── features (cached) ────────────────────────────────────────────────────────────────
    feats: Dict[str, FeatureSet] = {}
    for s in ("own", "nyu", "sun"):
        rs = by_src[s]
        ids = [r["id"] for r in rs]
        imgs = [r["img"] for r in rs]
        if s == "own":
            originals = [os.path.join(OWN_ORIGINALS, i + ".jpg") for i in ids]
            missing = [p for p in originals if not os.path.exists(p)]
            if missing:
                raise SystemExit(f"[build_roles] {len(missing)} own originals missing, e.g. {missing[:2]}")
            sha_paths = originals
        else:
            sha_paths = imgs
        key = {"source": s, "manifest_sha256": man_sha, "n": len(ids), "v": 2}
        feats[s] = cached_features(s, args.evidence, key,
                                   lambda ids=ids, imgs=imgs, sp=sha_paths:
                                   features_for_files(ids, imgs, args.workers, sp), args.recompute)
    F_own, F_nyu, F_sun = feats["own"], feats["nyu"], feats["sun"]

    # ── SUN 8x rows ──────────────────────────────────────────────────────────────────────
    sun_rows = by_src["sun"]

    def _s8(r):
        d = cv2.imread(r["depth"], cv2.IMREAD_UNCHANGED)
        if d is None:
            raise SystemExit(f"[build_roles] unreadable SUN depth {r['depth']}")
        return sun_8x_stats(d)
    with ThreadPoolExecutor(args.workers) as ex:
        s8 = list(ex.map(_s8, sun_rows))
    frac8 = np.array([a for a, _ in s8])
    med8 = np.array([b for _, b in s8])
    flag8 = frac8 > SUN8X_FRAC
    bad8 = sorted(r["id"] for r, f in zip(sun_rows, flag8) if f)
    good = frac8[~flag8]
    sun8_report = {
        "rule": f"fraction of valid stored-mm pixels with value % 8 == 0 > {SUN8X_FRAC}",
        "n_flagged": len(bad8),
        "flagged_frac8_min": round(float(frac8[flag8].min()), 4) if bad8 else None,
        "unflagged_frac8_max": round(float(good.max()), 4) if len(good) else None,
        "unflagged_frac8_median": round(float(np.median(good)), 4) if len(good) else None,
        "flagged_stored_median_m_p10_p90": ([round(float(np.percentile(med8[flag8], q)), 3)
                                             for q in (10, 90)] if bad8 else None),
        "rows_in_gap_0.5_to_0.9": int(((frac8 > 0.5) & (frac8 <= SUN8X_FRAC)).sum()),
        "action": "blacklisted from D_SUN (depth needs a rebuild from the raw SUN PNG, not a /8 rescale: "
                  "pixels above 8 m in the wrong scale were zeroed by the legacy build)",
    }
    _log(f"SUN 8x rows: {len(bad8)} {sun8_report}")

    # ── benchmarks (hash only) and SF6 videos ────────────────────────────────────────────
    reach, reach_msg = datasets_root_status()
    if not reach:
        print(f"[build_roles] WARNING: {reach_msg}", file=sys.stderr, flush=True)
    bench, bench_status, nyu_test_ids, bench_meta = load_benchmarks(args, reach, reach_msg)
    F_lab = bench.get("nyu_labelled")
    F_lab_test = F_lab.subset(nyu_test_ids) if F_lab is not None else None

    sf6_feats: Dict[str, FeatureSet] = {}
    sf6_meta: Dict[str, dict] = {}
    for name, vp in SF6_VIDEOS.items():
        if not os.path.exists(vp):
            raise SystemExit(f"[build_roles] SF6 video missing: {vp}")
        path = os.path.join(args.evidence, "features", f"sf6_{name}.npz")
        if os.path.exists(path) and not args.recompute:
            fs, meta = FeatureSet.load(path)
        else:
            _log(f"SF6 video {name}: decoding at 1 fps ...")
            vids, frames, meta = video_frames_1fps(vp)
            meta["sha256"] = sha256_file(vp)
            meta["bytes"] = os.path.getsize(vp)
            fs = FeatureSet.from_images(vids, frames)
            fs.save(path, meta)
        sf6_feats[name] = fs
        sf6_meta[name] = {"path": vp, **meta}
    F_sf6 = FeatureSet.concat(list(sf6_feats.values()))

    # ── D_SUN ────────────────────────────────────────────────────────────────────────────
    sun_ok = F_sun.subset(set(F_sun.ids) - set(bad8))
    m_sn = cross_match(sun_ok, F_nyu, "sun_vs_nyu")
    write_match_csv(os.path.join(args.evidence, "sun_vs_local_nyu.csv"), sun_ok, F_nyu, m_sn)
    hits_local = {sun_ok.ids[i] for i in np.nonzero(m_sn["hit"])[0]}
    dsun_report: Dict[str, object] = {
        "sun_rows": len(F_sun), "minus_8x": len(bad8), "after_8x": len(sun_ok),
        "vs_local_nyu": {"hit_union": len(hits_local),
                         "by_crop_phash_le12": int(m_sn["by_ham"].sum()),
                         "by_ncc_ge085": int(m_sn["by_ncc"].sum()),
                         "bins": bin_counts(m_sn)},
    }
    truth = m_sn["ncc"] >= 0.90
    raw = cross_match(sun_ok, F_nyu, "raw", ham_th=HAM_CROSS, ncc_th=-2)
    if truth.any():
        rec = {f"raw_phash_le{t}": round(float((raw["ham"][truth] <= t).mean()), 4) for t in (4, 8, 12)}
        rec.update({f"crop_phash_le{t}": round(float((m_sn["ham"][truth] <= t).mean()), 4) for t in (4, 8, 12)})
        dsun_report["recall_of_ncc090"] = rec
        dsun_report["n_ncc090"] = int(truth.sum())
    hits_bench: Set[str] = set()
    dsun_nyu_test_hits: List[str] = []
    if F_lab is not None:
        m_sl = cross_match(sun_ok, F_lab, "sun_vs_nyu")
        write_match_csv(os.path.join(args.evidence, "sun_vs_nyu_labelled.csv"), sun_ok, F_lab, m_sl)
        hits_bench = {sun_ok.ids[i] for i in np.nonzero(m_sl["hit"])[0]}
        m_st = cross_match(sun_ok, F_lab_test, "sun_vs_nyu")
        dsun_nyu_test_hits = sorted(sun_ok.ids[i] for i in np.nonzero(m_st["hit"])[0])
        dsun_report["vs_nyu_labelled_1449"] = {
            "hit_union": len(hits_bench), "match_an_eigen_test_frame_654": len(dsun_nyu_test_hits),
            "not_already_in_local_nyu_hits": len(hits_bench - hits_local), "bins": bin_counts(m_sl)}
    nyu_derived = sorted(hits_local | hits_bench)
    d_sun = sorted(set(sun_ok.ids) - set(nyu_derived))
    dsun_report["minus_nyu_derived"] = len(nyu_derived)
    dsun_report["d_sun"] = len(d_sun)
    _log(f"D_SUN: {len(d_sun)} (8x {len(bad8)}, NYU-derived {len(nyu_derived)})")
    for tag, mask in (("h0-8", m_sn["by_ham"] & (m_sn["ham"] <= HAM_CONFIRM) & ~m_sn["by_ncc"]),
                      ("h10-12_only", m_sn["by_ham"] & (m_sn["ham"] > HAM_CONFIRM) & ~m_sn["by_ncc"]),
                      ("ncc_only", ~m_sn["by_ham"] & m_sn["by_ncc"]),
                      ("both", m_sn["by_ham"] & m_sn["by_ncc"])):
        match_sheet(os.path.join(args.evidence, f"sheet_sun_nyu_{tag}.jpg"),
                    f"SUN (left) vs local NYU (right): {tag}", sun_ok, sun_path, F_nyu, nyu_path, m_sn, mask, rng)

    # D_SUN clusters: plan rule (pHash <= 8) unioned with NCC >= 0.85 (superset partition)
    F_d = F_sun.subset(d_sun)
    lab_plan = connected_components(F_d.ph, HAM_CLUSTER)
    lab_union = connected_components(F_d.ph, HAM_CLUSTER, thumbs=F_d.th, ncc_th=NCC_CLUSTER)

    def _clusters(labels: np.ndarray) -> List[List[str]]:
        cl: Dict[int, List[str]] = {}
        for i, lab in zip(F_d.ids, labels):
            cl.setdefault(int(lab), []).append(i)
        return sorted((sorted(v) for v in cl.values() if len(v) > 1), key=lambda v: (-len(v), v[0]))
    multi = _clusters(lab_union)
    cluster_report = {"rule": f"connected components of (raw pHash Hamming <= {HAM_CLUSTER}) OR "
                              f"(thumbnail NCC >= {NCC_CLUSTER}) within D_SUN",
                      "n_clusters": int(len(set(lab_union.tolist()))), "n_multi": len(multi),
                      "rows_in_multi": int(sum(len(v) for v in multi)),
                      "largest": sorted((len(v) for v in multi), reverse=True)[:5],
                      "plan_rule_pHash_le8_only": {"n_multi": len(_clusters(lab_plan)),
                                                   "rows_in_multi": int(sum(len(v) for v in _clusters(lab_plan)))},
                      "note": "the local SUN copy has no scene ids; D_SUN has almost no near-duplicate pairs, "
                              "so the cluster bootstrap is close to an image bootstrap"}
    _log(f"D_SUN clusters: {cluster_report}")

    # ── own: groups, held-out sets, SF6 footage, NYU family ──────────────────────────────
    own_ids = sorted(F_own.ids)
    vgroup = {i: video_group(i) for i in own_ids}
    group_key = {i: (vgroup[i] or "still:" + prehash_stem(i)) for i in own_ids}
    vcount: Dict[str, int] = {}
    vfrozen: Dict[str, int] = {}
    for i in own_ids:
        g = vgroup[i]
        if g:
            vcount[g] = vcount.get(g, 0) + 1
            vfrozen[g] = vfrozen.get(g, 0) + (i in frozen397)
    held_groups = {g for fam in VIDEO_TEST_FAMILIES.values() for g in fam}
    missing_fam = sorted(held_groups - set(vcount))
    if missing_fam:
        raise SystemExit(f"[build_roles] held-out video groups not found: {missing_fam}")
    video_test = sorted(i for i in own_ids if vgroup[i] in held_groups)

    # SF6 footage inside own (A21): group closure against the SF6 videos
    m_own_sf6 = cross_match(F_own, F_sf6, "symmetric")
    sf6_groups = group_closure(F_own.ids, vgroup, m_own_sf6)
    sf6_groups = {g: v for g, v in sf6_groups.items() if g not in held_groups}
    sf6_own = sorted(i for i in own_ids if vgroup[i] in sf6_groups)
    _log(f"SF6 footage groups in own: {sf6_groups} ({len(sf6_own)} rows)")

    # NYU family: Roboflow exports of NYU labelled frames
    nyu_family = sorted(i for i in own_ids if NYU_FAMILY_RE.match(i))
    F_fam = F_own.subset(nyu_family)
    fam_report: Dict[str, object] = {"rule": NYU_FAMILY_RE.pattern, "rows": len(nyu_family),
                                     "frozen397_rows": len(set(nyu_family) & frozen397)}
    for name, ref, mode in (("nyu_labelled_1449", F_lab, "symmetric"), ("local_nyu", F_nyu, "symmetric"),
                            ("sun_all", F_sun, "symmetric")):
        if ref is None or not len(F_fam):
            continue
        mf = cross_match(F_fam, ref, mode)
        fam_report[f"vs_{name}"] = {"a3": int(mf["hit"].sum()), "confirmed": int(mf["confirmed"].sum())}
        if name == "nyu_labelled_1449":
            frac = float(mf["hit"].mean())
            fam_report["a3_fraction_vs_nyu_labelled"] = round(frac, 4)
            if frac < 0.9:
                raise SystemExit(f"[build_roles] NYU family rule no longer verified: only {frac:.1%} of "
                                 f"'{NYU_FAMILY_RE.pattern}' rows match an NYU labelled frame")
            mt = cross_match(F_fam, F_lab_test, "symmetric")
            fam_report["confirmed_match_an_eigen_test_frame"] = int(mt["confirmed"].sum())
            fam_report["ids_confirmed_eigen_test"] = sorted(F_fam.ids[i] for i in np.nonzero(mt["confirmed"])[0])
    _log(f"NYU family: {fam_report}")

    # frozen test = legacy 397 stills (A2) minus NYU-derived rows
    frozen_still = sorted(i for i in frozen397 if not vgroup.get(i))
    frozen_in_video = sorted(i for i in frozen397 if vgroup.get(i))
    frozen_removed: Dict[str, List[str]] = {}

    def _rm(i: str, why: str) -> None:
        frozen_removed.setdefault(i, [])
        if why not in frozen_removed[i]:
            frozen_removed[i].append(why)
    for i in frozen_still:
        if i in set(nyu_family):
            _rm(i, "nyu_family")
    F_fs = F_own.subset(frozen_still)
    for name, ref in (("local_nyu", F_nyu), ("nyu_labelled", F_lab), ("sun_all", F_sun)):
        if ref is None:
            continue
        mfz = cross_match(F_fs, ref, "symmetric")
        for k in np.nonzero(mfz["confirmed"])[0]:
            _rm(F_fs.ids[k], f"confirmed_match_of_{name}")
    frozen_v2 = sorted(set(frozen_still) - set(frozen_removed))

    a2_named = {vgroup[i] for i in own_ids if vgroup[i] and VIDEO_PATTERNS[0].match(prehash_stem(i))}
    video_report = {
        "n_video_groups": len(vcount), "rows_in_video_groups": int(sum(vcount.values())),
        "frozen397_frames_in_video_groups": len(frozen_in_video),
        "a2_named_frame_groups": {"groups": len(a2_named), "rows": int(sum(vcount[g] for g in a2_named)),
                                  "frozen": int(sum(vfrozen[g] for g in a2_named))},
        "frozen397_stills": len(frozen_still), "frozen_own_after_nyu_cleaning": len(frozen_v2),
        "held_out_families": {k: {g: vcount[g] for g in v} for k, v in VIDEO_TEST_FAMILIES.items()},
        "video_test_rows": len(video_test),
        "groups": {g: {"rows": vcount[g], "frozen397": vfrozen[g]} for g in sorted(vcount, key=lambda g: (-vcount[g], g))},
    }
    _log(f"video groups {len(vcount)} rows {sum(vcount.values())}; frozen397 in videos {len(frozen_in_video)}; "
         f"frozen_own {len(frozen_v2)} (removed {len(frozen_removed)}); video_test {len(video_test)}")

    # ── legacy val carve (kept out of training, never used for selection) ────────────────
    val = legacy_val_carve(rows, frozen397)
    own_val = sorted(set(val["own"]) - set(video_test) - set(sf6_own) - set(nyu_family))
    nyu_val = val["nyu"]

    # ── own train candidates and leak exclusions ─────────────────────────────────────────
    held = set(frozen_v2) | set(video_test) | set(sf6_own)
    cand = sorted(set(own_ids) - held - set(val["own"]))
    excluded: Dict[str, Dict[str, List[str]]] = {"own": {}, "nyu": {}}

    def excl(src: str, ids: Iterable[str], reason: str) -> None:
        for i in ids:
            excluded[src].setdefault(i, [])
            if reason not in excluded[src][i]:
                excluded[src][i].append(reason)

    F_c = F_own.subset(cand)
    own_hits: Dict[str, Dict[str, int]] = {}

    def own_vs(ref_name: str, ref: Optional[FeatureSet], rule: str, rpath: Optional[Dict[str, str]] = None) -> None:
        """Exclude candidates matching ``ref`` (rule 'a3' or 'confirmed') + group closure."""
        if ref is None or not len(ref):
            return
        m = cross_match(F_c, ref, "symmetric")
        sel = m["hit"] if rule == "a3" else m["confirmed"]
        excl("own", [F_c.ids[k] for k in np.nonzero(sel)[0]], f"{rule}_match_of_{ref_name}")
        closed = group_closure(F_c.ids, vgroup, m)
        excl("own", [i for i in cand if vgroup[i] in closed], f"group_match_of_{ref_name}")
        own_hits[ref_name] = {"rule": rule, "a3": int(m["hit"].sum()), "confirmed": int(m["confirmed"].sum()),
                              "removed": int(sel.sum()), "groups_closed": len(closed),
                              "group_rows": int(sum(1 for i in cand if vgroup[i] in closed)), "bins": bin_counts(m)}
        write_match_csv(os.path.join(args.evidence, f"own_vs_{ref_name}.csv"), F_c, ref, m)
        if rpath is not None:
            match_sheet(os.path.join(args.evidence, f"sheet_own_vs_{ref_name}.jpg"),
                        f"own train candidate (left) vs {ref_name} (right), {rule} hits", F_c, own_path,
                        ref, rpath, m, sel, rng)

    # (a) stills sharing a pre-hash stem with a held-out still; (b) SHA-1 copies; (c) pHash <= 4
    held_keys = {group_key[i] for i in held if not vgroup[i]}
    excl("own", [i for i in cand if not vgroup[i] and group_key[i] in held_keys], "prehash_stem_of_heldout")
    sha_of = dict(zip(F_own.ids, F_own.sha1))
    held_sha = {sha_of[i] for i in held}
    excl("own", [i for i in cand if sha_of[i] in held_sha], "sha1_dup_of_heldout")
    m_dup = cross_match(F_c, F_own.subset(held), "raw", ham_th=HAM_DUP, ncc_th=-2)
    excl("own", [F_c.ids[i] for i in np.nonzero(m_dup["hit"])[0]], "phash_le4_of_heldout")
    # (d) evaluation-side references: A3 verbatim
    own_vs("frozen_own", F_own.subset(frozen_v2), "a3", own_path)
    own_vs("video_own", F_own.subset(video_test), "a3", own_path)
    own_vs("d_sun", F_d, "a3", sun_path)
    own_vs("nyu_labelled", F_lab, "a3")
    own_vs("ibims1", bench.get("ibims1"), "a3")
    own_vs("sf6_footage", F_sf6, "a3")
    # (e) NYU family and the training-side local NYU reference
    excl("own", [i for i in cand if i in set(nyu_family)], "nyu_family")
    own_vs("local_nyu", F_nyu, "a3" if args.strict_local_nyu else "confirmed", nyu_path)
    m_ln = cross_match(F_c, F_nyu, "symmetric")
    local_nyu_sensitivity = {"a3_rule_would_remove": int(m_ln["hit"].sum()),
                             "confirmed_rule_removes": int(m_ln["confirmed"].sum()),
                             "a3_only_rows_not_otherwise_excluded":
                                 int(sum(1 for k in np.nonzero(m_ln["hit"] & ~m_ln["confirmed"])[0]
                                         if F_c.ids[k] not in excluded["own"])),
                             "applied": "a3" if args.strict_local_nyu else "confirmed"}
    match_sheet(os.path.join(args.evidence, "sheet_own_vs_local_nyu_a3_not_confirmed.jpg"),
                "own (left) vs local NYU (right): a3 hits NOT confirmed (kept in training)", F_c, own_path,
                F_nyu, nyu_path, m_ln, m_ln["hit"] & ~m_ln["confirmed"], rng)
    own_train = sorted(set(cand) - set(excluded["own"]))

    # ── NYU train ────────────────────────────────────────────────────────────────────────
    nyu_ids = sorted(F_nyu.ids)
    nyu_cand = sorted(set(nyu_ids) - set(nyu_val))
    F_nc = F_nyu.subset(nyu_cand)
    nyu_hits: Dict[str, Dict[str, int]] = {}
    for name, ref in (("nyu_eigen_test_654", F_lab_test), ("ibims1", bench.get("ibims1"))):
        if ref is None:
            continue
        # local NYU frames legitimately match labelled TRAIN frames (same raw videos); only a match
        # to one of the 654 Eigen TEST frames is a benchmark leak, so the reference is test-only.
        m = cross_match(F_nc, ref, "symmetric")
        excl("nyu", [F_nc.ids[k] for k in np.nonzero(m["hit"])[0]], f"a3_match_of_{name}")
        nyu_hits[name] = {"a3": int(m["hit"].sum()), "confirmed": int(m["confirmed"].sum()), "bins": bin_counts(m)}
        write_match_csv(os.path.join(args.evidence, f"nyu_vs_{name}.csv"), F_nc, ref, m)
    nyu_train = sorted(set(nyu_cand) - set(excluded["nyu"]))

    # ── calibration (int8 only): train.own minus every legacy frozen id (P6) ──────────────
    luma = dict(zip(F_own.ids, map(float, F_own.luma)))
    calib_pool = sorted(set(own_train) - frozen397)
    calib = select_calibration(calib_pool, luma, group_key)
    calib_low = sum(luma[i] < LOWLIGHT_GRAY for i in calib)

    # ── pool (optional, after build_pool + labelling) ────────────────────────────────────
    pool_section: Dict[str, object] = {"status": "not built (A1: X-IN removed; MIT Indoor67 pending download)",
                                       "by_pool": {}}
    if args.pool_manifest:
        prow = read_manifest(args.pool_manifest)
        kept = [r for r in prow if str(r.get("keep", "0")) == "1"]
        pool_section = {"manifest": os.path.abspath(args.pool_manifest).replace("\\", "/"),
                        "manifest_sha256": sha256_file(args.pool_manifest), "status": "imported", "by_pool": {}}
        for r in kept:
            pool_section["by_pool"].setdefault(r["pool"], []).append(r["id"])
        for k in pool_section["by_pool"]:
            pool_section["by_pool"][k] = sorted(pool_section["by_pool"][k])

    # ── benchmark-side closure check: no benchmark image may match a final training row ──
    train_fs = FeatureSet.concat([F_own.subset(own_train), F_nyu.subset(nyu_train)])
    bench_drop: Dict[str, List[str]] = {}
    for name, ref in (("nyu_eigen_test_654", F_lab_test), ("ibims1", bench.get("ibims1"))):
        if ref is None:
            continue
        m = cross_match(ref, train_fs, "symmetric")
        bench_drop[name] = sorted(ref.ids[k] for k in np.nonzero(m["hit"])[0])
    _log(f"benchmark images still matching training rows (must be empty): "
         f"{ {k: len(v) for k, v in bench_drop.items()} }")

    # ── regression pairs for tests (known matched pairs found in the data) ───────────────
    reg: List[dict] = []
    strong = np.nonzero((m_sn["ham"] <= 2) & (m_sn["ncc"] >= 0.97))[0]
    for k in strong[:3]:
        reg.append({"kind": "sun_vs_local_nyu", "mode": "sun_vs_nyu", "query": sun_path[sun_ok.ids[k]],
                    "query_id": sun_ok.ids[k], "ref": nyu_path[F_nyu.ids[m_sn["ref"][k]]],
                    "ref_id": F_nyu.ids[m_sn["ref"][k]], "ham": int(m_sn["ham"][k]),
                    "ncc": round(float(m_sn["ncc"][k]), 4)})
    weak = np.nonzero((m_sn["ham"] >= 6) & (m_sn["ham"] <= 8) & (m_sn["ncc"] >= 0.9))[0]
    for k in weak[:2]:
        reg.append({"kind": "sun_vs_local_nyu_weak", "mode": "sun_vs_nyu", "query": sun_path[sun_ok.ids[k]],
                    "query_id": sun_ok.ids[k], "ref": nyu_path[F_nyu.ids[m_sn["ref"][k]]],
                    "ref_id": F_nyu.ids[m_sn["ref"][k]], "ham": int(m_sn["ham"][k]),
                    "ncc": round(float(m_sn["ncc"][k]), 4)})
    m_fs = cross_match(F_fam, F_sun, "symmetric")
    for k in np.nonzero((m_fs["ham"] <= 4) & (m_fs["ncc"] >= 0.95))[0][:2]:
        reg.append({"kind": "own_nyu_family_vs_sun", "mode": "symmetric", "query": own_path[F_fam.ids[k]],
                    "query_id": F_fam.ids[k], "ref": sun_path[F_sun.ids[m_fs["ref"][k]]],
                    "ref_id": F_sun.ids[m_fs["ref"][k]], "ham": int(m_fs["ham"][k]),
                    "ncc": round(float(m_fs["ncc"][k]), 4)})

    # ── assemble ─────────────────────────────────────────────────────────────────────────
    reason_counts: Dict[str, Dict[str, int]] = {}
    for src, d in excluded.items():
        c: Dict[str, int] = {}
        for rs in d.values():
            for r in rs:
                c[r] = c.get(r, 0) + 1
        reason_counts[src] = dict(sorted(c.items()))

    roles = {
        "schema": "v2-roles/2",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "frozen": True,
        "note": ("Frozen before R0. Benchmark-domain images never train, calibrate, select or gate. "
                 "SUN is development for V2 and is never reported as zero-shot."),
        "paths": {
            "manifest": os.path.abspath(args.manifest).replace("\\", "/"),
            "own_originals": OWN_ORIGINALS,
            "own_labels": os.path.join(OUTPUTS, "depth_mm").replace("\\", "/"),
            "own_conf": os.path.join(OUTPUTS, "conf").replace("\\", "/"),
            "cover_cache": os.path.join(OUTPUTS, "v2", "cache").replace("\\", "/"),
            "pool_labels": os.path.join(OUTPUTS, "v2", "labels", "v2l").replace("\\", "/"),
            "evidence": os.path.abspath(args.evidence).replace("\\", "/"),
            "datasets_root": DATASETS_ROOT,
        },
        "params": {
            "nyu_crop_rows_cols": list(NYU_CROP), "ham_dup": HAM_DUP, "ham_cross": HAM_CROSS,
            "ncc_cross": NCC_CROSS, "ham_confirm": HAM_CONFIRM, "ncc_confirm": NCC_CONFIRM,
            "ham_cluster": HAM_CLUSTER, "ncc_cluster": NCC_CLUSTER, "group_min_confirmed": GROUP_MIN_CONFIRMED,
            "group_min_frac": GROUP_MIN_FRAC, "sun8x_frac": SUN8X_FRAC,
            "phash": "64-bit DCT: 32x32 INTER_AREA gray, cv2.dct, 8x8 block, > median(AC); distances are even",
            "thumb": "40x30 INTER_AREA gray, zero-mean unit-norm, NCC = dot product",
            "leak_rule": {
                "a3": f"aligned pHash <= {HAM_CROSS} OR NCC >= {NCC_CROSS} (alignments raw-raw, raw-vs-crop, crop-vs-raw)",
                "confirmed": f"aligned pHash <= {HAM_CONFIRM} OR NCC >= {NCC_CONFIRM}",
                "evaluation_side_refs_use": "a3 (D_SUN vs NYU, own vs D_SUN / NYU labelled 1449 / IBims-1 / own tests / "
                                            "SF6, NYU train vs Eigen test 654 and IBims-1)",
                "training_side_refs_use": ("a3" if args.strict_local_nyu else "confirmed") + " (own vs local NYU train frames)",
                "group_closure": f">= {GROUP_MIN_CONFIRMED} confirmed hits and a3 hits on >= {GROUP_MIN_FRAC:.0%} of a video group",
            },
            "nyu_family_regex": NYU_FAMILY_RE.pattern,
            "lowlight_mean_gray": LOWLIGHT_GRAY, "calib_n": CALIB_N,
            "calib_lowlight_min": CALIB_LOWLIGHT_MIN, "calib_max_per_group": CALIB_MAX_PER_GROUP,
            "seed": SEED, "legacy_val_frac": LEGACY_VAL_FRAC, "legacy_val_seed": LEGACY_VAL_SEED,
            "video_test_families": VIDEO_TEST_FAMILIES,
            "video_patterns": [p.pattern for p in VIDEO_PATTERNS],
        },
        "counts": {},
        "train": {"own": own_train, "nyu": nyu_train, "pool": pool_section},
        "dev": {"d_sun": d_sun, "d_sun_clusters": multi, "d_sun_cluster_rule": cluster_report["rule"]},
        "test": {"frozen_own": frozen_v2, "video_own": video_test,
                 "frozen_own_removed": {k: frozen_removed[k] for k in sorted(frozen_removed)}},
        "val": {"own": own_val, "nyu": nyu_val,
                "rule": "legacy distill/dataset.py 3% carve (seed 1234) minus held-out/SF6/NYU-family rows; never "
                        "used for selection; own val is the PTQ MinMax-vs-Percentile comparison slice (plan 8.2)"},
        "calibration": {"own": calib, "n_lowlight": int(calib_low),
                        "luma": {i: round(luma[i], 2) for i in calib},
                        "rule": f"seeded {SEED}; subset of train.own minus all legacy 397 frozen ids; "
                                f">= {CALIB_LOWLIGHT_MIN} with mean gray < {LOWLIGHT_GRAY:g}; "
                                f"<= {CALIB_MAX_PER_GROUP} per video/pre-hash group"},
        "blacklist": {"sun_8x": bad8, "sun_8x_report": sun8_report},
        "nyu_derived_sun": {"ids": nyu_derived, "match_an_eigen_test_frame": dsun_nyu_test_hits,
                            "report": dsun_report},
        "nyu_family_own": {"ids": nyu_family, "report": fam_report,
                           "action": "excluded from train, val and the frozen test (NYU labelled frames inside own)"},
        "excluded": {"own": {k: excluded["own"][k] for k in sorted(excluded["own"])},
                     "nyu": {k: excluded["nyu"][k] for k in sorted(excluded["nyu"])},
                     "counts": reason_counts},
        "video_rule": {"rule": "inverted (A2): videos stay in training, their frames leave the frozen "
                               "test; whole small families are held out as test.video_own",
                       "frozen397_frames_moved_out_of_frozen_test": frozen_in_video, **video_report},
        "sf6_temporal": {"rule": "A21: held out of X-indoor and all training; used only for SF6",
                         "videos": sf6_meta, "own_groups": sf6_groups, "own_ids": sf6_own},
        "report_only": report_only_sets(bench_status),
        "leak_check": {
            "datasets_root": DATASETS_ROOT, "reachable_at_build": reach, "status": bench_status,
            "features": bench_meta,
            "complete": bool(all(str(bench_status.get(k, "")).startswith("done") for k in ("nyu_labelled", "ibims1"))),
            "own_train_vs_refs": own_hits, "nyu_train_vs_refs": nyu_hits,
            "own_vs_local_nyu_sensitivity": local_nyu_sensitivity,
            "benchmark_images_matching_final_training": bench_drop,
            "action_on_benchmark_hit": ("sets hashed at freeze time (NYU labelled, IBims-1): matching training rows "
                                        "are removed before training. Sets that become reachable after the freeze: "
                                        "run --check-benchmark; matching benchmark images are dropped from scoring "
                                        "and the drop is disclosed (A3)"),
        },
        "d_sun_clusters_report": cluster_report,
        "regression_pairs": reg,
    }
    roles["counts"] = {
        "train_own": len(own_train), "train_nyu": len(nyu_train), "d_sun": len(d_sun),
        "frozen_own": len(frozen_v2), "frozen_own_removed": len(frozen_removed), "video_own": len(video_test),
        "sf6_own": len(sf6_own), "calibration_own": len(calib), "calibration_lowlight": int(calib_low),
        "val_own": len(own_val), "val_nyu": len(nyu_val), "sun_8x": len(bad8),
        "nyu_derived_sun": len(nyu_derived), "nyu_family_own": len(nyu_family),
        "own_candidates": len(cand), "own_excluded": len(excluded["own"]),
        "nyu_excluded": len(excluded["nyu"]), "own_total": len(own_ids), "nyu_total": len(nyu_ids),
        "sun_total": len(F_sun), "d_sun_multi_clusters": len(multi),
    }
    roles["hashes"] = {
        "manifest_sha256": man_sha,
        "frozen_test_json_sha256": sha256_file(args.frozen),
        "splits_json_sha256": sha256_file(DEFAULT_SPLITS) if os.path.exists(DEFAULT_SPLITS) else None,
        "build_roles_py_sha256": sha256_file(os.path.abspath(__file__)),
        "roles": role_hashes(roles),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__},
    }
    check_disjoint(roles)
    return roles


ROLE_PATHS = {
    "train_own": ("train", "own"), "train_nyu": ("train", "nyu"), "d_sun": ("dev", "d_sun"),
    "frozen_own": ("test", "frozen_own"), "video_own": ("test", "video_own"),
    "sf6_own": ("sf6_temporal", "own_ids"), "calibration_own": ("calibration", "own"),
    "val_own": ("val", "own"), "val_nyu": ("val", "nyu"), "sun_8x": ("blacklist", "sun_8x"),
    "nyu_derived_sun": ("nyu_derived_sun", "ids"), "nyu_family_own": ("nyu_family_own", "ids"),
}


def role_hashes(roles: dict) -> Dict[str, str]:
    """ids_hash of every id-list role (what policy_check and the registry compare)."""
    return {k: ids_hash(roles[a][b]) for k, (a, b) in ROLE_PATHS.items()}


def report_only_sets(bench_status: Dict[str, str]) -> Dict[str, dict]:
    def on_disk(rel: str) -> str:
        if not os.path.isdir(DATASETS_ROOT):
            return "unknown (DATASETS_ROOT unreachable at build time)"
        return "on_disk" if os.path.exists(os.path.join(DATASETS_ROOT, rel)) else "not_on_disk"
    return {
        "ibims1": {"root": f"{DATASETS_ROOT}/IBims-1", "status": on_disk("IBims-1"),
                   "leak_check": bench_status.get("ibims1"), "zero_shot": "clean for CoDeQ"},
        "nyu_eigen_test_654": {"root": f"{DATASETS_ROOT}/NYUv2", "status": on_disk("NYUv2"),
                               "leak_check": bench_status.get("nyu_labelled"),
                               "zero_shot": "in-domain for keep-NYU; NYU-free only under the NYU leak rule"},
        "eth3d_indoor": {"root": f"{DATASETS_ROOT}/ETH3D", "status": on_disk("ETH3D"), "leak_check": "pending"},
        "diode_indoor_val": {"root": f"{DATASETS_ROOT}/DIODE", "status": on_disk("DIODE"), "leak_check": "pending"},
        "tum_rgbd_subset": {"root": f"{DATASETS_ROOT}/TUM", "status": on_disk("TUM"), "leak_check": "pending"},
        "diml_cvl_test": {"root": f"{DATASETS_ROOT}/DIML", "status": on_disk("DIML"), "leak_check": "pending"},
        "sun_official_test": {"zero_shot": "never (SUN is the V2 development domain)"},
        "rule": "scored once, after a recipe-freeze registry record (A9); never used for decisions; "
                "pending sets need --check-benchmark before their first score",
    }


def check_disjoint(roles: dict) -> None:
    """Raise if any two exclusive roles share an id, calibration leaves train.own or touches a
    legacy frozen id, or D_SUN holds a blacklisted / NYU-derived row."""
    own_sets = {"train.own": set(roles["train"]["own"]), "test.frozen_own": set(roles["test"]["frozen_own"]),
                "test.video_own": set(roles["test"]["video_own"]), "val.own": set(roles["val"]["own"]),
                "sf6_temporal.own_ids": set(roles["sf6_temporal"]["own_ids"])}
    names = list(own_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = own_sets[a] & own_sets[b]
            if inter:
                raise SystemExit(f"[build_roles] roles {a} and {b} overlap: {sorted(inter)[:3]}")
    if set(roles["train"]["own"]) & set(roles["excluded"]["own"]):
        raise SystemExit("[build_roles] train.own contains excluded ids")
    if set(roles["train"]["own"]) & set(roles["nyu_family_own"]["ids"]):
        raise SystemExit("[build_roles] train.own contains NYU-family ids")
    if not set(roles["calibration"]["own"]) <= own_sets["train.own"]:
        raise SystemExit("[build_roles] calibration ids outside train.own")
    if set(roles["train"]["nyu"]) & (set(roles["val"]["nyu"]) | set(roles["excluded"]["nyu"])):
        raise SystemExit("[build_roles] train.nyu overlaps val.nyu or excluded.nyu")
    if set(roles["dev"]["d_sun"]) & (set(roles["blacklist"]["sun_8x"]) | set(roles["nyu_derived_sun"]["ids"])):
        raise SystemExit("[build_roles] D_SUN contains blacklisted or NYU-derived rows")


def check_benchmark(args) -> dict:
    """Leak-check a benchmark image folder against the frozen training roles (never edits roles)."""
    roles = load_roles(args.roles)
    if not os.path.isdir(args.bench_dir):
        raise SystemExit(f"[build_roles] benchmark folder not found: {args.bench_dir} "
                         "(Z: may be disconnected; nothing is substituted)")
    fdir = os.path.join(roles["paths"]["evidence"], "features")
    own, _ = FeatureSet.load(os.path.join(fdir, "own.npz"))
    nyu, _ = FeatureSet.load(os.path.join(fdir, "nyu.npz"))
    train = FeatureSet.concat([own.subset(roles["train"]["own"]), nyu.subset(roles["train"]["nyu"])])
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.bench_dir) for f in fs
                   if f.lower().endswith(exts) and not any(t in f.lower() for t in ("depth", "mask", "normal")))
    if not paths:
        raise SystemExit(f"[build_roles] no RGB images under {args.bench_dir}")
    ids = [os.path.relpath(p, args.bench_dir).replace("\\", "/") for p in paths]
    fs = features_for_files(ids, paths, args.workers)
    m = cross_match(fs, train, "symmetric")
    drop = sorted(fs.ids[k] for k in np.nonzero(m["hit"])[0])
    out = {"benchmark": args.check_benchmark, "dir": args.bench_dir.replace("\\", "/"), "n_images": len(fs),
           "rule": "a3 (aligned pHash <= 12 OR NCC >= 0.85) against train.own + train.nyu",
           "drop_ids": drop, "n_drop": len(drop), "bins": bin_counts(m),
           "roles_sha256": sha256_file(args.roles), "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "action": "drop these images from scoring and disclose the drop (A3)"}
    path = os.path.join(roles["paths"]["evidence"], f"bench_{args.check_benchmark}_drop.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    _log(f"wrote {path}: {len(drop)} of {len(fs)} benchmark images match training rows")
    return out


def write_roles(roles: dict, path: str) -> None:
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
    head = ("# V2 (CoDeQ) frozen dataset roles. GENERATED by scripts/v2/build_roles.py - do not edit by hand.\n"
            "# Re-running the builder changes the hashes below; policy_check compares them.\n")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(head)
        yaml.dump(roles, f, Dumper=dumper, sort_keys=False, default_flow_style=False, width=200,
                  allow_unicode=False)
    os.replace(tmp, path)


def load_roles(path: str = DEFAULT_OUT) -> dict:
    """Read roles.yaml (C loader when available). Fails loudly if missing."""
    import yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"roles file not found: {path} (run python -m scripts.v2.build_roles)")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--frozen", default=DEFAULT_FROZEN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--benchmarks", choices=["auto", "require", "skip"], default="auto",
                    help="auto: hash benchmarks (or use their feature cache) and record what ran; "
                         "require: fail loudly when a benchmark cannot be hashed")
    ap.add_argument("--pool-manifest", default="")
    ap.add_argument("--strict-local-nyu", action="store_true",
                    help="apply the full A3 rule (not 'confirmed') to own vs the local NYU train frames")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--check-benchmark", default="", help="NAME: leak-check a benchmark folder (see --bench-dir)")
    ap.add_argument("--bench-dir", default="")
    ap.add_argument("--roles", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.check_benchmark:
        check_benchmark(args)
        return
    roles = build(args)
    write_roles(roles, args.out)
    _log(f"wrote {args.out}")
    print(json.dumps(roles["counts"], indent=1))


if __name__ == "__main__":
    main()
