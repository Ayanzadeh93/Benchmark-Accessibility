"""Resumable UniDepthV2-L (V2-L) teacher labelling for the V2 corpus (plan task 20). Env: ``depth``.

    python -m scripts.v2.label_unidepth --input outputs/v2/pool/manifest.csv --out outputs/v2/labels/v2l
    python -m scripts.v2.label_unidepth --input "C:/Tim/nature/database/Fianl Version/images" --limit 2

CONTRACT
  Input (--input): a CSV with an image-path column (``img`` | ``image`` | ``path`` | ``image_path``),
  an optional ``id`` column (default id = file stem) and an optional ``keep`` column (when present,
  only rows with keep == 1 are labelled; scripts/v2/build_pool.py writes it); or a .txt file with
  one image path per line; or a directory of images (searched recursively, sorted).
  Ids are used VERBATIM so they equal the manifest / roles.yaml ids (e.g. "..._000201 - Copy").
  An id that is not a safe Windows file name (``<>:"/\\|?*``, control characters, trailing dot or
  space, reserved device names) or that repeats with a different image stops the run before the
  GPU is touched; nothing is renamed silently.

  Teacher: ``lpiccinelli/unidepth-v2-vitl14`` only (no fallback to a smaller checkpoint), loaded
  through the legacy ``teacher.unidepth_teacher.UniDepthTeacher`` so the label pipeline is the
  one that produced ``outputs/depth_mm``: image resized (INTER_AREA) to longest side 1280 when
  larger, native ``UniDepthV2.infer`` (decorated with fp16 autocast inside UniDepth, camera
  self-calibrated, no intrinsics passed), depth bilinearly resized back to the original
  resolution. ``resolution_level`` is left unset by default (legacy parity: UniDepth's default
  pixel bounds) and recorded, together with the model's shape constraints, in every sidecar.

  Output, under --out (default ``outputs/v2/labels/v2l``):
    depth_mm/<id>.png   uint16 millimetres, 0 = invalid. Same encoding as outputs/depth_mm:
                        ``common.depth_ops.depth_m_to_png16`` (clip 60 m) then longest side
                        capped at 1024 with INTER_NEAREST.
    meta/<id>.json      sidecar (schema ``v2-teacher-label/1``): image path + SHA-1, original /
                        inference / saved sizes, teacher id and settings, PNG SHA-256, depth
                        stats, ``label_valid`` flag, optional pool-filter verdict, timings.
    done.csv            append-only manifest of finished ids (repaired from sidecars on start).
    failures.jsonl      one JSON line per failed image (the run exits with code 2 if any).
    run_<utc>.json      run summary with measured images/second (model-only and end-to-end)
                        and peak VRAM.

  Resumable / skip-if-done: an id counts as done when its sidecar exists with status "ok" and
  its PNG exists. PNG and sidecar are written atomically (temp file + os.replace), PNG first, so
  a sidecar always refers to a complete PNG. Interrupt at any point and re-run the same command.
  A done id whose sidecar names a DIFFERENT image is a conflict: if the SHA-1 of the requested
  image equals the recorded SHA-1 (the same file moved, e.g. to DATASETS_ROOT) it is accepted,
  otherwise the run aborts (use --overwrite to relabel).

  Unstable drives: every input drive is checked before the model loads, and again whenever an
  image fails to read. A missing drive (e.g. Z: disconnected) aborts the run with a clear
  message; nothing falls back to another dataset or another copy.

  ``label_valid`` (my definition, used for the T1-01 ">= 98% valid labels" gate): at least
  95% of pixels lie in [0.2, 30] m. The optional pool filter (--pool-filter, plan 5.2) records
  ``median in [0.5, 15] m and p95/p5 >= 2``; it never deletes a label.

  --vram-cap-gb N sets torch's per-process memory fraction so the run fails with an OOM instead
  of taking more than N GiB (for smoke tests on a shared GPU).

The shared labelling framework below (``read_items``, ``check_drives``, ``run_labelling`` and
helpers) is also used by ``scripts/v2/label_moge.py`` and ``scripts/v2/fuse_composite.py``; it
imports no torch at module level. Tests: tests/test_composite.py (CPU, offline, fake teacher).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.depth_ops import depth_m_to_png16, resize_longest_side  # noqa: E402

LABEL_SCHEMA = "v2-teacher-label/1"
DEFAULT_CHECKPOINT = "lpiccinelli/unidepth-v2-vitl14"
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "v2", "labels", "v2l")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PATH_COLUMNS = ("img", "image", "path", "image_path")
VALID_DMIN_M, VALID_DMAX_M = 0.2, 30.0          # V2 student range (log_min/log_max)
LABEL_VALID_MIN_FRAC = 0.95
POOL_MEDIAN_RANGE_M = (0.5, 15.0)
POOL_MIN_P95_P5 = 2.0
DEFAULT_CLIP_M = 60.0                            # same clamp as config.yaml teacher.depth_clip_m
DEFAULT_SAVE_LONGEST = 1024                      # same cap as config.yaml teacher.save_longest_side
DEFAULT_INFER_LONGEST = 1280
_ID_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                 *(f"LPT{i}" for i in range(1, 10))}


# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    id: str
    img: str


def validate_id(raw) -> str:
    """Return ``raw`` unchanged if it is usable as a file name on Windows, else raise ValueError.

    Ids are never rewritten: a rewritten id would no longer match manifest.csv / roles.yaml.
    """
    s = str(raw)
    if not s or s.strip() != s or s in (".", ".."):
        raise ValueError(f"unusable id {raw!r} (empty, or leading/trailing whitespace)")
    if _ID_FORBIDDEN.search(s) or s.endswith(".") or s.split(".")[0].upper() in _WIN_RESERVED:
        raise ValueError(f"id {raw!r} is not a safe Windows file name; give a safe id column")
    return s


def _resolve(path: str, base: str) -> str:
    path = path.strip().strip('"')
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.normpath(os.path.abspath(path))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def read_items(input_path: str, image_root: Optional[str] = None, start: int = 0,
               limit: int = 0, keep_column: str = "auto") -> List[Item]:
    """Parse --input (CSV, TXT or directory) into unique (id, image path) items, in file order.

    keep_column: "auto" filters on a ``keep`` CSV column when it exists; "ignore" never filters.
    Raises ValueError on a CSV without a path column, an unsafe id, or an id that repeats with
    a different image. Exact duplicate rows (same id, same image) collapse to one.
    """
    if not os.path.exists(input_path):
        check_drives([input_path])
        raise FileNotFoundError(f"--input {input_path} does not exist")
    items: List[Item] = []
    if os.path.isdir(input_path):
        found = []
        for dp, _dn, fns in os.walk(input_path):
            for fn in fns:
                if fn.lower().endswith(IMAGE_EXTS):
                    found.append(os.path.join(dp, fn))
        for p in sorted(found):
            items.append(Item(validate_id(os.path.splitext(os.path.basename(p))[0]),
                              os.path.normpath(os.path.abspath(p))))
    elif input_path.lower().endswith(".csv"):
        base = image_root or os.path.dirname(os.path.abspath(input_path))
        with open(input_path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            pcol = next((c for c in PATH_COLUMNS if c in cols), None)
            if pcol is None:
                raise ValueError(f"{input_path}: no image path column; expected one of {PATH_COLUMNS}, got {cols}")
            use_keep = keep_column == "auto" and "keep" in cols
            for r in rd:
                if use_keep and not _truthy(r.get("keep", "")):
                    continue
                p = _resolve(r[pcol], base)
                rid = r.get("id") or os.path.splitext(os.path.basename(p))[0]
                items.append(Item(validate_id(rid), p))
    else:
        base = image_root or os.path.dirname(os.path.abspath(input_path))
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = _resolve(line, base)
                items.append(Item(validate_id(os.path.splitext(os.path.basename(p))[0]), p))
    seen: Dict[str, str] = {}
    dups = []
    for it in items:
        if it.id in seen and os.path.normcase(seen[it.id]) != os.path.normcase(it.img):
            dups.append((it.id, seen[it.id], it.img))
        seen.setdefault(it.id, it.img)
    if dups:
        ex = "; ".join(f"{d[0]}: {d[1]} | {d[2]}" for d in dups[:5])
        raise ValueError(f"{len(dups)} ids repeat with different images (give an explicit id column): {ex}")
    uniq, used = [], set()
    for it in items:
        if it.id not in used:
            uniq.append(it)
            used.add(it.id)
    if start:
        uniq = uniq[start:]
    if limit and limit > 0:
        uniq = uniq[:limit]
    return uniq


def _drive_of(path: str) -> str:
    drive, _ = os.path.splitdrive(os.path.abspath(path))
    return drive or os.sep


def check_drives(paths: Iterable[str]) -> None:
    """Abort (SystemExit) if any drive or UNC share holding one of ``paths`` is not mounted."""
    missing = sorted({d for d in {_drive_of(p) for p in paths} if not os.path.exists(d + os.sep)})
    if missing:
        raise SystemExit(
            f"[label] drive(s) {', '.join(missing)} not available. The external drive (Z:) has "
            "disconnected before; reconnect it, or regenerate the input list against a local copy "
            "(DATASETS_ROOT). Refusing to continue: no fallback dataset is used.")


# ──────────────────────────────────────────────────────────────────────────────
# Atomic IO and hashing
# ──────────────────────────────────────────────────────────────────────────────

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_png(path: str, arr: np.ndarray) -> str:
    """Encode ``arr`` as PNG, write atomically, return the SHA-256 of the written bytes."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"PNG encode failed for {path}")
    data = buf.tobytes()
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: str, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=False).encode("utf-8"))


def file_sha256_short(path: str) -> str:
    try:
        return sha256_file(path)[:16]
    except OSError:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Label encoding and statistics
# ──────────────────────────────────────────────────────────────────────────────

def encode_label(depth_m: np.ndarray, valid: Optional[np.ndarray], clip_m: float = DEFAULT_CLIP_M,
                 save_longest: int = DEFAULT_SAVE_LONGEST) -> np.ndarray:
    """Metres (H,W) + validity -> uint16 mm PNG array; identical to the outputs/depth_mm chain."""
    d = np.asarray(depth_m, dtype=np.float32)
    if valid is not None:
        d = np.where(valid, d, 0.0).astype(np.float32)
    png16 = depth_m_to_png16(d, clip_m=clip_m)
    return resize_longest_side(png16, save_longest, cv2.INTER_NEAREST)


def read_label_png(path: str) -> np.ndarray:
    """uint16 mm PNG -> float32 metres (0 = invalid). Raises FileNotFoundError if unreadable."""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        check_drives([path])
        raise FileNotFoundError(f"cannot read label PNG {path}")
    if a.dtype != np.uint16 or a.ndim != 2:
        raise ValueError(f"{path}: expected single-channel uint16 PNG, got {a.dtype} {a.shape}")
    return a.astype(np.float32) / 1000.0


def depth_stats(png16: np.ndarray, clip_m: float = DEFAULT_CLIP_M) -> Dict[str, float]:
    """Statistics of a saved label (computed on the PNG, so they describe what is stored)."""
    d = png16.astype(np.float64) / 1000.0
    pos = d[d > 0]
    n = float(d.size)
    out = {
        "zero_frac": round(float((d <= 0).sum() / n), 6),
        "valid_frac_0.2_30": round(float(((d >= VALID_DMIN_M) & (d <= VALID_DMAX_M)).sum() / n), 6),
        "clipped_frac": round(float((d >= min(clip_m, 65.535) - 1e-3).sum() / n), 6),
    }
    if pos.size:
        p5, med, p95 = np.percentile(pos, [5, 50, 95])
        out.update(median_m=round(float(med), 4), p5_m=round(float(p5), 4), p95_m=round(float(p95), 4),
                   p95_over_p5=round(float(p95 / max(p5, 1e-6)), 4))
    else:
        out.update(median_m=None, p5_m=None, p95_m=None, p95_over_p5=None)
    return out


def label_is_valid(stats: Dict[str, float]) -> bool:
    return bool(stats.get("valid_frac_0.2_30", 0.0) >= LABEL_VALID_MIN_FRAC)


def pool_filter(stats: Dict[str, float]) -> Dict[str, object]:
    """Plan 5.2 pool filter: teacher median in [0.5, 15] m and p95/p5 >= 2 (recorded, not applied)."""
    med, ratio = stats.get("median_m"), stats.get("p95_over_p5")
    ok_med = med is not None and POOL_MEDIAN_RANGE_M[0] <= med <= POOL_MEDIAN_RANGE_M[1]
    ok_ratio = ratio is not None and ratio >= POOL_MIN_P95_P5
    return {"pass": bool(ok_med and ok_ratio), "median_ok": bool(ok_med), "p95_p5_ok": bool(ok_ratio),
            "rule": f"median in {list(POOL_MEDIAN_RANGE_M)} m and p95/p5 >= {POOL_MIN_P95_P5}"}


def resized_hw(h: int, w: int, longest: int) -> tuple:
    """Same rounding as teacher.unidepth_teacher / common.depth_ops.resize_longest_side."""
    if not longest or longest <= 0 or max(h, w) <= longest:
        return h, w
    s = longest / float(max(h, w))
    return int(round(h * s)), int(round(w * s))


# ──────────────────────────────────────────────────────────────────────────────
# Resumable labelling loop (teacher-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

DONE_HEADER = ["id", "image", "depth_png", "meta_json", "png_sha256", "label_valid", "utc"]

# predict_fn(rgb uint8 HxWx3) -> {"depth_m": float32 (H,W) metres at ORIGINAL resolution,
#                                 "valid": bool (H,W), "extra": dict (json-serialisable)}
PredictFn = Callable[[np.ndarray], Dict[str, object]]


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _read_sidecar(out_dir: str, item_id: str) -> Optional[Dict]:
    mp = os.path.join(out_dir, "meta", item_id + ".json")
    try:
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _is_done(out_dir: str, item_id: str) -> bool:
    if not os.path.exists(os.path.join(out_dir, "depth_mm", item_id + ".png")):
        return False
    m = _read_sidecar(out_dir, item_id)
    return bool(m and m.get("status") == "ok")


def done_ids(out_dir: str) -> set:
    """Ids whose sidecar says ok and whose PNG exists (the source of truth for resuming)."""
    md = os.path.join(out_dir, "meta")
    if not os.path.isdir(md):
        return set()
    ids = {f[:-5] for f in os.listdir(md) if f.endswith(".json")}
    return {i for i in ids if _is_done(out_dir, i)}


def find_conflicts(out_dir: str, items: Sequence[Item], ok_ids: set) -> List[Dict[str, str]]:
    """Done ids whose sidecar names another image whose content (SHA-1) also differs."""
    out = []
    for it in items:
        if it.id not in ok_ids:
            continue
        m = _read_sidecar(out_dir, it.id) or {}
        rec = m.get("image", "")
        if os.path.normcase(os.path.normpath(rec)) == os.path.normcase(os.path.normpath(it.img)):
            continue
        try:
            same = os.path.exists(it.img) and sha1_file(it.img) == m.get("image_sha1")
        except OSError:
            same = False
        if not same:
            out.append({"id": it.id, "labelled_image": rec, "requested_image": it.img})
    return out


def _repair_done_csv(out_dir: str, ok_ids: set) -> int:
    path = os.path.join(out_dir, "done.csv")
    listed = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            listed = {r["id"] for r in csv.DictReader(f)}
    missing = sorted(ok_ids - listed)
    if missing:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DONE_HEADER)
            if new:
                w.writeheader()
            for i in missing:
                m = _read_sidecar(out_dir, i) or {}
                w.writerow({"id": i, "image": m.get("image", ""), "depth_png": m.get("depth_png", ""),
                            "meta_json": os.path.join(out_dir, "meta", i + ".json"),
                            "png_sha256": m.get("png_sha256", ""), "label_valid": int(bool(m.get("label_valid"))),
                            "utc": m.get("utc", "")})
    return len(missing)


def _read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _env_stamp() -> Dict[str, str]:
    out = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
           "platform": platform.platform(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "") or
           os.path.basename(os.path.dirname(sys.executable))}
    try:  # torch is optional for this module (tests run without a GPU)
        import torch
        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return out


def _peak_vram_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def apply_vram_cap(cap_gb: Optional[float], device: str = "cuda") -> Optional[Dict[str, float]]:
    """Limit this process's CUDA caching allocator to ``cap_gb`` GiB (None/0 = no cap)."""
    if not cap_gb or not str(device).startswith("cuda"):
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, float(cap_gb) * 2 ** 30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return {"cap_gib": float(cap_gb), "fraction": round(frac, 4), "total_gib": round(total / 2 ** 30, 2)}


def run_labelling(items: Sequence[Item], out_dir: str, predict_fn: PredictFn, teacher_meta: Dict,
                  *, overwrite: bool = False, warmup: int = 1, clip_m: float = DEFAULT_CLIP_M,
                  save_longest: int = DEFAULT_SAVE_LONGEST, pool_filter_on: bool = False,
                  script_path: Optional[str] = None, log: Callable[[str], None] = print) -> Dict:
    """Label ``items`` with ``predict_fn`` into ``out_dir``; skip finished ids; return the summary.

    Timings: ``t_model_s`` is the predict_fn call (the teacher wrappers end with a .cpu() copy,
    so GPU work is synchronised, and include the resize back to the original size); ``t_total_s``
    adds image read, hashing, encoding and writes. The first ``warmup`` forward passes (on the
    first pending image) are run untimed and discarded. Raises SystemExit on a done-id conflict
    or a vanished drive.
    """
    os.makedirs(os.path.join(out_dir, "depth_mm"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    ok_before = done_ids(out_dir)
    if not overwrite:
        conflicts = find_conflicts(out_dir, items, ok_before)
        if conflicts:
            raise SystemExit(f"[label] {len(conflicts)} done ids in {out_dir} were labelled from a different "
                             f"image, e.g. {conflicts[:3]}. Use another --out or --overwrite.")
    repaired = _repair_done_csv(out_dir, ok_before)
    pending = [it for it in items if overwrite or it.id not in ok_before]
    log(f"[label] {len(items)} items, {len(items) - len(pending)} already done, {len(pending)} to label"
        + (f" (repaired {repaired} done.csv rows)" if repaired else ""))
    check_drives([it.img for it in pending])
    script_sha = file_sha256_short(script_path) if script_path else ""

    done_path = os.path.join(out_dir, "done.csv")
    new_done = not os.path.exists(done_path)
    fails_path = os.path.join(out_dir, "failures.jsonl")
    t_models, t_totals, failures = [], [], []
    n_ok = n_valid = n_pool = 0
    t_run0 = time.time()
    warm_s = None
    if pending and warmup > 0:
        rgb0 = _read_rgb(pending[0].img)
        if rgb0 is not None:
            tw = time.time()
            for _ in range(int(warmup)):
                predict_fn(rgb0)
            warm_s = round(time.time() - tw, 3)
    with open(done_path, "a", newline="", encoding="utf-8") as df:
        dw = csv.DictWriter(df, fieldnames=DONE_HEADER)
        if new_done:
            dw.writeheader()
        for k, it in enumerate(pending):
            t0 = time.time()
            try:
                rgb = _read_rgb(it.img)
                if rgb is None:
                    check_drives([it.img])               # drive vanished mid-run -> abort loudly
                    raise FileNotFoundError(f"cannot read image {it.img}")
                h, w = rgb.shape[:2]
                tm0 = time.time()
                pred = predict_fn(rgb)
                t_model = time.time() - tm0
                depth, valid = np.asarray(pred["depth_m"]), np.asarray(pred["valid"], bool)
                if depth.shape[:2] != (h, w) or valid.shape[:2] != (h, w):
                    raise RuntimeError(f"teacher returned {depth.shape} for a {h}x{w} image")
                png16 = encode_label(depth, valid, clip_m=clip_m, save_longest=save_longest)
                png_path = os.path.join(out_dir, "depth_mm", it.id + ".png")
                png_sha = atomic_write_png(png_path, png16)
                stats = depth_stats(png16, clip_m=clip_m)
                lv = label_is_valid(stats)
                meta = {
                    "schema": LABEL_SCHEMA, "status": "ok", "id": it.id, "image": it.img,
                    "image_sha1": sha1_file(it.img), "orig_hw": [h, w],
                    "saved_hw": list(png16.shape[:2]), "depth_png": png_path, "png_sha256": png_sha,
                    "encoding": f"uint16 millimetres, 0 = invalid, clipped at {clip_m} m, longest side "
                                f"<= {save_longest} (INTER_NEAREST)",
                    "teacher": teacher_meta, "stats": stats, "label_valid": lv,
                    "label_valid_rule": f">= {LABEL_VALID_MIN_FRAC:.0%} of pixels in "
                                        f"[{VALID_DMIN_M}, {VALID_DMAX_M}] m",
                    "teacher_valid_frac": round(float(valid.mean()), 6),
                    "extra": pred.get("extra", {}),
                    "script_sha256_16": script_sha, "utc": _utc(),
                }
                if pool_filter_on:
                    meta["pool_filter"] = pool_filter(stats)
                    n_pool += int(meta["pool_filter"]["pass"])
                t_total = time.time() - t0
                meta["timing"] = {"t_model_s": round(t_model, 4), "t_total_s": round(t_total, 4)}
                meta_path = os.path.join(out_dir, "meta", it.id + ".json")
                atomic_write_json(meta_path, meta)
                dw.writerow({"id": it.id, "image": it.img, "depth_png": png_path, "meta_json": meta_path,
                             "png_sha256": png_sha, "label_valid": int(lv), "utc": meta["utc"]})
                df.flush()
                t_models.append(t_model)
                t_totals.append(t_total)
                n_ok += 1
                n_valid += int(lv)
                if n_ok <= 3 or n_ok % 100 == 0:
                    log(f"[label] {k + 1}/{len(pending)} {it.id} {h}x{w} model {t_model:.3f}s "
                        f"total {t_total:.3f}s median {stats.get('median_m')} m valid={lv}")
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001 - record and continue, exit code reports it
                rec = {"id": it.id, "image": it.img, "error": f"{type(e).__name__}: {str(e)[:300]}", "utc": _utc()}
                failures.append(rec)
                with open(fails_path, "a", encoding="utf-8") as ff:
                    ff.write(json.dumps(rec) + "\n")
                log(f"[label] FAILED {it.id}: {rec['error']}")

    def rate(ts):
        return round(len(ts) / sum(ts), 3) if ts and sum(ts) > 0 else None

    summary = {
        "schema": "v2-label-run/1", "utc": _utc(), "out_dir": out_dir, "teacher": teacher_meta,
        "n_items": len(items), "n_already_done": len(items) - len(pending), "n_labelled": n_ok,
        "n_failed": len(failures), "n_label_valid": n_valid,
        "label_valid_pct": round(100.0 * n_valid / n_ok, 3) if n_ok else None,
        "n_pool_filter_pass": n_pool if pool_filter_on else None,
        "warmup_passes": int(warmup) if warm_s is not None else 0, "warmup_s": warm_s,
        "img_per_s_model": rate(t_models), "img_per_s_end_to_end": rate(t_totals),
        "median_t_model_s": round(float(np.median(t_models)), 4) if t_models else None,
        "median_t_total_s": round(float(np.median(t_totals)), 4) if t_totals else None,
        "wall_s_excl_model_load": round(time.time() - t_run0, 2),
        "peak_vram_mb": _peak_vram_mb(), "env": _env_stamp(), "failures": failures[:50],
        "script_sha256_16": script_sha,
    }
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(os.path.join(out_dir, f"run_{stamp}.json"), summary)
    log(f"[label] done: {n_ok} labelled, {len(failures)} failed, {summary['n_already_done']} skipped; "
        f"model {summary['img_per_s_model']} img/s, end-to-end {summary['img_per_s_end_to_end']} img/s, "
        f"peak VRAM {summary['peak_vram_mb']} MiB")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# UniDepthV2-L predictor
# ──────────────────────────────────────────────────────────────────────────────

def make_unidepth_predictor(checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda",
                            infer_longest_side: int = DEFAULT_INFER_LONGEST,
                            resolution_level: Optional[int] = None):
    """Load V2-L through the legacy wrapper; return (predict_fn, teacher_meta). Single checkpoint only."""
    import torch
    from teacher.unidepth_teacher import UniDepthTeacher
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[label] CUDA requested but not available; pass --device cpu explicitly to run on CPU")
    t0 = time.time()
    teacher = UniDepthTeacher([checkpoint], device=device)   # one entry -> no silent fallback
    teacher.load()
    if teacher.device != device:
        raise SystemExit(f"[label] teacher loaded on {teacher.device}, not {device}")
    if resolution_level is not None:
        teacher.model.resolution_level = int(resolution_level)
    load_s = round(time.time() - t0, 2)
    try:
        from importlib.metadata import version
        ud_ver = version("unidepth")
    except Exception:  # noqa: BLE001
        ud_ver = "unknown"
    shape_c = getattr(teacher.model, "shape_constraints", None)

    def predict(rgb: np.ndarray) -> Dict[str, object]:
        out = teacher.predict(rgb, infer_longest_side=infer_longest_side)
        d = np.asarray(out["depth_m"], dtype=np.float32)
        valid = np.isfinite(d) & (d > 0)
        ih, iw = resized_hw(rgb.shape[0], rgb.shape[1], infer_longest_side)
        return {"depth_m": d, "valid": valid, "extra": {"infer_hw": [ih, iw]}}

    meta = {"name": "unidepth-v2-l", "checkpoint": teacher.loaded_checkpoint,
            "infer_longest_side": int(infer_longest_side), "input_resize": "INTER_AREA",
            "resolution_level": resolution_level,
            "resolution_note": "None = UniDepthV2 default pixel bounds (legacy outputs/depth_mm parity)",
            "shape_constraints": {k: (v if isinstance(v, (int, float, str, list)) else str(v))
                                  for k, v in (shape_c or {}).items()},
            "interpolation_mode": str(getattr(teacher.model, "interpolation_mode", "")),
            "precision": "fp32 weights, fp16 autocast inside UniDepthV2.infer",
            "camera": "self-calibrated (no intrinsics passed)", "output_resize": "bilinear to original",
            "device": device, "unidepth_version": ud_ver, "model_load_s": load_s,
            "licence": "CC BY-NC 4.0 (research-only)"}
    return predict, meta


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="CSV / TXT list / directory of images")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--image-root", default=None, help="base for relative paths in CSV/TXT")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ignore-keep", action="store_true", help="label every CSV row even if a keep column exists")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--warmup", type=int, default=1, help="untimed forward passes before timing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-cap-gb", type=float, default=0.0, help="per-process CUDA memory cap (0 = none)")
    ap.add_argument("--save-longest-side", type=int, default=DEFAULT_SAVE_LONGEST)
    ap.add_argument("--clip-m", type=float, default=DEFAULT_CLIP_M)
    ap.add_argument("--pool-filter", action="store_true", help="record the plan 5.2 pool filter verdict")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face network access")
    return ap


def prepare_run(args, tag: str = "label") -> Optional[List[Item]]:
    """Shared CLI prologue: offline mode, items, drive check, done/conflict check. None = all done."""
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    items = read_items(args.input, args.image_root, args.start, args.limit,
                       keep_column="ignore" if args.ignore_keep else "auto")
    if not items:
        raise SystemExit(f"[{tag}] no images found in {args.input}")
    check_drives([it.img for it in items])
    if not args.overwrite:
        ok = done_ids(args.out)
        conflicts = find_conflicts(args.out, items, ok)
        if conflicts:
            raise SystemExit(f"[{tag}] {len(conflicts)} done ids in {args.out} were labelled from a different "
                             f"image, e.g. {conflicts[:3]}. Use another --out or --overwrite.")
        if all(it.id in ok for it in items):
            print(f"[{tag}] all {len(items)} items already labelled in {args.out}")
            return None
    return items


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--infer-longest-side", type=int, default=DEFAULT_INFER_LONGEST)
    ap.add_argument("--resolution-level", type=int, default=None,
                    help="UniDepthV2 resolution_level 0..9; default unset = legacy outputs/depth_mm parity")
    args = ap.parse_args(argv)
    items = prepare_run(args, "label")
    if items is None:
        return 0
    cap = apply_vram_cap(args.vram_cap_gb, args.device)
    predict, meta = make_unidepth_predictor(args.checkpoint, args.device, args.infer_longest_side,
                                            args.resolution_level)
    if cap:
        meta["vram_cap"] = cap
    summary = run_labelling(items, args.out, predict, meta, overwrite=args.overwrite, warmup=args.warmup,
                            clip_m=args.clip_m, save_longest=args.save_longest_side,
                            pool_filter_on=args.pool_filter, script_path=__file__)
    return 2 if summary["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
