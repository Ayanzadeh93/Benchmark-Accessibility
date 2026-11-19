"""Aspect-preserving "cover" cache for V2 training/eval images and teacher labels.

Contract
--------
CLI (env ``depth``; CPU only)::

    python -m scripts.v2.build_cover_cache [--roles configs/v2/roles.yaml] [--source own]
        [--input-csv rows.csv] [--out outputs/v2/cache] [--limit N] [--workers 8] [--overwrite]

For every row it reads the ORIGINAL-aspect image and its label, scales both by
``s = max(640 / W, 480 / H)`` (aspect kept, so the result covers a 480x640 crop window) and writes

    <out>/<source>/images/<id>.jpg       RGB, JPEG q95 (INTER_AREA when shrinking, INTER_LINEAR when growing)
    <out>/<source>/depth_mm/<id>.png     16-bit mm, INTER_NEAREST (no depth mixing across edges)
    <out>/<source>/conf/<id>.png         8-bit, INTER_NEAREST (when a conf map exists)
    <out>/<source>/manifest.csv          id,source,img,depth,conf,is_gt,dmin,dmax,orig_w,orig_h,
                                         label_w,label_h,cover_w,cover_h,scale
    <out>/<source>/build.json            parameters, counts, failures, code hash

Sources
  own (default): ids = every own id in roles.yaml (train, val, frozen test, video test);
                 images from ``paths.own_originals``; teacher labels from ``paths.own_labels``
                 (UniDepthV2-L, inferred at longest side 1280, SAVED at longest side <= 1024 with
                 the original aspect -> checked per row, a mismatch above 1 % fails that row).
  --input-csv:   any rows with columns id,source,img,depth[,conf,is_gt,dmin,dmax] (e.g. the
                 labelled pool), written under <out>/<source>/.

NYU and SUN rows are already 640x480 (4:3), for which the cover transform is the identity, so
distill/v2/data.py reads them from outputs/data_gt directly. Resumable: existing outputs are kept
unless --overwrite. A row failure is recorded and never silently replaced by another image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "outputs", "v2", "cache")
DEFAULT_ROLES = os.path.join(REPO, "configs", "v2", "roles.yaml")
OUT_W, OUT_H = 640, 480
ASPECT_TOL = 0.01
HEADER = ["id", "source", "img", "depth", "conf", "is_gt", "dmin", "dmax", "orig_w", "orig_h",
          "label_w", "label_h", "cover_w", "cover_h", "scale"]


def cover_size(w: int, h: int, out_w: int = OUT_W, out_h: int = OUT_H) -> Tuple[int, int, float]:
    """(cover_w, cover_h, scale) with scale = max(out_w/w, out_h/h); both sides >= the target."""
    s = max(out_w / float(w), out_h / float(h))
    cw = max(out_w, int(round(w * s)))
    ch = max(out_h, int(round(h * s)))
    return cw, ch, s


def resize_image(img: np.ndarray, cw: int, ch: int) -> np.ndarray:
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (cw * ch) < (w * h) else cv2.INTER_LINEAR
    return cv2.resize(img, (cw, ch), interpolation=interp)


def resize_label(lab: np.ndarray, cw: int, ch: int) -> np.ndarray:
    return cv2.resize(lab, (cw, ch), interpolation=cv2.INTER_NEAREST)


def cover_one(row: dict, out_root: str, overwrite: bool = False) -> dict:
    """Cache one row. Returns the output manifest row, or {'id', 'error'} on failure."""
    src = row.get("source", "own")
    base = os.path.join(out_root, src)
    ip = os.path.join(base, "images", row["id"] + ".jpg")
    dp = os.path.join(base, "depth_mm", row["id"] + ".png")
    cp = os.path.join(base, "conf", row["id"] + ".png")
    img = cv2.imread(row["img"], cv2.IMREAD_COLOR)          # same EXIF handling as the teacher run
    if img is None:
        return {"id": row["id"], "error": f"unreadable image {row['img']}"}
    lab = cv2.imread(row["depth"], cv2.IMREAD_UNCHANGED)
    if lab is None or lab.dtype != np.uint16 or lab.ndim != 2:
        return {"id": row["id"], "error": f"unreadable / non-uint16 label {row['depth']}"}
    H, W = img.shape[:2]
    lh, lw = lab.shape
    if abs((W / H) / (lw / lh) - 1.0) > ASPECT_TOL:
        return {"id": row["id"], "error": f"aspect mismatch image {W}x{H} vs label {lw}x{lh}"}
    cw, ch, s = cover_size(W, H)
    have_conf = bool(row.get("conf")) and os.path.exists(row["conf"])
    if overwrite or not (os.path.exists(ip) and os.path.exists(dp) and (os.path.exists(cp) or not have_conf)):
        for d in ("images", "depth_mm", "conf"):
            os.makedirs(os.path.join(base, d), exist_ok=True)
        ok = cv2.imwrite(ip, resize_image(img, cw, ch), [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok &= cv2.imwrite(dp, resize_label(lab, cw, ch))
        if have_conf:
            c8 = cv2.imread(row["conf"], cv2.IMREAD_UNCHANGED)
            if c8 is not None:
                ok &= cv2.imwrite(cp, resize_label(c8, cw, ch))
        if not ok:
            return {"id": row["id"], "error": "cv2.imwrite failed"}
    return {"id": row["id"], "source": src, "img": ip.replace("\\", "/"), "depth": dp.replace("\\", "/"),
            "conf": cp.replace("\\", "/") if os.path.exists(cp) else "",
            "is_gt": row.get("is_gt", "0"), "dmin": row.get("dmin", "0.2"), "dmax": row.get("dmax", "30.0"),
            "orig_w": W, "orig_h": H, "label_w": lw, "label_h": lh, "cover_w": cw, "cover_h": ch,
            "scale": round(s, 6)}


def own_rows_from_roles(roles: dict) -> List[dict]:
    p = roles["paths"]
    ids = set(roles["train"]["own"]) | set(roles["val"]["own"]) | set(roles["test"]["frozen_own"]) \
        | set(roles["test"]["video_own"])
    ids |= set(roles.get("excluded", {}).get("own", {}))           # cache everything own; roles decide use
    rows = []
    for i in sorted(ids):
        rows.append({"id": i, "source": "own", "img": f"{p['own_originals']}/{i}.jpg",
                     "depth": f"{p['own_labels']}/{i}.png", "conf": f"{p['own_conf']}/{i}.png",
                     "is_gt": "0", "dmin": "0.2", "dmax": "30.0"})
    return rows


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(rows: Sequence[dict], out_root: str, workers: int = 8, overwrite: bool = False,
          tag: str = "") -> dict:
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda r: cover_one(r, out_root, overwrite), rows))
    good = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    by_src: Dict[str, List[dict]] = {}
    for r in good:
        by_src.setdefault(r["source"], []).append(r)
    summary = {"n_rows": len(rows), "n_ok": len(good), "n_failed": len(bad), "failures": bad[:50],
               "out_root": out_root.replace("\\", "/"), "target": [OUT_H, OUT_W],
               "rule": "scale = max(640/W, 480/H), aspect kept; image AREA/LINEAR, labels NEAREST",
               "code_sha256": _sha256(os.path.abspath(__file__)), "tag": tag, "manifests": {}}
    for src, rs in by_src.items():
        base = os.path.join(out_root, src)
        os.makedirs(base, exist_ok=True)
        name = "manifest.csv" if not tag else f"manifest_{tag}.csv"
        mpath = os.path.join(base, name)
        # merge with an existing manifest so partial builds accumulate
        merged: Dict[str, dict] = {}
        if os.path.exists(mpath):
            with open(mpath, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    merged[r["id"]] = r
        for r in rs:
            merged[r["id"]] = r
        tmp = mpath + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            for k in sorted(merged):
                w.writerow({h: merged[k].get(h, "") for h in HEADER})
        os.replace(tmp, mpath)
        aspects = np.array([int(r["cover_w"]) / int(r["cover_h"]) for r in merged.values()])
        summary["manifests"][src] = {"path": mpath.replace("\\", "/"), "rows": len(merged),
                                     "sha256": _sha256(mpath),
                                     "cover_aspect_min_max": [round(float(aspects.min()), 4),
                                                              round(float(aspects.max()), 4)]}
        with open(os.path.join(base, "build.json" if not tag else f"build_{tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="V2 aspect-preserving cover cache")
    ap.add_argument("--roles", default=DEFAULT_ROLES)
    ap.add_argument("--source", default="own", choices=["own"])
    ap.add_argument("--input-csv", default="", help="generic rows (id,source,img,depth[,conf,...])")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="", help="write manifest_<tag>.csv (e.g. 'smoke') instead of manifest.csv")
    args = ap.parse_args(argv)
    if args.input_csv:
        with open(args.input_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        from scripts.v2.build_roles import load_roles
        rows = own_rows_from_roles(load_roles(args.roles))
    if args.limit:
        rows = rows[:args.limit]
    s = build(rows, args.out, args.workers, args.overwrite, args.tag)
    print(json.dumps({k: v for k, v in s.items() if k != "failures"}, indent=1))
    if s["n_failed"]:
        print(f"[build_cover_cache] {s['n_failed']} rows FAILED, first: {s['failures'][:3]}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
