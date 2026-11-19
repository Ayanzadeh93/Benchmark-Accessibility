"""Candidate pool manifest (MIT Indoor67, extra indoor stills, indoor video frames) with dedupe
and the V2 leak blocker.

Contract
--------
CLI (env ``depth``; CPU only)::

    python -m scripts.v2.build_pool [--roles configs/v2/roles.yaml]
        [--indoor67 Z:/Datasets/MITIndoor67] [--stills DIR ...] [--video FILE ...] [--fps 1]
        [--out outputs/v2/pool] [--min-side 300] [--workers 16] [--require-benchmarks]

Output ``<out>/manifest.csv`` has one row per candidate::

    id,pool,img,src,w,h,sha1,keep,reason,ham,ham_ref,ncc,ncc_ref

``keep == 1`` rows go to teacher labelling (scripts/v2/label_*.py) and are later imported into
roles.yaml with ``python -m scripts.v2.build_roles --pool-manifest <out>/manifest.csv``.
``<out>/build.json`` holds counts per pool and per drop reason, parameters and hashes.

Pools
  x_in67     every image under --indoor67 (category = parent folder name), the A1 replacement
             for the unusable 160 px ImageNet-100 copy.
  x_indoor   --stills folders (recursive) and --video files decoded at --fps (frames written to
             <out>/frames/<video>/ at longest side <= 1920). The SF6 videos recorded in
             roles.yaml (Dollar_Tree, Airport_BWI; amendment A21) are refused by path and SHA-256.

Drop reasons, checked in this order (first hit wins):
  unreadable, min_side (< --min-side), sha1_dup (exact duplicate inside the pool),
  phash_dup (pHash Hamming <= 4 of an earlier pool row, sorted-id order),
  own_dup (SHA-1 or pHash <= 4 of any own row: redundant with the existing corpus),
  leak_<ref> (A3 rule: crop-aligned pHash <= 12 OR thumbnail NCC >= 0.85) for references
  d_sun, own_heldout (frozen + video test), local_nyu, sf6, and every benchmark hashed by
  build_roles (bench_nyu_labelled, bench_ibims1).
A missing reference (e.g. benchmarks unreachable when roles were built) is reported loudly in
build.json and, with --require-benchmarks, is fatal.
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

from scripts.v2.build_roles import (DEFAULT_OUT as DEFAULT_ROLES, HAM_CROSS, HAM_DUP, NCC_CROSS,
                                    FeatureSet, cross_match, image_features, load_roles, sha1_file,
                                    sha256_file)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_POOL_OUT = os.path.join(REPO, "outputs", "v2", "pool")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HEADER = ["id", "pool", "img", "src", "w", "h", "sha1", "keep", "reason", "ham", "ham_ref", "ncc", "ncc_ref"]


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def enumerate_images(root: str, pool: str, prefix: str) -> List[dict]:
    """Recursive image listing -> candidate rows (id = <prefix>_<parent>_<stem>)."""
    if not os.path.isdir(root):
        raise SystemExit(f"[build_pool] source folder not found: {root} "
                         "(Z: may be disconnected; nothing is substituted)")
    out = []
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if f.lower().endswith(IMG_EXTS):
                p = os.path.join(dp, f)
                parent = os.path.basename(dp)
                out.append({"id": _safe(f"{prefix}_{parent}_{os.path.splitext(f)[0]}"), "pool": pool,
                            "img": p.replace("\\", "/"), "src": p.replace("\\", "/")})
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):          # disambiguate collisions deterministically
        seen: Dict[str, int] = {}
        for r in out:
            k = seen.get(r["id"], 0)
            seen[r["id"]] = k + 1
            if k:
                r["id"] = f"{r['id']}__{k}"
    return out


def extract_video(path: str, out_dir: str, fps_out: float = 1.0, max_side: int = 1920) -> List[dict]:
    """Decode ``path`` and write one JPEG per 1/fps_out seconds. Resumable (existing kept)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[build_pool] cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / fps_out)))
    name = _safe(os.path.splitext(os.path.basename(path))[0])
    vdir = os.path.join(out_dir, "frames", name)
    os.makedirs(vdir, exist_ok=True)
    rows, k = [], 0
    while True:
        if not cap.grab():
            break
        if k % step == 0:
            fid = f"vid_{name}_t{k / fps:07.1f}".replace(".", "p")
            fp = os.path.join(vdir, fid + ".jpg")
            if not os.path.exists(fp):
                ok, fr = cap.retrieve()
                if not ok or fr is None:
                    k += 1
                    continue
                h, w = fr.shape[:2]
                s = min(1.0, max_side / float(max(h, w)))
                if s < 1.0:
                    fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(fp, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"id": fid, "pool": "x_indoor", "img": fp.replace("\\", "/"),
                         "src": f"{path}#t={k / fps:.2f}".replace("\\", "/")})
        k += 1
    cap.release()
    return rows


def _feat_row(r: dict) -> Tuple[Optional[dict], int, int, str]:
    img = cv2.imread(r["img"], cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, 0, ""
    h, w = img.shape[:2]
    return image_features(img), w, h, sha1_file(r["img"])


def load_references(roles: dict) -> Tuple[Dict[str, FeatureSet], Dict[str, Optional[FeatureSet]], List[str]]:
    """Reference collections from the build_roles feature cache.

    Returns (dup_refs, leak_refs, missing): ``dup_refs['own']`` = all own rows (redundancy check);
    ``leak_refs`` = name -> FeatureSet (None when the cache for it is absent).
    """
    fdir = os.path.join(roles["paths"]["evidence"], "features")
    if not os.path.isdir(fdir):
        raise SystemExit(f"[build_pool] feature cache missing: {fdir} (run scripts.v2.build_roles first)")

    def ld(name: str) -> Optional[FeatureSet]:
        p = os.path.join(fdir, name + ".npz")
        return FeatureSet.load(p)[0] if os.path.exists(p) else None

    own, nyu, sun = ld("own"), ld("nyu"), ld("sun")
    if own is None or nyu is None or sun is None:
        raise SystemExit(f"[build_pool] own/nyu/sun feature caches missing in {fdir}")
    leak: Dict[str, Optional[FeatureSet]] = {
        "d_sun": sun.subset(roles["dev"]["d_sun"]),
        "own_heldout": own.subset(set(roles["test"]["frozen_own"]) | set(roles["test"]["video_own"])),
        "local_nyu": nyu,
    }
    sf6 = [ld(f"sf6_{n}") for n in roles.get("sf6_temporal", {}).get("videos", {})]
    sf6 = [s for s in sf6 if s is not None]
    if sf6:
        leak["sf6"] = FeatureSet(sum((s.ids for s in sf6), []), np.concatenate([s.ph for s in sf6]),
                                 np.concatenate([s.phc for s in sf6]), np.concatenate([s.th for s in sf6]),
                                 np.concatenate([s.thc for s in sf6]), np.concatenate([s.luma for s in sf6]))
    else:
        leak["sf6"] = None
    for b in ("nyu_labelled", "ibims1"):
        leak[f"bench_{b}"] = ld(f"bench_{b}")
    missing = [k for k, v in leak.items() if v is None]
    return {"own": own}, leak, missing


def decide(cands: List[dict], feats: FeatureSet, sizes: Dict[str, Tuple[int, int]],
           dup_refs: Dict[str, FeatureSet], leak_refs: Dict[str, Optional[FeatureSet]],
           min_side: int = 300) -> List[dict]:
    """Pure decision step (no IO): returns manifest rows with keep/reason. ``feats`` must hold the
    readable candidates (ids subset of ``cands``); others are 'unreadable'."""
    fidx = {i: k for k, i in enumerate(feats.ids)}
    out: Dict[str, dict] = {}
    for r in cands:
        w, h = sizes.get(r["id"], (0, 0))
        row = {**r, "w": w, "h": h, "sha1": "", "keep": 0, "reason": "", "ham": "", "ham_ref": "",
               "ncc": "", "ncc_ref": ""}
        if r["id"] not in fidx:
            row["reason"] = "unreadable"
        elif min(w, h) < min_side:
            row["reason"] = "min_side"
        else:
            row["sha1"] = feats.sha1[fidx[r["id"]]] if feats.sha1 else ""
        out[r["id"]] = row
    order = sorted(i for i, r in out.items() if not r["reason"])
    # in-pool exact and near duplicates (deterministic: first in sorted-id order is kept)
    seen_sha: Dict[str, str] = {}
    kept_idx: List[int] = []
    for i in order:
        r = out[i]
        if r["sha1"] and r["sha1"] in seen_sha:
            r["reason"], r["ham_ref"] = "sha1_dup", seen_sha[r["sha1"]]
            continue
        seen_sha[r["sha1"]] = i
        k = fidx[i]
        if kept_idx:
            from scripts.v2.build_roles import hamming
            d = hamming(feats.ph[k][None, :], feats.ph[kept_idx])
            j = int(d.argmin())
            if d[j] <= HAM_DUP:
                r["reason"], r["ham"], r["ham_ref"] = "phash_dup", int(d[j]), feats.ids[kept_idx[j]]
                continue
        kept_idx.append(k)
    live = [i for i in order if not out[i]["reason"]]
    q = feats.subset(live)
    # redundancy against the existing own corpus
    if live and "own" in dup_refs:
        own = dup_refs["own"]
        own_sha = set(own.sha1)
        m = cross_match(q, own, "raw", ham_th=HAM_DUP, ncc_th=-2)
        for k, i in enumerate(q.ids):
            r = out[i]
            if r["sha1"] in own_sha or m["hit"][k]:
                r["reason"], r["ham"], r["ham_ref"] = "own_dup", int(m["ham"][k]), own.ids[m["ham_ref"][k]]
    # leak blocker (A3 rule) against every available reference
    for name, ref in leak_refs.items():
        live = [i for i in q.ids if not out[i]["reason"]]
        if not live or ref is None or len(ref) == 0:
            continue
        qq = q.subset(live)
        m = cross_match(qq, ref, "symmetric", ham_th=HAM_CROSS, ncc_th=NCC_CROSS)
        for k, i in enumerate(qq.ids):
            if m["hit"][k]:
                r = out[i]
                r.update(reason=f"leak_{name}", ham=int(m["ham"][k]), ham_ref=ref.ids[m["ham_ref"][k]],
                         ncc=round(float(m["ncc"][k]), 4), ncc_ref=ref.ids[m["ncc_ref"][k]])
    for r in out.values():
        r["keep"] = int(not r["reason"])
    return [out[r["id"]] for r in cands]


def refuse_sf6(videos: Sequence[str], roles: dict) -> List[str]:
    """Drop SF6 hold-out videos (A21) by path or SHA-256, loudly."""
    sf6 = roles.get("sf6_temporal", {}).get("videos", {})
    paths = {os.path.normcase(os.path.abspath(v["path"])) for v in sf6.values()}
    shas = {v.get("sha256") for v in sf6.values()}
    keep = []
    for v in videos:
        if os.path.normcase(os.path.abspath(v)) in paths or sha256_file(v) in shas:
            print(f"[build_pool] REFUSED SF6 hold-out video (A21): {v}", file=sys.stderr)
            continue
        keep.append(v)
    return keep


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="V2 candidate pool with dedupe and leak blocker")
    ap.add_argument("--roles", default=DEFAULT_ROLES)
    ap.add_argument("--indoor67", default="")
    ap.add_argument("--stills", nargs="*", default=[])
    ap.add_argument("--video", nargs="*", default=[])
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", default=DEFAULT_POOL_OUT)
    ap.add_argument("--min-side", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--require-benchmarks", action="store_true")
    args = ap.parse_args(argv)

    roles = load_roles(args.roles)
    dup_refs, leak_refs, missing = load_references(roles)
    if missing:
        msg = f"[build_pool] WARNING: leak references unavailable: {missing}"
        if args.require_benchmarks and any(m.startswith("bench_") for m in missing):
            raise SystemExit(msg + " (--require-benchmarks)")
        print(msg, file=sys.stderr, flush=True)
    cands: List[dict] = []
    if args.indoor67:
        cands += enumerate_images(args.indoor67, "x_in67", "in67")
    for d in args.stills:
        cands += enumerate_images(d, "x_indoor", "still_" + _safe(os.path.basename(os.path.normpath(d))))
    for v in refuse_sf6(args.video, roles):
        cands += extract_video(v, args.out, args.fps)
    if not cands:
        raise SystemExit("[build_pool] no candidates (pass --indoor67, --stills or --video)")
    ids = [c["id"] for c in cands]
    if len(ids) != len(set(ids)):
        raise SystemExit("[build_pool] duplicate candidate ids across sources")

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(_feat_row, cands))
    ok = [(c, r) for c, r in zip(cands, res) if r[0] is not None]
    sizes = {c["id"]: (r[1], r[2]) for c, r in ok}
    feats = FeatureSet.from_feature_dicts([c["id"] for c, _ in ok], [r[0] for _, r in ok],
                                          [r[3] for _, r in ok])
    rows = decide(cands, feats, sizes, dup_refs, leak_refs, args.min_side)

    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.csv")
    tmp = mpath + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in HEADER})
    os.replace(tmp, mpath)
    reasons: Dict[str, Dict[str, int]] = {}
    for r in rows:
        reasons.setdefault(r["pool"], {}).setdefault(r["reason"] or "kept", 0)
        reasons[r["pool"]][r["reason"] or "kept"] += 1
    mins = sorted(min(s) for s in sizes.values())
    summary = {"manifest": mpath.replace("\\", "/"), "manifest_sha256": sha256_file(mpath),
               "n_candidates": len(rows), "n_kept": sum(r["keep"] for r in rows), "by_pool": reasons,
               "min_side_px": args.min_side,
               "min_side_quantiles_p1_p50_p99": ([mins[int(.01 * len(mins))], mins[len(mins) // 2],
                                                   mins[int(.99 * (len(mins) - 1))]] if mins else None),
               "leak_refs_used": sorted(k for k, v in leak_refs.items() if v is not None),
               "leak_refs_missing": missing,
               "roles_sha256": sha256_file(args.roles),
               "code_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               "params": {"ham_dup": HAM_DUP, "ham_cross": HAM_CROSS, "ncc_cross": NCC_CROSS, "fps": args.fps}}
    with open(os.path.join(args.out, "build.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
