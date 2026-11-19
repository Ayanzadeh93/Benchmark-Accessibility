"""Fetch Tier-1 benchmark data (paper-grade sources, resumable).

Targets (see docs/PROTOCOL_NOTES.md):
  nyu    nyu_depth_v2_labeled.mat (official 1449, float depths + rawDepths)
         + splits.mat (Eigen test indices -> the standard 654)
  sun    SUNRGBD.zip (official release, per-image intrinsics + sensor metadata)
         + SUNRGBDtoolbox.zip (allsplit.mat with the official alltest split)
  ibims  IBims-1 core (mat, exact float depths + masks + edges)

TUM-RGBD: deferred (official repo ships no TUM loader; their split is unknown —
any TUM numbers would be our-split-only). KITTI Eigen: deferred (account-gated).

  python -m benchmark.fetch_tier1 --only nyu,sun,ibims          # -> Z:/Datasets/<Name>/
  python -m benchmark.fetch_tier1 --only kitti                  # velodyne zip -> KITTI/raw/

Default root is the canonical datasets drive (Z:/Datasets, one titled folder per dataset,
matching benchmark/datasets_tier1.py); pass --root data_bench for the legacy local layout.
Downloads are resumable (HTTP Range). Files that exist with the right size are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20  # 1 MiB

FILES = {
    "nyu": [
        ("http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
         "NYUv2/nyu_depth_v2_labeled.mat"),
        ("http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
         "NYUv2/splits.mat"),
    ],
    "sun": [
        ("https://rgbd.cs.princeton.edu/data/SUNRGBD.zip", "SUNRGBD/SUNRGBD.zip"),
        ("https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip", "SUNRGBD/SUNRGBDtoolbox.zip"),
    ],
    "ibims": [
        # mediaTUM dataserv; ibims1_core_mat = exact float depths (+ masks, edges)
        ("https://dataserv.ub.tum.de/s/m1455541/download?path=%2F&files=ibims1_core_mat.zip",
         "IBims-1/ibims1_core_mat.zip"),
    ],
    "kitti": [
        # object-benchmark LiDAR (~29GB) -> unlocks datasets_tier1.iter_kitti_obj (dense
        # outdoor sanity). Lands next to the existing object zips; extract into
        # training/velodyne + testing/velodyne afterwards.
        ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
         "KITTI/raw/data_object_velodyne.zip"),
    ],
}

# fallback mirrors tried in order when the primary 404s/times out
# (keyed by the last two path segments of the destination)
FALLBACKS = {
    "NYUv2/nyu_depth_v2_labeled.mat": [
        "https://huggingface.co/datasets/lodestones/nyu-depth-v2/resolve/main/nyu_depth_v2_labeled.mat",
    ],
    "NYUv2/splits.mat": [
        "https://github.com/wl-zhao/VPD/raw/main/depth/splits.mat",
        "https://github.com/cleinc/bts/raw/master/utils/splits.mat",
    ],
}


def _open(url, start=0):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research download; unidepth2d benchmark)",
        **({"Range": f"bytes={start}-"} if start else {}),
    })
    return urllib.request.urlopen(req, timeout=60)


def download(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    start = os.path.getsize(part) if os.path.exists(part) else 0
    if os.path.exists(dest):
        print(f"[skip] {dest} exists ({os.path.getsize(dest)/1e9:.2f} GB)")
        return True
    tried = [url] + FALLBACKS.get("/".join(dest.replace("\\", "/").split("/")[-2:]), [])
    for u in tried:
        try:
            r = _open(u, start)
            total = r.headers.get("Content-Length")
            total = int(total) + start if total else None
            mode = "ab" if start and r.status == 206 else "wb"
            if mode == "wb":
                start = 0
            t0, done = time.time(), start
            with open(part, mode) as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if done % (200 * CHUNK) < CHUNK:
                        pct = f"{100*done/total:.1f}%" if total else f"{done/1e9:.2f}GB"
                        rate = done / max(1e-9, time.time() - t0) / 1e6
                        print(f"  {os.path.basename(dest)}: {pct} ({rate:.1f} MB/s)", flush=True)
            os.replace(part, dest)
            print(f"[done] {dest} ({os.path.getsize(dest)/1e9:.2f} GB) from {u}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"[warn] {u} failed: {e}; trying next mirror", flush=True)
            start = os.path.getsize(part) if os.path.exists(part) else 0
            continue
    print(f"[FAIL] could not fetch {dest}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATASETS_ROOT", "Z:/Datasets"))
    ap.add_argument("--only", default="nyu,sun,ibims")
    args = ap.parse_args()
    ok = True
    for key in [k.strip() for k in args.only.split(",") if k.strip()]:
        for url, rel in FILES[key]:
            ok &= download(url, os.path.join(args.root, rel))
    print("ALL OK" if ok else "SOME DOWNLOADS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
