"""Fetch the V2 zero-shot benchmark sets and the MIT Indoor67 training pool (resumable).

Contract
--------
* ``SETS`` maps a set key to its exact remote files (URL, destination relative to the datasets
  root, byte size). Every direct URL below was checked on 2026-11-19 with an HTTP HEAD (and a
  1 KiB range request where noted): status 200, ``Accept-Ranges: bytes``, and the listed
  Content-Length. Sizes are exact bytes, not estimates.
* Destination root: ``--root`` > ``$DATASETS_ROOT`` > ``Z:/Datasets`` (one titled folder per set,
  the same layout as benchmark/fetch_tier1.py). If the root's drive is missing the script stops
  with a clear message; it never falls back to another location.
* ``download()`` resumes ``<dest>.part`` with HTTP Range, refuses a 200-instead-of-206 resume
  silently (restarts from 0 and says so), checks the final byte count against ``Remote.size``
  and only then renames ``.part`` to the final name. An existing final file with the wrong size
  is reported as ``BAD-SIZE`` and is not overwritten unless ``--redownload`` is passed. The same
  ``.part`` naming as fetch_tier1 is used, so a transfer started by either tool can be resumed.
* ``--dry-run`` (the default when no action flag is given) lists every file with expected and
  present sizes and the total still to fetch. Nothing is written.
* ``--extract`` unpacks into the set folder with path-traversal checks; DIODE keeps only
  ``val/indoors/`` members. ETH3D ``.7z`` needs ``7z``/``7za`` on PATH, the ``py7zr`` package or
  the Windows 11 ``tar.exe`` (bsdtar/libarchive 3.8 reads 7z); the first available is used.
* ``LINEAGE`` records, per evaluation set, whether it appears in the published training lists of
  UniDepthV2 (docs/PROTOCOL_NOTES.md), MoGe-2 (arXiv 2507.02546 Table A.1) and YOLO26-Depth
  (Ultralytics docs, research report in outputs/v2/design_workflow_result.json). It is data for
  benchmark/policy_check.py; this module draws no fairness conclusion itself.

Sets (``--only`` keys)
----------------------
diode_indoor   DIODE val (indoor + outdoor archive, 2,774,625,282 B); extract keeps 325 indoor
eth3d_indoor   ETH3D high-res multi-view TRAINING, the 7 indoor scenes (219 images). Depth maps
               match the ORIGINAL (distorted) DSLR JPGs, not the undistorted ones (ETH3D
               documentation), so the ``*_dslr_jpg.7z`` archives are fetched with
               ``*_dslr_depth.7z``. The undistorted archives are listed in ETH3D_UNDISTORTED
               for reference only.
tum            TUM RGB-D fr1/desk + fr3/long_office_household (plan minimum).
tum_fr2_desk   optional third TUM sequence (1.89 GB), not part of the plan minimum.
diml_indoor    DIML/CVL indoor TEST. MANUAL: hosted only as a Naver MYBOX share (JavaScript app,
               no stable direct URL), so this script cannot fetch or size-check it; it prints
               the instructions and validates the extracted layout instead.
indoor67       MIT Indoor67 (training pool only, never a benchmark). The official csail URL
               returns 404 (checked 2026-11-19); the byte-identical original is served by the
               Internet Archive (range request returned a tar header 'Images/airport_inside/').

Usage::

  python -m benchmark.fetch_v2                                   # dry run of every set
  python -m benchmark.fetch_v2 --only tum,eth3d_indoor --dry-run
  python -m benchmark.fetch_v2 --only tum --download             # resumable, size-checked
  python -m benchmark.fetch_v2 --only tum --extract
  python -m benchmark.fetch_v2 --check-remote                    # re-HEAD every URL now
  python -m benchmark.fetch_v2 --lineage                         # print the lineage table

Tests: tests/test_datasets_v2.py (env depth or depth_cmp; offline, CPU).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

CHUNK = 1 << 20  # 1 MiB
UA = "Mozilla/5.0 (research dataset download)"
VERIFIED_ON = "2026-11-19"


@dataclass(frozen=True)
class Remote:
    url: str
    rel: str            # destination, relative to the datasets root (forward slashes)
    size: int           # exact bytes from HEAD on VERIFIED_ON
    note: str = ""


ETH3D_INDOOR_SCENES = ("delivery_area", "kicker", "office", "pipes", "relief", "relief_2",
                       "terrains")
# images per scene from https://www.eth3d.net/datasets (219 in total)
ETH3D_INDOOR_COUNTS = {"delivery_area": 44, "kicker": 31, "office": 26, "pipes": 14,
                       "relief": 31, "relief_2": 31, "terrains": 42}
_ETH3D_JPG = {"delivery_area": 372958192, "kicker": 408604363, "office": 231734054,
              "pipes": 109668393, "relief": 287532557, "relief_2": 286463962,
              "terrains": 357963793}
_ETH3D_DEPTH = {"delivery_area": 388144128, "kicker": 415014798, "office": 441352937,
                "pipes": 127725253, "relief": 686900728, "relief_2": 646673138,
                "terrains": 529179696}
# reference only: undistorted images do NOT match the GT depth maps (see module docstring)
ETH3D_UNDISTORTED = {"delivery_area": 478076254, "kicker": 433074686, "office": 288765137,
                     "pipes": 145321540, "relief": 360586534, "relief_2": 358562735,
                     "terrains": 461761123}

DIML_SHARE_URL = ("https://mybox.naver.com/share/list?shareKey="
                  "0kC6LSsKNBKT4i6JoOP3TLkGoQgtiuM38m3jMYa45CIB")   # http://naver.me/xsYr9yJ4

SETS: Dict[str, dict] = {
    "diode_indoor": {
        "folder": "DIODE",
        "files": [Remote("http://diode-dataset.s3.amazonaws.com/val.tar.gz", "DIODE/val.tar.gz",
                         2774625282, "ETag a6e107704c4512441f2a65878e0e84b8-331; "
                         "gzip magic 1f8b checked by range request")],
        "extract": {"kind": "tar", "archive": "DIODE/val.tar.gz", "prefix": "val/indoors/",
                    "dest": "DIODE"},
        "expect": "DIODE/val/indoors/scene_*/scan_*/*.png (+ _depth.npy, _depth_mask.npy); 325 images",
        "licence": "MIT", "access": "direct",
    },
    "eth3d_indoor": {
        "folder": "ETH3D",
        "files": ([Remote(f"https://www.eth3d.net/data/{s}_dslr_jpg.7z",
                          f"ETH3D/{s}_dslr_jpg.7z", _ETH3D_JPG[s], "distorted originals")
                   for s in ETH3D_INDOOR_SCENES]
                  + [Remote(f"https://www.eth3d.net/data/{s}_dslr_depth.7z",
                            f"ETH3D/{s}_dslr_depth.7z", _ETH3D_DEPTH[s], "GT depth, float32 dumps")
                     for s in ETH3D_INDOOR_SCENES]),
        "extract": {"kind": "7z", "dest": "ETH3D"},
        "expect": "ETH3D/<scene>/images/dslr_images/*.JPG + ETH3D/<scene>/ground_truth*/dslr_images/*.JPG",
        "licence": "not stated on eth3d.net (Ultralytics lists CC BY-NC-SA 4.0)", "access": "direct",
    },
    "tum": {
        "folder": "TUM-RGBD",
        "files": [
            Remote("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
                   "TUM-RGBD/rgbd_dataset_freiburg1_desk.tgz", 344011403,
                   "302 -> webshare.cvg.cit.tum.de (followed)"),
            Remote("https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
                   "rgbd_dataset_freiburg3_long_office_household.tgz",
                   "TUM-RGBD/rgbd_dataset_freiburg3_long_office_household.tgz", 1483556251,
                   "302 -> webshare.cvg.cit.tum.de (followed)"),
        ],
        "extract": {"kind": "tar", "dest": "TUM-RGBD"},
        "expect": "TUM-RGBD/rgbd_dataset_freiburg*/{rgb.txt,depth.txt,groundtruth.txt,rgb/,depth/}",
        "licence": "CC BY 4.0", "access": "direct",
    },
    "tum_fr2_desk": {
        "folder": "TUM-RGBD",
        "files": [Remote("https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
                         "TUM-RGBD/rgbd_dataset_freiburg2_desk.tgz", 1893351095, "optional")],
        "extract": {"kind": "tar", "dest": "TUM-RGBD"},
        "expect": "TUM-RGBD/rgbd_dataset_freiburg2_desk/",
        "licence": "CC BY 4.0", "access": "direct",
    },
    "diml_indoor": {
        "folder": "DIML-CVL",
        "files": [],
        "manual": (
            "DIML/CVL indoor TEST is distributed only through a Naver MYBOX share "
            f"({DIML_SHARE_URL}, short link http://naver.me/xsYr9yJ4 on dimlrgbd.github.io). "
            "The share is a JavaScript web app with no stable direct file URL, so it cannot be "
            "fetched or size-checked by script. The project page states 847 MB for the HR test "
            "split and 208 MB for LR (an independent search summary said 3.5 GB / 610 MB; the "
            "size is therefore UNVERIFIED). Download the indoor test archive in a browser into "
            "<root>/DIML-CVL/, extract it so that <root>/DIML-CVL/**/HR/<scene>/color/*_c.png "
            "and .../depth_filled/*_depth_filled.png exist, then run "
            "`python -m benchmark.fetch_v2 --only diml_indoor --verify`."),
        "extract": {"kind": "zip-any", "dest": "DIML-CVL"},
        "expect": "DIML-CVL/**/HR/<scene>/{color/*_c.png, depth_filled/*_depth_filled.png, depth_raw/}",
        "licence": "authors' copyright; attribution, non-commercial, share-alike", "access": "manual",
    },
    "indoor67": {
        "folder": "MITIndoor67",
        "files": [
            Remote("https://web.archive.org/web/20210422054918id_/"
                   "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar",
                   "MITIndoor67/indoorCVPR_09.tar", 2592010240,
                   "official csail URL 404 on 2026-11-19; Internet Archive raw snapshot, "
                   "x-archive-orig-content-length 2592010240, range request -> 206 + tar header"),
            Remote("https://web.mit.edu/torralba/www/TrainImages.txt",
                   "MITIndoor67/TrainImages.txt", 172080, "official split list"),
            Remote("https://web.mit.edu/torralba/www/TestImages.txt",
                   "MITIndoor67/TestImages.txt", 43475, "official split list"),
        ],
        "extract": {"kind": "tar", "archive": "MITIndoor67/indoorCVPR_09.tar", "dest": "MITIndoor67"},
        "expect": "MITIndoor67/Images/<category>/*.jpg (15,620 images, 67 categories)",
        "licence": "research purposes only (MIT page); training pool, never a benchmark",
        "access": "direct",
    },
}

# Published-training-list lineage of each evaluation set. Values:
#   "absent"   not in the model's published training list
#   "present"  listed as a training source
#   "pooled"   not listed itself, but a listed source contains its images
#   "unlisted" neither listed nor named as an evaluation set (absence from a list, not a promise)
# "evidence" says where each statement comes from. MoGe-2 entries come from the paper table only;
# the released checkpoint's lineage is not independently verifiable, so policy_check should keep
# the plan's "lineage unverified" qualifier for MoGe-2 rows (plan 9.1 matrix).
LINEAGE: Dict[str, Dict[str, str]] = {
    "diode_indoor": {"unidepthv2": "absent", "moge2": "absent (evaluation set)",
                     "yolo26_depth": "present", "role_v2": "report-only zero-shot"},
    "eth3d_indoor": {"unidepthv2": "absent (declared zero-shot)", "moge2": "absent (evaluation set)",
                     "yolo26_depth": "absent (declared zero-shot)", "role_v2": "report-only zero-shot"},
    "tum": {"unidepthv2": "absent (declared zero-shot; TUM is in its Table I)", "moge2": "unlisted",
            "yolo26_depth": "unlisted", "role_v2": "report-only zero-shot"},
    "diml_indoor": {"unidepthv2": "unlisted (a DIML loader exists in UniDepth-main, but DIML is not "
                                  "in the 23-dataset V2 list)",
                    "moge2": "unlisted", "yolo26_depth": "unlisted",
                    "role_v2": "report-only zero-shot (or dev upgrade, owner decision)"},
    "ibims": {"unidepthv2": "absent (declared zero-shot)", "moge2": "absent (evaluation set)",
              "yolo26_depth": "absent (declared zero-shot)", "role_v2": "report-only zero-shot"},
    "nyu": {"unidepthv2": "absent (declared zero-shot)", "moge2": "absent (evaluation set)",
            "yolo26_depth": "pooled (SUN RGB-D is a training source and contains NYU frames)",
            "role_v2": "train (keep-NYU branch)"},
    "sun": {"unidepthv2": "absent (declared zero-shot)", "moge2": "unlisted",
            "yolo26_depth": "present", "role_v2": "development (D_SUN), never zero-shot"},
    "indoor67": {"unidepthv2": "unlisted", "moge2": "unlisted", "yolo26_depth": "unlisted",
                 "role_v2": "training pool (never a benchmark)"},
}
LINEAGE_EVIDENCE = {
    "unidepthv2": "docs/PROTOCOL_NOTES.md section 1 (arXiv 2502.20110v2 Sec. IV-A: 23 training "
                  "datasets; NOT in training: NYU, KITTI, SUN RGB-D, IBims-1, TUM-RGBD, ETH3D, ...)",
    "moge2": "arXiv 2507.02546 Table A.1 (24 training sets: A2D2, Argoverse2, ARKitScenes, "
             "BlendedMVS, MegaDepth, ScanNet++, Taskonomy, Waymo + 16 synthetic); evaluation sets "
             "NYUv2, KITTI, ETH3D, iBims-1, GSO, Sintel, DDAD, DIODE, Spring, HAMMER. Read through "
             "an automated HTML reader on 2026-11-19; checkpoint lineage itself unverified.",
    "yolo26_depth": "Ultralytics docs via outputs/v2/design_workflow_result.json research[0]: "
                    "training mix ARKitScenes, SUN RGB-D, DIODE, Hypersim, TartanAir, Virtual "
                    "KITTI 2, KITTI, pseudo-labelled ImageNet; NYU, iBims-1, ETH3D, Make3D declared "
                    "zero-shot; SUN RGB-D pools NYU scenes, so NYU is not clean for it.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Root resolution (fail loudly when the drive is missing)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_root(root: Optional[str] = None, must_exist: bool = True) -> str:
    """--root > $DATASETS_ROOT > Z:/Datasets. Raises if its drive/parent is unreachable."""
    r = root or os.environ.get("DATASETS_ROOT") or "Z:/Datasets"
    drive = os.path.splitdrive(os.path.abspath(r))[0]
    if drive and not os.path.exists(drive + os.sep):
        raise SystemExit(f"datasets root {r!r}: drive {drive} is not mounted. Reconnect it or set "
                         "DATASETS_ROOT / --root explicitly; no fallback location is used.")
    if must_exist and not os.path.isdir(r):
        raise SystemExit(f"datasets root {r!r} does not exist (create it or pass --root)")
    return r


def _dest(root: str, rel: str) -> str:
    return os.path.join(root, *rel.split("/"))


# ──────────────────────────────────────────────────────────────────────────────
# Listing / dry run
# ──────────────────────────────────────────────────────────────────────────────

def file_status(root: str, rf: Remote) -> dict:
    d = _dest(root, rf.rel)
    part = d + ".part"
    have = os.path.getsize(d) if os.path.exists(d) else None
    have_part = os.path.getsize(part) if os.path.exists(part) else 0
    if have is not None:
        status = "OK" if have == rf.size else "BAD-SIZE"
    elif have_part:
        status = "PARTIAL"
    else:
        status = "MISSING"
    return {"url": rf.url, "dest": d, "expected": rf.size, "present": have, "part": have_part,
            "status": status}


def dry_run(names: Iterable[str], root: str) -> List[dict]:
    rows = []
    for n in names:
        s = SETS[n]
        if s.get("access") == "manual":
            rows.append({"set": n, "url": None, "dest": _dest(root, s["folder"]), "expected": None,
                         "present": None, "part": 0, "status": "MANUAL"})
            continue
        for rf in s["files"]:
            rows.append({"set": n, **file_status(root, rf)})
    return rows


def _fmt_bytes(b: Optional[int]) -> str:
    return "-" if b is None else f"{b / 1e9:.3f} GB"


def print_dry_run(rows: List[dict]) -> int:
    todo = 0
    for r in rows:
        print(f"[{r['status']:8s}] {r['set']:13s} {_fmt_bytes(r['expected']):>10s}  {r['dest']}")
        if r["url"]:
            print(f"           <- {r['url']}")
        if r["status"] in ("MISSING", "PARTIAL") and r["expected"]:
            todo += r["expected"] - (r["part"] or 0)
    print(f"remaining to download: {todo / 1e9:.3f} GB ({todo} bytes)")
    return todo


# ──────────────────────────────────────────────────────────────────────────────
# Remote check / download
# ──────────────────────────────────────────────────────────────────────────────

def _open(url: str, start: int = 0, method: str = "GET", timeout: int = 60):
    headers = {"User-Agent": UA}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def head(url: str) -> dict:
    """HEAD (redirects followed). Returns {status, size, ranges, final_url}."""
    with _open(url, method="HEAD", timeout=60) as r:
        n = r.headers.get("Content-Length")
        return {"status": r.status, "size": int(n) if n else None,
                "ranges": r.headers.get("Accept-Ranges"), "final_url": r.geturl()}


def check_remote(names: Iterable[str]) -> bool:
    ok = True
    for n in names:
        for rf in SETS[n]["files"]:
            try:
                h = head(rf.url)
                good = h["status"] == 200 and h["size"] == rf.size
                print(f"[{'OK' if good else 'MISMATCH'}] {rf.rel}: HTTP {h['status']} size {h['size']} "
                      f"(expected {rf.size}) ranges={h['ranges']}")
                ok &= good
            except (urllib.error.URLError, OSError) as e:
                print(f"[FAIL] {rf.rel}: {e}")
                ok = False
    return ok


def download(rf: Remote, root: str, redownload: bool = False, opener=None) -> bool:
    """Resumable, size-checked download of one Remote into root. Returns True when complete."""
    opener = opener or _open
    dest = _dest(root, rf.rel)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        have = os.path.getsize(dest)
        if have == rf.size:
            print(f"[skip] {dest} complete ({have} bytes)")
            return True
        if not redownload:
            print(f"[BAD-SIZE] {dest} has {have} bytes, expected {rf.size}; pass --redownload")
            return False
        os.replace(dest, part + ".bad")
    start = os.path.getsize(part) if os.path.exists(part) else 0
    if start > rf.size:
        print(f"[warn] {part} larger than expected ({start} > {rf.size}); restarting")
        os.remove(part)
        start = 0
    if start == rf.size:
        os.replace(part, dest)
        print(f"[done] {dest} ({rf.size} bytes, already complete in .part)")
        return True
    try:
        r = opener(rf.url, start)
    except (urllib.error.URLError, OSError) as e:
        print(f"[FAIL] {rf.url}: {e}")
        return False
    status = getattr(r, "status", 200)
    if start and status != 206:
        print(f"[warn] server ignored Range for {rf.rel} (HTTP {status}); restarting from byte 0")
        start = 0
    mode = "ab" if start else "wb"
    done, t0, last = start, time.time(), time.time()
    try:
        with r, open(part, mode) as f:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if time.time() - last > 10:
                    last = time.time()
                    rate = (done - start) / max(1e-9, last - t0) / 1e6
                    print(f"  {os.path.basename(dest)}: {100 * done / rf.size:.1f}% ({rate:.1f} MB/s)",
                          flush=True)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"[FAIL] interrupted at {done} bytes: {e}; rerun to resume")
        return False
    got = os.path.getsize(part)
    if got != rf.size:
        print(f"[FAIL] {part}: {got} bytes, expected {rf.size}; rerun to resume")
        return False
    os.replace(part, dest)
    print(f"[done] {dest} ({got} bytes)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _safe_member(name: str) -> bool:
    n = name.replace("\\", "/")
    return not (n.startswith("/") or ":" in n.split("/")[0] or ".." in n.split("/"))


def extract_tar(archive: str, dest: str, prefix: str = "") -> int:
    """Extract members of a (gz) tar under dest, optionally only those starting with prefix."""
    n = 0
    with tarfile.open(archive, "r:*") as tf:
        members = []
        for m in tf:
            if prefix and not m.name.startswith(prefix):
                continue
            if not _safe_member(m.name) or m.issym() or m.islnk():
                raise SystemExit(f"refusing unsafe tar member {m.name!r} in {archive}")
            members.append(m)
        for m in members:
            tf.extract(m, dest, filter="data")
            n += 1
    return n


def _find_7z_tool() -> Optional[List[str]]:
    for exe in ("7z", "7za"):
        p = shutil.which(exe)
        if p:
            return [p, "x", "-y"]
    try:
        import py7zr  # noqa: F401
        return ["py7zr"]
    except ImportError:
        pass
    win_tar = os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "System32", "tar.exe")
    if os.path.exists(win_tar):
        return [win_tar, "-xf"]
    return None


def extract_7z(archive: str, dest: str) -> None:
    tool = _find_7z_tool()
    if tool is None:
        raise SystemExit("no 7z extractor: install 7-Zip, `pip install py7zr`, or use Windows 11 tar.exe")
    os.makedirs(dest, exist_ok=True)
    if tool == ["py7zr"]:
        import py7zr
        with py7zr.SevenZipFile(archive, "r") as z:
            bad = [n for n in z.getnames() if not _safe_member(n)]
            if bad:
                raise SystemExit(f"refusing unsafe 7z members {bad[:3]} in {archive}")
            z.extractall(dest)
        return
    if tool[0].lower().endswith("tar.exe"):
        cmd = tool + [archive, "-C", dest]
    else:
        cmd = tool + [archive, f"-o{dest}"]
    subprocess.run(cmd, check=True)


def extract(name: str, root: str) -> None:
    s = SETS[name]
    ex = s["extract"]
    dest = _dest(root, ex["dest"])
    if ex["kind"] == "tar":
        archives = [ex["archive"]] if "archive" in ex else [f.rel for f in s["files"]
                                                            if f.rel.endswith((".tgz", ".tar.gz", ".tar"))]
        for a in archives:
            ap = _dest(root, a)
            if not os.path.exists(ap):
                raise SystemExit(f"{ap} missing; download first")
            n = extract_tar(ap, dest, ex.get("prefix", ""))
            print(f"[extract] {ap}: {n} members -> {dest}")
    elif ex["kind"] == "7z":
        for f in s["files"]:
            ap = _dest(root, f.rel)
            if not os.path.exists(ap):
                raise SystemExit(f"{ap} missing; download first")
            extract_7z(ap, dest)
            print(f"[extract] {ap} -> {dest}")
    elif ex["kind"] == "zip-any":
        import zipfile
        zips = [os.path.join(dest, z) for z in os.listdir(dest) if z.lower().endswith(".zip")] \
            if os.path.isdir(dest) else []
        if not zips:
            raise SystemExit(s.get("manual", f"no archive in {dest}"))
        for zp in zips:
            with zipfile.ZipFile(zp) as z:
                bad = [n for n in z.namelist() if not _safe_member(n)]
                if bad:
                    raise SystemExit(f"refusing unsafe zip members {bad[:3]} in {zp}")
                z.extractall(dest)
            print(f"[extract] {zp} -> {dest}")


def verify_layout(name: str, root: str) -> dict:
    """Count what the matching datasets_v2 loader would see (no GT decoding)."""
    from benchmark import datasets_v2 as dv2
    finder = {"diode_indoor": dv2.list_diode, "eth3d_indoor": dv2.list_eth3d,
              "tum": dv2.list_tum, "tum_fr2_desk": dv2.list_tum,
              "diml_indoor": dv2.list_diml, "indoor67": dv2.list_indoor67}[name]
    folder = _dest(root, SETS[name]["folder"])
    items = finder(folder)
    return {"set": name, "folder": folder, "n": len(items)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--only", default=",".join(k for k in SETS if k != "tum_fr2_desk"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--verify", action="store_true", help="count loader-visible items on disk")
    ap.add_argument("--check-remote", action="store_true")
    ap.add_argument("--redownload", action="store_true")
    ap.add_argument("--lineage", action="store_true")
    args = ap.parse_args(argv)
    names = [k.strip() for k in args.only.split(",") if k.strip()]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; choose from {sorted(SETS)}")
    if args.lineage:
        for k, v in LINEAGE.items():
            print(f"{k:13s} " + "  ".join(f"{m}={s}" for m, s in v.items()))
        for m, e in LINEAGE_EVIDENCE.items():
            print(f"  [{m}] {e}")
        return 0
    if args.check_remote:
        return 0 if check_remote(names) else 1
    acting = args.download or args.extract or args.verify
    root = resolve_root(args.root, must_exist=not args.download)
    if args.dry_run or not acting:
        print_dry_run(dry_run(names, root))
        for n in names:
            if SETS[n].get("access") == "manual":
                print(f"[MANUAL] {n}: {SETS[n]['manual']}")
        return 0
    ok = True
    if args.download:
        for n in names:
            if SETS[n].get("access") == "manual":
                print(f"[MANUAL] {n}: {SETS[n]['manual']}")
                continue
            for rf in SETS[n]["files"]:
                ok &= download(rf, root, redownload=args.redownload)
    if args.extract and ok:
        for n in names:
            extract(n, root)
    if args.verify:
        for n in names:
            try:
                print(verify_layout(n, root))
            except (FileNotFoundError, SystemExit) as e:
                print(f"[verify] {n}: {e}")
                ok = False
    print("ALL OK" if ok else "SOME STEPS FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
