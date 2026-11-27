"""Tier-1 eval datasets from OFFICIAL sources (see docs/PROTOCOL_NOTES.md).

Canonical dataset locations: Z:\\Datasets\\{NYUv2, SUNRGBD, IBims-1, KITTI, OCMR-Gold}
(one titled folder per dataset; override drive with the DATASETS_ROOT env var). The legacy
data_bench/{nyu,sun,ibims} dirs are used automatically as a fallback when Z: is absent.
Derived artifacts (OCMR GT caches, gold manifests) stay in data_bench/.

  nyu    official nyu_depth_v2_labeled.mat + splits.mat (Eigen test = 654)
         variants: filled (`depths`, the standard eval target) | raw (`rawDepths`)
         K = standard NYU Kinect intrinsics; Eigen border crop applied to valid mask.
  sun    official SUNRGBD release + toolbox allsplit.mat alltest, with kv1/NYUdata
         (the NYU-origin frames) EXCLUDED per UniDepth's stated protocol.
         variants: bfx (`depth_bfx`, refined) | raw (`depth`); per-image intrinsics.txt.
  ibims  IBims-1 core (mat files: float depth + invalid/transparency masks + calib).
  diode  DIODE val/indoors (325; FARO depth + validity mask; fixed DIODE K), 0.005/50.
  eth3d  ETH3D high-res training, 7 indoor scenes (219 DSLR; laser GT aligned to the distorted
         jpgs; per-image COLMAP pinhole K); evaluated at 1/4 resolution, 0.005/50.

Every sample -> EvalSample(id, rgb, gt, valid, K, min_depth, max_depth).
`python -m benchmark.datasets_tier1 inspect <name>` prints structure/counts for sanity.

Protocol constants (official loaders): NYU min/max 0.005/10.0 + border crop
[45:-9, 41:-39]; SUN 0.005/8.0 no crop; IBims 0.005/25.0 no crop.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# Derived artifacts (OCMR GT caches, gold captures) stay small and local:
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data_bench")
# Source datasets live on the shared datasets drive, one titled folder per dataset
# (override with the DATASETS_ROOT env var). Legacy data_bench/<short> locations are
# used as an automatic fallback so nothing breaks if Z: is unavailable.
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "Z:/Datasets")


def _pick(new: str, old: str) -> str:
    """Canonical Z:\\Datasets\\<Name> if present, else the legacy data_bench dir."""
    return new if os.path.isdir(new) else old


NYU_DIR = _pick(os.path.join(DATASETS_ROOT, "NYUv2"), os.path.join(DATA_ROOT, "nyu"))
SUN_DIR = _pick(os.path.join(DATASETS_ROOT, "SUNRGBD"), os.path.join(DATA_ROOT, "sun"))
IBIMS_DIR = _pick(os.path.join(DATASETS_ROOT, "IBims-1"), os.path.join(DATA_ROOT, "ibims"))

NYU_K = np.array([[518.857901, 0.0, 325.582245],
                  [0.0, 519.469611, 253.736166],
                  [0.0, 0.0, 1.0]])


@dataclass
class EvalSample:
    id: str
    rgb: np.ndarray           # (H,W,3) uint8
    gt: np.ndarray            # (H,W) float32 metres
    valid: np.ndarray         # (H,W) bool (sensor validity + crops; NO max-depth cut)
    K: Optional[np.ndarray]   # (3,3) or None
    min_depth: float
    max_depth: float


# ──────────────────────────────────────────────────────────────────────────────
# NYU Depth v2 — official labeled.mat + Eigen test split
# ──────────────────────────────────────────────────────────────────────────────

def _nyu_border_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[45:h - 9, 41:w - 39] = True      # official eval_mask: [..., 45:-9, 41:-39]
    return m


def iter_nyu(root: str = "", variant: str = "filled",
             limit: int = 0) -> Iterator[EvalSample]:
    import h5py
    from scipy.io import loadmat
    root = root or NYU_DIR
    mat = os.path.join(root, "nyu_depth_v2_labeled.mat")
    splits = os.path.join(root, "splits.mat")
    test_idx = loadmat(splits)["testNdxs"].ravel() - 1          # one-based -> zero-based
    key = {"filled": "depths", "raw": "rawDepths"}[variant]
    with h5py.File(mat, "r") as f:
        images, depths = f["images"], f[key]
        n = 0
        for i in test_idx:
            # MATLAB column-major: images (N,3,W,H) -> (H,W,3); depths (N,W,H) -> (H,W)
            rgb = np.transpose(images[i], (2, 1, 0))
            gt = np.asarray(depths[i], dtype=np.float32).T
            h, w = gt.shape
            valid = (gt > 0.005) & _nyu_border_mask(h, w)
            yield EvalSample(f"nyu_{i:04d}", np.ascontiguousarray(rgb), gt, valid,
                             NYU_K.copy(), 0.005, 10.0)
            n += 1
            if limit and n >= limit:
                return


# ──────────────────────────────────────────────────────────────────────────────
# SUN RGB-D — official release + toolbox alltest, NYU-origin excluded
# ──────────────────────────────────────────────────────────────────────────────

def _sun_alltest_paths(root: str = "") -> list[str]:
    """Official test list from SUNRGBDtoolbox allsplit.mat, as SUNRGBD-relative paths."""
    from scipy.io import loadmat
    toolbox_zip = os.path.join(root or SUN_DIR, "SUNRGBDtoolbox.zip")
    inner = "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
    with zipfile.ZipFile(toolbox_zip) as z, z.open(inner) as fh:
        allsplit = loadmat(io.BytesIO(fh.read()))
    raw = [str(p[0]) for p in allsplit["alltest"].ravel()]
    # entries look like '/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/...' -> strip prefix
    out = []
    for p in raw:
        k = p.find("/SUNRGBD/")
        out.append(p[k + len("/SUNRGBD/"):].strip("/") if k >= 0 else p.strip("/"))
    return out


def _sun_decode_depth(png_u16: np.ndarray) -> np.ndarray:
    """Official SUN encoding: depth_mm = bitshift(x,-3) | bitshift(x,13); metres."""
    d = np.bitwise_or(np.right_shift(png_u16, 3),
                      np.left_shift(png_u16, 16 - 3)).astype(np.float32) / 1000.0
    d[d > 8.0] = 0.0            # beyond Kinect-family range -> invalid (official max 8)
    return d


def sun_extract(root: str = "") -> str:
    """One-time extraction of SUNRGBD.zip -> <SUN_DIR>/SUNRGBD. Returns the dir."""
    dest = root or SUN_DIR
    out = os.path.join(dest, "SUNRGBD")
    if os.path.isdir(out):
        return out
    zp = os.path.join(dest, "SUNRGBD.zip")
    print(f"extracting {zp} (one-time, ~10 min) ...", flush=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    return out


def iter_sun(root: str = "", variant: str = "bfx", exclude_nyu: bool = True,
             limit: int = 0) -> Iterator[EvalSample]:
    import cv2
    root = root or SUN_DIR
    base = sun_extract(root)
    depth_dir_name = {"bfx": "depth_bfx", "raw": "depth"}[variant]
    n = 0
    for rel in _sun_alltest_paths(root):
        if exclude_nyu and "NYUdata" in rel:
            continue
        sdir = os.path.join(base, rel)
        if not os.path.isdir(sdir):        # tolerate path quirks; count in inspect
            continue
        try:
            img_dir = os.path.join(sdir, "image")
            dep_dir = os.path.join(sdir, depth_dir_name)
            img_path = os.path.join(img_dir, os.listdir(img_dir)[0])
            dep_path = os.path.join(dep_dir, os.listdir(dep_dir)[0])
            K = np.loadtxt(os.path.join(sdir, "intrinsics.txt")).reshape(3, 3)
        except (OSError, IndexError, ValueError):
            continue
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        png = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if bgr is None or png is None:
            continue
        gt = _sun_decode_depth(png.astype(np.uint16))
        valid = gt > 0.005
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != gt.shape[:2]:  # rare size mismatch -> skip, log in inspect
            continue
        yield EvalSample(rel.replace("/", "_"), rgb, gt, valid, K, 0.005, 8.0)
        n += 1
        if limit and n >= limit:
            return


# ──────────────────────────────────────────────────────────────────────────────
# IBims-1 — core mat files (float depth + masks + calib)
# ──────────────────────────────────────────────────────────────────────────────

def ibims_extract(root: str = "") -> str:
    dest = root or IBIMS_DIR
    for cand in ("ibims1_core_mat", "ibims1-core-mat"):
        out = os.path.join(dest, cand)
        if os.path.isdir(out):
            return out
    zp = os.path.join(dest, "ibims1_core_mat.zip")
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    for cand in ("ibims1_core_mat", "ibims1-core-mat"):
        out = os.path.join(dest, cand)
        if os.path.isdir(out):
            return out
    return dest


def iter_ibims(root: str = "", limit: int = 0) -> Iterator[EvalSample]:
    """IBims-1 core: per-image .mat with struct `data` (rgb, depth, calib, masks).

    Fields (IBims-1 readme): depth [m] float; mask_invalid; mask_transp; calib K.
    Valid = depth>0 & mask_invalid==1?  -> resolved in `inspect` on first run; this
    loader auto-detects the exact field names and stores them on first sample.
    """
    from scipy.io import loadmat
    base = ibims_extract(root)
    mats = sorted(
        os.path.join(dp, f) for dp, _, fs in os.walk(base) for f in fs if f.endswith(".mat")
    )
    n = 0
    for mp in mats:
        m = loadmat(mp, simplify_cells=True)
        data = m.get("data", m)
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
        gt = np.asarray(data["depth"], dtype=np.float32)
        valid = gt > 0.005
        # IBims-1 convention (verified on the release): BOTH masks are 1 = keep,
        # 0 = exclude (invalid sensor reading / transparent surface respectively).
        for mask_key in ("mask_invalid", "mask_transp"):
            if mask_key in data:
                valid &= np.asarray(data[mask_key]).astype(bool)
        K = None
        if "calib" in data:
            # stored MATLAB-transposed: [[fx,0,0],[0,fy,0],[cx,cy,1]] -> K = calib.T
            K = np.asarray(data["calib"], dtype=np.float64).reshape(3, 3).T
        yield EvalSample(os.path.splitext(os.path.basename(mp))[0], rgb, gt, valid,
                         K, 0.005, 25.0)
        n += 1
        if limit and n >= limit:
            return


# ──────────────────────────────────────────────────────────────────────────────
# KITTI object-benchmark frames + velodyne-projected sparse depth (outdoor sanity)
# ──────────────────────────────────────────────────────────────────────────────

KITTI_ROOT = os.path.join(DATASETS_ROOT, "KITTI")
# Official UniDepth KITTI constants (UniDepth-main/unidepth/datasets/kitti.py):
# min_depth 0.05, max_depth 80.0, garg eval crop fractions below.
KITTI_MIN_D, KITTI_MAX_D = 0.05, 80.0
_GARG = (0.40810811, 0.99189189, 0.03594771, 0.96405229)


def _garg_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[int(_GARG[0] * h):int(_GARG[1] * h), int(_GARG[2] * w):int(_GARG[3] * w)] = True
    return m


def _kitti_calib(path: str):
    vals = {}
    for line in open(path, encoding="utf-8"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        vals[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    P2 = vals["P2"].reshape(3, 4)
    R0 = vals["R0_rect"].reshape(3, 3)
    Tr = vals["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0, Tr


def _velo_to_depth(bin_path: str, P2, R0, Tr, h: int, w: int) -> np.ndarray:
    """Project a velodyne sweep into cam2 -> sparse z-depth map (min z-buffer)."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    cam = (R0 @ (Tr[:, :3] @ pts.T + Tr[:, 3:4]))          # (3,N) rectified cam0
    keep = cam[2] > 0.1
    cam = cam[:, keep]
    y = P2 @ np.vstack([cam, np.ones((1, cam.shape[1]))])   # (3,N) cam2 pixels
    z = y[2]
    u = np.round(y[0] / z).astype(int)
    v = np.round(y[1] / z).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    depth = np.zeros((h, w), dtype=np.float32)
    # min z-buffer for duplicate pixels (near surface wins)
    order = np.argsort(-z[ok])                               # far first, near overwrites
    uu, vv, zz = u[ok][order], v[ok][order], z[ok][order]
    depth[vv, uu] = zz.astype(np.float32)
    return depth


def iter_kitti_obj(root: str = KITTI_ROOT, limit: int = 0) -> Iterator[EvalSample]:
    """KITTI object-benchmark frames with velodyne-projected SPARSE depth GT.

    Requires training/velodyne/*.bin — the single missing zip `data_object_velodyne.zip`
    (~29 GB, same KITTI page as the present object zips; fetchable via
    `python -m benchmark.fetch_tier1 --only kitti`).
    Until it exists this loader raises with that instruction (self-gating).

    Protocol: official UniDepth KITTI constants (min 0.05 / max 80) + garg crop; GT is a
    single-sweep LiDAR projection (sparse), so boundary-F1 must NOT be run on this slice.
    NOT the Eigen split (no raw sequences on disk) — a documented our-protocol outdoor
    sanity slice over the 7,481 object-benchmark frames; no published row corresponds to
    it, so its rows are always [ours]-only (comparability contract §10 applies).
    """
    velo_dir = os.path.join(root, "training", "velodyne")
    if not os.path.isdir(velo_dir):
        raise FileNotFoundError(
            f"{velo_dir} missing — download data_object_velodyne.zip "
            "(fetch_tier1 --root Z:/Datasets/KITTI --only kitti) and extract; "
            "dense KITTI eval is gated on it. OCMR-KITTI (benchmark.ocmr_kitti) "
            "works today without it.")
    import cv2
    img_dir = os.path.join(root, "training", "image_2")
    cal_dir = os.path.join(root, "training", "calib")
    n = 0
    for f in sorted(os.listdir(velo_dir)):
        if not f.endswith(".bin"):
            continue
        fr = os.path.splitext(f)[0]
        bgr = cv2.imread(os.path.join(img_dir, fr + ".png"), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        try:
            P2, R0, Tr = _kitti_calib(os.path.join(cal_dir, fr + ".txt"))
        except (OSError, KeyError, ValueError):
            continue
        gt = _velo_to_depth(os.path.join(velo_dir, f), P2, R0, Tr, h, w)
        valid = (gt > KITTI_MIN_D) & _garg_mask(h, w)
        if valid.sum() < 50:                     # official loader's too-few-points guard
            continue
        yield EvalSample(f"kittiobj_{fr}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt,
                         valid, P2[:, :3].copy(), KITTI_MIN_D, KITTI_MAX_D)
        n += 1
        if limit and n >= limit:
            return


# ──────────────────────────────────────────────────────────────────────────────
# DIODE val / indoors — 325 images, FARO laser GT (benchmark/fetch_v2.py diode_indoor)
# ──────────────────────────────────────────────────────────────────────────────

DIODE_DIR = os.path.join(DATASETS_ROOT, "DIODE")
# DIODE camera (dataset authors, github.com/diode-dataset/diode-devkit issue #9; same values in
# Metric3D / UniDepth configs): fx 886.81, fy 927.06, cx 512, cy 384 at 1024x768.
DIODE_K = np.array([[886.81, 0.0, 512.0], [0.0, 927.06, 384.0], [0.0, 0.0, 1.0]])
DIODE_MIN_D, DIODE_MAX_D = 0.005, 50.0       # scanner range; mask already drops no-return pixels


def list_diode(root: str = "") -> list[str]:
    """Sorted RGB paths of DIODE val/indoors that have both _depth.npy and _depth_mask.npy."""
    base = os.path.join(root or DIODE_DIR, "val", "indoors")
    out = []
    for dp, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(".png"):
                stem = os.path.join(dp, f[:-4])
                if os.path.isfile(stem + "_depth.npy") and os.path.isfile(stem + "_depth_mask.npy"):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def iter_diode(root: str = "", limit: int = 0) -> Iterator[EvalSample]:
    import cv2
    n = 0
    for p in list_diode(root):
        stem = p[:-4]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gt = np.load(stem + "_depth.npy").astype(np.float32).squeeze()
        mask = np.load(stem + "_depth_mask.npy").astype(bool).squeeze()
        valid = mask & np.isfinite(gt) & (gt > DIODE_MIN_D)
        gt = np.where(np.isfinite(gt), gt, 0.0).astype(np.float32)
        if gt.shape != bgr.shape[:2] or valid.sum() < 50:
            continue
        yield EvalSample("diode_" + os.path.basename(stem), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                         gt, valid, DIODE_K.copy(), DIODE_MIN_D, DIODE_MAX_D)
        n += 1
        if limit and n >= limit:
            return


# ──────────────────────────────────────────────────────────────────────────────
# ETH3D high-res multi-view TRAINING, 7 indoor scenes — 219 DSLR images, laser-scan GT
# ──────────────────────────────────────────────────────────────────────────────

ETH3D_DIR = os.path.join(DATASETS_ROOT, "ETH3D")
ETH3D_SCENES = ("delivery_area", "kicker", "office", "pipes", "relief", "relief_2", "terrains")
ETH3D_W, ETH3D_H = 6048, 4032                 # raw float32 depth dumps (row-major, little-endian)
ETH3D_DOWN = 4                                # evaluate at 1512x1008 (see iter_eth3d docstring)
ETH3D_MIN_D, ETH3D_MAX_D = 0.005, 50.0


def _eth3d_cameras(scene_dir: str) -> dict:
    """{image basename: (fx, fy, cx, cy, W, H)} from the COLMAP text model of the DISTORTED jpgs.

    The jpg archive ships dslr_calibration_jpg/{cameras,images}.txt. The camera model is
    THIN_PRISM_FISHEYE; only its pinhole part (fx fy cx cy) is used (distortion ignored, the GT
    depth is aligned to the distorted image pixel-for-pixel, so per-pixel metrics are exact)."""
    cal = None
    for c in ("dslr_calibration_jpg", "dslr_calibration_undistorted"):
        if os.path.isdir(os.path.join(scene_dir, c)):
            cal = os.path.join(scene_dir, c)
            break
    if cal is None:
        return {}
    cams = {}
    for line in open(os.path.join(cal, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[t[0]] = (float(t[4]), float(t[5]), float(t[6]), float(t[7]), int(t[2]), int(t[3]))
    out = {}
    lines = [ln for ln in open(os.path.join(cal, "images.txt"), encoding="utf-8")
             if not ln.startswith("#")]
    for ln in lines[0::2]:                    # COLMAP images.txt: pose line, then 2D points line
        t = ln.split()
        if len(t) >= 10 and t[8] in cams:
            out[os.path.basename(t[9])] = cams[t[8]]
    return out


def list_eth3d(root: str = "") -> list[tuple]:
    """[(scene, rgb_path, depth_path, (fx,fy,cx,cy,W,H))] for every image with GT + calibration."""
    base = root or ETH3D_DIR
    out = []
    for s in ETH3D_SCENES:
        sd = os.path.join(base, s)
        img_dir = os.path.join(sd, "images", "dslr_images")
        dep_dir = os.path.join(sd, "ground_truth_depth", "dslr_images")
        if not (os.path.isdir(img_dir) and os.path.isdir(dep_dir)):
            continue
        cams = _eth3d_cameras(sd)
        for f in sorted(os.listdir(dep_dir)):
            rp = os.path.join(img_dir, f)
            if os.path.isfile(rp) and f in cams:
                out.append((s, rp, os.path.join(dep_dir, f), cams[f]))
    return out


def iter_eth3d(root: str = "", limit: int = 0, down: int = ETH3D_DOWN) -> Iterator[EvalSample]:
    """ETH3D indoor. GT dumps are float32 6048x4032 (inf/0 = no GT) aligned to the DISTORTED jpg.

    Evaluated at 1/`down` resolution (default 4 -> 1512x1008, ~1.5 MP): RGB INTER_AREA, depth = the
    nearest valid laser point in every down x down block (GT is sparse: 7-16 % of full-res pixels,
    60-85 % of 4x4 blocks; no interpolation), K scaled with the pixel-centre convention. Every model only sees <=0.6 MP internally, so this changes
    no prediction materially and keeps per-image metrics tractable (24 MP -> 1.5 MP)."""
    import cv2
    n = 0
    for scene, rp, dp, (fx, fy, cx, cy, W, H) in list_eth3d(root):
        raw = np.fromfile(dp, dtype="<f4")
        if raw.size != W * H:
            continue
        gt = raw.reshape(H, W)
        bgr = cv2.imread(rp, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (H, W):
            continue
        if down > 1:
            h2, w2 = H // down, W // down
            bgr = cv2.resize(bgr, (w2, h2), interpolation=cv2.INTER_AREA)
            # GT is projected laser points (7-16 % of full-res pixels): keep the NEAREST valid point
            # per down x down block (z-buffer semantics, as _velo_to_depth), so no GT point is dropped
            blk = np.where(np.isfinite(gt) & (gt > 0), gt, np.inf)[:h2 * down, :w2 * down]
            gt = blk.reshape(h2, down, w2, down).min(axis=(1, 3))
            fx, fy = fx / down, fy / down
            cx, cy = (cx + 0.5) / down - 0.5, (cy + 0.5) / down - 0.5
        valid = np.isfinite(gt) & (gt > ETH3D_MIN_D)
        gt = np.where(np.isfinite(gt), gt, 0.0).astype(np.float32)
        if valid.sum() < 50:
            continue
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        yield EvalSample(f"eth3d_{scene}_{os.path.splitext(os.path.basename(rp))[0]}",
                         cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.ascontiguousarray(gt), valid, K,
                         ETH3D_MIN_D, ETH3D_MAX_D)
        n += 1
        if limit and n >= limit:
            return


LOADERS = {"nyu": iter_nyu, "sun": iter_sun, "ibims": iter_ibims, "kitti_obj": iter_kitti_obj,
           "diode": iter_diode, "eth3d": iter_eth3d}


def _inspect(name: str):
    import itertools
    it = LOADERS[name]()
    counts, hw = 0, None
    for s in itertools.islice(it, 3):
        counts += 1
        hw = s.gt.shape
        v = s.gt[s.valid]
        print(f"{s.id}: rgb{s.rgb.shape} gt{s.gt.shape} valid={s.valid.mean():.2%} "
              f"gt-range=[{v.min():.3f},{v.max():.3f}] med={np.median(v):.3f} "
              f"K={'yes' if s.K is not None else 'none'}")
    total = sum(1 for _ in LOADERS[name]()) if name != "sun" else "(skip full count)"
    print(f"{name}: first-3 ok, shape {hw}; total={total}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        _inspect(sys.argv[2])
    else:
        print("usage: python -m benchmark.datasets_tier1 inspect {nyu|sun|ibims}")
