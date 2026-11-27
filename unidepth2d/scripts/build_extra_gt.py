"""Convert the downloaded ARKitScenes (depth-upsampling, Training) and Hypersim (train split) subsets
into the trainer's GT format (640x480 jpg + 16-bit mm png + normalised K), on the external SSD, and
write a NEW manifest = the original outputs/data_gt/manifest.csv rows + the new rows.

Sources are named 'xarkit' and 'xhyper' on purpose: MultiSourceDataset carves its per-source val slice
by iterating sources in SORTED order with one shared RNG, so names that sort after 'sun' leave the
nyu/own/sun train/val split bit-identical to the one b20dws was trained with.

ARKitScenes: RGB = upsampling/<vid>/wide/*.png (1920x1440); GT = highres_depth (Faro laser, mm);
             K = raw/<vid>/wide_intrinsics/<vid>_<ts>.pincam (w h fx fy cx cy), nearest timestamp.
Hypersim:    RGB = scene_cam_XX_final_preview/frame.IIII.tonemap.jpg (1024x768);
             depth_meters.hdf5 = Euclidean distance to the camera centre -> converted to PLANAR depth
             with the scene's M_cam_from_uv (contrib/mikeroberts3000/metadata_camera_parameters.csv);
             scenes whose camera is not a plain pinhole (skew / tilt terms) are skipped.
Depth labels are resized with INTER_NEAREST, identical to scripts/build_gt_dataset.py.

  python -m scripts.build_extra_gt --which all
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np

OUT_W, OUT_H = 640, 480
FIELDS = ["id", "source", "img", "depth", "conf", "is_gt", "dmin", "dmax", "fx", "fy", "cx", "cy"]


def save_pair(root, src, sid, rgb_bgr, depth_m, clip_m):
    img_d, dep_d = os.path.join(root, src, "images"), os.path.join(root, src, "depth_mm")
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(dep_d, exist_ok=True)
    ip, dp = os.path.join(img_d, sid + ".jpg"), os.path.join(dep_d, sid + ".png")
    if os.path.isfile(ip) and os.path.isfile(dp):
        return ip, dp
    rgb_r = cv2.resize(rgb_bgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    d = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, min(clip_m, 65.535))
    d_r = cv2.resize(d, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(ip, rgb_r, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(dp, (d_r * 1000.0).astype(np.uint16))
    return ip, dp


def build_arkit(arkit_root, out_root, min_valid=0.3, max_per_video=40):
    rows = []
    for vdir in sorted(glob.glob(os.path.join(arkit_root, "upsampling", "Training", "*"))):
        vid = os.path.basename(vdir)
        if not os.path.isdir(os.path.join(vdir, "highres_depth")):
            continue
        idir = os.path.join(arkit_root, "raw", "Training", vid, "wide_intrinsics")
        pin = {}
        for p in glob.glob(os.path.join(idir, "*.pincam")):
            ts = float(os.path.basename(p)[len(vid) + 1:-len(".pincam")])
            pin[ts] = p
        if not pin:
            print(f"  skip {vid}: no intrinsics")
            continue
        tss = np.array(sorted(pin))
        dpaths = sorted(glob.glob(os.path.join(vdir, "highres_depth", "*.png")))
        if max_per_video > 0 and len(dpaths) > max_per_video:  # consecutive video frames are near-duplicates
            dpaths = [dpaths[int(i)] for i in np.linspace(0, len(dpaths) - 1, max_per_video)]
        for dpath in dpaths:
            name = os.path.basename(dpath)
            rpath = os.path.join(vdir, "wide", name)
            if not os.path.isfile(rpath):
                continue
            ts = float(name[len(vid) + 1:-4])
            near = tss[np.argmin(np.abs(tss - ts))]
            if abs(near - ts) > 0.05:
                continue
            w, h, fx, fy, cx, cy = map(float, open(pin[near]).read().split())
            dep = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            if (dep > 0).mean() < min_valid:
                continue
            rgb = cv2.imread(rpath, cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[1] != int(w) or rgb.shape[0] != int(h):
                continue
            sid = f"xarkit_{vid}_{name[len(vid) + 1:-4].replace('.', '_')}"
            ip, dp = save_pair(out_root, "xarkit", sid, rgb, dep, 10.0)
            rows.append([sid, "xarkit", ip, dp, "", 1, 0.2, 10.0, fx / w, fy / h, cx / w, cy / h])
    print(f"[xarkit] {len(rows)} rows")
    return rows


def hypersim_cams(repo):
    cams = {}
    p = os.path.join(repo, "contrib", "mikeroberts3000", "metadata_camera_parameters.csv")
    for r in csv.DictReader(open(p)):
        m = {k: float(r[f"M_cam_from_uv_{k}"]) for k in ("00", "01", "02", "10", "11", "12", "20", "21", "22")}
        if max(abs(m["01"]), abs(m["10"]), abs(m["20"]), abs(m["21"]), abs(m["22"] + 1.0)) > 1e-6:
            continue  # not a plain pinhole -> skip scene
        cams[r["scene_name"]] = m
    return cams


def build_hyper(hyp_root, repo, out_root):
    import h5py
    cams = hypersim_cams(repo)
    rows = []
    skipped = set()
    for jpg in sorted(glob.glob(os.path.join(hyp_root, "scenes", "ai_*", "images", "scene_cam_*_final_preview",
                                             "frame.*.tonemap.jpg"))):
        scene = jpg.replace("\\", "/").split("/scenes/")[1].split("/")[0]
        if scene not in cams:
            skipped.add(scene)
            continue
        cam = jpg.split("scene_cam_")[1][:2]
        fr = os.path.basename(jpg).split(".")[1]
        h5 = os.path.join(os.path.dirname(os.path.dirname(jpg)), f"scene_cam_{cam}_geometry_hdf5",
                          f"frame.{fr}.depth_meters.hdf5")
        if not os.path.isfile(h5):
            continue
        try:
            with h5py.File(h5, "r") as f:
                dist = np.array(f["dataset"], dtype=np.float32)
        except OSError:
            print(f"  skip unreadable {h5}")
            continue
        H, W = dist.shape
        m = cams[scene]
        u = (np.arange(W) + 0.5) / W * 2.0 - 1.0
        v = 1.0 - (np.arange(H) + 0.5) / H * 2.0
        xr = m["00"] * u[None, :] + m["02"]
        yr = m["11"] * v[:, None] + m["12"]
        planar = dist / np.sqrt(xr ** 2 + yr ** 2 + 1.0)
        if np.isfinite(planar).mean() < 0.3:
            continue
        fx_n, fy_n = 1.0 / (2.0 * m["00"]), 1.0 / (2.0 * m["11"])
        cx_n, cy_n = 0.5 - m["02"] / (2.0 * m["00"]), 0.5 + m["12"] / (2.0 * m["11"])
        rgb = cv2.imread(jpg, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"  skip unreadable {jpg}")
            continue
        sid = f"xhyper_{scene}_{cam}_{fr}"
        ip, dp = save_pair(out_root, "xhyper", sid, rgb, planar, 20.0)
        rows.append([sid, "xhyper", ip, dp, "", 1, 0.2, 20.0, fx_n, fy_n, cx_n, cy_n])
    print(f"[xhyper] {len(rows)} rows; skipped non-pinhole scenes: {sorted(skipped)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "arkit", "hyper"])
    ap.add_argument("--arkit-root", default="Z:/Datasets/ARKitScenes")
    ap.add_argument("--hyper-root", default="Z:/Datasets/Hypersim")
    ap.add_argument("--hyper-repo", default="Z:/Datasets/Hypersim/repo")
    ap.add_argument("--out-root", default="Z:/Datasets/student_gt_extra")
    ap.add_argument("--base-manifest", default="C:/Tim/nature/unidepth2d/outputs/data_gt/manifest.csv")
    a = ap.parse_args()
    os.makedirs(a.out_root, exist_ok=True)
    rows = []
    if a.which in ("all", "arkit"):
        rows += build_arkit(a.arkit_root, a.out_root)
    if a.which in ("all", "hyper"):
        rows += build_hyper(a.hyper_root, a.hyper_repo, a.out_root)
    out = os.path.join(a.out_root, "manifest_plus.csv")
    with open(a.base_manifest, newline="", encoding="utf-8") as f:
        base = list(csv.reader(f))
    assert base[0] == FIELDS, base[0]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(base)
        w.writerows(rows)
    print(f"wrote {out}: {len(base) - 1} base rows + {len(rows)} new rows")


if __name__ == "__main__":
    main()
