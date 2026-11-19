"""SUN RGB-D test, streamed from HF wyrx/SUNRGBD_seg (config default, split test). Depth is the
bit-shifted uint16 Kinect encoding (auto-detected). 8 m cap. This repackage lacks per-image
intrinsics, so K is a nominal Kinect matrix (used only for F_A back-projection / rho as approximate;
2D metrics are unaffected)."""

from __future__ import annotations

import numpy as np

# Nominal SUN-RGBD Kinect-v1 intrinsics (varies per sensor; approximate for F_A/rho only).
SUN_K = np.array([[529.5, 0.0, 320.0],
                  [0.0, 529.5, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
MAX_D = 8.0


def _decode(arr_u16):
    plain = arr_u16.astype(np.float32) / 1000.0
    med = float(np.median(plain[plain > 0])) if (plain > 0).any() else 0.0
    if 0.4 <= med <= 8.0:
        return plain
    return np.bitwise_or(np.right_shift(arr_u16, 3), np.left_shift(arr_u16, 13)).astype(np.float32) / 1000.0


def iter_samples(limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("wyrx/SUNRGBD_seg", "default", split="test", streaming=True)
    n = 0
    for s in ds:
        if limit and n >= limit:
            break
        rgb = np.array(s["image"].convert("RGB"))
        depth = _decode(np.array(s["depth"]).astype(np.uint16))
        depth[depth > MAX_D] = 0.0
        H, W = depth.shape[:2]
        valid = (depth > 1e-3) & (depth < MAX_D)
        if valid.sum() < 100:
            continue
        K = SUN_K.copy()
        if (W, H) != (640, 480):
            K[0] *= W / 640.0
            K[1] *= H / 480.0
        yield {"rgb": rgb, "depth": depth, "K": K, "valid": valid,
               "max_depth": MAX_D, "name": "sun", "K_approx": True}
        n += 1
