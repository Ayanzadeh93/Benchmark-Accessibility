"""NYU Depth v2 validation (654), streamed from HF vikhyatk/nyu_depth_v2. Eigen crop, 10 m cap.
GT camera = NYU Kinect intrinsics (standard constants) at the native 480x640."""

from __future__ import annotations

import numpy as np

# Standard NYU-v2 Kinect intrinsics at 640x480 (Silberman toolbox constants).
NYU_K = np.array([[518.8579, 0.0, 325.5824],
                  [0.0, 519.4696, 253.7362],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
MAX_D = 10.0


def _eigen_crop(H, W):
    m = np.zeros((H, W), dtype=bool)
    if (H, W) == (480, 640):
        m[45:471, 41:601] = True
    else:
        m[:] = True
    return m


def iter_samples(limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("vikhyatk/nyu_depth_v2", split="validation", streaming=True)
    n = 0
    for s in ds:
        if limit and n >= limit:
            break
        rgb = np.array(s["image"].convert("RGB") if hasattr(s["image"], "convert") else s["image"])
        depth = np.array(s["depth_map"]).astype(np.float32)
        H, W = depth.shape[:2]
        valid = (depth > 1e-3) & (depth < MAX_D) & _eigen_crop(H, W)
        if valid.sum() < 100:
            continue
        # Scale K if the delivered resolution differs from 640x480.
        K = NYU_K.copy()
        if (W, H) != (640, 480):
            K[0] *= W / 640.0
            K[1] *= H / 480.0
        yield {"rgb": rgb, "depth": depth, "K": K, "valid": valid,
               "max_depth": MAX_D, "name": "nyu"}
        n += 1
