"""Resumable MoGe-2-L teacher labelling for the composite labels (plan task 21). Env: ``depth_cmp``.

    C:/Users/aydina1/AppData/Local/anaconda3/envs/depth_cmp/python.exe -m scripts.v2.label_moge \
        --input outputs/v2/pool/manifest.csv --out outputs/v2/labels/moge

CONTRACT
  Teacher: ``Ruicheng/moge-2-vitl-normal`` (MoGe-2 ViT-L, metric scale head), native
  ``MoGeModel.infer`` at ``resolution_level=9`` (num_tokens = the model's maximum), with
  ``apply_mask=False``, ``force_projection=True`` (MoGe default) and ``use_fp16=True``; no
  intrinsics or FoV are passed. MoGe-2's training-data lineage is recorded as unverified
  (plan 9.1 zero-shot matrix) in every sidecar.

  Same original-aspect image as V2-L: the RGB is resized (INTER_AREA, same rounding as
  ``teacher.unidepth_teacher``) to longest side 1280 when larger, MoGe returns depth at that
  input size, and depth is bilinearly resized back to the original resolution (validity with
  INTER_NEAREST). The saved PNG then goes through the same chain as the V2-L label (clip 60 m,
  longest side 1024, INTER_NEAREST), so both labels of one id share one pixel grid.

  Invalid pixels are recorded, never guessed: a pixel is invalid when MoGe's own mask
  (``mask > 0.5`` and depth > 0) is false, or its depth is non-finite or <= 0. Invalid pixels
  are stored as 0 in the PNG, and each sidecar's ``extra`` carries ``mask_invalid_frac``,
  ``nonfinite_frac`` and ``invalid_frac`` (at the inference resolution), plus MoGe's
  normalised focal estimate (fx_norm, fy_norm). ``teacher_valid_frac`` is the valid fraction
  at the original resolution.

  Inputs (ids verbatim, CSV ``keep`` column honoured), outputs, resuming / skip-if-done,
  conflict and drive checks, --vram-cap-gb and the run summary are identical to
  ``scripts/v2/label_unidepth.py`` (shared framework): depth_mm/<id>.png, meta/<id>.json,
  done.csv, failures.jsonl, run_<utc>.json under --out (default ``outputs/v2/labels/moge``).
  Tests: tests/test_composite.py (``moge_postprocess``; CPU, no model).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v2.label_unidepth import (  # noqa: E402
    DEFAULT_INFER_LONGEST, apply_vram_cap, build_argparser, prepare_run, resized_hw, run_labelling,
)

DEFAULT_HUB = "Ruicheng/moge-2-vitl-normal"
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "v2", "labels", "moge")
DEFAULT_RES_LEVEL = 9


def moge_postprocess(depth: np.ndarray, mask: Optional[np.ndarray], orig_hw) -> Dict[str, object]:
    """MoGe output at inference size -> {"depth_m", "valid", "extra"} at the original size.

    Pure numpy/cv2 (unit-tested without the model). Invalid = mask false | non-finite | <= 0.
    Invalid pixels get the median valid depth before the bilinear resize (so they do not bleed
    inf/0 into valid neighbours); their validity is resized with INTER_NEAREST and they are
    written as 0 by the framework.
    """
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    m = np.ones(depth.shape, bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != depth.shape:
        raise ValueError(f"mask {m.shape} does not match depth {depth.shape}")
    valid = finite & m
    extra = {
        "infer_hw": list(depth.shape[:2]),
        "mask_invalid_frac": round(float(1.0 - m.mean()), 6),
        "nonfinite_frac": round(float(1.0 - finite.mean()), 6),
        "invalid_frac": round(float(1.0 - valid.mean()), 6),
        "mask_available": mask is not None,
    }
    fill = float(np.median(depth[valid])) if valid.any() else 0.0
    dense = np.where(valid, depth, fill).astype(np.float32)
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if dense.shape[:2] != (h, w):
        dense = cv2.resize(dense, (w, h), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return {"depth_m": dense, "valid": valid, "extra": extra}


def make_moge_predictor(hub_id: str = DEFAULT_HUB, device: str = "cuda",
                        resolution_level: int = DEFAULT_RES_LEVEL,
                        infer_longest_side: int = DEFAULT_INFER_LONGEST, use_fp16: bool = True):
    """Load MoGe-2 and return (predict_fn, teacher_meta)."""
    import torch
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as e:  # wrong env
        raise SystemExit(f"[label_moge] cannot import moge ({e}); run this script in the depth_cmp env") from e
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[label_moge] CUDA requested but not available; pass --device cpu explicitly")
    t0 = time.time()
    model = MoGeModel.from_pretrained(hub_id).to(device).eval()
    load_s = round(time.time() - t0, 2)
    try:
        from importlib.metadata import version
        moge_ver = version("moge")
    except Exception:  # noqa: BLE001
        moge_ver = "unknown"
    tok_range = [int(x) for x in getattr(model, "num_tokens_range", (0, 0))]

    @torch.inference_mode()
    def predict(rgb: np.ndarray) -> Dict[str, object]:
        h, w = rgb.shape[:2]
        ih, iw = resized_hw(h, w, infer_longest_side)
        rgb_in = rgb if (ih, iw) == (h, w) else cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(rgb_in)).permute(2, 0, 1).to(device).float() / 255.0
        out = model.infer(t, resolution_level=int(resolution_level), apply_mask=False, use_fp16=use_fp16)
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool) if "mask" in out else None
        res = moge_postprocess(depth, mask, (h, w))
        K = out.get("intrinsics")
        if K is not None:
            K = K.float().cpu().numpy()
            res["extra"]["fx_norm"] = round(float(K[0, 0]), 6)
            res["extra"]["fy_norm"] = round(float(K[1, 1]), 6)
        return res

    meta = {"name": "moge-2-l", "checkpoint": hub_id, "resolution_level": int(resolution_level),
            "num_tokens_range": tok_range,
            "infer_longest_side": int(infer_longest_side), "input_resize": "INTER_AREA",
            "apply_mask": False, "force_projection": True, "use_fp16": bool(use_fp16),
            "camera": "self-estimated (no FoV passed)",
            "output_resize": "bilinear depth / nearest validity to original", "device": device,
            "moge_version": moge_ver, "model_load_s": load_s,
            "lineage": "training-data lineage unverified (plan 9.1 zero-shot matrix)"}
    return predict, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    ap.set_defaults(out=DEFAULT_OUT)
    ap.add_argument("--checkpoint", default=DEFAULT_HUB)
    ap.add_argument("--infer-longest-side", type=int, default=DEFAULT_INFER_LONGEST)
    ap.add_argument("--resolution-level", type=int, default=DEFAULT_RES_LEVEL)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args(argv)
    items = prepare_run(args, "label_moge")
    if items is None:
        return 0
    cap = apply_vram_cap(args.vram_cap_gb, args.device)
    predict, meta = make_moge_predictor(args.checkpoint, args.device, args.resolution_level,
                                        args.infer_longest_side, not args.no_fp16)
    if cap:
        meta["vram_cap"] = cap
    summary = run_labelling(items, args.out, predict, meta, overwrite=args.overwrite, warmup=args.warmup,
                            clip_m=args.clip_m, save_longest=args.save_longest_side,
                            pool_filter_on=args.pool_filter, script_path=__file__)
    return 2 if summary["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
