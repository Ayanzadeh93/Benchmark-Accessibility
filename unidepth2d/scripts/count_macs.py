"""Static cost of a student: parameters and GMAC at 480x640 (b20 spec section 5, step 6a).

MACs are counted with forward hooks on every nn.Conv2d / nn.Linear (bias adds and elementwise
ops are not counted; attention matmuls inside the encoder are not Conv/Linear and are therefore
not counted either — the same convention as the spec's 6.84 GMAC b10 figure). rep4 students are
re-parameterised first, so the deploy (fused) cost is reported.

  python -m scripts.count_macs --preset b10 --preset b20
  python -m scripts.count_macs --ckpt outputs/checkpoints/b10/best.pt --ckpt outputs/checkpoints/b1_mix/best.pt

Presets build a fresh model (no weights needed; pretrained_backbone off):
  b10      B1 + ray + FiLM                         (recipe s4,s5,ray,range,prox)
  b20core  b10 + dsh + cstm                        (no rep4)
  b20      b10 + dsh + cstm + rep4                 (fused)
CPU only; nothing is written.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict

import torch
import torch.nn as nn

PRESETS = {
    "b10": {},
    "b20core": {"scale_head": True, "cstm_fc": 518.9},
    "b20dws": {"scale_head": True, "cstm_fc": 518.9, "dws4": True},
    "b20": {"scale_head": True, "cstm_fc": 518.9, "rep4": True},
}
BASE_STUDENT = {"backbone": "efficientvit_b1.r224_in1k", "pretrained_backbone": False,
                "decoder_channels": 96, "train_size": [480, 640], "depth_min_m": 0.2,
                "depth_max_m": 30.0, "use_intrinsics": True, "ray_map": True}


def count(model: nn.Module, h: int = 480, w: int = 640, intr=None):
    """Return (params, total_macs, {top-level module: macs}, {module name: macs})."""
    per_mod = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def hk(m, inp, out, name=name):
                kh, kw = m.kernel_size
                per_mod[name] = per_mod.get(name, 0) + out.numel() // out.shape[0] * (
                    m.in_channels // m.groups) * kh * kw
            hooks.append(mod.register_forward_hook(hk))
        elif isinstance(mod, nn.Linear):
            def hk(m, inp, out, name=name):
                per_mod[name] = per_mod.get(name, 0) + out.numel() // out.shape[0] * m.in_features
            hooks.append(mod.register_forward_hook(hk))
    model.eval()
    x = torch.zeros(1, 3, h, w)
    if intr is None:
        intr = torch.tensor([[0.81, 1.08, 0.5, 0.53]])
    with torch.no_grad():
        model(x, intr)
    for hk in hooks:
        hk.remove()
    groups = defaultdict(int)
    for n, v in per_mod.items():
        top = n.split(".")[0]
        if top in ("smooths", "head") and len(n.split(".")) > 1:
            top = ".".join(n.split(".")[:2])
        groups[top] += v
    params = sum(p.numel() for p in model.parameters())
    return params, sum(per_mod.values()), dict(groups), per_mod


def _build(student_cfg: dict, state=None):
    from distill.student import build_student
    m = build_student({"student": student_cfg}).eval()
    if state is not None:
        m.load_state_dict(state)
    m.reparameterize()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", action="append", default=[], choices=sorted(PRESETS))
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--hw", type=int, nargs=2, default=[480, 640])
    ap.add_argument("--breakdown", action="store_true", help="print per-group GMAC")
    a = ap.parse_args()
    if not a.preset and not a.ckpt:
        a.preset = ["b10", "b20core", "b20"]
    h, w = a.hw
    rows = []
    for p in a.preset:
        s = copy.deepcopy(BASE_STUDENT)
        s.update(PRESETS[p])
        rows.append((f"preset:{p}", _build(s)))
    for c in a.ckpt:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        s = copy.deepcopy(ck["cfg"]["student"])
        s["pretrained_backbone"] = False
        rows.append((c, _build(s, ck["model"])))
    print(f"{'model':<60} {'params_M':>9} {'GMAC@%dx%d' % (w, h):>12}")
    for name, m in rows:
        params, macs, groups, _ = count(m, h, w)
        print(f"{name:<60} {params / 1e6:>9.3f} {macs / 1e9:>12.3f}")
        if a.breakdown:
            enc = groups.get("encoder", 0)
            print(f"    encoder {enc / 1e9:.3f}  decoder+heads {(macs - enc) / 1e9:.3f}")
            for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
                if g != "encoder":
                    print(f"    {g:<20} {v / 1e9:.4f}")


if __name__ == "__main__":
    main()
