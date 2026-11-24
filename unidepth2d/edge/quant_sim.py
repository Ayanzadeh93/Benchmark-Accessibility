"""Weight-only uniform quantization, simulated in PyTorch (quantize -> dequantize in place).

This answers "what happens to accuracy if the weights are stored in N bits?" without any
export tool chain: every Conv2d / Linear weight is rounded onto a symmetric signed integer
grid and written back as float, so the model then runs through the normal harness.

Per-channel mode (one scale per output channel) is what int8 back-ends actually use for
convolutions (ONNX Runtime QDQ, Core ML, TFLite), and it is invariant to BatchNorm folding:
scaling an output channel by a constant scales its weights and its quantization step by the
same factor, so quantizing before or after folding gives the same relative error. Per-tensor
mode (one scale per layer) is the pessimistic variant and is NOT folding-invariant.

Biases, BatchNorm parameters and activations stay in floating point: this is the
"8-bit weights" storage scheme (Core ML linear weight quantization), not full integer
inference. Full int8 (weights + activations) lives in edge/ort_int8.py.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

QUANT_TYPES = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
# b20 dsh: the decoupled scale head (~64k params) stays float under weight-only quantisation
# (spec section 2, supporting change 4). Students without these modules are unaffected.
FLOAT_KEEP_PREFIXES = ("scale_mlp", "k_embed")


def quantize_tensor(w: torch.Tensor, bits: int = 8, per_channel: bool = True,
                    channel_dim: int = 0) -> torch.Tensor:
    """Symmetric uniform quantize-dequantize of a weight tensor.

    Integer grid is [-(2^(b-1) - 1), 2^(b-1) - 1] (restricted range, e.g. +-127 for 8 bits).
    All-zero channels are returned unchanged.
    """
    if bits < 2 or bits > 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")
    qmax = float(2 ** (bits - 1) - 1)
    if per_channel and w.dim() > 1:
        dims = [d for d in range(w.dim()) if d != channel_dim]
        amax = w.detach().abs().amax(dim=dims, keepdim=True)
    else:
        amax = w.detach().abs().amax()
    scale = amax / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / safe), -qmax, qmax)
    return torch.where(scale > 0, q * safe, w)


@torch.no_grad()
def quantize_weights_(model: nn.Module, bits: int = 8, per_channel: bool = True) -> Dict[str, float]:
    """Quantize every Conv2d / Linear / ConvTranspose2d weight of `model` in place.

    Returns storage statistics: how many parameters were quantized, how many stay float,
    the resulting size in MB, and the mean relative weight error introduced.
    """
    n_q = n_fp = 0
    rel_err_num = rel_err_den = 0.0
    seen = set()
    for name, mod in model.named_modules():
        if any(part in FLOAT_KEEP_PREFIXES for part in name.split(".")):
            continue                                      # b20 scale head stays float
        if isinstance(mod, QUANT_TYPES) and id(mod.weight) not in seen:
            seen.add(id(mod.weight))
            w = mod.weight
            # ConvTranspose2d stores weights as (in, out/groups, kH, kW): out channels on dim 1
            cdim = 1 if isinstance(mod, nn.ConvTranspose2d) else 0
            wq = quantize_tensor(w, bits, per_channel, cdim)
            rel_err_num += float((wq - w).pow(2).sum())
            rel_err_den += float(w.pow(2).sum())
            w.copy_(wq)
            n_q += w.numel()
    for p in model.parameters():
        if id(p) not in seen:
            n_fp += p.numel()
    size_mb = (n_q * bits / 8.0 + n_fp * 4.0) / 1e6
    return {
        "bits": bits,
        "per_channel": bool(per_channel),
        "params_quantized": n_q,
        "params_float": n_fp,
        "size_mb": round(size_mb, 3),
        "size_mb_fp32": round((n_q + n_fp) * 4.0 / 1e6, 3),
        "weight_rel_rmse": (rel_err_num / rel_err_den) ** 0.5 if rel_err_den > 0 else 0.0,
    }
