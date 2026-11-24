"""MobileOne-style re-parameterisable depthwise-separable block (b20 recipe tag ``rep4``).

Replaces a dense 3x3 96->96 ConvBNReLU at stride 4 (1.593 GMAC at 480x640) with a
depthwise-separable block whose *deploy* form is

    DW3x3(+bias) -> ReLU -> PW1x1(+bias) -> ReLU            (0.194 GMAC at 120x160, C=96)

and whose *training* form carries extra linear branches that fold away exactly:

    DW stage: y = ReLU( sum_j BN_j(DW3x3_j(x)) + BN_3(DW1x1(x)) + BN_id(x) )
    PW stage: z = ReLU( sum_j BN_j(PW1x1_j(y)) + BN_id(y) )

Fusion (per branch, eval-mode BN):  K' = K * g/sqrt(v+eps),  b' = beta - mu * g/sqrt(v+eps).
The 1x1 depthwise kernel is zero-padded to 3x3 at the centre; the identity branch is a
centre-delta depthwise kernel (DW stage) or I_C (PW stage). K = sum K', b = sum b'.

``reparameterize()`` is irreversible: it replaces the branches with two plain convs. Checkpoints
are always saved in the (unfused) training form, so old/new state dicts stay compatible.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fuse_bn(kernel: Optional[torch.Tensor], bn: nn.BatchNorm2d,
             identity_kernel: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold an eval-mode BN into the preceding (bias-free) conv kernel.

    kernel: (Cout, Cin/groups, k, k) or None for an identity branch (then ``identity_kernel``)."""
    k = identity_kernel if kernel is None else kernel
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return k * t, bn.bias - bn.running_mean * bn.weight / std


class RepDWSep(nn.Module):
    """Re-parameterisable DW3x3 -> PW1x1 block, C -> C channels, stride 1, ends in ReLU."""

    def __init__(self, c: int, n_branches: int = 2):
        super().__init__()
        self.c = int(c)
        self.n_branches = int(n_branches)
        self.deployed = False
        # depthwise stage
        self.dw3 = nn.ModuleList([nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
                                  for _ in range(self.n_branches)])
        self.dw3_bn = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(self.n_branches)])
        self.dw1 = nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False)
        self.dw1_bn = nn.BatchNorm2d(c)
        self.dw_id_bn = nn.BatchNorm2d(c)
        # pointwise stage
        self.pw = nn.ModuleList([nn.Conv2d(c, c, 1, 1, 0, bias=False) for _ in range(self.n_branches)])
        self.pw_bn = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(self.n_branches)])
        self.pw_id_bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=False)

    # ── training form ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.pw_fused(self.act(self.dw_fused(x))))
        y = self.dw_id_bn(x) + self.dw1_bn(self.dw1(x))
        for conv, bn in zip(self.dw3, self.dw3_bn):
            y = y + bn(conv(x))
        y = self.act(y)
        z = self.pw_id_bn(y)
        for conv, bn in zip(self.pw, self.pw_bn):
            z = z + bn(conv(y))
        return self.act(z)

    # ── fusion ────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fused_params(self):
        """Return (dw_k (C,1,3,3), dw_b (C,), pw_k (C,C,1,1), pw_b (C,)) of the deploy form."""
        c = self.c
        ref = self.dw1.weight
        dw_k = torch.zeros(c, 1, 3, 3, dtype=ref.dtype, device=ref.device)
        dw_b = torch.zeros(c, dtype=ref.dtype, device=ref.device)
        for conv, bn in zip(self.dw3, self.dw3_bn):
            k, b = _fuse_bn(conv.weight, bn)
            dw_k += k
            dw_b += b
        k1, b1 = _fuse_bn(self.dw1.weight, self.dw1_bn)
        dw_k += F.pad(k1, [1, 1, 1, 1])
        dw_b += b1
        delta = torch.zeros(c, 1, 3, 3, dtype=ref.dtype, device=ref.device)
        delta[:, 0, 1, 1] = 1.0
        kid, bid = _fuse_bn(None, self.dw_id_bn, identity_kernel=delta)
        dw_k += kid
        dw_b += bid

        pw_k = torch.zeros(c, c, 1, 1, dtype=ref.dtype, device=ref.device)
        pw_b = torch.zeros(c, dtype=ref.dtype, device=ref.device)
        for conv, bn in zip(self.pw, self.pw_bn):
            k, b = _fuse_bn(conv.weight, bn)
            pw_k += k
            pw_b += b
        eye = torch.eye(c, dtype=ref.dtype, device=ref.device).view(c, c, 1, 1)
        kid, bid = _fuse_bn(None, self.pw_id_bn, identity_kernel=eye)
        pw_k += kid
        pw_b += bid
        return dw_k, dw_b, pw_k, pw_b

    @torch.no_grad()
    def reparameterize(self) -> None:
        if self.deployed:
            return
        dw_k, dw_b, pw_k, pw_b = self.fused_params()
        c = self.c
        self.dw_fused = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True).to(dw_k.device, dw_k.dtype)
        self.dw_fused.weight.copy_(dw_k)
        self.dw_fused.bias.copy_(dw_b)
        self.pw_fused = nn.Conv2d(c, c, 1, 1, 0, bias=True).to(pw_k.device, pw_k.dtype)
        self.pw_fused.weight.copy_(pw_k)
        self.pw_fused.bias.copy_(pw_b)
        for name in ("dw3", "dw3_bn", "dw1", "dw1_bn", "dw_id_bn", "pw", "pw_bn", "pw_id_bn"):
            delattr(self, name)
        self.deployed = True

    @staticmethod
    def deploy_param_count(c: int) -> int:
        """Analytic parameter count of the fused block: DW(9C + C) + PW(C^2 + C)."""
        return 9 * c + c + c * c + c


def fuse_with_parity(m: nn.Module, x: torch.Tensor, intr: Optional[torch.Tensor],
                     tol: Optional[float] = None) -> int:
    """Re-parameterise a rep4 student in place (no-op for students without RepDWSep) and assert
    that the fused graph reproduces the unfused one: max |d log D| < tol. Used by
    benchmark/fps_bench.py and edge/export_onnx.py. Returns the number of fused blocks.

    The check runs with TF32 disabled (cuDNN convs and CUDA matmuls) and restores the caller's
    flags afterwards: under TF32 (PyTorch's cuDNN default) fused and unfused graphs differ by
    ~5e-3 in log-depth from the 10-bit-mantissa rounding alone, which would fail a real fusion."""
    if not has_unfused_rep(m):
        return 0
    m.eval()
    # Exactness is proven in float64 on a CPU copy (fusion is algebraically exact, so this must
    # agree to ~1e-9). The on-device fp32 comparison below then only has to stay within ordinary
    # fp32 accumulation-order noise: a trained net on GPU differs by ~1e-4 log-depth (0.01% depth)
    # even with TF32 off, which is not a fusion error.
    import copy
    m64 = copy.deepcopy(m).to("cpu", torch.float64).eval()
    x64 = x.detach().to("cpu", torch.float64)
    k64 = intr.detach().to("cpu", torch.float64) if intr is not None else None
    tf32_0 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            ref64 = m64(x64, k64)[:, :1]
            m64.reparameterize() if hasattr(m64, "reparameterize") else reparameterize_model(m64)
            err64 = float((m64(x64, k64)[:, :1] - ref64).abs().max())
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32_0
    del m64
    if not err64 < 1e-6:
        raise AssertionError(f"rep4 fusion is not exact in float64: max|dlogD|={err64:.3e}")
    if tol is None:
        tol = 2e-3                      # fp32 on-device check; exactness was proven in fp64 above
    tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            ref = m(x, intr)[:, :1].float()
            n = m.reparameterize() if hasattr(m, "reparameterize") else reparameterize_model(m)
            got = m(x, intr)[:, :1].float()
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    err = float((got - ref).abs().max())
    if not err < tol:
        raise AssertionError(f"rep4 fusion parity failed: max|dlogD|={err:.3e} >= {tol:.0e}")
    print(f"[rep4] fused {n} blocks, parity max|dlogD|={err:.2e}", flush=True)
    return n


def has_unfused_rep(m: nn.Module) -> bool:
    """True when ``m`` contains a RepDWSep still in its multi-branch training form."""
    return any(isinstance(mod, RepDWSep) and not mod.deployed for mod in m.modules())


def reparameterize_model(m: nn.Module) -> int:
    """Fuse every RepDWSep inside ``m`` in place. Returns the number of blocks fused."""
    blocks: List[RepDWSep] = [mod for mod in m.modules() if isinstance(mod, RepDWSep) and not mod.deployed]
    for b in blocks:
        b.reparameterize()
    return len(blocks)
