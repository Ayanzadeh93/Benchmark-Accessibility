"""Unit tests for edge/quant_sim.py (weight-only N-bit quantization, simulated)."""
import pytest
import torch
import torch.nn as nn

from edge.quant_sim import quantize_tensor, quantize_weights_


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.Conv2d(8, 4, 1), nn.Flatten(), nn.Linear(4 * 8 * 8, 5))


def test_values_lie_on_integer_grid_per_channel():
    torch.manual_seed(0)
    w = torch.randn(6, 4, 3, 3)
    wq = quantize_tensor(w, bits=8, per_channel=True)
    scale = w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0
    q = wq / scale
    assert torch.allclose(q, q.round(), atol=1e-4)
    assert q.abs().max() <= 127 + 1e-4


def test_error_bounded_by_half_step():
    torch.manual_seed(1)
    w = torch.randn(5, 7)
    for bits in (4, 6, 8):
        wq = quantize_tensor(w, bits=bits, per_channel=True)
        step = w.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
        assert ((wq - w).abs() <= step / 2 + 1e-6).all()


def test_idempotent():
    torch.manual_seed(2)
    w = torch.randn(4, 3, 3, 3)
    once = quantize_tensor(w, 8, True)
    twice = quantize_tensor(once, 8, True)
    assert torch.allclose(once, twice, atol=1e-6)


def test_more_bits_means_less_error():
    torch.manual_seed(3)
    w = torch.randn(16, 8, 3, 3)
    errs = [float((quantize_tensor(w, b) - w).pow(2).mean()) for b in (4, 6, 8, 10)]
    assert errs == sorted(errs, reverse=True)


def test_per_channel_is_invariant_to_channel_scaling():
    """The property that makes per-channel results valid before OR after BatchNorm folding."""
    torch.manual_seed(4)
    w = torch.randn(6, 4, 3, 3)
    k = torch.tensor([0.01, 0.5, 1.0, 3.0, 40.0, 900.0]).view(6, 1, 1, 1)
    a = quantize_tensor(w * k, 8, per_channel=True)
    b = quantize_tensor(w, 8, per_channel=True) * k
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_per_channel_beats_per_tensor_on_skewed_channels():
    torch.manual_seed(5)
    w = torch.randn(6, 4, 3, 3) * torch.tensor([0.01, 0.1, 1, 1, 10, 100.]).view(6, 1, 1, 1)
    amax = w.abs().amax(dim=(1, 2, 3), keepdim=True)

    def rel(q):
        return float(((q - w) / amax).pow(2).mean())

    assert rel(quantize_tensor(w, 8, True)) < rel(quantize_tensor(w, 8, False))


def test_zero_channel_stays_zero_and_finite():
    w = torch.randn(3, 2, 3, 3)
    w[1] = 0
    wq = quantize_tensor(w, 8, True)
    assert torch.isfinite(wq).all() and (wq[1] == 0).all()


def test_bits_out_of_range_raises():
    with pytest.raises(ValueError):
        quantize_tensor(torch.randn(2, 2), bits=1)


def test_model_stats_and_size_accounting():
    net = _net()
    n_total = sum(p.numel() for p in net.parameters())
    n_w = sum(m.weight.numel() for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    stats = quantize_weights_(net, bits=8, per_channel=True)
    assert stats["params_quantized"] == n_w
    assert stats["params_quantized"] + stats["params_float"] == n_total
    assert stats["size_mb"] == pytest.approx((n_w * 1 + (n_total - n_w) * 4) / 1e6, abs=1e-3)
    assert stats["size_mb"] < stats["size_mb_fp32"]
    assert 0 < stats["weight_rel_rmse"] < 0.05


def test_8bit_output_stays_close_and_4bit_is_worse():
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(7))
    ref = _net().eval()(x)
    errs = {}
    for bits in (8, 4):
        net = _net().eval()
        quantize_weights_(net, bits=bits)
        errs[bits] = float((net(x) - ref).abs().mean() / ref.abs().mean())
    assert errs[8] < 0.05
    assert errs[4] > errs[8]


def test_shared_weight_quantized_once():
    conv = nn.Conv2d(3, 3, 3, padding=1)
    net = nn.Sequential(conv, nn.ReLU(), conv)
    stats = quantize_weights_(net, 8)
    assert stats["params_quantized"] == conv.weight.numel()
