"""b20 C2 cstm: canonical-focal term c = log(f_px / 518.9). CPU only."""
import math

import torch

from distill.student import _cstm_term, build_student
from _orig import b10_student_cfg

FC = 518.9
NYU_K = [0.81071875, 1.0822291666666668, 0.50871875, 0.528625]     # data_gt manifest, 640x480


def test_zoom_shifts_c_by_log_k():
    H, W = 480, 640
    k0 = torch.tensor([NYU_K], dtype=torch.float64)
    # s5-style crop of linear size 1/1.5 at an arbitrary offset, resized back to (H, W):
    # normalised K update fx' = fx / nw, cx' = (cx - nx0) / nw  (dataset.py S5 block)
    s = 1.0 / 1.5
    nx0, ny0 = 0.13, 0.07
    k1 = k0.clone()
    k1[:, 0] = k0[:, 0] / s
    k1[:, 1] = k0[:, 1] / s
    k1[:, 2] = (k0[:, 2] - nx0) / s
    k1[:, 3] = (k0[:, 3] - ny0) / s
    d = _cstm_term(k1, H, W, FC) - _cstm_term(k0, H, W, FC)
    assert abs(float(d) - math.log(1.5)) < 1e-12


def test_uncrop_shifts_c_by_log_z():
    k0 = torch.tensor([NYU_K], dtype=torch.float64)
    z = 0.7
    k1 = k0.clone()
    k1[:, 0] *= z
    k1[:, 1] *= z
    d = _cstm_term(k1, 480, 640, FC) - _cstm_term(k0, 480, 640, FC)
    assert abs(float(d) - math.log(z)) < 1e-12


def test_nyu_k_is_canonical():
    c = _cstm_term(torch.tensor([NYU_K]), 480, 640, FC)
    assert abs(float(c)) < 1e-3


def test_resolution_consistency():
    # same normalised K at half resolution -> f_px halves -> c drops by log 2
    k = torch.tensor([NYU_K], dtype=torch.float64)
    d = _cstm_term(k, 240, 320, FC) - _cstm_term(k, 480, 640, FC)
    assert abs(float(d) + math.log(2.0)) < 1e-12


def test_cstm_without_dsh_adds_exactly_c():
    cfg0 = b10_student_cfg()
    cfg1 = b10_student_cfg()
    cfg1["student"]["cstm_fc"] = FC
    torch.manual_seed(0)
    m0 = build_student(cfg0).eval()
    torch.manual_seed(0)
    m1 = build_student(cfg1).eval()
    m1.load_state_dict(m0.state_dict())            # cstm adds no parameters
    x = torch.randn(2, 3, 128, 160)
    k = torch.tensor([NYU_K, [0.5, 0.9, 0.5, 0.5]])
    with torch.no_grad():
        o0, o1 = m0(x, k), m1(x, k)
    c = _cstm_term(k, 128, 160, FC).view(2, 1, 1, 1)
    assert torch.allclose(o1, o0 + c, atol=1e-6)


# ── one intrinsics-input rule for every loader / exporter ───────────────────────────────────
def test_needs_intrinsics_rule():
    from distill.student import needs_intrinsics
    for ui in (False, True):
        for ray in (False, True):
            s = {"use_intrinsics": ui, "ray_map": ray}
            assert needs_intrinsics(s) == (ui or ray)                 # run_bench / fps / exporters
            assert needs_intrinsics(s, legacy_ray=False) == ui        # gt_eval / evaluate / ...
            for b20 in ({"scale_head": True}, {"cstm_fc": 518.9}):   # b20 always needs K
                assert needs_intrinsics({**s, **b20}) and needs_intrinsics({**s, **b20}, legacy_ray=False)
    assert not needs_intrinsics({"cstm_fc": 0.0, "scale_head": False})
    assert not needs_intrinsics({})


def test_loaders_use_needs_intrinsics():
    """No loader / exporter keeps its own copy of the rule (they used to disagree on dsh/cstm)."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("benchmark/run_bench.py", "benchmark/gt_eval.py", "benchmark/fps_bench.py",
                "distill/evaluate.py", "distill/predict_objects.py", "scripts/cache_assistant.py",
                "edge/export_onnx.py", "edge/export_coreml.py", "edge/export_v2.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        assert "needs_intrinsics(" in src, rel
        assert 'get("use_intrinsics"' not in src, rel
