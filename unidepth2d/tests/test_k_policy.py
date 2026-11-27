"""A05: one K resolution path (StudentConfig.k_none_policy). CPU only.

* default_k: forward(x, None) == forward(x, DEFAULT_K) exactly, for every b10/b20 config;
* legacy (default): outputs equal the ORIGINAL code (branch b20-canonscale) for K=None and GT K;
* GT-K outputs are identical under both policies;
* there is one DEFAULT_K object across the eval modules.
Real checkpoints are used when present (non-zero FiLM weights); random-init configs otherwise.
"""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from distill import student as S  # noqa: E402

ORIG_BRANCH = "b20-canonscale"
CK = os.path.join(ROOT, "outputs", "checkpoints")
H, W = 96, 128
BASE = {"backbone": "efficientvit_b1.r224_in1k", "pretrained_backbone": False,
        "use_intrinsics": True, "ray_map": True, "train_size": [H, W]}
CONFIGS = {   # b10 and every b20 variant (random init)
    "b10": {},
    "b20core": {"scale_head": True, "cstm_fc": 518.9},
    "b20dws": {"scale_head": True, "cstm_fc": 518.9, "dws4": True},
    "b20rep4": {"scale_head": True, "cstm_fc": 518.9, "rep4": True},
    "cstm_only": {"cstm_fc": 518.9},
    "film_only": {"ray_map": False},
}
REAL = ("b10", "b20core", "b20dws", "b20", "b1_mix")


def _orig_student():
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{ORIG_BRANCH}:./distill/student.py"],
                             check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"original branch {ORIG_BRANCH} unavailable: {e}")
    name = "_orig_student_a05"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    exec(compile(src, f"{ORIG_BRANCH}:distill/student.py", "exec"), mod.__dict__)
    return mod


def _models(build):
    """[(name, model)] in eval mode: random-init configs + real checkpoints if present."""
    out = []
    for name, kw in CONFIGS.items():
        torch.manual_seed(0)
        out.append((name, build({"student": {**BASE, **kw}}).eval()))
    for name in REAL:
        p = os.path.join(CK, name, "last.pt")
        if os.path.isfile(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = {"student": {**ck["cfg"]["student"], "pretrained_backbone": False}}
            m = build(cfg).eval()
            m.load_state_dict(ck["model"])
            out.append((f"ckpt:{name}", m))
    return out


def _x(B=2):
    return torch.randn(B, 3, H, W, generator=torch.Generator().manual_seed(3))


def _gtk(B=2):
    return torch.tensor([[0.81, 1.08, 0.51, 0.53], [1.02, 1.36, 0.49, 0.52]][:B])


def _dk(B):
    return torch.tensor([S.DEFAULT_K] * B, dtype=torch.float32)


@pytest.mark.parametrize("B", [1, 2])
def test_default_k_none_equals_explicit_default_k(B):
    x = _x(B)
    for name, m in _models(S.build_student):
        m.set_k_none_policy("default_k")
        with torch.no_grad():
            a, b = m(x, None), m(x, _dk(B))
        assert torch.equal(a, b), name


def test_legacy_equals_original_code_and_gt_k_identical_under_both():
    orig = _orig_student()
    x, k = _x(), _gtk()
    new_models = dict(_models(S.build_student))
    for name, mo in _models(orig.build_student):
        mn = new_models[name]
        mn.load_state_dict(mo.state_dict())                 # same weights for random-init too
        assert mn.k_none_policy == "legacy"                 # the default
        with torch.no_grad():
            ref_none, ref_k = mo(x, None), mo(x, k)
            assert torch.equal(mn(x, None), ref_none), name
            assert torch.equal(mn(x, k), ref_k), name
            mn.set_k_none_policy("default_k")
            assert torch.equal(mn(x, k), ref_k), name        # GT K: identical under both
            # FiLM is what legacy dropped: with K=None the two policies differ iff FiLM exists
            differ = not torch.equal(mn(x, None), ref_none)
            film_live = mn.use_intrinsics and float(mn.film[-1].weight.abs().sum()) > 0
            assert differ == film_live, name


def test_default_policy_reproduces_b10_init_and_state_dict():
    torch.manual_seed(0)
    a = S.build_student({"student": {**BASE, **CONFIGS["b20core"]}})
    torch.manual_seed(0)
    b = S.build_student({"student": {**BASE, **CONFIGS["b20core"], "k_none_policy": "default_k"}})
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[n], b.state_dict()[n]) for n in a.state_dict())
    assert a.k_none_policy == "legacy" and b.k_none_policy == "default_k"


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        S.DepthStudent(S.StudentConfig(backbone=BASE["backbone"], pretrained_backbone=False,
                                       k_none_policy="uncalibrated"))
    m = S.build_student({"student": BASE})
    with pytest.raises(ValueError):
        m.set_k_none_policy("none")


def test_one_default_k_object():
    assert S.DepthStudent._DEFAULT_K is S.DEFAULT_K
    assert S.DEFAULT_K == (0.8, 1.08, 0.5, 0.5)
    import benchmark.nyu_eval as nyu
    import distill.visualize as viz
    assert nyu.DEFAULT_K is S.DEFAULT_K and viz.DEFAULT_K is S.DEFAULT_K


def test_eval_callers_default_unchanged_and_opt_in():
    import benchmark.nyu_eval as nyu
    s = {"use_intrinsics": True, "ray_map": True}
    assert nyu.student_intrinsics("none", s, 480, 640, "cpu") is None       # historical default
    assert torch.equal(nyu.student_intrinsics("default", s, 480, 640, "cpu"), _dk(1))
    gt = nyu.student_intrinsics("gt", s, 480, 640, "cpu")
    from benchmark.datasets_tier1 import NYU_K
    exp = torch.tensor([[NYU_K[0, 0] / 640, NYU_K[1, 1] / 480, NYU_K[0, 2] / 640,
                         NYU_K[1, 2] / 480]], dtype=torch.float32)
    assert torch.equal(gt, exp)
    assert nyu.student_intrinsics("gt", {"use_intrinsics": False}, 480, 640, "cpu") is None


def test_safety_default_k_builds_default_k_students(monkeypatch):
    import benchmark.run_bench as rb
    import benchmark.safety as sf
    seen = []
    monkeypatch.setattr(rb, "build_model",
                        lambda spec, dev, gt_cam, res, k_none_policy="legacy":
                        seen.append((gt_cam, k_none_policy)))
    for mode in ("none", "default-k", "gt-k"):
        sf._build("student:x.pt", "cpu", mode)
    assert seen == [(False, "legacy"), (False, "default_k"), (True, "legacy")]


def test_run_bench_runner_policy_and_name():
    p = os.path.join(CK, "b20dws", "last.pt")
    if not os.path.isfile(p):
        pytest.skip(f"checkpoint missing: {p}")
    import numpy as np
    from benchmark.run_bench import StudentModelRunner
    leg = StudentModelRunner(p, "cpu")
    kd = StudentModelRunner(p, "cpu", k_none_policy="default_k")
    assert leg.name == "student-b20dws" and kd.name == "student-b20dws+kdef"
    assert leg.model.k_none_policy == "legacy" and kd.model.k_none_policy == "default_k"
    rgb = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    assert np.array_equal(leg.predict(rgb, K)["depth"], kd.predict(rgb, K)["depth"])
    assert not np.array_equal(leg.predict(rgb, None)["depth"], kd.predict(rgb, None)["depth"])
