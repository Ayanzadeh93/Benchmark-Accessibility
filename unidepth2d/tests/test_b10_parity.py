"""b10 parity: with no b20 recipe tags the new code is bit-identical to the ORIGINAL code in
C:/Tim/nature/unidepth2d (loaded read-only via importlib). CPU only.

(a) fresh model, b10 config, same seed -> identical weights and outputs
(b) S16_prox / b1_mix checkpoints -> identical outputs under old and new code
(c) DistillLoss on random tensors -> identical loss and terms
(d) MultiSourceDataset with the b10 data recipe -> identical samples (same numpy seed)
"""
import copy
import os

import numpy as np
import pytest
import torch

from distill.dataset import MultiSourceDataset
from distill.losses import DistillLoss
from distill.student import build_student
from _orig import CKPTS, b10_student_cfg, load_orig


def _orig_student():
    return load_orig("distill/student.py", "orig_b10_student")


def _orig_losses():
    return load_orig("distill/losses.py", "orig_b10_losses")


def test_a_fresh_model_identical():
    orig = _orig_student()
    cfg = b10_student_cfg()
    torch.manual_seed(1234)
    m_old = orig.build_student(copy.deepcopy(cfg)).eval()
    torch.manual_seed(1234)
    m_new = build_student(copy.deepcopy(cfg)).eval()
    sd_o, sd_n = m_old.state_dict(), m_new.state_dict()
    assert list(sd_o.keys()) == list(sd_n.keys())
    for k in sd_o:
        assert torch.equal(sd_o[k], sd_n[k]), k
    torch.manual_seed(7)
    x = torch.randn(2, 3, 480, 640)
    k = torch.tensor([[0.81, 1.08, 0.5, 0.53], [0.6, 0.8, 0.45, 0.55]])
    with torch.no_grad():
        assert torch.equal(m_old(x, k), m_new(x, k))
        assert torch.equal(m_old(x, None), m_new(x, None))
        assert torch.equal(m_old.infer_metres(x, k), m_new.infer_metres(x, k))
    # train-mode forward (BN batch stats) and gradients are identical too
    m_old.train()
    m_new.train()
    yo, yn = m_old(x, k), m_new(x, k)
    assert torch.equal(yo, yn)
    yo.mean().backward()
    yn.mean().backward()
    for (no, po), (nn_, pn) in zip(m_old.named_parameters(), m_new.named_parameters()):
        assert no == nn_
        assert (po.grad is None and pn.grad is None) or torch.equal(po.grad, pn.grad), no


@pytest.mark.parametrize("name", ["S16_prox", "b1_mix"])
def test_b_old_checkpoints_identical(name):
    p = os.path.join(CKPTS, name, "best.pt")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    orig = _orig_student()
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ck["cfg"])
    cfg["student"]["pretrained_backbone"] = False
    m_old = orig.build_student(copy.deepcopy(cfg)).eval()
    m_new = build_student(copy.deepcopy(cfg)).eval()
    m_old.load_state_dict(ck["model"], strict=True)
    m_new.load_state_dict(ck["model"], strict=True)
    assert m_new.reparameterize() == 0                      # nothing to fuse in a b10-family ckpt
    torch.manual_seed(11)
    x = torch.randn(2, 3, 480, 640)
    k = torch.tensor([[0.81, 1.08, 0.5, 0.53], [0.7, 0.9, 0.5, 0.5]])
    with torch.no_grad():
        assert torch.equal(m_old(x, k), m_new(x, k))
        assert torch.equal(m_old(x, None), m_new(x, None))


def _b10_loss_cfg():
    return {"student": {"depth_min_m": 0.2, "depth_max_m": 30.0},
            "distill": {"use_teacher_confidence": False,
                        "loss": {"silog_weight": 1.0, "silog_lambda": 0.85, "grad_weight": 0.5,
                                 "grad_scales": 4, "ssi_weight": 0.0, "eg_ssi_weight": 0.3,
                                 "prox_silog_weight": 0.5}}}


@pytest.mark.parametrize("variant", ["b10", "default", "allterms"])
def test_c_distill_loss_identical(variant):
    orig = _orig_losses()
    cfg = _b10_loss_cfg()
    if variant == "default":
        cfg["distill"]["loss"] = {}
    elif variant == "allterms":
        cfg["distill"]["loss"].update(ssi_weight=0.2, normal_weight=0.3, scale_bias_weight=0.3)
        cfg["distill"]["use_teacher_confidence"] = True
    g = torch.Generator().manual_seed(0)
    B, H, W = 3, 96, 128
    pred = (0.8 + 0.5 * torch.randn(B, 1, H, W, generator=g)).requires_grad_(True)
    pred2 = pred.detach().clone().requires_grad_(True)
    tgt = 0.9 + 0.4 * torch.randn(B, 1, H, W, generator=g)
    valid = torch.rand(B, 1, H, W, generator=g) > 0.15
    conf = torch.rand(B, 1, H, W, generator=g)
    weight = 1.0 + torch.rand(B, 1, H, W, generator=g)
    intr = torch.tensor([[0.81, 1.08, 0.5, 0.53]] * B)
    is_gt = torch.tensor([1, 0, 1])
    for wt in (weight, None):
        lo, to = orig.DistillLoss(cfg)(pred, tgt, valid, conf, weight=wt, intrinsics=intr, is_gt=is_gt)
        ln, tn = DistillLoss(cfg)(pred2, tgt, valid, conf, weight=wt, intrinsics=intr, is_gt=is_gt)
        assert torch.equal(lo, ln)
        assert to.keys() == tn.keys()
        for k in to:
            assert torch.equal(to[k], tn[k]), k
        lo.backward()
        ln.backward()
        assert torch.equal(pred.grad, pred2.grad)
        pred.grad = None
        pred2.grad = None


def test_d_dataset_identical():
    from common.io_utils import load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.yaml"))
    if not os.path.exists(cfg["data_gt"]["manifest"]):
        pytest.skip("real corpus not available")
    orig = load_orig("distill/dataset.py", "orig_b10_dataset")
    # b10 data recipe: s5 crop + range weighting, intrinsics on (ray/FiLM)
    cfg["student"]["use_intrinsics"] = True
    cfg["distill"]["crop_context"] = {"p": 0.5, "min_scale": 0.5}
    cfg["distill"]["range_weight_alpha"] = 1.0
    for split, aug in (("train", True), ("val", False)):
        d_old = orig.MultiSourceDataset(copy.deepcopy(cfg), split, augment=aug)
        d_new = MultiSourceDataset(copy.deepcopy(cfg), split, augment=aug)
        assert [r["id"] for r in d_old.rows] == [r["id"] for r in d_new.rows]
        rng = np.random.RandomState(0)
        idxs = rng.choice(len(d_old), size=12, replace=False)
        for seed, i in enumerate(idxs):
            np.random.seed(seed)
            a = d_old[int(i)]
            np.random.seed(seed)
            b = d_new[int(i)]
            assert a.keys() == b.keys()                     # no 'pad' key without b20 tags
            for k in a:
                if torch.is_tensor(a[k]):
                    assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), (split, i, k)
                else:
                    assert a[k] == b[k], (split, i, k)
        # identical RNG consumption: after the same sample both streams are at the same state
        np.random.seed(99)
        d_old[int(idxs[0])]
        s_old = np.random.get_state()
        np.random.seed(99)
        d_new[int(idxs[0])]
        s_new = np.random.get_state()
        assert s_new[2] == s_old[2] and np.array_equal(s_new[1], s_old[1])


# ── (d') dataset parity on the REAL b10 checkpoint cfg, every source visited (folded from
# reviewer scratch; reads outputs/ only) ────────────────────────────────────────────────────
B10_CKPT = os.path.join(CKPTS, "b10", "best.pt")


def _b10_real_cfg():
    if not os.path.exists(B10_CKPT):
        pytest.skip("no b10 ckpt")
    return torch.load(B10_CKPT, map_location="cpu", weights_only=False)["cfg"]


def _cmp_sample(a, b, tag):
    assert a.keys() == b.keys(), tag
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), (tag, k)
        else:
            assert a[k] == b[k], (tag, k)


@pytest.mark.parametrize("variant", ["b10ckpt", "s6s7"])
def test_dataset_parity_real_cfg(variant):
    cfg = _b10_real_cfg()
    if variant == "s6s7":       # other legacy paths that touch the edited code (conf / taskw / boxes)
        cfg["distill"]["conf_mode"] = "rank"
        cfg["distill"]["task_weight_maps"] = True
        cfg["distill"]["loss"]["obj_dist_weight"] = 0.3
    orig = load_orig("distill/dataset.py", "orig_b10_dataset")
    for split, aug, n in (("train", True, 40), ("val", False, 15)):
        d_old = orig.MultiSourceDataset(copy.deepcopy(cfg), split, augment=aug)
        d_new = MultiSourceDataset(copy.deepcopy(cfg), split, augment=aug)
        assert [r["id"] for r in d_old.rows] == [r["id"] for r in d_new.rows]
        # stratify: make sure every source is visited
        by = {}
        for i, s in enumerate(d_old.sources):
            by.setdefault(s, []).append(i)
        rng = np.random.RandomState(5)
        idxs = []
        for s, ii in sorted(by.items()):
            idxs += list(rng.choice(ii, size=min(len(ii), max(2, n // len(by))), replace=False))
        for seed, i in enumerate(idxs):
            np.random.seed(1000 + seed)
            a = d_old[int(i)]
            st_a = np.random.get_state()
            np.random.seed(1000 + seed)
            b = d_new[int(i)]
            st_b = np.random.get_state()
            _cmp_sample(a, b, (variant, split, int(i)))
            assert st_a[2] == st_b[2] and np.array_equal(st_a[1], st_b[1])


# ── (e) benchmark/run_bench.py StudentModelRunner: identical predictions to the original runner
@pytest.mark.parametrize("name", ["S16_prox", "b1_mix"])
def test_e_run_bench_runner_identical(name):
    p = os.path.join(CKPTS, name, "best.pt")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    orig_rb = load_orig("benchmark/run_bench.py", "orig_b10_run_bench")
    from benchmark.run_bench import StudentModelRunner
    torch.set_num_threads(4)
    old, new = orig_rb.StudentModelRunner(p, "cpu"), StudentModelRunner(p, "cpu")
    assert old.use_intr == new.use_intr
    rng = np.random.RandomState(0)
    imgs = [rng.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            rng.randint(0, 256, (375, 500, 3), dtype=np.uint8)]
    Ks = [np.array([[518.9, 0, 325.6], [0, 519.5, 253.7], [0, 0, 1.0]]), None]
    for im, K in zip(imgs, Ks):
        assert np.array_equal(old.predict(im, K)["depth"], new.predict(im, K)["depth"])
