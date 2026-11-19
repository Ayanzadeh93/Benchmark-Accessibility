"""Resume restores RNG state even when last.pt was loaded with map_location on the GPU (PR #1 review)."""
import random

import numpy as np
import pytest
import torch

from distill.v2.train import rng_state, set_rng_state


def _draws():
    out = [random.random(), float(np.random.rand()), float(torch.rand(1))]
    if torch.cuda.is_available():
        out.append(float(torch.rand(1, device="cuda")))
    return out


def test_set_rng_state_roundtrip_cpu():
    st = rng_state()
    a = _draws()
    set_rng_state(st)
    assert _draws() == a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_set_rng_state_accepts_states_mapped_to_cuda():
    st = rng_state()
    a = _draws()
    mapped = dict(st, torch=st["torch"].cuda(), cuda=[s.cuda() for s in st["cuda"]])
    set_rng_state(mapped)                      # previously: TypeError (RNG state must be a ByteTensor)
    assert _draws() == a                       # CPU and CUDA streams both restored
