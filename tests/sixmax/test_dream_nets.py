"""
Tier 1 unit tests for DreamMLP (Task 8).

DreamMLP.forward (via __call__) accepts at::Tensor, but Python torch tensors
cannot be passed across the pybind11 boundary when sixmax.so is linked to a
separate libtorch copy.  All forward-pass tests therefore use forward_vec(),
which accepts a flat Python float list and returns a flat float list.

Tests verify:
  - DreamMLP can be constructed with arbitrary shapes
  - forward_vec returns the correct number of elements
  - output values are all finite
  - output shape matches vocab.size() when n_actions=vocab.size()
"""
import math
import random

import sixmax_dream


def _randf(n: int) -> list:
    rng = random.Random(42)
    return [rng.gauss(0, 1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construct_6_actions():
    net = sixmax_dream.DreamMLP(154, 256, 3, 6)
    assert net is not None


def test_construct_custom_shape():
    net = sixmax_dream.DreamMLP(154, 64, 2, 10)
    assert net is not None


# ---------------------------------------------------------------------------
# Forward pass via forward_vec (avoids Python-tensor bridge)
# ---------------------------------------------------------------------------

def test_output_shape_6_actions():
    """forward_vec batch=4, 6 outputs → flat list of length 4*6=24."""
    net = sixmax_dream.DreamMLP(154, 256, 3, 6)
    batch = 4
    x = _randf(batch * 154)
    out = net.forward_vec(x, batch)
    assert len(out) == batch * 6


def test_output_shape_10_actions():
    """forward_vec batch=1, 10 outputs → 10 elements."""
    net = sixmax_dream.DreamMLP(154, 256, 3, 10)
    x = _randf(154)
    out = net.forward_vec(x, 1)
    assert len(out) == 10


def test_output_shape_matches_vocab(default_vocab):
    """Output dim equals vocab.size() for a net built with that n_actions."""
    n = default_vocab.size()
    net = sixmax_dream.DreamMLP(sixmax_dream.FEATURE_DIM, 256, 3, n)
    batch = 2
    x = _randf(batch * sixmax_dream.FEATURE_DIM)
    out = net.forward_vec(x, batch)
    assert len(out) == batch * n


def test_output_all_finite():
    """No NaN or inf in forward output."""
    net = sixmax_dream.DreamMLP(154, 128, 2, 6)
    x = _randf(4 * 154)
    out = net.forward_vec(x, 4)
    for v in out:
        assert math.isfinite(v), f"non-finite value {v} in output"


def test_different_seeds_give_different_output():
    """Two randomly-initialised nets should produce different outputs (very likely)."""
    net1 = sixmax_dream.DreamMLP(154, 256, 3, 6)
    net2 = sixmax_dream.DreamMLP(154, 256, 3, 6)
    x = _randf(154)
    out1 = net1.forward_vec(x, 1)
    out2 = net2.forward_vec(x, 1)
    # At least one element should differ
    assert any(abs(a - b) > 1e-6 for a, b in zip(out1, out2)), \
        "two nets gave identical outputs (extremely unlikely if weights differ)"
