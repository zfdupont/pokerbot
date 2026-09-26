"""
Tier 1 unit tests for WeightedReservoir (Task 8).

add() and sample_batch() accept at::Tensor but cannot receive Python torch
tensors across the pybind11 ABI boundary.  Tests use add_vec() and
sample_batch_vec(), which accept/return Python float lists.

Tests verify:
  - Reservoir caps at its stated capacity
  - add_vec + sample_batch_vec return correct shapes
  - Returned weights are all positive
  - clear() resets size to 0
  - Sampling fewer items than capacity works
"""
import random

import sixmax_dream

_FEAT_DIM = sixmax_dream.FEATURE_DIM   # 154
_TGT_DIM  = 6                          # 6 actions (default)


def _rand_feat(seed=None):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(_FEAT_DIM)]


def _rand_tgt(seed=None):
    rng = random.Random(seed)
    return [rng.random() for _ in range(_TGT_DIM)]


# ---------------------------------------------------------------------------
# Capacity cap
# ---------------------------------------------------------------------------

def test_size_cap():
    """Reservoir with capacity 10 must not grow beyond 10."""
    r = sixmax_dream.WeightedReservoir(capacity=10, seed=42)
    for i in range(50):
        r.add_vec(_rand_feat(i), _rand_tgt(i), float(i + 1))
    assert r.size() == 10


def test_size_below_cap():
    """Reservoir reports exact size when still below capacity."""
    r = sixmax_dream.WeightedReservoir(capacity=100, seed=0)
    for i in range(30):
        r.add_vec(_rand_feat(i), _rand_tgt(i), 1.0)
    assert r.size() == 30


# ---------------------------------------------------------------------------
# add_vec + sample_batch_vec shapes
# ---------------------------------------------------------------------------

def test_sample_batch_shapes():
    """sample_batch_vec returns flat lists with the expected element counts."""
    r = sixmax_dream.WeightedReservoir(capacity=1000, seed=7)
    for i in range(100):
        r.add_vec(_rand_feat(i), _rand_tgt(i), 1.0)

    feat_f, tgt_f, w_f = r.sample_batch_vec(32)
    assert len(feat_f)  == 32 * _FEAT_DIM, f"feat len {len(feat_f)} != {32*_FEAT_DIM}"
    assert len(tgt_f)   == 32 * _TGT_DIM,  f"tgt len {len(tgt_f)} != {32*_TGT_DIM}"
    assert len(w_f)     == 32,              f"weight len {len(w_f)} != 32"


def test_weights_positive():
    """All sampled weights must be strictly positive."""
    r = sixmax_dream.WeightedReservoir(capacity=200, seed=99)
    for i in range(100):
        r.add_vec(_rand_feat(i), _rand_tgt(i), float(i + 1))

    _, _, w_f = r.sample_batch_vec(50)
    assert all(w > 0 for w in w_f), "some sampled weight is non-positive"


def test_sample_size_1():
    """sample_batch_vec(1) works correctly."""
    r = sixmax_dream.WeightedReservoir(capacity=50, seed=1)
    r.add_vec(_rand_feat(0), _rand_tgt(0), 2.5)
    feat_f, tgt_f, w_f = r.sample_batch_vec(1)
    assert len(feat_f) == _FEAT_DIM
    assert len(tgt_f)  == _TGT_DIM
    assert len(w_f)    == 1
    assert w_f[0] > 0


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear():
    """clear() resets the reservoir to size 0."""
    r = sixmax_dream.WeightedReservoir(capacity=100, seed=5)
    r.add_vec(_rand_feat(0), _rand_tgt(0), 1.0)
    r.clear()
    assert r.size() == 0


def test_clear_then_add():
    """After clear(), new items can be added normally."""
    r = sixmax_dream.WeightedReservoir(capacity=100, seed=3)
    for i in range(20):
        r.add_vec(_rand_feat(i), _rand_tgt(i), 1.0)
    r.clear()
    r.add_vec(_rand_feat(99), _rand_tgt(99), 1.0)
    assert r.size() == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_sample():
    """Two reservoirs with the same seed and identical inserts return the
    same sample order."""
    def _fill(seed):
        r = sixmax_dream.WeightedReservoir(capacity=20, seed=seed)
        for i in range(50):
            r.add_vec(_rand_feat(i), _rand_tgt(i), float(i + 1))
        feat_f, _, _ = r.sample_batch_vec(5)
        return feat_f

    assert _fill(42) == _fill(42)
