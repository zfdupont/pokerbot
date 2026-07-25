"""
Tier 1 unit tests for DREAM feature encoding (Task 8).

encode_state is not exposed as a standalone Python function; it is called
internally by DreamStrategy.get_probs.  Tests here verify:
  - FEATURE_DIM constant equals 154
  - CHIP_NORM and RAISE_NORM constants are correct
  - encode_state is exercised indirectly via DreamStrategy.get_probs:
      * output is a probability vector of length vocab.size()
      * every entry is in [0, 1]
      * entries sum to 1.0 (within floating-point tolerance)
      * illegal actions (mask==0) have zero probability
"""
import os
import math
import tempfile

import pytest
import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_preflop_state(vocab):
    """Return an EngineGameState at preflop (heads-up, fixed deck)."""
    cfg = sixmax.EngineConfig(num_players=2)
    deck = list(range(52))
    return sixmax.EngineGameState(cfg, 0, deck, vocab, [])


def _make_dream_strategy(vocab, iterations=0):
    """Save a randomly-initialised DreamStrategy and reload it."""
    adv = sixmax.DreamMLP(sixmax.FEATURE_DIM, 256, 3, vocab.size())
    strat = sixmax.DreamMLP(sixmax.FEATURE_DIM, 256, 3, vocab.size())
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        sixmax.save_dream_checkpoint(path, adv, strat, iterations, vocab.hash())
        return sixmax.DreamStrategy.load(path, "cpu", vocab)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Constant tests
# ---------------------------------------------------------------------------

def test_feature_dim():
    assert sixmax.FEATURE_DIM == 154


def test_chip_norm():
    assert sixmax.CHIP_NORM == 100.0


def test_raise_norm():
    assert sixmax.RAISE_NORM == 5.0


# ---------------------------------------------------------------------------
# Indirect encode_state tests via DreamStrategy.get_probs
# ---------------------------------------------------------------------------

def test_get_probs_length(default_vocab):
    """get_probs returns exactly vocab.size() values."""
    ds = _make_dream_strategy(default_vocab)
    state = _make_preflop_state(default_vocab)
    probs = ds.get_probs(state)
    assert len(probs) == default_vocab.size()


def test_get_probs_in_unit_interval(default_vocab):
    """Every probability is in [0, 1]."""
    ds = _make_dream_strategy(default_vocab)
    state = _make_preflop_state(default_vocab)
    probs = ds.get_probs(state)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"probability {p} out of [0, 1]"


def test_get_probs_sum_to_one(default_vocab):
    """Probabilities over legal actions sum to 1.0."""
    ds = _make_dream_strategy(default_vocab)
    state = _make_preflop_state(default_vocab)
    probs = ds.get_probs(state)
    total = sum(probs)
    assert math.isclose(total, 1.0, abs_tol=1e-5), f"probs sum to {total}"


def test_get_probs_illegal_actions_zero(default_vocab):
    """Illegal actions must have zero probability."""
    ds = _make_dream_strategy(default_vocab)
    state = _make_preflop_state(default_vocab)
    mask = state.legal_mask()
    probs = ds.get_probs(state)
    for i, (m, p) in enumerate(zip(mask, probs)):
        if not m:
            assert p == 0.0, f"illegal action {i} has non-zero prob {p}"


def test_get_probs_all_finite(default_vocab):
    """No NaN or inf in the output."""
    ds = _make_dream_strategy(default_vocab)
    state = _make_preflop_state(default_vocab)
    probs = ds.get_probs(state)
    for p in probs:
        assert math.isfinite(p), f"non-finite probability: {p}"
