"""
Tier 1 unit tests for DREAM checkpoint save/load (Task 8).

DreamStrategy.load() hard-codes the strategy-net architecture as
DreamMLP(FEATURE_DIM, 256, 3, n_actions).  Tests must save with the same
shape or the torch-archive load will raise.

Verified behaviour:
  - save_dream_checkpoint + DreamStrategy.load round-trip without error
  - ds.iterations (property) equals the saved value
  - vocab hash mismatch raises RuntimeError with 'vocab hash mismatch' in msg
  - File must exist after save
  - Checkpoint file size is > 0
"""
import math
import os
import tempfile

import pytest
import sixmax
import sixmax_dream

_FEAT_DIM = sixmax_dream.FEATURE_DIM   # 154
_HIDDEN   = 256                         # must match DreamStrategy::load hard-coded arch
_LAYERS   = 3                           # must match DreamStrategy::load hard-coded arch


def _make_nets(n_actions: int):
    """Return a (adv, strat) pair of DreamMLPs with the canonical architecture."""
    adv  = sixmax_dream.DreamMLP(_FEAT_DIM, _HIDDEN, _LAYERS, n_actions)
    strat = sixmax_dream.DreamMLP(_FEAT_DIM, _HIDDEN, _LAYERS, n_actions)
    return adv, strat


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(default_vocab, tmp_path):
    """Saving then loading a checkpoint preserves iterations."""
    path = str(tmp_path / "dream_test.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)

    sixmax_dream.save_dream_checkpoint(path, adv, strat, 999, default_vocab.hash())

    assert os.path.exists(path), "checkpoint file was not created"
    assert os.path.getsize(path) > 0, "checkpoint file is empty"

    ds = sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)
    assert ds.iterations == 999


def test_checkpoint_iterations_zero(default_vocab, tmp_path):
    """iterations=0 round-trips correctly."""
    path = str(tmp_path / "zero.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)
    sixmax_dream.save_dream_checkpoint(path, adv, strat, 0, default_vocab.hash())
    ds = sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)
    assert ds.iterations == 0


def test_checkpoint_large_iterations(default_vocab, tmp_path):
    """Large iteration count survives serialisation."""
    path = str(tmp_path / "big.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)
    big = 1_000_000
    sixmax_dream.save_dream_checkpoint(path, adv, strat, big, default_vocab.hash())
    ds = sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)
    assert ds.iterations == big


# ---------------------------------------------------------------------------
# Vocab hash guard
# ---------------------------------------------------------------------------

def test_vocab_hash_mismatch(default_vocab, tmp_path):
    """Loading a checkpoint saved with a wrong hash raises RuntimeError."""
    path = str(tmp_path / "bad.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)
    wrong_hash = 0xDEADBEEF
    sixmax_dream.save_dream_checkpoint(path, adv, strat, 1, wrong_hash)

    with pytest.raises(RuntimeError, match="vocab hash mismatch"):
        sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)


def test_correct_hash_does_not_raise(default_vocab, tmp_path):
    """Loading with the correct hash must NOT raise."""
    path = str(tmp_path / "ok.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)
    sixmax_dream.save_dream_checkpoint(path, adv, strat, 5, default_vocab.hash())
    ds = sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)
    assert ds is not None


# ---------------------------------------------------------------------------
# Inference smoke-test
# ---------------------------------------------------------------------------

def test_get_probs_after_load(default_vocab, tmp_path):
    """A freshly-loaded DreamStrategy returns valid probabilities for a live state."""
    path = str(tmp_path / "inf.pt")
    n = default_vocab.size()
    adv, strat = _make_nets(n)
    sixmax_dream.save_dream_checkpoint(path, adv, strat, 42, default_vocab.hash())
    ds = sixmax_dream.DreamStrategy.load(path, "cpu", default_vocab)

    cfg = sixmax.EngineConfig(num_players=2)
    state = sixmax.EngineGameState(cfg, 0, list(range(52)), default_vocab, [])
    probs = ds.get_probs(state)

    assert len(probs) == n
    assert all(math.isfinite(p) for p in probs), "non-finite prob in output"
    assert all(0.0 <= p <= 1.0 for p in probs), "prob out of [0,1]"
    total = sum(probs)
    assert math.isclose(total, 1.0, abs_tol=1e-5), f"probs sum {total} != 1"
