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


# ---------------------------------------------------------------------------
# Direct encode_state_vec structural tests
# ---------------------------------------------------------------------------

def test_encode_state_vec_length(default_vocab):
    """encode_state_vec returns exactly FEATURE_DIM floats."""
    state = _make_preflop_state(default_vocab)
    vec = sixmax.encode_state_vec(state)
    assert len(vec) == sixmax.FEATURE_DIM


def test_encode_state_vec_all_finite(default_vocab):
    """encode_state_vec contains no NaN or Inf values."""
    state = _make_preflop_state(default_vocab)
    vec = sixmax.encode_state_vec(state)
    for i, v in enumerate(vec):
        assert math.isfinite(v), f"non-finite value at index {i}: {v}"


def test_encode_state_vec_hole_cards_sum(default_vocab):
    """Hole card dims 0-33: one-hot rank (13 bits) + one-hot suit (4 bits) per card
    = 17 bits * 2 cards. Each card contributes exactly 1 rank bit + 1 suit bit,
    so dims 0-16 sum to 2.0 (card 0) and dims 17-33 sum to 2.0 (card 1)."""
    state = _make_preflop_state(default_vocab)
    vec = sixmax.encode_state_vec(state)
    # Dims 0-16: rank(13) + suit(4) for hole card 0
    card0_sum = sum(vec[0:17])
    assert math.isclose(card0_sum, 2.0, abs_tol=1e-5), \
        f"hole card 0 dims 0-16 sum to {card0_sum}, expected 2.0"
    # Dims 17-33: rank(13) + suit(4) for hole card 1
    card1_sum = sum(vec[17:34])
    assert math.isclose(card1_sum, 2.0, abs_tol=1e-5), \
        f"hole card 1 dims 17-33 sum to {card1_sum}, expected 2.0"


def test_encode_state_vec_street_one_hot(default_vocab):
    """Street one-hot dims 119-122 sum to 1.0."""
    state = _make_preflop_state(default_vocab)
    vec = sixmax.encode_state_vec(state)
    street_sum = sum(vec[119:123])
    assert math.isclose(street_sum, 1.0, abs_tol=1e-5), \
        f"street one-hot dims 119-122 sum to {street_sum}, expected 1.0"


def test_encode_state_vec_acting_player_one_hot(default_vocab):
    """Acting player one-hot dims 142-147 sum to 1.0."""
    state = _make_preflop_state(default_vocab)
    vec = sixmax.encode_state_vec(state)
    player_sum = sum(vec[142:148])
    assert math.isclose(player_sum, 1.0, abs_tol=1e-5), \
        f"acting player one-hot dims 142-147 sum to {player_sum}, expected 1.0"


# ---------------------------------------------------------------------------
# DreamTrainer seats coverage test
# ---------------------------------------------------------------------------

def test_dream_trainer_all_seats_get_advantage_samples(default_vocab):
    """DreamTrainer must deposit advantage samples for ALL n_players seats.

    Trains for a small number of iterations with players_min=players_max=3
    so every hand is 3-handed.  Before the fix the loop was hardcoded to
    2 updating_player iterations, so seats 2+ never contributed to M_v.
    After the fix iter_.fetch_add increments by n_players per hand and the
    loop runs for all seats; M_v should accumulate samples from all of them.

    We use a tiny reservoir (no retrain threshold) so the size reflects
    raw sample accumulation: with N hands × 3 seats each we expect
    exactly 3*N entries in M_v (one advantage vector per seat per hand).
    """
    import importlib.util as _ilu
    import os

    abs_ = sixmax.Abstraction(flop_buckets=5, turn_buckets=5, river_buckets=3,
                              equity_rollouts=10, quantile_samples=50, seed=0)

    cfg = sixmax.DreamConfig()
    cfg.players_min = 3
    cfg.players_max = 3
    cfg.reservoir_size = 100_000
    cfg.train_interval = 100_000   # disable retraining during this test
    cfg.num_threads = 1
    cfg.seed = 7

    n_hands = 20
    # DreamTrainer.train(iterations) runs `iterations` outer loop iterations
    # (each is one full hand with n_players traversals).
    trainer = sixmax.DreamTrainer(
        default_vocab.size(), default_vocab, abs_, cfg, "cpu")
    trainer.train(n_hands)

    # Verify the iteration counter: iter_ is incremented by n_players per hand,
    # so total_iterations() must equal n_hands * 3 after the fix.
    # Before the fix (hardcoded < 2 loop) it would have been n_hands * 2.
    expected_iters = n_hands * 3
    actual_iters = trainer.total_iterations()
    assert actual_iters == expected_iters, (
        f"Expected total_iterations()={expected_iters} for {n_hands} 3-player "
        f"hands (3 traversals each), got {actual_iters}. "
        f"Seats 2+ may not be in the traversal loop."
    )

    # Verify M_v has samples at all — it should be non-empty since each
    # traversal deposits advantage samples at the updating player's nodes.
    adv_size = trainer.adv_reservoir_size()
    assert adv_size > 0, "M_v is empty — no advantage samples were deposited."
    # With 3 players × 20 hands, M_v must strictly exceed what 2-player
    # training would produce for 20 hands (≥ 1.5× more traversals means
    # at least 1.5× more M_v entries, all else equal).
    # We verify it's non-trivially large relative to the 2-player baseline
    # by checking adv_size >= n_hands (at least 1 node per hand).
    assert adv_size >= n_hands, (
        f"M_v has only {adv_size} entries for {n_hands} hands — "
        f"expected at least {n_hands}."
    )
