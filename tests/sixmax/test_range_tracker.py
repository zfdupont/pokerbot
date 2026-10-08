"""HU range tracker invariants: exact card removal, an always-normalised
posterior, and a Bayes update that moves mass toward the observed action."""
import importlib.util
import math
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab(section="blueprint"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


VOCAB = _load_vocab()


def _tracker():
    return sixmax.RangeTracker(sixmax.EngineConfig(num_players=2), VOCAB)


def test_combo_index_round_trips():
    n = 0
    for c0 in range(52):
        for c1 in range(c0 + 1, 52):
            idx = sixmax.RangeTracker.combo_index(c0, c1)
            assert idx == n
            assert tuple(sixmax.RangeTracker.combo_cards(idx)) == (c0, c1)
            n += 1
    assert n == 1326


def test_reset_zeroes_blocked_cards_and_normalises():
    t = _tracker()
    board, hole = [0, 1, 2], (4, 5)
    t.reset(board, hole)
    r = list(t.range(1))
    assert math.isclose(sum(r), 1.0, abs_tol=1e-9)
    for blocked in (*board, *hole):
        for other in range(52):
            if other == blocked:
                continue
            lo, hi = min(blocked, other), max(blocked, other)
            assert r[sixmax.RangeTracker.combo_index(lo, hi)] == 0.0


def test_bayes_update_collapses_onto_the_observed_hand():
    """Closed-form toy, no engine needed: the Bayes step multiplies by the
    per-combo action probability and renormalises."""
    t = _tracker()
    t.set_range_for_test(1, [0.5, 0.5, 0.0])
    t.observe_with_probs(1, [1.0, 0.0, 0.0])
    r = list(t.range(1))
    assert math.isclose(r[0], 1.0, abs_tol=1e-12)
    assert math.isclose(r[1], 0.0, abs_tol=1e-12)
    assert math.isclose(sum(r), 1.0, abs_tol=1e-12)


def test_degenerate_update_keeps_a_valid_distribution():
    """An observation impossible on the current support must not produce NaNs
    or a zero vector: the prior is kept, still normalised."""
    t = _tracker()
    t.set_range_for_test(1, [0.5, 0.5, 0.0])
    t.observe_with_probs(1, [0.0, 0.0, 1.0])
    r = list(t.range(1))
    assert math.isclose(sum(r), 1.0, abs_tol=1e-12)
    assert all(v >= 0.0 and not math.isnan(v) for v in r)
