"""Orchestration contract: the deadline is honoured, every failure path
degrades to an empty result (never an exception, never NaN), and a search
result is a distribution over the *legal* actions only. Requires the
gitignored blueprint checkpoint, so the module skips when it is absent."""
import importlib.util
import math
import os
import time

import pytest

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")

pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT),
                                reason="blueprint checkpoint is gitignored")


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _strategy(budget_ms):
    vocab = _load_vocab("search")
    bp = sixmax.BlueprintStrategy.load(_CKPT, vocab)
    cfg = sixmax.SearchConfig()
    cfg.enabled = True
    cfg.budget_ms = budget_ms
    strat = sixmax.SearchStrategy(bp, vocab,
                                  sixmax.EngineConfig(num_players=2), cfg)
    return strat, vocab


def _flop(vocab):
    eng = sixmax.EngineGameState(sixmax.EngineConfig(num_players=2), 0,
                                 list(range(52)), vocab, [])
    eng.apply(CALL)
    eng.apply(CHECK)
    assert eng.street() == 1
    return eng


def test_tiny_budget_returns_without_overrunning():
    """A 1 ms budget must return promptly and must not raise."""
    strat, vocab = _strategy(1.0)
    eng = _flop(vocab)
    t0 = time.time()
    probs = list(strat.probs(eng, 0, []))
    assert time.time() - t0 < 5.0
    assert probs == [] or math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_incomplete_search_falls_back_to_blueprint():
    """Safety invariant (the plan's 'never worse than blueprint by construction'):
    a budget too small to evaluate the leaves must FALL BACK — an empty result so
    the blueprint is used — never act on a zero-truncated subgame. Before the fix
    the solver returned a non-empty uniform distribution over the garbage tree."""
    strat, vocab = _strategy(1.0)
    eng = _flop(vocab)
    probs = list(strat.probs(eng, 0, []))
    assert probs == []
    assert strat.last_decision_searched() is False


def test_result_is_empty_or_a_legal_distribution():
    strat, vocab = _strategy(500.0)
    eng = _flop(vocab)
    probs = list(strat.probs(eng, 0, []))
    if probs:
        assert all(p >= 0.0 for p in probs)
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
        mask = list(eng.legal_mask())
        assert all(p == 0.0 for p, m in zip(probs, mask) if not m)


def test_searched_flag_agrees_with_the_result():
    strat, vocab = _strategy(1.0)
    eng = _flop(vocab)
    probs = list(strat.probs(eng, 0, []))
    assert strat.last_decision_searched() == (probs != [])


def test_opp_prior_narrows_support():
    """A sparse opponent prior (a handful of unblocked combos) still yields a
    legal, normalised distribution or an empty fallback — never an exception."""
    strat, vocab = _strategy(500.0)
    eng = _flop(vocab)                       # hero=0,1 opp=2,3 board=4,5,6
    prior = [0.0] * 1326
    for (c0, c1) in [(7, 8), (9, 10), (11, 12)]:   # unblocked combos
        prior[sixmax.RangeTracker.combo_index(c0, c1)] = 1.0
    p = list(strat.probs(eng, 0, [], opp_prior=prior))
    assert p == [] or (abs(sum(p) - 1.0) < 1e-6 and all(x >= 0 for x in p))


def test_empty_opp_prior_is_backcompat():
    """An omitted or empty opp_prior preserves today's uniform-reset behaviour."""
    strat, vocab = _strategy(500.0)
    eng = _flop(vocab)
    p_default = list(strat.probs(eng, 0, []))
    p_explicit = list(strat.probs(eng, 0, [], opp_prior=[]))
    assert p_default == p_explicit
