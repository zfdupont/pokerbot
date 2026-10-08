"""Leaf-evaluation contract. The board-complete (river) branch is an EXACT
showdown via safe_eval — it needs no blueprint and no engine settlement (the
state is NOT terminal; betting has not closed), so these tests never depend on
the gitignored checkpoint. The sampled flop/turn branch is covered by a
module-level skip when the checkpoint is absent."""
import importlib.util
import math
import os

import pytest

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _street_state(target_street):
    """Drive the engine from preflop to `target_street` (1=flop .. 3=river)."""
    vocab = _load_vocab("search")
    eng = sixmax.EngineGameState(sixmax.EngineConfig(num_players=2), 0,
                                 list(range(52)), vocab, [])
    while eng.street() < target_street:
        eng.apply(CALL)
        eng.apply(CHECK)
    assert eng.street() == target_street
    return eng


def test_terminal_leaf_zeroes_conflicting_hands():
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())      # exact-only (board-complete)
    hands = [0, 1, 2, 3]
    m = list(ev.matrix(eng, 0, hands, hands))
    n = len(hands)
    for i in range(n):
        assert m[i * n + i] == 0.0


def test_terminal_leaf_is_antisymmetric_between_players():
    """Zero-sum: player 0's payoff for (h0,h1) must be minus player 1's for
    (h1,h0), i.e. m0[i][j] == -m1[j][i]."""
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())
    hands = [0, 1, 2, 3]
    n = len(hands)
    m0 = list(ev.matrix(eng, 0, hands, hands))
    m1 = list(ev.matrix(eng, 1, hands, hands))
    for i in range(n):
        for j in range(n):
            assert math.isclose(m0[i * n + j], -m1[j * n + i], abs_tol=1e-9)


def test_terminal_leaf_is_deterministic():
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())
    a = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    b = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    assert a == b


def test_leaf_zeroes_card_removal_conflicts():
    """T4-5: a combo that overlaps the board, or shares a card with the
    opponent's combo (without being identical), must contribute exactly 0 —
    not only the h0==h1 diagonal. Without this the solver sees impossible hands."""
    ci = sixmax.RangeTracker.combo_index
    eng = _street_state(3)
    board = list(eng.board())                       # 5 cards on the river
    free = [c for c in range(52) if c not in board]
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())

    # (a) our combo uses a board card -> impossible -> 0
    our = [ci(min(board[0], free[0]), max(board[0], free[0]))]
    opp = [ci(min(free[1], free[2]), max(free[1], free[2]))]
    assert list(ev.matrix(eng, 0, our, opp))[0] == 0.0

    # (b) our and opp share one card (not identical combos) -> 0
    our = [ci(free[0], free[1])]
    opp = [ci(free[1], free[2])]                     # shares free[1]
    assert list(ev.matrix(eng, 0, our, opp))[0] == 0.0


@pytest.mark.skipif(not os.path.exists(_CKPT),
                    reason="blueprint checkpoint is gitignored")
def test_sampled_leaf_is_deterministic():
    """A turn leaf uses the blueprint continuation; the same seed and inputs
    must give the same matrix."""
    vocab = _load_vocab("search")
    bp = sixmax.BlueprintStrategy.load(_CKPT, vocab)
    ev = sixmax.LeafEval(bp, sixmax.LeafEvalConfig())
    eng = _street_state(2)
    a = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    b = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    assert a == b
