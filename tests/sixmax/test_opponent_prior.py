"""Opponent prior: a preflop line narrows the blueprint opponent range, combos
blocked by the board/hero are exactly zero, and mass-based pruning keeps the
smallest covering set. Requires the gitignored HU checkpoint."""
import os

import pytest

from agents.sixmax_agent import SixmaxAgent
from models.enums import Action
from models.player import Player
from models.state import GameState

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "hu_blueprint.bin")
_TOML = os.path.join(_ROOT, "sixmax", "configs", "hu.toml")
pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint gitignored")


def _scripted_flop_state():
    """HU GameState on the flop after a preflop SB-open / BB-call line.
    button_pos=0 so player 0 (hero) is SB and raised first preflop."""
    p0 = Player("hero", 100, None)
    p1 = Player("opp", 100, None)
    gs = GameState([p0, p1], small_blind=1)       # big_blind = 2
    gs.betting_round = 1                          # flop
    gs.button_pos = 0
    gs.action_history = [
        (0, Action.RAISE, 6, 0),                  # SB opens to 6 (3bb)
        (1, Action.CALL, 6, 0),                   # BB calls
    ]
    return gs, p0


def _agent():
    return SixmaxAgent(_CKPT, config_toml=_TOML, search_toml=_TOML)


def test_prune_keeps_smallest_mass_covering_set():
    prior = [0.0] * 1326
    prior[0] = 0.5
    prior[1] = 0.3
    prior[2] = 0.15
    prior[3] = 0.05
    out = SixmaxAgent._prune_prior(prior, 0.9)     # 0.5+0.3+0.15 = 0.95 >= 0.9
    assert out[3] == 0.0                           # tail dropped
    assert abs(sum(out) - 1.0) < 1e-9
    assert out[0] > out[1] > out[2] > 0.0


def test_board_blocked_combos_are_zero():
    import sixmax
    a = _agent()
    gs, hero = _scripted_flop_state()
    a._hero = hero
    hole = [51, 50]
    board = [0, 1, 2]
    prior = a._opponent_prior(gs, hole, board)
    for c in board + hole:
        for other in range(52):
            if other == c:
                continue
            lo, hi = min(c, other), max(c, other)
            i = sixmax.RangeTracker.combo_index(lo, hi)
            assert prior[i] == 0.0


def test_preflop_call_concentrates_relative_to_uniform():
    a = _agent()
    gs, hero = _scripted_flop_state()
    a._hero = hero
    hole = [51, 47]
    board = [0, 13, 26]
    prior = a._opponent_prior(gs, hole, board)
    nonzero = [p for p in prior if p > 0]
    assert nonzero, "prior collapsed to all-zero"
    total = sum(nonzero)
    acc = 0.0
    k = 0
    for v in sorted(nonzero, reverse=True):
        acc += v / total
        k += 1
        if acc >= 0.99:
            break
    # Top-99% mass sits in materially fewer combos than the ~1081 unblocked ones.
    assert k < 0.6 * len(nonzero)
