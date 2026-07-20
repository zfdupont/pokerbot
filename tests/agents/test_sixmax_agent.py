"""SixmaxDeployStrategy reconstructs trained infoset keys, respects the legal
mask, and translates bets through the vocab."""
import os
import random

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")

from agents.sixmax_agent import SixmaxDeployStrategy, canonical_live_after


def card(rank, suit):
    return (rank - 2) * 4 + suit


def test_canonical_live_after_hu_preflop():
    # HU, button 0, hero is button/seat 0 acting first preflop: one opponent (BB)
    # still to act after.
    live, after = canonical_live_after(2, button=0, hero=0,
                                       folded=[False, False],
                                       all_in=[False, False], street=0)
    assert live == 1 and after == 1


def test_canonical_live_after_6max_after_utg_fold():
    # 6-handed, button 0, hero=HJ(seat 4), UTG(seat 3) folded.
    live, after = canonical_live_after(
        6, button=0, hero=4,
        folded=[False, False, False, True, False, False],
        all_in=[False] * 6, street=0)
    assert live == 4      # everyone but hero and folded UTG
    assert after == 4     # CO, BTN, SB, BB act after HJ preflop


def test_decide_returns_legal_action(blueprint_6max_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_6max_ckpt, _TOML)
    rng = random.Random(0)
    n = strat.num_players
    legal = [0, 0, 1, 1, 1, 1, 1, 1, 1]  # call + all bets/all-in legal
    idx, raise_to = strat.decide(
        hole=[card(14, 3), card(13, 1)], board=[], street=0,
        raises_per_street=[0, 0, 0, 0], pot_bb=1.5, current_bet_bb=1.0,
        to_call_bb=1.0, stack_bb=99.0, live=n - 1, after=n - 1,
        legal=legal, rng=rng)
    assert legal[idx] == 1
    if raise_to > 0.0:
        assert raise_to >= 1.0  # a raise-to is at least the current bet
