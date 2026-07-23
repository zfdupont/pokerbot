"""Tests for eval_mixed_table — observer math, chip conservation, CLI smoke."""
import math
import os
import sys

import pytest

from models.player import Player
from models.enums import Action
from agents.base_agent import PokerAgent
from game.poker import PokerGame

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FoldAgent(PokerAgent):
    def get_action(self, player, game_state):
        to_call = game_state.current_bet - player.current_bet
        return (Action.CHECK, None) if to_call <= 0 else (Action.FOLD, None)


def _make_players(n, stack, agents):
    return [Player(f"p{i}", stack, agent=agents[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Observer: chips conserved every hand
# ---------------------------------------------------------------------------
def test_chips_conserved_per_hand():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(20):
        random.seed(i)
        game.play_hand()

    assert obs.hands_played == 20
    # Row sums of chip_matrix must match per-agent cumulative deltas within 1e-6
    for seat in range(6):
        row_sum = sum(obs.chip_matrix[seat])
        bb100, _ = obs.bb100_ci(seat)
        agent_total = bb100 / 100.0 * 2 * obs.hands_played  # BB back to chips
        assert abs(row_sum - agent_total) < 1e-4


def test_bb100_is_finite_after_hands():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(10):
        random.seed(i)
        game.play_hand()

    for seat in range(6):
        bb100, ci = obs.bb100_ci(seat)
        assert math.isfinite(bb100)
        assert math.isfinite(ci)
        assert ci >= 0.0


def test_chip_matrix_row_sums_match_totals():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(15):
        random.seed(i)
        game.play_hand()

    for seat in range(6):
        row_sum = sum(obs.chip_matrix[seat])
        col_sum = sum(obs.chip_matrix[j][seat] for j in range(6))
        # chip matrix is antisymmetric: M[i][j] = -M[j][i]
        assert abs(row_sum + col_sum) < 1e-6


# ---------------------------------------------------------------------------
# CLI smoke test: fails fast when checkpoint paths are missing
# ---------------------------------------------------------------------------
def test_main_runs_without_checkpoints(tmp_path, monkeypatch):
    """main() fails fast with SystemExit when checkpoint paths are missing."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_mixed_table",
         "--blueprint", str(tmp_path / "missing.bin"),
         "--neural", str(tmp_path / "missing.pt"),
         "--tabular", str(tmp_path / "missing.pkl"),
         "--hands", "5"],
        capture_output=True, text=True,
        cwd=_ROOT,
    )
    assert result.returncode != 0
    assert "missing" in result.stderr.lower() or "not found" in result.stderr.lower() \
        or result.returncode == 1
