"""Tests for the six-max blueprint-vs-baseline eval harness.

The pure functions (block_stats, play_deck, parse_iters, expand_checkpoints,
build_villain) are tested with no buck2 build or checkpoint. A final smoke test
runs a real SixmaxAgent over a few decks using the tiny 6-max fixture.
"""
import math
import random

import pytest

from models.enums import Action
from models.player import Player
from game.poker import PokerGame
from agents.base_agent import PokerAgent

from scripts.eval_sixmax_baseline import (
    block_stats,
    build_villain,
    expand_checkpoints,
    parse_iters,
    play_deck,
    run_match,
)


class FoldAgent(PokerAgent):
    """Deterministic, RNG-free: check when free, fold to any bet. Keeps hands
    short and makes deck-driven outcomes fully reproducible."""
    def get_action(self, player, game_state):
        to_call = game_state.current_bet - player.current_bet
        return (Action.CHECK, None) if to_call <= 0 else (Action.FOLD, None)


def _hand_key(player):
    return tuple(sorted((c.rank, c.suit.value) for c in player.hole_cards))


# ---------------------------------------------------------------------------
# block_stats: the BB/100 + standard-error math
# ---------------------------------------------------------------------------
def test_block_stats_matches_hand_computation():
    samples = [1.0, -1.0, 2.0, 0.0]  # per-deck mean BB/hand
    bb100, stderr = block_stats(samples)
    mean = 0.5
    var = ((1 - mean) ** 2 + (-1 - mean) ** 2 + (2 - mean) ** 2
           + (0 - mean) ** 2) / 3  # ddof=1
    assert bb100 == pytest.approx(100.0 * mean)
    assert stderr == pytest.approx(100.0 * math.sqrt(var) / math.sqrt(4))


def test_block_stats_single_deck_has_zero_stderr():
    bb100, stderr = block_stats([2.5])
    assert bb100 == pytest.approx(250.0)
    assert stderr == 0.0


def test_block_stats_empty_is_zero():
    assert block_stats([]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Seat rotation: the duplicate-deal variance-reduction invariant
# ---------------------------------------------------------------------------
def test_hero_sees_every_dealt_hand_once_per_deck():
    """Across the n rotations of one deck, the hero's hole cards are exactly the
    multiset of all n seats' dealt hands — the property that cancels card luck.

    Seat i always gets the same cards regardless of which agent sits there,
    because agents never touch the global RNG that drives the shuffle."""
    n, seed, stack, sb = 6, 12345, 200, 1

    # Hero's hand in each rotation.
    hero_hands = []
    for hero_seat in range(n):
        random.seed(seed)
        players = [Player(f"p{i}", stack, agent=FoldAgent()) for i in range(n)]
        PokerGame(players, small_blind=sb).play_hand()
        hero_hands.append(_hand_key(players[hero_seat]))

    # All seats' hands in a single hand of the same deck.
    random.seed(seed)
    players = [Player(f"p{i}", stack, agent=FoldAgent()) for i in range(n)]
    PokerGame(players, small_blind=sb).play_hand()
    all_hands = [_hand_key(p) for p in players]

    assert sorted(hero_hands) == sorted(all_hands)
    assert len(set(hero_hands)) == n  # n distinct hands, no duplicates


def test_play_deck_is_reproducible():
    n, stack, sb = 6, 200, 1
    hero, villain = FoldAgent(), FoldAgent()
    a = play_deck(hero, villain, seed=7, n=n, small_blind=sb, stack=stack)
    b = play_deck(hero, villain, seed=7, n=n, small_blind=sb, stack=stack)
    assert a == b


def test_play_deck_is_zero_sum_for_identical_agents():
    """With the same agent in every seat, summing the hero delta over all n
    rotations covers every seat once, so the total is chip-conserving (0)."""
    total = play_deck(FoldAgent(), FoldAgent(), seed=99, n=6,
                      small_blind=1, stack=200)
    assert total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Checkpoint / villain plumbing
# ---------------------------------------------------------------------------
def test_parse_iters():
    assert parse_iters("sixmax/checkpoints/blueprint_1000000.bin") == 1000000
    assert parse_iters("checkpoint_09040000.pkl") == 9040000
    assert parse_iters("best_checkpoint.bin") is None


def test_expand_checkpoints_dedupes_and_globs(tmp_path):
    a = tmp_path / "blueprint_100.bin"
    b = tmp_path / "blueprint_200.bin"
    a.write_text("x")
    b.write_text("x")
    out = expand_checkpoints(str(a), [str(tmp_path / "blueprint_*.bin"), str(a)])
    assert out == [str(a), str(b)]  # single first, glob expands, dupe dropped


def test_build_villain_known_and_unknown():
    assert build_villain("potodds") is not None
    assert build_villain("simple") is not None
    with pytest.raises(SystemExit):
        build_villain("nope")


# ---------------------------------------------------------------------------
# Smoke: a real blueprint over a few decks vs the baseline
# ---------------------------------------------------------------------------
def test_run_match_smoke(blueprint_6max_ckpt):
    import os
    from agents.sixmax_agent import SixmaxAgent

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    toml = os.path.join(root, "sixmax", "configs", "default.toml")
    hero = SixmaxAgent(blueprint_6max_ckpt, config_toml=toml)
    bb100, stderr, n_hands = run_match(
        hero, build_villain("potodds"), hands=5, seed=1, n=6,
        small_blind=1, stack=200)
    assert math.isfinite(bb100)
    assert stderr >= 0.0
    assert n_hands == 30
