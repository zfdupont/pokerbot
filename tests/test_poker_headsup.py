"""Heads-up (n==2) blind/position ordering in the live engine.

Standard heads-up: the button is the small blind and acts first preflop; the
big blind acts first on every postflop street. The generic 6-max formulas in
`game/poker.py` degenerate wrongly at n==2 (button posts the big blind, and the
same seat acts first on every street), which desynchronizes the six-max
blueprint's position/`after` context. Regression for that fix.
"""
from models.player import Player
from models.enums import Action, Position
from agents.base_agent import PokerAgent
from game.poker import PokerGame


class _Recorder(PokerAgent):
    """Records (name, street, to_call, position) for each decision; check/calls."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def get_action(self, player, state):
        to_call = state.current_bet - player.current_bet
        self.log.append((self.name, state.betting_round, to_call,
                         player.position))
        if to_call > 0:
            return Action.CALL, None
        return Action.CHECK, None


def _play(button):
    log = []
    players = [
        Player("A", 200, _Recorder("A", log)),
        Player("B", 200, _Recorder("B", log)),
    ]
    game = PokerGame(players, small_blind=1)
    game.state.button_pos = button
    game.play_hand()
    return game, players, log


def test_hu_positions_are_small_and_big_blind():
    _, players, _ = _play(0)
    assert players[0].position == Position.SMALL_BLIND
    assert players[1].position == Position.BIG_BLIND


def test_hu_button_posts_small_blind_and_acts_first_preflop():
    _, _, log = _play(0)
    name, street, to_call, position = log[0]
    assert name == "A"                        # the button acts first preflop
    assert street == 0                        # preflop
    assert to_call == 1                       # SB faces the BB
    assert position == Position.SMALL_BLIND


def test_hu_big_blind_acts_first_postflop():
    _, _, log = _play(0)
    name, _street, _to_call, position = next(row for row in log if row[1] == 1)
    assert name == "B"
    assert position == Position.BIG_BLIND


def test_hu_button_rotation_flips_small_blind():
    _, players, _ = _play(1)
    assert players[1].position == Position.SMALL_BLIND
    assert players[0].position == Position.BIG_BLIND
