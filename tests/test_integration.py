import pytest
from models.player import Player
from agents.simple_agent import SimpleAgent
from agents.position_agent import PositionBasedAgent
from game.poker import PokerGame


def make_game(stacks=None):
    if stacks is None:
        stacks = [1000, 1000, 1000, 1000]
    players = [
        Player("Simple 1", stacks[0], SimpleAgent()),
        Player("Position 1", stacks[1], PositionBasedAgent()),
        Player("Simple 2", stacks[2], SimpleAgent()),
        Player("Position 2", stacks[3], PositionBasedAgent()),
    ]
    return PokerGame(players, small_blind=5)


def total_chips(game: PokerGame) -> int:
    return sum(p.stack for p in game.state.players)


class TestMultiHandIntegration:
    def test_single_hand_completes(self):
        game = make_game()
        game.play_hand()  # must not raise

    def test_chips_conserved_single_hand(self):
        game = make_game()
        before = total_chips(game)
        game.play_hand()
        assert total_chips(game) == before

    def test_no_negative_stacks_single_hand(self):
        game = make_game()
        game.play_hand()
        for p in game.state.players:
            assert p.stack >= 0, f"{p.name} has negative stack {p.stack}"

    def test_chips_conserved_many_hands(self):
        game = make_game()
        before = total_chips(game)
        for _ in range(20):
            game.play_hand()
        assert total_chips(game) == before

    def test_no_negative_stacks_many_hands(self):
        game = make_game()
        for _ in range(20):
            game.play_hand()
            for p in game.state.players:
                assert p.stack >= 0, f"{p.name} has negative stack {p.stack}"

    def test_button_advances_each_hand(self):
        game = make_game()
        num_players = len(game.state.players)
        positions = []
        for _ in range(num_players):
            positions.append(game.state.button_pos)
            game.play_hand()
        # Button should have advanced once per hand, wrapping around
        for i in range(1, len(positions)):
            expected = (positions[i - 1] + 1) % num_players
            assert positions[i] == expected

    def test_hand_reset_clears_hole_cards(self):
        game = make_game()
        game.play_hand()
        game.play_hand()
        # After each hand reset, every player should have exactly 2 hole cards
        for p in game.state.players:
            assert len(p.hole_cards) == 2, f"{p.name} has {len(p.hole_cards)} hole cards"

    def test_all_players_active_after_reset(self):
        game = make_game()
        game.play_hand()
        # After reset (at start of next play_hand), all players should be active
        # We verify by calling play_hand again — if reset didn't work, folded players
        # would skip the betting round and chips wouldn't balance
        before = total_chips(game)
        game.play_hand()
        assert total_chips(game) == before

    def test_short_stack_survives_many_hands(self):
        # Ensures all-in / side pot path doesn't corrupt chip counts
        game = make_game(stacks=[50, 1000, 50, 1000])
        before = total_chips(game)
        for _ in range(10):
            game.play_hand()
        assert total_chips(game) == before
