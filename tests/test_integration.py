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


def test_raises_per_street_is_faithful_past_two():
    """After the uncap, raises_per_street counts the true number of raises on a
    street *while that street is live* (old code capped at 2). Consumers read
    the count mid-street (that is the only value they ever see), so the contract
    is tested by what an agent observes at decision time — not the end-of-hand
    value, which the engine's per-street reset (_reset_street_bets zeroes the
    just-finished street when advancing) always wipes to 0.

    Both players min-raise preflop while the live count is below 3, so exactly 3
    preflop raises go in and then the action closes with a call. A later actor
    therefore observes raises_per_street[0] == 3. On the old capped engine the
    count sticks at 2, the `< 3` gate never closes, and the agents raise until
    all-in — so the observed max is 2 and this test fails (assert 2 == 3)."""
    from models.enums import Action
    from models.player import Player
    from game.poker import PokerGame

    observed_preflop = []

    class Raiser:
        def get_action(self, player, game_state):
            to_call = game_state.current_bet - player.current_bet
            can_raise = player.stack > to_call + game_state.big_blind
            if game_state.betting_round == 0:
                observed_preflop.append(game_state.raises_per_street[0])
                if game_state.raises_per_street[0] < 3 and can_raise:
                    return (Action.BET,
                            game_state.current_bet + 2 * game_state.big_blind)
            return (Action.CALL, None) if to_call > 0 else (Action.CHECK, None)

    p0, p1 = Player("a", 200, Raiser()), Player("b", 200, Raiser())
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()
    assert max(observed_preflop) == 3   # count passes 2 while the street is live
    assert p0.stack + p1.stack == 400   # chips conserved
