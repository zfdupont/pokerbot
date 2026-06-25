import pytest
from cfr.abstract_state import AbstractState, deal_heads_up


class TestDealHeadsUp:
    def test_each_player_gets_two_cards(self):
        s = deal_heads_up()
        assert len(s.hole_cards[0]) == 2
        assert len(s.hole_cards[1]) == 2

    def test_no_duplicate_cards(self):
        s = deal_heads_up()
        all_cards = list(s.hole_cards[0]) + list(s.hole_cards[1]) + list(s.deck)
        ids = [(c.rank, c.suit) for c in all_cards]
        assert len(ids) == len(set(ids))

    def test_initial_to_act(self):
        # Preflop: player 0 (BTN/SB) acts first, then player 1 (BB)
        s = deal_heads_up()
        assert s.to_act == (0, 1)

    def test_initial_bets(self):
        s = deal_heads_up()
        assert s.player_bets[0] == pytest.approx(0.5)
        assert s.player_bets[1] == pytest.approx(1.0)
        assert s.current_bet == pytest.approx(1.0)


class TestApplyAction:
    def test_fold_sets_terminal(self):
        s = deal_heads_up().apply_action("fold")
        assert s.is_terminal()

    def test_fold_winner_gets_pot(self):
        s = deal_heads_up()
        folder = s.to_act[0]
        winner = 1 - folder
        s2 = s.apply_action("fold")
        assert s2.payoff(winner) > 0
        assert s2.payoff(folder) < 0

    def test_call_equalizes_bets(self):
        s = deal_heads_up()  # player 0 faces 0.5 to call
        s2 = s.apply_action("call")
        assert s2.player_bets[0] == pytest.approx(s2.player_bets[1])

    def test_call_pops_to_act(self):
        s = deal_heads_up()  # to_act = (0, 1)
        s2 = s.apply_action("call")
        assert s2.to_act == (1,)  # player 1 (BB) still needs to act

    def test_check_pops_to_act(self):
        # Get to a state where player can check (no bet outstanding)
        s = deal_heads_up().apply_action("call")  # to_act = (1,)
        s2 = s.apply_action("check")
        # Street should be over — to_act empty
        assert len(s2.to_act) == 0

    def test_bet_rebuilds_to_act(self):
        s = deal_heads_up().apply_action("call")  # to_act = (1,)
        s2 = s.apply_action("b1.0")               # player 1 bets
        # Player 0 must now respond
        assert 0 in s2.to_act
        assert 1 not in s2.to_act

    def test_bet_increases_pot(self):
        s = deal_heads_up().apply_action("call")
        pot_before = s.pot
        s2 = s.apply_action("b0.5")
        assert s2.pot > pot_before

    def test_allin_empties_stack(self):
        s = deal_heads_up().apply_action("call").apply_action("allin")
        # The player who went all-in has 0 stack
        allin_player = 1  # player 1 went allin after call
        assert s.stacks[allin_player] == pytest.approx(0.0)


class TestAdvanceStreet:
    def test_advance_deals_flop(self):
        # Play through preflop and advance
        s = deal_heads_up().apply_action("call").apply_action("check")
        assert len(s.to_act) == 0
        s2 = s.advance_street()
        assert s2.street == 1
        assert len(s2.board) == 3

    def test_advance_resets_bets(self):
        s = deal_heads_up().apply_action("call").apply_action("check").advance_street()
        assert s.player_bets[0] == pytest.approx(0.0)
        assert s.player_bets[1] == pytest.approx(0.0)
        assert s.current_bet == pytest.approx(0.0)

    def test_postflop_to_act_order(self):
        # Postflop: player 1 (BB/OOP) acts first in HU
        s = deal_heads_up().apply_action("call").apply_action("check").advance_street()
        assert s.to_act[0] == 1

    def test_betting_history_tracks_raises_per_street(self):
        s = deal_heads_up().apply_action("call").apply_action("b1.0").apply_action("allin")
        # preflop: one raise (b1.0), then allin → capped at 2
        assert s.betting_history[0] == 2
        assert s.betting_history[1:] == (0, 0, 0)


class TestIsTerminal:
    def test_not_terminal_at_start(self):
        assert not deal_heads_up().is_terminal()

    def test_terminal_after_fold(self):
        assert deal_heads_up().apply_action("fold").is_terminal()

    def test_not_terminal_mid_hand(self):
        s = deal_heads_up().apply_action("call")
        assert not s.is_terminal()
