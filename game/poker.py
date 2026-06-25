from collections import deque
from typing import List, Optional

from models.player import Player
from models.state import GameState
from models.enums import Action
from util.evaluator import HandEvaluator
from util.observer import GameObserver
from game.pot_manager import PotManager


class PokerGame:
    def __init__(
        self,
        players: List[Player],
        small_blind: int = 1,
        observers: Optional[List[GameObserver]] = None,
    ):
        self.state = GameState(players, small_blind)
        self.observers = observers or []

    def _emit(self, event: str, *args, **kwargs) -> None:
        for obs in self.observers:
            getattr(obs, event)(*args, **kwargs)

    def _assign_positions(self) -> None:
        from models.enums import Position
        num = len(self.state.players)
        positions = list(Position)[-num:]
        for i, player in enumerate(self.state.players):
            idx = (i + self.state.button_pos) % num
            player.position = positions[idx]

    def _post_blinds(self, pot_manager: PotManager) -> None:
        num = len(self.state.players)
        sb_pos = (self.state.button_pos + 1) % num
        bb_pos = (self.state.button_pos + 2) % num

        sb = self.state.players[sb_pos]
        bb = self.state.players[bb_pos]

        sb_amount = min(self.state.small_blind, sb.stack)
        sb.stack -= sb_amount
        sb.current_bet = sb_amount
        pot_manager.contribute(sb, sb_amount)

        bb_amount = min(self.state.big_blind, bb.stack)
        bb.stack -= bb_amount
        bb.current_bet = bb_amount
        pot_manager.contribute(bb, bb_amount)

        self.state.current_bet = self.state.big_blind

    def _reset_street_bets(self) -> None:
        for p in self.state.players:
            p.current_bet = 0
        self.state.current_bet = 0
        self.state.raises_per_street[self.state.betting_round] = 0

    def _first_to_act(self, street: int) -> int:
        num = len(self.state.players)
        start = (self.state.button_pos + 3) % num if street == 0 else (self.state.button_pos + 1) % num
        for i in range(num):
            idx = (start + i) % num
            p = self.state.players[idx]
            if p.is_active and not p.is_all_in:
                return idx
        return start

    def _build_action_queue(self, start_idx: int, exclude_idx: Optional[int] = None):
        num = len(self.state.players)
        all_idxs = list(range(num))
        ordered = all_idxs[start_idx:] + all_idxs[:start_idx]
        return deque(
            i for i in ordered
            if self.state.players[i].is_active
            and not self.state.players[i].is_all_in
            and i != exclude_idx
        )

    def _betting_round(self, street: int, pot_manager: PotManager) -> None:
        active_non_allin = [p for p in self.state.players if p.is_active and not p.is_all_in]
        if len(active_non_allin) <= 1:
            return

        first_idx = self._first_to_act(street)
        to_act = self._build_action_queue(first_idx)

        while to_act:
            idx = to_act.popleft()
            player = self.state.players[idx]

            if not player.is_active or player.is_all_in:
                continue

            self.state.current_player_idx = idx
            self.state.pot = pot_manager.total
            action, amount = player.make_decision(self.state)
            self._process_action(player, action, amount, pot_manager)
            self._emit("on_player_action", player, action, amount)

            if action in (Action.BET, Action.RAISE):
                next_start = (idx + 1) % len(self.state.players)
                to_act = self._build_action_queue(next_start, exclude_idx=idx)

            if len([p for p in self.state.players if p.is_active and not p.is_all_in]) <= 1:
                break

    def _process_action(
        self,
        player: Player,
        action: Action,
        amount: Optional[int],
        pot_manager: PotManager,
    ) -> None:
        if action == Action.FOLD:
            player.is_active = False

        elif action == Action.CHECK:
            pass

        elif action == Action.CALL:
            to_call = min(self.state.current_bet - player.current_bet, player.stack)
            player.stack -= to_call
            player.current_bet += to_call
            pot_manager.contribute(player, to_call)
            if player.stack == 0:
                player.is_all_in = True

        elif action in (Action.BET, Action.RAISE):
            if amount is None:
                raise ValueError("Amount required for bet/raise")
            max_bet = player.stack + player.current_bet
            amount = min(amount, max_bet)
            additional = amount - player.current_bet
            player.stack -= additional
            player.current_bet = amount
            self.state.current_bet = amount
            pot_manager.contribute(player, additional)
            if player.stack == 0:
                player.is_all_in = True
            street = self.state.betting_round
            self.state.raises_per_street[street] = min(
                self.state.raises_per_street[street] + 1, 2
            )

    def _determine_winners(self, pot_manager: PotManager) -> None:
        active_players = [p for p in self.state.players if p.is_active]

        if len(active_players) == 1:
            player_ranks = [(active_players[0], 0)]
        else:
            player_ranks = HandEvaluator.evaluate_hands(active_players, self.state.community_cards)

        pot_manager.award(player_ranks)
        self._emit("on_hand_complete", player_ranks, pot_manager.total)

    def _reset_for_hand(self) -> None:
        self.state.deck = self.state._create_deck()
        self.state.community_cards = []
        self.state.current_bet = 0
        self.state.betting_round = 0
        self.state.raises_per_street = [0, 0, 0, 0]
        for p in self.state.players:
            p.hole_cards = []
            p.current_bet = 0
            p.is_active = p.stack > 0
            p.is_all_in = False
            p.total_contributed = 0

    def play_hand(self) -> None:
        pot_manager = PotManager()

        self._reset_for_hand()
        self._assign_positions()
        self.state.deal_hole_cards()
        self._emit("on_hand_start", self.state.players, self.state.button_pos)

        self._post_blinds(pot_manager)

        active = lambda: [p for p in self.state.players if p.is_active]

        # Preflop (blinds already set current_bet — no reset)
        self._emit("on_street_start", "Pre-flop", self.state.community_cards)
        self._betting_round(0, pot_manager)

        # Flop
        if len(active()) > 1:
            self._reset_street_bets()
            self.state.betting_round = 1
            self.state.deal_community_cards()
            self._emit("on_street_start", "Flop", self.state.community_cards)
            self._betting_round(1, pot_manager)

        # Turn
        if len(active()) > 1:
            self._reset_street_bets()
            self.state.betting_round = 2
            self.state.deal_community_cards()
            self._emit("on_street_start", "Turn", self.state.community_cards)
            self._betting_round(2, pot_manager)

        # River
        if len(active()) > 1:
            self._reset_street_bets()
            self.state.betting_round = 3
            self.state.deal_community_cards()
            self._emit("on_street_start", "River", self.state.community_cards)
            self._betting_round(3, pot_manager)

        self._determine_winners(pot_manager)
        self.state.button_pos = (self.state.button_pos + 1) % len(self.state.players)
