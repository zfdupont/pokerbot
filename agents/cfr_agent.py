from typing import Tuple, Optional, List
import numpy as np

from models.enums import Action
from agents.base_agent import PokerAgent
from cfr.regret_table import RegretTable
from cfr.info_set import InfoSet, stack_bucket
from cfr.abstraction import hand_to_bucket, board_to_bucket, legal_abstract_actions

_ABSTRACT_TO_ACTION = {
    "fold":  Action.FOLD,
    "check": Action.CHECK,
    "call":  Action.CALL,
    "b0.5":  Action.BET,
    "b1.0":  Action.BET,
    "allin": Action.BET,
}


class CFRAgent(PokerAgent):
    def __init__(self, strategy_path: str):
        self.table = RegretTable()
        self.table.load(strategy_path)

    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        infoset = self._encode(player, game_state)
        legal = self._legal_abstract(player, game_state)
        probs = self.table.get_average_strategy(infoset, legal)
        abstract = np.random.choice(legal, p=probs)
        return self._translate(abstract, player, game_state)

    def _encode(self, player, game_state) -> InfoSet:
        big_blind = game_state.big_blind
        return InfoSet(
            player=0,
            hand_bucket=hand_to_bucket(
                player.hole_cards,
                game_state.community_cards,
                game_state.betting_round,
            ),
            street=game_state.betting_round,
            board_bucket=board_to_bucket(
                game_state.community_cards,
                game_state.betting_round,
            ),
            betting_history=(),
            stack_bucket=stack_bucket(player.stack, big_blind),
        )

    def _legal_abstract(self, player, game_state) -> List[str]:
        to_call = game_state.current_bet - player.current_bet
        pot = game_state.pot
        return legal_abstract_actions(
            to_call=float(to_call),
            pot=float(pot),
            stack=float(player.stack),
            current_bet=float(game_state.current_bet),
            player_bet=float(player.current_bet),
        )

    def _translate(self, abstract: str, player, game_state) -> Tuple[Action, Optional[int]]:
        pot = game_state.pot
        to_call = game_state.current_bet - player.current_bet

        if abstract == "fold":
            return Action.FOLD, None
        if abstract == "check":
            return Action.CHECK, None
        if abstract == "call":
            return Action.CALL, int(to_call)
        if abstract == "allin":
            return Action.BET, int(player.stack + player.current_bet)
        if abstract in ("b0.5", "b1.0"):
            size = float(abstract[1:])
            amount = int(to_call + size * (pot + to_call * 2))
            amount = min(amount, player.stack + player.current_bet)
            return Action.BET, amount
        return Action.CALL, int(to_call)
