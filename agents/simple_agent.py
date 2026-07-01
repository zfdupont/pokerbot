from typing import Tuple, Optional
from models.enums import Action
from agents.base_agent import PokerAgent

class SimpleAgent(PokerAgent):
    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        to_call = game_state.current_bet - player.current_bet
        
        if to_call == 0:
            return Action.CHECK, None
        elif to_call <= player.stack * 0.25:
            return Action.CALL, to_call
        else:
            return Action.FOLD, None