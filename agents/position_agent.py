from typing import Tuple, Optional
from models.enums import Action, Position
from agents.base_agent import PokerAgent

class PositionBasedAgent(PokerAgent):
    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        to_call = game_state.current_bet - player.current_bet
        
        if player.position in [Position.BUTTON, Position.CUTOFF]:
            if to_call == 0:
                return Action.BET, game_state.big_blind * 3
            elif to_call <= player.stack * 0.3:
                return Action.CALL, to_call
        
        if player.position in [Position.UNDER_THE_GUN, Position.MIDDLE_POSITION]:
            if to_call > player.stack * 0.2:
                return Action.FOLD, None
        
        if to_call == 0:
            return Action.CHECK, None
        elif to_call <= player.stack * 0.25:
            return Action.CALL, to_call
        return Action.FOLD, None