from typing import Tuple, Optional
from models.enums import Action
from models.player import Player
from models.state import GameState

class PokerAgent:
    """Base class for poker agents"""
    def get_action(self, player: Player, game_state: GameState) -> Tuple[Action, Optional[int]]:
        raise NotImplementedError("Implement this method in your custom agent")
