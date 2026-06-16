from typing import List, Optional, Tuple
from models.card import Card
from models.enums import Position, Action


class Player:
    def __init__(self, name: str, stack: int, agent):
        self.name = name
        self.stack = stack
        self.hole_cards: List[Card] = []
        self.agent = agent
        self.position: Optional[Position] = None
        self.current_bet = 0
        self.is_active = True
        self.is_all_in: bool = False
        self.total_contributed: int = 0

    def make_decision(self, game_state) -> Tuple[Action, Optional[int]]:
        return self.agent.get_action(self, game_state)
