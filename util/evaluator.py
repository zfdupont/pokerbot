from typing import List, Tuple

from models.player import Player
from models.card import Card
from util.util import hand_value


class HandEvaluator:
    @staticmethod
    def evaluate_hands(players: List[Player], community_cards: List[Card]) -> List[Tuple[Player, int]]:
        results = [
            (p, hand_value(p.hole_cards, community_cards))
            for p in players if p.is_active
        ]
        return sorted(results, key=lambda x: x[1])

    @staticmethod
    def rank_to_hand_name(rank: int) -> str:
        if rank == 1:
            return "Royal Flush"
        if rank <= 10:
            return "Straight Flush"
        if rank <= 166:
            return "Four of a Kind"
        if rank <= 322:
            return "Full House"
        if rank <= 1599:
            return "Flush"
        if rank <= 1609:
            return "Straight"
        if rank <= 2467:
            return "Three of a Kind"
        if rank <= 3325:
            return "Two Pair"
        if rank <= 6185:
            return "Pair"
        return "High Card"
