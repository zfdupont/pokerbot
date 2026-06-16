from typing import List, Tuple, Optional

from models.player import Player
from models.card import Card
from models.enums import Action
from util.evaluator import HandEvaluator


class GameObserver:
    def on_hand_start(self, players: List[Player], button_pos: int) -> None:
        pass

    def on_hole_cards_dealt(self, player: Player, cards: List[Card]) -> None:
        pass

    def on_street_start(self, street_name: str, community_cards: List[Card]) -> None:
        pass

    def on_player_action(self, player: Player, action: Action, amount: Optional[int]) -> None:
        pass

    def on_hand_complete(self, winners: List[Tuple[Player, int]], pot: int) -> None:
        pass


class ConsoleObserver(GameObserver):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_hand_start(self, players: List[Player], button_pos: int) -> None:
        print("\n=== New Hand ===")
        print("Players:")
        for player in players:
            cards_str = " ".join(str(card) for card in player.hole_cards)
            pos = player.position.value if player.position else "?"
            print(f"  {player.name} ({pos}): {cards_str} - Stack: ${player.stack}")

    def on_street_start(self, street_name: str, community_cards: List[Card]) -> None:
        print(f"\n--- {street_name} ---")
        if community_cards:
            cards_str = " ".join(str(c) for c in community_cards)
            print(f"Community: {cards_str}")

    def on_player_action(self, player: Player, action: Action, amount: Optional[int]) -> None:
        action_str = action.value
        if amount is not None:
            action_str = f"{action_str} ${amount}"
        pos = player.position.value if player.position else "?"
        print(f"  {player.name} ({pos}): {action_str}")
        if self.verbose:
            print(f"    Stack: ${player.stack}")

    def on_hand_complete(self, winners: List[Tuple[Player, int]], pot: int) -> None:
        print("\n=== Hand Complete ===")
        print(f"Pot: ${pot}")
        if len(winners) > 1:
            split = pot // len(winners)
            print(f"Split pot ({len(winners)} players):")
            for player, rank in winners:
                print(f"  {player.name} wins ${split} with {HandEvaluator.rank_to_hand_name(rank)}")
        else:
            player, rank = winners[0]
            print(f"  {player.name} wins ${pot} with {HandEvaluator.rank_to_hand_name(rank)}")
        print("=" * 40)
