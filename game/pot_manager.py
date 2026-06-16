from dataclasses import dataclass
from typing import Dict, List, Tuple

from models.player import Player


@dataclass
class SidePot:
    amount: int
    eligible: List[Player]


class PotManager:
    def __init__(self):
        self._contributions: Dict[Player, int] = {}

    def contribute(self, player: Player, amount: int) -> None:
        self._contributions[player] = self._contributions.get(player, 0) + amount
        player.total_contributed += amount

    @property
    def total(self) -> int:
        return sum(self._contributions.values())

    def calculate_side_pots(self) -> List[SidePot]:
        contributions = self._contributions
        all_players = list(contributions.keys())

        all_in_levels = sorted(set(
            contributions[p] for p in all_players if p.is_all_in
        ))

        if not all_in_levels:
            return [SidePot(amount=self.total, eligible=list(all_players))]

        pots = []
        prev_level = 0

        for level in all_in_levels:
            eligible = [p for p in all_players if contributions[p] >= level]
            amount = sum(
                min(contributions[p], level) - min(contributions[p], prev_level)
                for p in all_players
            )
            if amount > 0:
                pots.append(SidePot(amount=amount, eligible=eligible))
            prev_level = level

        remaining = sum(max(0, contributions[p] - prev_level) for p in all_players)
        if remaining > 0:
            eligible = [p for p in all_players if contributions[p] > prev_level]
            pots.append(SidePot(amount=remaining, eligible=eligible))

        return pots

    def award(self, player_ranks: List[Tuple[Player, int]]) -> None:
        pots = self.calculate_side_pots()
        rank_map = {p: r for p, r in player_ranks}

        for pot in pots:
            eligible_ranked = [
                (p, rank_map[p]) for p in pot.eligible if p in rank_map
            ]
            if not eligible_ranked:
                continue

            best_rank = min(r for _, r in eligible_ranked)
            winners = [p for p, r in eligible_ranked if r == best_rank]

            split = pot.amount // len(winners)
            remainder = pot.amount % len(winners)
            for w in winners:
                w.stack += split
            winners[0].stack += remainder
