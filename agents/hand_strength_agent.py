from typing import Tuple, Optional, List

from models.card import Card
from models.enums import Action
from agents.base_agent import PokerAgent
from util.util import hand_value

# Preflop hand strength tiers (0.0–1.0 equity estimate, heads-up vs random hand)
# Pairs: rank index = card.rank - 2  (so 2=0, A=12)
# Suited/offsuit connectors grouped by gap and high card
_PREFLOP_PREMIUM = frozenset([
    (14, 14), (13, 13), (12, 12), (11, 11),     # AA KK QQ JJ
    ("s", 14, 13), ("s", 14, 12),                # AKs AQs
    ("o", 14, 13),                               # AKo
])

def _preflop_equity(hole_cards: List[Card]) -> float:
    c1, c2 = sorted(hole_cards, key=lambda c: c.rank, reverse=True)
    r1, r2 = c1.rank, c2.rank
    suited = c1.suit == c2.suit
    gap = r1 - r2

    # Pairs
    if r1 == r2:
        if r1 >= 11:   return 0.82   # JJ+
        if r1 >= 8:    return 0.70   # 88-TT
        if r1 >= 5:    return 0.60   # 55-77
        return 0.52                  # 22-44

    # High-card hands
    prefix = "s" if suited else "o"

    if r1 == 14:
        if r2 >= 12:   return 0.74 if suited else 0.67   # AQ+
        if r2 >= 10:   return 0.65 if suited else 0.58   # AT-AJ
        return 0.58 if suited else 0.52                  # A2-A9

    if r1 == 13 and r2 >= 10:
        return 0.64 if suited else 0.57                  # KT-KQ

    if suited and gap == 1 and r1 >= 9:
        return 0.57                                      # suited connectors JT+

    if suited and r1 >= 10:
        return 0.54                                      # other broadway suited

    return 0.48                                          # everything else


def _postflop_equity(hole_cards: List[Card], community: List[Card]) -> float:
    rank = hand_value(hole_cards, community)
    # Cactus Kev: 1 = best (royal flush), 7462 = worst; invert to 0–1
    return 1.0 - (rank - 1) / 7461.0


def estimate_equity(hole_cards: List[Card], community: List[Card]) -> float:
    if not community:
        return _preflop_equity(hole_cards)
    return _postflop_equity(hole_cards, community)


class HandStrengthAgent(PokerAgent):
    """
    Makes decisions by comparing estimated hand equity to pot odds.
    Preflop equity comes from a heuristic tier table.
    Postflop equity is derived from the Cactus Kev hand rank.
    """

    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        equity = estimate_equity(player.hole_cards, game_state.community_cards)
        to_call = game_state.current_bet - player.current_bet
        pot = game_state.pot if hasattr(game_state, "pot") else 0
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        big_blind = game_state.big_blind

        # TODO(human): implement action selection
        # Given equity (0–1), pot_odds (0–1), to_call, and big_blind,
        # decide what action to return.
        #
        # Things to consider:
        #   - When should you raise vs just call?
        #   - How much should you bet/raise?
        #   - When is folding correct even if equity > pot_odds (marginal edges)?
        #   - What's a reasonable minimum equity to voluntarily put chips in?

        RAISE_THRESHOLD = 0.65   # raise/bet with strong hands
        CALL_THRESHOLD  = 0.40   # minimum equity to voluntarily call

        if to_call == 0:
            if equity >= RAISE_THRESHOLD:
                return Action.BET, big_blind * 3
            return Action.CHECK, None

        if equity >= RAISE_THRESHOLD and player.stack >= pot // 3:
            return Action.RAISE, big_blind * 3
        if equity > pot_odds and equity >= CALL_THRESHOLD:
            return Action.CALL, to_call
        return Action.FOLD, None