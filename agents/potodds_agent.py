"""A fixed, non-adaptive pot-odds baseline for six-max evaluation.

Deliberately minimal: it NEVER bets or raises, CHECKS when it can act for
free, and when facing a bet calls only when a crude equity estimate beats the
pot odds it is being offered. Because it is fixed and simple, it is a stable
yardstick — a blueprint that improves with training should beat it by a
widening margin, turning "is training helping?" into a measurable curve.

It is intentionally weak (never bets/raises; equity is a rough heuristic).
That's fine for a reference opponent; it is not meant to be strong.
"""
from collections import Counter
from typing import Tuple, Optional

from models.enums import Action
from agents.base_agent import PokerAgent


class PotOddsAgent(PokerAgent):
    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        to_call = game_state.current_bet - player.current_bet

        # Never bet or raise: if nothing to call, take the free card.
        if to_call <= 0:
            return Action.CHECK, None

        # Facing a bet: call iff our estimated equity beats the pot odds.
        equity = self._calculate_equity(player.hole_cards,
                                        game_state.community_cards)
        if self._should_call(game_state.pot, to_call, equity):
            return Action.CALL, to_call
        return Action.FOLD, None

    def _should_call(self, pot: float, to_call: float,
                     hand_equity: float) -> bool:
        """Call iff estimated equity beats the pot odds being offered.

        Pot odds = to_call / (pot + to_call): the fraction of the eventual pot
        we must contribute. Calling is +EV when our equity meets or exceeds
        that fraction, so we call when hand_equity >= pot_odds.
        """
        denom = pot + to_call
        if denom <= 0:
            return False
        return hand_equity >= to_call / denom

    def _calculate_equity(self, hole_cards, board) -> float:
        """Crude, dependency-free equity estimate in [0.05, 0.95].

        NOT a real equity — it ignores opponent range, kickers, and board
        pairing. Preflop it is a high-card / pair / suited heuristic; postflop
        it is a made-hand baseline plus draw outs via the 4/2 rule (outs*4 on
        the flop, outs*2 on the turn). Good enough only as this agent's fixed,
        throwaway yardstick, not as real hand-strength evaluation.
        """
        ranks = sorted((c.rank for c in hole_cards), reverse=True)
        suited = hole_cards[0].suit == hole_cards[1].suit

        if not board:  # preflop heuristic
            hi, lo = ranks
            eq = 0.35 + (hi + lo - 4) / 40.0        # two high cards
            if hi == lo:                             # pocket pair
                eq = 0.50 + (hi - 2) / 24.0          # 22 ~0.50 .. AA ~0.95
            elif suited:
                eq += 0.03
            if hi != lo and 0 < hi - lo <= 4:        # connected-ish
                eq += 0.02
            return _clamp(eq)

        # postflop: made-hand baseline + draw outs (4/2 rule)
        cards = list(hole_cards) + list(board)
        rank_counts = Counter(c.rank for c in cards)
        suit_counts = Counter(c.suit for c in cards)
        best_kind = max(rank_counts.values())        # 2=pair 3=trips 4=quads
        pairs = sum(1 for n in rank_counts.values() if n == 2)
        made_flush = max(suit_counts.values()) >= 5
        made_straight = _has_straight(set(rank_counts))

        if best_kind >= 4 or (made_straight and made_flush):
            made = 0.95
        elif best_kind == 3:
            made = 0.80
        elif made_flush or made_straight:
            made = 0.78
        elif pairs >= 2:
            made = 0.62
        elif best_kind == 2:
            made = 0.52
        else:
            made = 0.30

        outs = 0
        if not made_flush and max(suit_counts.values()) == 4:
            outs += 9                                # flush draw
        if not made_straight:
            outs += 4 * _straight_completions(set(rank_counts))  # 8 OESD/4 gut
        mult = 4 if len(board) == 3 else 2 if len(board) == 4 else 0
        return _clamp(made + outs * mult / 100.0)


def _clamp(equity: float) -> float:
    return max(0.05, min(equity, 0.95))


def _has_straight(rank_set) -> bool:
    """True if the ranks already contain 5 consecutive (ace plays high or low)."""
    s = set(rank_set)
    if 14 in s:
        s.add(1)  # wheel: A-2-3-4-5
    return any(all(lo + k in s for k in range(5)) for lo in range(1, 11))


def _straight_completions(rank_set) -> int:
    """Distinct ranks that would complete a straight: 2 = open-ended, 1 =
    gutshot, 0 = none. Multiplied by 4 (suits) to approximate straight outs."""
    present = set(rank_set)
    return sum(1 for r in range(2, 15)
               if r not in present and _has_straight(present | {r}))