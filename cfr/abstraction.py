import random
from functools import lru_cache
from typing import List, Tuple
from models.card import Card
from models.enums import Suit
from util.util import hand_value

NUM_PREFLOP_BUCKETS = 8
NUM_POSTFLOP_BUCKETS = 5
MONTE_CARLO_SAMPLES = 100  # rollouts per equity estimate (higher = more stable)

BET_SIZES = {
    "fold":  None,
    "check": 0.0,
    "call":  None,
    "b0.5":  0.5,
    "b1.0":  1.0,
    "allin": None,
}

_DECK = [Card(r, s) for r in range(2, 15) for s in list(Suit)]


def _random_cards(exclude: List[Card], n: int) -> List[Card]:
    exclude_set = {(c.rank, c.suit) for c in exclude}
    pool = [c for c in _DECK if (c.rank, c.suit) not in exclude_set]
    return random.sample(pool, n)


def _equity(hole: List[Card], board: List[Card]) -> float:
    """Monte Carlo equity estimate: fraction of rollouts this hand wins."""
    known = hole + board
    wins = 0
    for _ in range(MONTE_CARLO_SAMPLES):
        remaining = _random_cards(known, 2 + (5 - len(board)))
        opp_hole = remaining[:2]
        runout = board + remaining[2:]
        my_rank = hand_value(hole, runout)
        opp_rank = hand_value(opp_hole, runout)
        if my_rank < opp_rank:  # lower Cactus Kev rank = better hand
            wins += 1
        elif my_rank == opp_rank:
            wins += 0.5
    return wins / MONTE_CARLO_SAMPLES


def _card_key(card: Card) -> Tuple[int, int]:
    return (card.rank, card.suit.value)


@lru_cache(maxsize=None)
def _hand_to_bucket_cached(hole_key: Tuple, board_key: Tuple, street: int) -> int:

    hole = [Card(r, Suit(s)) for r,s in hole_key]
    board = [Card(r, Suit(s)) for r,s in board_key]
    eq = _equity(hole, board)

    if street == 0:
        n = NUM_PREFLOP_BUCKETS
    else:
        n = NUM_POSTFLOP_BUCKETS

    bucket = int((1.0 - eq) * n)
    return min(bucket, n-1)


@lru_cache(maxsize=None)
def _board_to_bucket_cached(board_key: Tuple, street: int) -> int:

    if street == 0 or not board_key:
        return 0
    
    suits = [suit for _, suit in board_key]
    ranks = [rank for rank, _ in board_key]

    flush_draw = max(suits.count(s) for s in set(suits)) >= 2
    straight_draw = any(ranks[i+1] - ranks[i] <= 2 for i in range(len(ranks) - 1))

    return int(flush_draw) + int(straight_draw)


def hand_to_bucket(hole: List[Card], board: List[Card], street: int) -> int:
    """Map hole cards + board to an equity bucket (0 = strongest)."""
    hole_key = tuple(sorted(_card_key(c) for c in hole))
    board_key = tuple(sorted(_card_key(c) for c in board))
    return _hand_to_bucket_cached(hole_key, board_key, street)


def board_to_bucket(board: List[Card], street: int) -> int:
    """Classify board texture: 0=dry, 1=semi-wet, 2=wet."""
    board_key = tuple(sorted(_card_key(c) for c in board))
    return _board_to_bucket_cached(board_key, street)


def legal_abstract_actions(
    to_call: float,
    pot: float,
    stack: float,
    current_bet: float,
    player_bet: float,
) -> List[str]:
    """Return list of legal abstract action strings for this decision point."""
    actions = []
    if to_call > 0:
        actions.append("fold")
        if stack >= to_call:
            actions.append("call")
        # Raise options (only if stack exceeds call + minimum raise)
        effective_pot = pot + to_call * 2
        for size in [0.5, 1.0]:
            raise_additional = to_call + size * effective_pot
            if stack > raise_additional:
                actions.append(f"b{size:.1f}")
        # allin is available if there's any stack left (including when stack == to_call)
        if stack >= to_call:
            actions.append("allin")
    else:
        actions.append("check")
        for size in [0.5, 1.0]:
            if stack > size * pot:
                actions.append(f"b{size:.1f}")
        if stack > 0:
            actions.append("allin")
    return actions


def snap_to_abstract_bet(fraction: float) -> str:
    """Snap a real bet fraction (bet/pot) to nearest abstract bet size."""
    if fraction >= 2.0:
        return "allin"
    elif fraction >= 0.7:  # Bias toward larger bets when >= 0.7
        return "b1.0"
    else:
        return "b0.5"
