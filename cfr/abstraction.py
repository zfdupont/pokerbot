import random
from functools import lru_cache
from itertools import combinations
from typing import List, Tuple
from models.card import Card
from models.enums import Suit
from util.util import hand_value, hand_value_fast, hand_value_from_mask, _RANK7
from util.index52c7 import index52c7
from util.vec_equity import equity_vec as _equity_vec

NUM_PREFLOP_BUCKETS = 6
NUM_POSTFLOP_BUCKETS = 5
# Percentile-based thresholds per street (descending equity order).
# Each divides hands into equal-frequency buckets: bucket = sum(1 for t in T if equity < t)
# 0 = strongest bucket, N-1 = weakest.
# Preflop: computed from all 1,326 HU combos at 500 MC samples (analyze_equity_distribution.py).
# Postflop: sampled from 4,000 random (hole, board) pairs per street at 500 MC samples.
PREFLOP_THRESHOLDS  = (0.593, 0.544, 0.499, 0.448, 0.393)
POSTFLOP_THRESHOLDS = {
    1: (0.698, 0.544, 0.433, 0.314),  # flop
    2: (0.736, 0.548, 0.405, 0.263),  # turn
    3: (0.808, 0.594, 0.385, 0.187),  # river
}
MONTE_CARLO_SAMPLES = 500  # rollouts per equity estimate (higher = more stable)

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
    exclude_set = {c.key for c in exclude}
    result = []
    needed = n
    while needed:
        c = _DECK[int(random.random() * 52)]
        k = c.key
        if k not in exclude_set:
            result.append(c)
            exclude_set.add(k)
            needed -= 1
    return result


def _draw_two_cards(exclude_set: set) -> List[Card]:
    """Draw 2 distinct cards not in exclude_set. No set allocation in hot loop."""
    while True:
        c1 = _DECK[int(random.random() * 52)]
        if c1.key not in exclude_set:
            k1 = c1.key
            while True:
                c2 = _DECK[int(random.random() * 52)]
                k2 = c2.key
                if k2 not in exclude_set and k2 != k1:
                    return [c1, c2]


def _equity(hole: List[Card], board: List[Card]) -> float:
    """Monte Carlo equity estimate: fraction of rollouts this hand wins."""
    known = hole + board
    board_len = len(board)
    wins = 0

    if _RANK7 is not None:
        hole_mask = hole[0].card_mask | hole[1].card_mask
        board_mask = 0
        for c in board:
            board_mask |= c.card_mask

        if board_len == 5:
            my_rank = _RANK7[index52c7(hole_mask | board_mask)]
            known_set = {c.key for c in known}
            for _ in range(MONTE_CARLO_SAMPLES):
                opp = _draw_two_cards(known_set)
                opp_rank = _RANK7[index52c7(board_mask | opp[0].card_mask | opp[1].card_mask)]
                if my_rank < opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    wins += 0.5
        else:
            n_draw = 2 + (5 - board_len)
            for _ in range(MONTE_CARLO_SAMPLES):
                remaining = _random_cards(known, n_draw)
                runout_mask = board_mask
                for c in remaining[2:]:
                    runout_mask |= c.card_mask
                my_rank = _RANK7[index52c7(hole_mask | runout_mask)]
                opp_rank = _RANK7[index52c7(remaining[0].card_mask | remaining[1].card_mask | runout_mask)]
                if my_rank < opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    wins += 0.5
    else:
        if board_len == 5:
            my_rank = hand_value_fast(hole, board)
            known_set = {c.key for c in known}
            for _ in range(MONTE_CARLO_SAMPLES):
                opp_hole = _draw_two_cards(known_set)
                opp_rank = hand_value_fast(opp_hole, board)
                if my_rank < opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    wins += 0.5
        else:
            n_draw = 2 + (5 - board_len)
            for _ in range(MONTE_CARLO_SAMPLES):
                remaining = _random_cards(known, n_draw)
                runout = board + remaining[2:]
                my_rank = hand_value_fast(hole, runout)
                opp_rank = hand_value_fast(remaining[:2], runout)
                if my_rank < opp_rank:
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
    eq = _equity_vec(hole, board) if _RANK7 is not None else _equity(hole, board)

    if street == 0:
        return sum(1 for t in PREFLOP_THRESHOLDS if eq < t)
    return sum(1 for t in POSTFLOP_THRESHOLDS[street] if eq < t)


_FULL_DECK = [Card(r, s) for r in range(2, 15) for s in list(Suit)]


def prewarm_preflop_buckets() -> int:
    """Pre-compute equity buckets for all C(52,2)=1326 preflop hole-card combos.

    Populates the lru_cache on _hand_to_bucket_cached so every preflop bucket
    lookup during training is a free cache hit (~5M/sec vs ~1.6k/sec cold).
    Returns the number of entries warmed.
    """
    count = 0
    for h0, h1 in combinations(_FULL_DECK, 2):
        hole_key = tuple(sorted([h0.key, h1.key]))
        _hand_to_bucket_cached(hole_key, (), 0)
        count += 1
    return count


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
    hole_key = tuple(sorted(c.key for c in hole))
    board_key = tuple(sorted(c.key for c in board))
    return _hand_to_bucket_cached(hole_key, board_key, street)


def board_to_bucket(board: List[Card], street: int) -> int:
    """Classify board texture: 0=dry, 1=semi-wet, 2=wet."""
    board_key = tuple(sorted(c.key for c in board))
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
