import array
import os

import numpy as np

from models.card import Card
from util.lookup_table import LookupTables
from util.ck_tables import FLUSHES, UNIQUE5, HASH_ADJUST, HASH_VALUES
from util.index52c7 import index52c7, cards_to_bitmask

_PRIMES = LookupTables.primes  # [2,3,5,7,11,13,17,19,23,29,31,37,41], indexed by rank-2

# Straight detection: A-high (index 0) down to 5-high/wheel (last)
_STRAIGHT_MASKS = (
    [(1 << h) | (1 << (h-1)) | (1 << (h-2)) | (1 << (h-3)) | (1 << (h-4))
     for h in range(14, 5, -1)]
    + [(1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)]
)

_hand_value_cache: dict = {}

# Precomputed 133,784,560-entry rank table: RANK7[index52c7(mask)] → CK rank.
# When present, eliminates _best_5_of_7 and the dict cache entirely.
_TABLE_PATH = os.path.join(os.path.dirname(__file__), '7card_rank_table.npy')
_RANK7: array.array | None = None
if os.path.exists(_TABLE_PATH):
    _data = np.load(_TABLE_PATH)
    _RANK7 = array.array('H', _data)
    del _data


def _hash_lookup(prime_product: int) -> int:
    """Perfect hash from senzee.blogspot.com — simulates 32-bit unsigned overflow."""
    u = (prime_product + 0xe91aaa35) & 0xFFFFFFFF
    u ^= u >> 16
    u = (u + (u << 8)) & 0xFFFFFFFF
    u ^= u >> 4
    return HASH_VALUES[(((u + (u << 2)) & 0xFFFFFFFF) >> 19) ^ HASH_ADJUST[(u >> 8) & 0x1ff]]


def _bits_to_rank_q(bits: int, n: int) -> int:
    """Return bitmask-for-FLUSHES/UNIQUE5 using top-n ranks set in `bits`."""
    q = 0
    for r in range(14, 1, -1):
        if bits & (1 << r):
            q |= 1 << (r - 2)
            n -= 1
            if n == 0:
                break
    return q


def _best_5_of_7(cards) -> int:
    """
    Best CK rank from 7 cards via single-pass preprocessing + CK table lookup.

    Processes rank/suit information in O(7), then determines category and
    computes the exact CK rank without enumerating all 21 subsets.
    """
    rank_count = [0] * 15
    suit_count = [0] * 5
    suit_rank_bits = [0] * 5
    rank_bits = 0

    for c in cards:
        r, si = c.rank, c.suit_index
        rank_count[r] += 1
        suit_count[si] += 1
        suit_rank_bits[si] |= 1 << r
        rank_bits |= 1 << r

    # --- Flush / Straight Flush ---
    flush_si = 0
    flush_bits = 0
    for si in range(1, 5):
        if suit_count[si] >= 5:
            flush_si = si
            flush_bits = suit_rank_bits[si]
            break

    if flush_si:
        for mask in _STRAIGHT_MASKS:
            if flush_bits & mask == mask:
                q = sum(1 << (r - 2) for r in range(2, 15) if mask & (1 << r))
                return FLUSHES[q]
        # Plain flush: top 5 ranks in flush suit
        return FLUSHES[_bits_to_rank_q(flush_bits, 5)]

    # --- Quads ---
    for r in range(14, 1, -1):
        if rank_count[r] == 4:
            kicker = next(r2 for r2 in range(14, 1, -1) if r2 != r and rank_count[r2] > 0)
            return _hash_lookup((_PRIMES[r - 2] ** 4) * _PRIMES[kicker - 2])

    # --- Full House ---
    trips_r = next((r for r in range(14, 1, -1) if rank_count[r] >= 3), None)
    if trips_r is not None:
        pair_r = next((r for r in range(14, 1, -1) if r != trips_r and rank_count[r] >= 2), None)
        if pair_r is not None:
            return _hash_lookup((_PRIMES[trips_r - 2] ** 3) * (_PRIMES[pair_r - 2] ** 2))

    # --- Straight ---
    for mask in _STRAIGHT_MASKS:
        if rank_bits & mask == mask:
            q = sum(1 << (r - 2) for r in range(2, 15) if mask & (1 << r))
            return UNIQUE5[q]

    # --- Three of a Kind ---
    if trips_r is not None:
        kickers = [r for r in range(14, 1, -1) if r != trips_r and rank_count[r] > 0][:2]
        return _hash_lookup((_PRIMES[trips_r - 2] ** 3)
                            * _PRIMES[kickers[0] - 2] * _PRIMES[kickers[1] - 2])

    # --- Two Pair ---
    pairs = [r for r in range(14, 1, -1) if rank_count[r] >= 2]
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = next(r for r in range(14, 1, -1) if r not in (p1, p2) and rank_count[r] > 0)
        return _hash_lookup((_PRIMES[p1 - 2] ** 2) * (_PRIMES[p2 - 2] ** 2) * _PRIMES[kicker - 2])

    # --- One Pair ---
    if pairs:
        p1 = pairs[0]
        kickers = [r for r in range(14, 1, -1) if r != p1 and rank_count[r] > 0][:3]
        return _hash_lookup((_PRIMES[p1 - 2] ** 2)
                            * _PRIMES[kickers[0] - 2] * _PRIMES[kickers[1] - 2] * _PRIMES[kickers[2] - 2])

    # --- High Card ---
    return UNIQUE5[_bits_to_rank_q(rank_bits, 5)]


def hand_value(hole_cards, community) -> int:
    """Best 5-card CK rank from 7 cards (2 hole + 5 board). Lower = better hand."""
    mask = 0
    for c in hole_cards:
        mask |= c.card_mask
    for c in community:
        mask |= c.card_mask
    if _RANK7 is not None:
        return _RANK7[index52c7(mask)]
    key = index52c7(mask)
    cached = _hand_value_cache.get(key)
    if cached is not None:
        return cached
    result = _best_5_of_7(list(hole_cards) + list(community))
    _hand_value_cache[key] = result
    return result


def hand_value_fast(hole_cards, community) -> int:
    """Same as hand_value — unified CK evaluator. Lower = better hand."""
    return hand_value(hole_cards, community)


def hand_value_from_key(key: int, hole_cards, community) -> int:
    """Direct lookup using a precomputed index52c7 integer key."""
    if _RANK7 is not None:
        return _RANK7[key]
    cached = _hand_value_cache.get(key)
    if cached is not None:
        return cached
    result = _best_5_of_7(list(hole_cards) + list(community))
    _hand_value_cache[key] = result
    return result


def hand_value_from_mask(mask: int) -> int:
    """Best CK rank from a precomputed 52-bit card bitmask. Requires _RANK7 table."""
    return _RANK7[index52c7(mask)]


def card_to_binary(card: Card):
    b_mask = 1 << (14 + card.rank)
    q_mask = LookupTables.primes[card.suit_index - 1] << 12
    r_mask = (card.rank - 2) << 8
    p_mask = LookupTables.primes[card.rank - 2]
    return b_mask | q_mask | r_mask | p_mask


def card_to_binary_lookup(card: Card):
    return LookupTables.card_to_binary[card.rank][card.suit_index]
