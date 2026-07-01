"""
Generate missing entries for prime_products_to_rank lookup table.

The XOR hand evaluator has several branches that use prime_products_to_rank,
but the existing table (21,463 entries) is incomplete. Since the prime product
key depends only on rank (not suit), we can enumerate all C(19,7)=50,388 rank
multisets of size 7 instead of all C(52,7)=133M card combinations.

Outputs util/prime_products_patch.py which is applied at import time in util.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from itertools import combinations_with_replacement
from models.enums import Suit
from models.card import Card
from util.lookup_table import LookupTables
from util.util import popcount, _fallback_hand_value, _STRAIGHT_MASKS, _HANDRANK_TO_CK
import functools
import operator

RANKS = list(range(2, 15))
SUITS = list(Suit)

# Suit primes: suit_index 1-4 → primes 2,3,5,7
SUIT_PRIMES = [LookupTables.primes[i] for i in range(4)]  # [2, 3, 5, 7]

# Representative non-flush suit assignments for 7 cards.
# We vary the distribution to cover different XOR routing branches.
# Round-robin ensures max 2 per suit → never flush.
SUIT_ASSIGNMENTS = [
    [0, 1, 2, 3, 0, 1, 2],  # (2,2,2,1) — most common non-flush
    [0, 0, 0, 1, 1, 2, 3],  # (3,2,1,1)
    [0, 0, 0, 0, 1, 2, 3],  # (4,1,1,1)
    [0, 0, 0, 1, 1, 1, 2],  # (3,3,1,0)
    [0, 0, 0, 0, 1, 1, 2],  # (4,2,1,0)
    [0, 0, 0, 0, 1, 1, 1],  # (4,3,0,0)
    [0, 0, 1, 1, 2, 2, 3],  # (2,2,2,1) different order
    [1, 0, 2, 0, 3, 1, 2],  # interleaved
]


def compute_prime_product(binhand):
    return functools.reduce(operator.mul, [card & 0xFF for card in binhand])


def routes_to_prime_products(binhand):
    """Return True if this hand routes to the prime_products_to_rank branch."""
    flush_prime = functools.reduce(operator.mul, [(card >> 12) & 0xF for card in binhand])
    if flush_prime in LookupTables.prime_products_to_flush:
        return False  # flush path

    odd_xor = functools.reduce(operator.xor, binhand)
    even_xor = (functools.reduce(operator.or_, binhand) >> 16) ^ odd_xor

    odd_pc = popcount(odd_xor)

    if even_xor == 0:
        return odd_pc != 7  # branch: even_xor==0, non-all-distinct → prime_products
    else:
        even_pc = popcount(even_xor)
        if odd_pc == 5:
            return False  # → even_xors_to_odd_xors_to_rank
        elif odd_pc == 3:
            return even_pc != 2  # even_pc==2 → even_xors table; else → prime_products
        else:  # odd_pc == 1
            if even_pc == 3:
                return False  # → even_xors table
            elif even_pc == 2:
                return True   # → prime_products
            else:
                return False  # → even_xors table


def fast_rank_from_ranks(rank_list):
    """Determine coarse CK rank using bitmask classifier (no Card objects needed)."""
    rank_count = [0] * 15
    rank_bits = 0
    for r in rank_list:
        rank_count[r] += 1
        rank_bits |= 1 << r

    quads = trips = pairs = 0
    for cnt in rank_count:
        if cnt == 4:
            quads = 1
        elif cnt == 3:
            trips += 1
        elif cnt == 2:
            pairs += 1

    # No flush possible for rank-only analysis (use non-flush assumption)
    if quads:
        from models.enums import HandRank
        return _HANDRANK_TO_CK[HandRank.FOUR_OF_KIND]
    if trips and (pairs or trips >= 2):
        from models.enums import HandRank
        return _HANDRANK_TO_CK[HandRank.FULL_HOUSE]
    for mask in _STRAIGHT_MASKS:
        if rank_bits & mask == mask:
            from models.enums import HandRank
            return _HANDRANK_TO_CK[HandRank.STRAIGHT]
    if trips:
        from models.enums import HandRank
        return _HANDRANK_TO_CK[HandRank.THREE_OF_KIND]
    if pairs >= 2:
        from models.enums import HandRank
        return _HANDRANK_TO_CK[HandRank.TWO_PAIR]
    if pairs:
        from models.enums import HandRank
        return _HANDRANK_TO_CK[HandRank.PAIR]
    from models.enums import HandRank
    return _HANDRANK_TO_CK[HandRank.HIGH_CARD]


def main():
    patch = {}
    checked = 0
    routes = 0

    print("Enumerating rank multisets...", flush=True)

    for rank_combo in combinations_with_replacement(RANKS, 7):
        checked += 1

        # Compute rank prime product (independent of suit)
        prime_product = 1
        for r in rank_combo:
            prime_product *= LookupTables.primes[r - 2]

        # Skip if already in table
        if prime_product in LookupTables.prime_products_to_rank:
            continue

        # Try multiple suit assignments to find if any routes to prime_products_to_rank
        for suit_assignment in SUIT_ASSIGNMENTS:
            cards = [Card(r, SUITS[suit_assignment[i]]) for i, r in enumerate(rank_combo)]
            binhand = [c.binary for c in cards]

            if routes_to_prime_products(binhand):
                routes += 1
                rank = fast_rank_from_ranks(list(rank_combo))
                patch[prime_product] = rank
                break  # one representative per rank multiset is enough

    print(f"Checked {checked:,} rank multisets", flush=True)
    print(f"Found {routes:,} new prime_products_to_rank entries", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'util', 'prime_products_patch.py')
    with open(out_path, 'w') as f:
        f.write("# Auto-generated — do not edit manually.\n")
        f.write("# Patches prime_products_to_rank with missing 7-card hand entries.\n")
        f.write("PRIME_PRODUCTS_PATCH = {\n")
        for product, rank in sorted(patch.items()):
            f.write(f"    {product}: {rank},\n")
        f.write("}\n")

    print(f"Written to util/prime_products_patch.py", flush=True)


if __name__ == "__main__":
    main()
