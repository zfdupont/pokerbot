"""
Generate missing entries for even_xors_to_odd_xors_to_rank lookup table.
Outputs util/missing_entries.py which patches LookupTables at import time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from itertools import combinations
from collections import defaultdict
import operator, functools

from models.card import Card
from models.enums import Suit, HandRank
from util.util import card_to_binary, popcount
from util.lookup_table import LookupTables

# Midpoint of each Cactus Kev rank range (lower = better)
_HANDRANK_TO_CK = {
    HandRank.ROYAL_FLUSH: 1,
    HandRank.STRAIGHT_FLUSH: 6,
    HandRank.FOUR_OF_KIND: 88,
    HandRank.FULL_HOUSE: 244,
    HandRank.FLUSH: 961,
    HandRank.STRAIGHT: 1604,
    HandRank.THREE_OF_KIND: 2038,
    HandRank.TWO_PAIR: 2896,
    HandRank.PAIR: 4755,
    HandRank.HIGH_CARD: 6824,
}

def fallback_rank(cards):
    from models.hand import Hand
    return _HANDRANK_TO_CK[Hand(list(cards)).rank]

deck = [Card(r, s) for r in range(2, 15) for s in list(Suit)]

# Collect all missing (even_xor, odd_xor) -> representative card set
missing = {}  # (even_xor, odd_xor) -> cards

print("Enumerating 7-card hands...", flush=True)
for i, combo in enumerate(combinations(deck, 7)):
    if i % 5_000_000 == 0:
        print(f"  {i:,} hands processed, {len(missing)} missing found so far...", flush=True)

    binhand = [card_to_binary(c) for c in combo]
    flush_prime = functools.reduce(operator.mul, [(c >> 12) & 0xF for c in binhand])
    if flush_prime in LookupTables.prime_products_to_flush:
        continue

    odd_xor = functools.reduce(operator.xor, binhand)
    even_xor = (functools.reduce(operator.or_, binhand) >> 16) ^ odd_xor
    if even_xor == 0:
        continue

    odd_pc = popcount(odd_xor)
    even_pc = popcount(even_xor)

    key = (even_xor, odd_xor)
    if key in missing:
        continue

    # Mirror the exact dispatch in hand_value to find ALL missing keys
    try:
        if odd_pc == 5:
            _ = LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]  # branch A
        elif odd_pc == 3:
            if even_pc == 2:
                _ = LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]  # branch B
            # else branch C uses prime_products_to_rank — no lookup gap
        else:
            if even_pc == 3:
                _ = LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]  # branch D
            elif even_pc != 2:
                _ = LookupTables.even_xors_to_odd_xors_to_rank[even_xor][odd_xor]  # branch F
            # else branch E uses prime_products_to_rank — no lookup gap
    except KeyError:
        missing[key] = combo

print(f"Found {len(missing)} unique missing (even_xor, odd_xor) pairs", flush=True)

# Evaluate rank for each missing entry
print("Evaluating ranks...", flush=True)
patch = defaultdict(dict)  # even_xor -> {odd_xor: rank}
for (even_xor, odd_xor), cards in missing.items():
    rank = fallback_rank(cards)
    patch[even_xor][odd_xor] = rank

print(f"Writing util/missing_entries.py...", flush=True)
out_path = os.path.join(os.path.dirname(__file__), '..', 'util', 'missing_entries.py')
with open(out_path, 'w') as f:
    f.write("# Auto-generated — do not edit manually.\n")
    f.write("# Patches even_xors_to_odd_xors_to_rank with missing 7-card hand entries.\n")
    f.write("PATCH = {\n")
    for even_xor, inner in sorted(patch.items()):
        f.write(f"    {even_xor}: {{\n")
        for odd_xor, rank in sorted(inner.items()):
            f.write(f"        {odd_xor}: {rank},\n")
        f.write("    },\n")
    f.write("}\n")

print("Done.", flush=True)
