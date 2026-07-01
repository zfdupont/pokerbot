"""
Build the precomputed 7-card rank table.

Enumerates all C(52,7) = 133,784,560 7-card hands, evaluates each with the
CK evaluator, and writes a numpy uint16 array to util/7card_rank_table.npy.

After building, hand_value() uses the table for O(1) evaluation with no cache.

Run from the repo root:
    uv run python scripts/build_7card_table.py

Expected time: ~12–15 minutes on a single core.
"""
import os
import sys
import time
import random
from itertools import combinations

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.card import Card
from models.enums import Suit
from util.index52c7 import index52c7
from util.util import _best_5_of_7

N_HANDS = 133_784_560
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'util', '7card_rank_table.npy')

# Flat list of all 52 cards, ordered so card i → bit i in the 52-bit mask.
# bit = (rank - 2) * 4 + (suit_index - 1), matching cards_to_bitmask().
DECK = [Card(r, s) for r in range(2, 15) for s in list(Suit)]


def build_table() -> np.ndarray:
    table = np.zeros(N_HANDS, dtype=np.uint16)

    for combo in tqdm(combinations(range(52), 7), total=N_HANDS, desc='Building'):
        cards = [DECK[i] for i in combo]
        mask = 0
        for c in cards:
            mask |= c.card_mask
        table[index52c7(mask)] = _best_5_of_7(cards)

    return table


def verify(table: np.ndarray) -> None:
    """Verify table completeness and spot-check correctness against _best_5_of_7."""
    zeros = (table == 0).sum()
    assert zeros == 0, f"Table has {zeros} unfilled entries — index collision or missing hands"
    assert 1 <= table.min() and table.max() <= 7462, \
        f"Rank out of range: min={table.min()}, max={table.max()}"

    print("Spot-checking 10,000 random hands...")
    rng = random.Random(42)
    deck_indices = list(range(52))
    for _ in range(10_000):
        combo = tuple(rng.sample(deck_indices, 7))
        cards = [DECK[i] for i in combo]
        mask = 0
        for c in cards:
            mask |= c.card_mask
        expected = _best_5_of_7(cards)
        got = int(table[index52c7(mask)])
        assert got == expected, f"Mismatch: expected {expected}, got {got} for {combo}"

    print("All checks passed.")


def main():
    t0 = time.perf_counter()
    table = build_table()
    elapsed = time.perf_counter() - t0
    print(f"Built in {elapsed:.1f}s")

    verify(table)

    np.save(OUT_PATH, table)
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"Saved {size_mb:.0f} MB → {os.path.abspath(OUT_PATH)}")
    print("Restart your Python process to load the table automatically.")


if __name__ == '__main__':
    main()
