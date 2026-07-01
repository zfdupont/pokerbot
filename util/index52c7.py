"""
Port of Paul D. Senzee's index52c7.h (2007).
Maps a 7-card hand to a unique index in [0, C(52,7)) = [0, 133784560).

Card-to-bit mapping: bit = (rank - 2) * 4 + (suit_index - 1)
  ranks 2-5  → bits  0-15  (chunk D)
  ranks 6-9  → bits 16-31  (chunk C)
  ranks 10-K → bits 32-47  (chunk B)
  rank  A    → bits 48-51  (chunk A)

_TABLE and _BITCOUNT are built once and cached to disk at _CACHE_FILE.
"""
import os
import pickle
from math import comb

_CACHE_FILE = os.path.join(os.path.dirname(__file__), '_index52c7_cache.pkl')

# Verbatim from index52c7.h
_CHOOSE16X = (1, 16, 120, 560, 1820, 4368, 8008, 11440)
_CHOOSE32X = (1, 32, 496, 4960, 35960, 201376, 906192, 3365856)
_CHOOSE48X = (1, 48, 1128, 17296, 194580, 1712304, 12271512, 73629072)
_TABLE4 = (0, 0, 1, 5, 2, 3, 4, 3, 3, 0, 1, 2, 2, 0, 1, 0, 0)
_OFFSETS52C = (0, 73629072, 122715120, 132988944, 133767264)
_OFFSETS48C = (
    0, 3365856, 17864928, 42030048, 62167648, 71194848, 73361376, 73617632, 73629072, 0, 0, 0, 0, 0, 0, 0,
    0,  906192,  4128208,  8443408, 11221008, 12123728, 12263504, 12271512,        0, 0, 0, 0, 0, 0, 0, 0,
    0,  201376,   776736,  1371936,  1649696,  1707936,  1712304,        0,        0, 0, 0, 0, 0, 0, 0, 0,
    0,   35960,   115320,   174840,   192760,   194580,        0,        0,        0, 0, 0, 0, 0, 0, 0, 0,
    0,    4960,    12896,    16736,    17296,        0,        0,        0,        0, 0, 0, 0, 0, 0, 0, 0,
    0,     496,     1008,     1128,        0,        0,        0,        0,        0, 0, 0, 0, 0, 0, 0, 0,
    0,      32,       48,        0,        0,        0,        0,        0,        0, 0, 0, 0, 0, 0, 0, 0,
    0,       1,        0,        0,        0,        0,        0,        0,        0, 0, 0, 0, 0, 0, 0, 0,
)
_OFFSETS32C = (
    0, 11440, 139568, 663728, 1682928, 2702128, 3226288, 3354416, 3365856, 0, 0, 0, 0, 0, 0, 0,
    0,  8008,  77896, 296296,  609896,  828296,  898184,  906192,       0, 0, 0, 0, 0, 0, 0, 0,
    0,  4368,  33488, 100688,  167888,  197008,  201376,       0,       0, 0, 0, 0, 0, 0, 0, 0,
    0,  1820,  10780,  25180,   34140,   35960,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0,
    0,   560,   2480,   4400,    4960,       0,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0,
    0,   120,    376,    496,       0,       0,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0,
    0,    16,     32,      0,       0,       0,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0,
    0,     1,      0,      0,       0,       0,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0,
)


def _build_table_and_bitcount():
    """
    Build _table[65536] and _bitcount[65536].

    _bitcount[v] = popcount of v.
    _table[v] for v with k bits set = combinatorial rank of v among all 16-bit values
    with k bits set, using the recursive formula:
      rank = C(16,k) - C(b_{k-1}+1, k)
           + C(b_{k-1}, k-1) - C(b_{k-2}+1, k-1)
           + ... + b_0
    where b_0 < ... < b_{k-1} are the set-bit positions.
    """
    # Precompute C[n][k] for n in 0..17, k in 0..7 to avoid repeated comb() calls.
    C = [[comb(n, k) for k in range(8)] for n in range(18)]

    table = [0] * 65536
    bitcount = [0] * 65536
    for v in range(1, 65536):
        bits = []
        w = v
        while w:
            lb = w & (-w)
            bits.append(lb.bit_length() - 1)
            w ^= lb
        k = len(bits)
        bitcount[v] = k
        if k == 1:
            table[v] = bits[0]
        elif k <= 7:
            n = 16
            rank = C[n][k] - C[bits[-1] + 1][k]
            n = bits[-1]
            for idx in range(k - 2, 0, -1):
                j = idx + 1
                rank += C[n][j] - C[bits[idx] + 1][j]
                n = bits[idx]
            rank += bits[0]
            table[v] = rank
        # k > 7: not reachable for valid 7-card hands; leave table[v]=0
    return table, bitcount


def _load_or_build():
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    result = _build_table_and_bitcount()
    with open(_CACHE_FILE, 'wb') as f:
        pickle.dump(result, f, protocol=5)
    return result


_TABLE, _BITCOUNT = _load_or_build()


def index52c7(x: int) -> int:
    """Map a 52-bit integer with exactly 7 bits set to a unique index in [0, 133784560)."""
    D = x & 0xFFFF
    C = (x >> 16) & 0xFFFF
    B = (x >> 32) & 0xFFFF
    A = (x >> 48) & 0xFFFF

    bcA = _BITCOUNT[A]
    bcB = _BITCOUNT[B]
    bcC = _BITCOUNT[C]
    bcD = _BITCOUNT[D]

    mulA = _CHOOSE48X[7 - bcA]
    mulB = _CHOOSE32X[7 - bcA - bcB]
    mulC = _CHOOSE16X[bcD]

    return (
        _OFFSETS52C[bcA]                          + _TABLE4[A] * mulA
        + _OFFSETS48C[(bcA << 4) + bcB]           + _TABLE[B]  * mulB
        + _OFFSETS32C[((bcA + bcB) << 4) + bcC]   + _TABLE[C]  * mulC
        + _TABLE[D]
    )


def cards_to_bitmask(cards) -> int:
    """Convert an iterable of Card objects to a 52-bit integer (one bit per card).

    Bit assignment: (rank - 2) * 4 + (suit_index - 1).
    Uses the precomputed card_mask attribute for speed.
    """
    mask = 0
    for c in cards:
        mask |= c.card_mask
    return mask
