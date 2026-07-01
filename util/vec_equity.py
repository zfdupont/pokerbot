"""
Vectorized equity estimator using numpy.

Replaces the 500-iteration Python loop in _equity with batched numpy operations:
  - Draw all n_samples opponent hands at once via argsort on a random matrix
  - Build all card masks in a single numpy broadcast
  - Run index52c7 over the entire (n_samples,) array element-wise
  - Look up all ranks in one numpy fancy-index on _RANK7_NP
  - Compare and sum with numpy reductions

Requires the precomputed rank table (run scripts/build_7card_table.py first).
"""
import numpy as np
from typing import List

from models.card import Card
from models.enums import Suit
from util.index52c7 import (
    _TABLE, _BITCOUNT,
    _CHOOSE16X, _CHOOSE32X, _CHOOSE48X,
    _TABLE4, _OFFSETS52C, _OFFSETS48C, _OFFSETS32C,
)
from util.util import _RANK7

# Convert Python lists → int64 numpy arrays for vectorized table lookups
_NP_TABLE     = np.array(_TABLE,      dtype=np.int64)
_NP_BITCOUNT  = np.array(_BITCOUNT,   dtype=np.int64)
_NP_C16       = np.array(_CHOOSE16X,  dtype=np.int64)
_NP_C32       = np.array(_CHOOSE32X,  dtype=np.int64)
_NP_C48       = np.array(_CHOOSE48X,  dtype=np.int64)
_NP_TABLE4    = np.array(_TABLE4,     dtype=np.int64)
_NP_OFF52     = np.array(_OFFSETS52C, dtype=np.int64)
_NP_OFF48     = np.array(_OFFSETS48C, dtype=np.int64)
_NP_OFF32     = np.array(_OFFSETS32C, dtype=np.int64)

# numpy view of the rank table (same data as array.array _RANK7, no copy)
_RANK7_NP: np.ndarray | None = (
    np.frombuffer(bytes(_RANK7), dtype=np.uint16) if _RANK7 is not None else None
)

_FULL_DECK   = [Card(r, s) for r in range(2, 15) for s in list(Suit)]
_DECK_MASKS  = np.array([c.card_mask for c in _FULL_DECK], dtype=np.int64)
_DECK_KEYS   = [c.key for c in _FULL_DECK]

_rng = np.random.default_rng()


def _index52c7_vec(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized index52c7: maps a (N,) int64 array of 52-bit card masks
    to a (N,) int64 array of unique indices in [0, 133_784_560).

    This is the exact same formula as the scalar index52c7() in util/index52c7.py,
    applied element-wise using numpy fancy indexing.

    Tables available (all int64 numpy arrays):
        _NP_BITCOUNT[v]  — popcount of v (for v in 0..65535)
        _NP_TABLE[v]     — combinatorial rank of v among same-popcount 16-bit values
        _NP_TABLE4[v]    — combinatorial rank for the 4-bit A chunk (v in 0..16)
        _NP_C16[k]       — C(16, k)  for k in 0..7  (_CHOOSE16X)
        _NP_C32[k]       — C(32, k)  for k in 0..7  (_CHOOSE32X)
        _NP_C48[k]       — C(48, k)  for k in 0..7  (_CHOOSE48X)
        _NP_OFF52[bcA]   — offset into index space based on # bits in chunk A (0..4)
        _NP_OFF48[i]     — flat lookup: i = (bcA << 4) + bcB
        _NP_OFF32[i]     — flat lookup: i = ((bcA + bcB) << 4) + bcC

    Hint: split masks into four 16-bit chunks with >> and & 0xFFFF, then use
    _NP_BITCOUNT for popcounts, _NP_C* for multipliers, and fancy indexing
    throughout. Watch out for products that can exceed int32 range — keep int64.
    """
    D = (masks & 0xFFFF).astype(np.int64)
    C = ((masks >> 16) & 0xFFFF).astype(np.int64)
    B = ((masks >> 32) & 0xFFFF).astype(np.int64)
    A = ((masks >> 48) & 0xFFFF).astype(np.int64)

    bcA = _NP_BITCOUNT[A]
    bcB = _NP_BITCOUNT[B]
    bcC = _NP_BITCOUNT[C]
    bcD = _NP_BITCOUNT[D]

    return (
        _NP_OFF52[bcA]                        + _NP_TABLE4[A] * _NP_C48[7 - bcA]
        + _NP_OFF48[(bcA << 4) + bcB]         + _NP_TABLE[B]  * _NP_C32[7 - bcA - bcB]
        + _NP_OFF32[((bcA + bcB) << 4) + bcC] + _NP_TABLE[C]  * _NP_C16[bcD]
        + _NP_TABLE[D]
    )


def equity_vec(hole: List[Card], board: List[Card], n_samples: int = 500) -> float:
    """
    Vectorized equity estimate. Drop-in replacement for _equity() in abstraction.py.

    Samples all n_samples opponent hands at once (no Python loop),
    evaluates all masks in one numpy pass, and returns win fraction.
    Requires _RANK7 table; raises if table is absent.
    """
    assert _RANK7_NP is not None, "equity_vec requires the rank table — run build_7card_table.py"

    known_keys = {c.key for c in hole + board}
    avail_idx  = [i for i, k in enumerate(_DECK_KEYS) if k not in known_keys]
    avail      = _DECK_MASKS[avail_idx]   # (n_avail,) card masks for remaining deck
    n_avail    = len(avail)

    board_len = len(board)
    n_draw    = 2 + (5 - board_len)      # 2 opp cards + cards needed to complete board

    # Vectorized partial Fisher-Yates: draw n_draw distinct indices per row.
    # Builds a (n_samples, n_avail) index matrix, then swaps n_draw columns in-place.
    # O(n_samples × n_draw) vs argsort's O(n_samples × n_avail × log n_avail).
    indices = np.tile(np.arange(n_avail, dtype=np.int32), (n_samples, 1))
    _rows   = np.arange(n_samples)
    for k in range(n_draw):
        j        = _rng.integers(k, n_avail, size=n_samples)
        col_k    = indices[_rows, k].copy()
        indices[_rows, k] = indices[_rows, j]
        indices[_rows, j] = col_k
    perm = indices[:, :n_draw]
    drawn = avail[perm]                   # (n_samples, n_draw): each row = one deal

    # Precompute fixed parts of the mask (board + our hole cards)
    board_mask = np.int64(0)
    for c in board:
        board_mask |= np.int64(c.card_mask)
    hole_mask = np.int64(hole[0].card_mask | hole[1].card_mask)

    # Build runout mask: board | extra community cards drawn per sample
    runout = np.full(n_samples, board_mask, dtype=np.int64)
    for col in range(2, n_draw):
        runout |= drawn[:, col]

    my_masks  = runout | hole_mask                     # (n_samples,)
    opp_masks = runout | drawn[:, 0] | drawn[:, 1]    # (n_samples,)

    my_idx  = _index52c7_vec(my_masks)
    opp_idx = _index52c7_vec(opp_masks)

    my_ranks  = _RANK7_NP[my_idx]
    opp_ranks = _RANK7_NP[opp_idx]

    wins = float(np.sum(my_ranks < opp_ranks)) + 0.5 * float(np.sum(my_ranks == opp_ranks))
    return wins / n_samples
