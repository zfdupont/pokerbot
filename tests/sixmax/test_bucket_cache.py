"""Bounded, swappable bucket-memo cache.

The cache is a pure accelerator: bounding it (and evicting) must change memory
only, never outputs. These tests pin that contract — a tiny cap forces constant
eviction, and the buckets must still equal an effectively-unbounded cache's.
"""
import sixmax


def card(rank, suit):
    return (rank - 2) * 4 + suit


TINY = dict(flop_buckets=10, turn_buckets=10, river_buckets=5,
            equity_rollouts=40, quantile_samples=300, seed=42)


def _cases():
    """A spread of hole/board pairs across flop, turn, and river."""
    holes = [[card(14, 3), card(14, 1)],       # AA
             [card(7, 0), card(2, 3)],         # 72o
             [card(13, 0), card(11, 2)]]       # KQo
    flops = [[card(13, 0), card(8, 1), card(3, 2)],    # Kc 8d 3h
             [card(10, 0), card(9, 1), card(5, 2)]]   # Ts 9h 5d
    turns = [f + [card(4, 3)] for f in flops]
    rivers = [t + [card(6, 0)] for t in turns]
    boards = flops + turns + rivers
    for h in holes:
        for b in boards:
            yield h, b


def test_tiny_cap_matches_unbounded():
    """Constant eviction must not change any bucket assignment."""
    small = sixmax.Abstraction(cache_cap=1, **TINY)
    large = sixmax.Abstraction(cache_cap=10_000_000, **TINY)
    for hole, board in _cases():
        assert small.bucket(hole, board) == large.bucket(hole, board)


def test_cache_size_bounded_by_cap():
    cap = 2
    a = sixmax.Abstraction(cache_cap=cap, **TINY)
    for hole, board in _cases():
        a.bucket(hole, board)
    assert a.bucket_cache_size() <= cap * 64


def test_cap_does_not_affect_hash():
    """The cap is not part of the artifact contract."""
    a1 = sixmax.Abstraction(cache_cap=1, **TINY)
    a2 = sixmax.Abstraction(cache_cap=999, **TINY)
    assert a1.hash() == a2.hash()


def test_default_cap_matches_explicit():
    """Default construction (no cache_cap) points at the same abstraction."""
    default = sixmax.Abstraction(**TINY)
    explicit = sixmax.Abstraction(cache_cap=4096, **TINY)
    assert default.hash() == explicit.hash()
