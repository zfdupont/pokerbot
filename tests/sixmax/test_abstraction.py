"""169-class preflop indexing + deterministic MC equity-percentile buckets."""
import itertools

import sixmax

# card(r, s) = (r-2)*4 + s ; suits 0=c 1=d 2=h 3=s
def card(rank, suit):
    return (rank - 2) * 4 + suit


TINY = dict(flop_buckets=10, turn_buckets=10, river_buckets=5,
            equity_rollouts=40, quantile_samples=300, seed=42)


def test_preflop_class_covers_exactly_169():
    classes = set()
    for c0, c1 in itertools.combinations(range(52), 2):
        classes.add(sixmax.preflop_class([c0, c1]))
    assert len(classes) == 169
    assert min(classes) == 0 and max(classes) == 168


def test_preflop_class_known_values():
    # pairs on the diagonal: class = r*13 + r
    assert sixmax.preflop_class([card(14, 3), card(14, 2)]) == 12 * 13 + 12  # AA
    assert sixmax.preflop_class([card(2, 0), card(2, 1)]) == 0              # 22
    # suited above the diagonal (hi*13+lo), offsuit below (lo*13+hi)
    assert sixmax.preflop_class([card(14, 3), card(13, 3)]) == 12 * 13 + 11  # AKs
    assert sixmax.preflop_class([card(14, 3), card(13, 1)]) == 11 * 13 + 12  # AKo
    # order of the two cards must not matter
    assert (sixmax.preflop_class([card(13, 1), card(14, 3)])
            == sixmax.preflop_class([card(14, 3), card(13, 1)]))


BOARD = [card(13, 0), card(8, 1), card(3, 2)]  # Kc 8d 3h — dry rainbow


def test_hand_equity_orders_hands_and_is_deterministic():
    aa = [card(14, 3), card(14, 1)]
    trash = [card(7, 0), card(2, 3)]
    e_aa = sixmax.hand_equity(aa, BOARD, 200, 7)
    e_tr = sixmax.hand_equity(trash, BOARD, 200, 7)
    assert 0.0 <= e_tr < e_aa <= 1.0
    assert e_aa > 0.75
    assert e_tr < 0.5
    # deterministic: same inputs -> bitwise-same estimate
    assert e_aa == sixmax.hand_equity(aa, BOARD, 200, 7)
    # hole-card order must not matter (seed uses sorted cards)
    assert e_aa == sixmax.hand_equity([card(14, 1), card(14, 3)], BOARD, 200, 7)


def test_abstraction_buckets_in_range_and_monotone():
    abs_ = sixmax.Abstraction(**TINY)
    aa = [card(14, 3), card(14, 1)]
    trash = [card(7, 0), card(2, 3)]
    b_aa = abs_.bucket(aa, BOARD)
    b_tr = abs_.bucket(trash, BOARD)
    assert 0 <= b_tr <= b_aa < 10
    # river board: bucket range obeys river_buckets
    river = BOARD + [card(9, 3), card(4, 1)]
    assert 0 <= abs_.bucket(aa, river) < 5
    assert abs_.num_buckets(1) == 10
    assert abs_.num_buckets(3) == 5


def test_abstraction_edges_sorted_and_hash_stable():
    a1 = sixmax.Abstraction(**TINY)
    a2 = sixmax.Abstraction(**TINY)
    assert a1.hash() == a2.hash()          # same config -> same edges -> same hash
    a3 = sixmax.Abstraction(**{**TINY, "seed": 43})
    assert a1.hash() != a3.hash()
    for street_edges in a1.edges():
        assert street_edges == sorted(street_edges)
