"""Suit-canonical hand indexer (Waugh 2013 hand isomorphism)."""
import random

import pytest

import sixmax


def card(rank, suit):
    return (rank - 2) * 4 + suit


def test_version():
    assert sixmax.HAND_INDEX_VERSION == 1


def test_sizes():
    assert [sixmax.hand_index_size(s) for s in (1, 2, 3)] == [
        1_286_792,
        13_960_050,
        123_156_254,
    ]


def test_suit_relabel_and_order_invariant():
    # As Kd | 7s 7h 2c  vs  suits permuted (s<->h, d<->c) + cards reordered
    a = sixmax.hand_index(
        1, [card(14, 3), card(13, 1)], [card(7, 3), card(7, 2), card(2, 0)]
    )
    b = sixmax.hand_index(
        1, [card(13, 0), card(14, 2)], [card(2, 1), card(7, 3), card(7, 2)]
    )
    assert a == b


def _random_deal(street, rng):
    deck = list(range(52))
    rng.shuffle(deck)
    n_board = street + 2
    return deck[:2], deck[2 : 2 + n_board]


def test_roundtrip_random_samples():
    rng = random.Random(12345)
    for street in (1, 2, 3):
        size = sixmax.hand_index_size(street)
        for _ in range(20000):
            hole, board = _random_deal(street, rng)
            idx = sixmax.hand_index(street, hole, board)
            assert 0 <= idx < size
            h2, b2 = sixmax.hand_unindex(street, idx)
            assert sixmax.hand_index(street, h2, b2) == idx


def test_dense_on_flop():
    for i in range(sixmax.hand_index_size(1)):
        h, b = sixmax.hand_unindex(1, i)
        assert sixmax.hand_index(1, h, b) == i


def test_multiplicity_sums_to_raw_deals():
    total = sum(
        sixmax.hand_multiplicity(1, i)
        for i in range(sixmax.hand_index_size(1))
    )
    assert total == 1326 * 19600


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        sixmax.hand_index(
            1, [card(14, 3), card(14, 3)], [card(7, 3), card(7, 2), card(2, 0)]
        )
    with pytest.raises(ValueError):
        sixmax.hand_index(
            1, [card(14, 3), card(13, 1)], [card(7, 3), card(7, 2)]
        )
    with pytest.raises(ValueError):
        sixmax.hand_index(
            1, [card(14, 3), 52], [card(7, 3), card(7, 2), card(2, 0)]
        )
    with pytest.raises(ValueError):
        sixmax.hand_index(
            1,
            [card(14, 3), card(13, 1)],
            [card(7, 3), card(7, 2), card(2, 0), card(3, 1)],
        )
