from models.card import Card
from models.enums import Suit


def test_suit_index_values():
    assert Card(2, Suit.HEARTS).suit_index == 1
    assert Card(2, Suit.DIAMONDS).suit_index == 2
    assert Card(2, Suit.CLUBS).suit_index == 3
    assert Card(2, Suit.SPADES).suit_index == 4
