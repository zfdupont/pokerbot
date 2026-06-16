from models.enums import Suit

_SUIT_INDEX = {Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3, Suit.SPADES: 4}


class Card:
    def __init__(self, rank: int, suit: Suit):
        self.rank = rank
        self.suit = suit

    @property
    def suit_index(self) -> int:
        return _SUIT_INDEX[self.suit]

    def __str__(self):
        ranks = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = ranks.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit.value}"
