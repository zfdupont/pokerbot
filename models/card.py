from models.enums import Suit

_SUIT_INDEX = {Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3, Suit.SPADES: 4}
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


class Card:
    def __init__(self, rank: int, suit: Suit):
        self.rank = rank
        self.suit = suit
        si = _SUIT_INDEX[suit]
        self.suit_index = si
        self.binary = (1 << (14 + rank)) | (_PRIMES[si - 1] << 12) | ((rank - 2) << 8) | _PRIMES[rank - 2]
        self.key = (rank, suit.value)
        self.card_mask = 1 << ((rank - 2) * 4 + (si - 1))

    def __str__(self):
        ranks = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = ranks.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit.value}"
