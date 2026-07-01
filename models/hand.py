from typing import List, Optional
from models.enums import HandRank
from models.card import Card

class Hand:
    def __init__(self, cards: List[Card]):
        self.cards = sorted(cards, key=lambda x: x.rank, reverse=True)
        self.rank: Optional[HandRank] = None
        self.rank_cards: List[Card] = []  # Cards that make up the hand rank
        self.kickers: List[Card] = []     # Remaining cards used for tiebreaks
        self._evaluate()

    def _evaluate(self):
        """Evaluate the hand and set rank, rank_cards, and kickers."""
        if self._is_royal_flush():
            self.rank = HandRank.ROYAL_FLUSH
        elif self._is_straight_flush():
            self.rank = HandRank.STRAIGHT_FLUSH
        elif self._is_four_of_kind():
            self.rank = HandRank.FOUR_OF_KIND
        elif self._is_full_house():
            self.rank = HandRank.FULL_HOUSE
        elif self._is_flush():
            self.rank = HandRank.FLUSH
        elif self._is_straight():
            self.rank = HandRank.STRAIGHT
        elif self._is_three_of_kind():
            self.rank = HandRank.THREE_OF_KIND
        elif self._is_two_pair():
            self.rank = HandRank.TWO_PAIR
        elif self._is_pair():
            self.rank = HandRank.PAIR
        else:
            self.rank = HandRank.HIGH_CARD
            self.rank_cards = self.cards[:1]
            self.kickers = self.cards[1:6]

    def _is_royal_flush(self) -> bool:
        if not self._is_straight_flush():
            return False
        return self.cards[0].rank == 14  # Ace-high

    def _is_straight_flush(self) -> bool:
        if not self._is_flush():
            return False
        return self._is_straight()

    def _is_four_of_kind(self) -> bool:
        for i in range(len(self.cards) - 3):
            if all(self.cards[i].rank == self.cards[i + j].rank for j in range(4)):
                self.rank_cards = self.cards[i:i + 4]
                self.kickers = self.cards[:i] + self.cards[i + 4:]
                return True
        return False

    def _is_full_house(self) -> bool:
        ranks = {}
        for card in self.cards:
            ranks[card.rank] = ranks.get(card.rank, 0) + 1

        three_rank = None
        pair_rank = None
        for rank, count in ranks.items():
            if count >= 3 and (three_rank is None or rank > three_rank):
                three_rank = rank
            elif count >= 2 and (pair_rank is None or rank > pair_rank):
                pair_rank = rank

        if three_rank and pair_rank:
            # Set rank cards (three of a kind first, then pair)
            self.rank_cards = [card for card in self.cards if card.rank == three_rank]
            self.rank_cards.extend([card for card in self.cards if card.rank == pair_rank][:2])
            return True
        return False

    def _is_flush(self) -> bool:
        for i in range(len(self.cards) - 4):
            if all(self.cards[i].suit == self.cards[i + j].suit for j in range(5)):
                self.rank_cards = self.cards[i:i + 5]
                self.kickers = self.cards[:i] + self.cards[i + 5:]
                return True
        return False

    def _is_straight(self) -> bool:
        # Handle Ace-low straight (A,2,3,4,5)
        if (self.cards[0].rank == 14 and  # Ace
            any(self.cards[i].rank == 2 and 
                all(self.cards[j].rank == self.cards[j-1].rank - 1 
                    for j in range(i+1, i+4))
                for i in range(1, len(self.cards)-3))):
            ace = self.cards[0]
            straight_cards = sorted([card for card in self.cards[1:] 
                                  if card.rank <= 5], 
                                 key=lambda x: x.rank)[:4]
            straight_cards.append(ace)
            self.rank_cards = straight_cards
            self.kickers = [card for card in self.cards 
                          if card not in straight_cards][:5]
            return True

        # Normal straight
        for i in range(len(self.cards) - 4):
            if all(self.cards[i + j].rank == self.cards[i].rank - j 
                  for j in range(5)):
                self.rank_cards = self.cards[i:i + 5]
                self.kickers = self.cards[:i] + self.cards[i + 5:]
                return True
        return False

    def _is_three_of_kind(self) -> bool:
        for i in range(len(self.cards) - 2):
            if all(self.cards[i].rank == self.cards[i + j].rank for j in range(3)):
                self.rank_cards = self.cards[i:i + 3]
                self.kickers = self.cards[:i] + self.cards[i + 3:]
                return True
        return False

    def _is_two_pair(self) -> bool:
        pairs = []
        i = 0
        while i < len(self.cards) - 1 and len(pairs) < 2:
            if self.cards[i].rank == self.cards[i + 1].rank:
                pairs.extend([self.cards[i], self.cards[i + 1]])
                i += 2
            else:
                i += 1

        if len(pairs) == 4:
            self.rank_cards = pairs
            self.kickers = [card for card in self.cards if card not in pairs]
            return True
        return False

    def _is_pair(self) -> bool:
        for i in range(len(self.cards) - 1):
            if self.cards[i].rank == self.cards[i + 1].rank:
                self.rank_cards = self.cards[i:i + 2]
                self.kickers = self.cards[:i] + self.cards[i + 2:]
                return True
        return False

    def __gt__(self, other: 'Hand') -> bool:
        if self.rank.value != other.rank.value:
            return self.rank.value > other.rank.value

        # Compare rank cards
        for self_card, other_card in zip(self.rank_cards, other.rank_cards):
            if self_card.rank != other_card.rank:
                return self_card.rank > other_card.rank

        # Compare kickers
        for self_card, other_card in zip(self.kickers, other.kickers):
            if self_card.rank != other_card.rank:
                return self_card.rank > other_card.rank

        return False  # Hands are equal

    def __eq__(self, other: 'Hand') -> bool:
        return (not self > other) and (not other > self)