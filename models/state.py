from typing import List
import random
from models.card import Card
from models.enums import Suit
from models.player import Player

class GameState:
    def __init__(self, players: List[Player], small_blind: int = 1):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.deck = self._create_deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.button_pos = 0
        self.current_player_idx = 0
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.raises_per_street = [0, 0, 0, 0]  # raise counts per street, capped at 2
        
    def _create_deck(self) -> List[Card]:
        deck = []
        for suit in Suit:
            for rank in range(2, 15):
                deck.append(Card(rank, suit))
        random.shuffle(deck)
        return deck
    
    def deal_hole_cards(self):
        for _ in range(2):
            for player in self.players:
                if player.is_active:
                    player.hole_cards.append(self.deck.pop())
    
    def deal_community_cards(self):
        if self.betting_round == 1:  # Flop
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
        else:  # Turn or River
            self.community_cards.append(self.deck.pop())
