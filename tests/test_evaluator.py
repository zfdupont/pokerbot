from models.card import Card
from models.enums import Suit
from models.player import Player
from agents.simple_agent import SimpleAgent
from util.util import hand_value


def test_hand_value_returns_int():
    hole = [Card(14, Suit.SPADES), Card(13, Suit.SPADES)]
    community = [
        Card(12, Suit.SPADES), Card(11, Suit.SPADES), Card(10, Suit.SPADES),
        Card(2, Suit.HEARTS), Card(3, Suit.DIAMONDS),
    ]
    result = hand_value(hole, community)
    assert isinstance(result, int)
    assert result == 1  # Royal Flush = rank 1


def _make_player(name, hole_ranks_suits):
    p = Player(name, 1000, SimpleAgent())
    p.hole_cards = [Card(r, s) for r, s in hole_ranks_suits]
    return p


def test_evaluate_hands_ranks_correctly():
    from util.evaluator import HandEvaluator
    royal = _make_player("Royal", [(14, Suit.SPADES), (13, Suit.SPADES)])
    pair = _make_player("Pair", [(2, Suit.HEARTS), (2, Suit.DIAMONDS)])
    community = [
        Card(12, Suit.SPADES), Card(11, Suit.SPADES), Card(10, Suit.SPADES),
        Card(5, Suit.HEARTS), Card(7, Suit.CLUBS),
    ]
    results = HandEvaluator.evaluate_hands([royal, pair], community)
    assert results[0][0].name == "Royal"


def test_rank_to_hand_name_royal_flush():
    from util.evaluator import HandEvaluator
    assert HandEvaluator.rank_to_hand_name(1) == "Royal Flush"


def test_rank_to_hand_name_high_card():
    from util.evaluator import HandEvaluator
    assert HandEvaluator.rank_to_hand_name(7462) == "High Card"


def test_rank_to_hand_name_pair():
    from util.evaluator import HandEvaluator
    assert HandEvaluator.rank_to_hand_name(3326) == "Pair"
