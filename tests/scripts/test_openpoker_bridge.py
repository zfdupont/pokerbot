import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.openpoker_bot import HandTracker
from models.card import Card
from models.enums import Suit


class SpyStrategy:
    def __init__(self):
        self.kwargs = None
        self.args = None
    def get_action_probs(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return {"fold": 0.5, "call": 0.5}


def make_tracker(big_blind):
    t = HandTracker()
    t.big_blind = big_blind
    t.my_stack = 50 * big_blind          # 50BB stack in chips
    t.my_committed = big_blind
    t.hole_cards = [Card(14, Suit.HEARTS), Card(13, Suit.HEARTS)]
    t.community_cards = []
    return t


def test_scale_derives_from_big_blind_not_buy_in():
    spy = SpyStrategy()
    t = make_tracker(big_blind=20)
    # buy_in deliberately NOT 100*bb: the old bug scaled by buy_in/100.
    t._decide_neural(spy, to_call_chips=40, pot=60, buy_in=1000,
                     abstract_legal=["fold", "call"])
    # Positional args to get_action_probs:
    #   args[0]=hole_ints, args[1]=board_ints, args[2]=street,
    #   args[3]=pot/scale, args[4]=my_stack/scale, args[5]=to_call/scale,
    #   args[6]=raises_per_street, args[7]=my_position
    # stack must arrive in the training frame: 50BB stack -> 50.0
    assert spy.args[4] == 50.0, f"expected stack=50.0, got {spy.args[4]}"   # my_stack / scale
    assert spy.args[3] == 3.0, f"expected pot=3.0, got {spy.args[3]}"       # pot 60 chips / bb20
    assert spy.args[5] == 2.0, f"expected to_call=2.0, got {spy.args[5]}"   # to_call 40 chips / bb20


def test_hu_position_returns_my_position():
    t = make_tracker(big_blind=20)
    t.my_position = 0
    assert t.hu_position() == 0
    t.my_position = 1
    assert t.hu_position() == 1
