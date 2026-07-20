"""The six-max openpoker decode path turns a scripted message stream into a
legal action dict via the blueprint bridge."""
import os

from scripts.openpoker_bot import SixmaxHandTracker
from agents.sixmax_agent import SixmaxDeployStrategy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_sixmax_tracker_decides_preflop(blueprint_6max_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_6max_ckpt, _TOML)
    t = SixmaxHandTracker()
    t.on_hand_start({
        "hand_id": 1, "seat": 2, "dealer_seat": 0,
        "blinds": {"small_blind": 10, "big_blind": 20},
        "players": [{"seat": s, "stack": 2000} for s in range(6)],
    })
    t.on_hole_cards({"cards": ["Ah", "Kh"]})
    action = t.decide({
        "hand_id": 1, "pot": 30,
        "players": [{"seat": s, "stack": 2000} for s in range(6)],
        "valid_actions": [
            {"action": "fold"}, {"action": "call", "amount": 20},
            {"action": "raise", "min": 40, "max": 2000},
            {"action": "all_in"}],
    }, strat, buy_in=2000)
    assert action["action"] in ("fold", "call", "raise", "all_in")
    if action["action"] == "raise":
        assert 40 <= action["amount"] <= 2000
