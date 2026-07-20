"""NeuralAgent plays legal actions in the live engine (HU-mode eval opponent)."""
import os

import pytest

from models.enums import Action

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _neural_ckpt():
    d = os.path.join(_ROOT, "neural_cfr", "checkpoints")
    if not os.path.isdir(d):
        return None
    pts = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    return os.path.join(d, pts[-1]) if pts else None


@pytest.mark.skipif(_neural_ckpt() is None, reason="no neural checkpoint")
def test_neural_agent_get_action_is_legal():
    from agents.neural_agent import NeuralAgent
    from models.player import Player
    from game.poker import PokerGame

    a = NeuralAgent(_neural_ckpt())
    p0 = Player("hero", 200, agent=a)
    p1 = Player("villain", 200, agent=a)  # Player.agent is a required arg
    game = PokerGame([p0, p1], small_blind=1)
    game.state.deck = game.state._create_deck()
    game.state.deal_hole_cards()
    action, amount = a.get_action(p0, game.state)
    assert action in (Action.FOLD, Action.CHECK, Action.CALL, Action.BET)
