"""Faithful street-root reconstruction: replaying the live action_history into
a fresh engine state must reproduce the live pot (in bb) at the start of the
current betting round. Requires the gitignored HU checkpoint."""
import os

import pytest

from models.enums import Action

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "hu_blueprint.bin")
_TOML = os.path.join(_ROOT, "sixmax", "configs", "hu.toml")
pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint gitignored")


def test_street_root_pot_matches_live_hand():
    """Script a limp/check preflop line (no bet quantization), capture the live
    GameState at the hero's flop turn, and assert the reconstructed flop-root pot
    equals the live pot going into the flop."""
    from tests.test_action_history import ScriptedAgent
    from game.poker import PokerGame
    from models.player import Player
    from agents.sixmax_agent import SixmaxAgent, _card_to_int

    sx = SixmaxAgent(_CKPT, config_toml=_TOML, search_toml=_TOML)
    captured = {}

    class CapturingAgent(ScriptedAgent):
        def get_action(self, player, game_state):
            # Reconstruct DURING the decision (production timing): the live
            # GameState is mutable and keeps changing after this call returns.
            if game_state.betting_round == 1 and "root_pot" not in captured:
                hole = [_card_to_int(c) for c in player.hole_cards]
                board = [_card_to_int(c) for c in game_state.community_cards]
                sx._hero = player
                root, prefix = sx._replay_to_street_root(game_state, hole, board)
                captured["live_pot_bb"] = game_state.pot / game_state.big_blind
                captured["root_pot"] = root.bet_context().pot
                captured["street"] = root.street()
                captured["prefix"] = list(prefix)
            return super().get_action(player, game_state)

    # HU, button_pos=0: p0 is button/SB (acts first preflop). Limp + check -> flop.
    # On the flop the BB (p1) acts first; reconstruct there, then fold to end.
    p0 = Player("A", 100, CapturingAgent([(Action.CALL, None), (Action.CHECK, None)]))
    p1 = Player("B", 100, CapturingAgent([(Action.CHECK, None), (Action.FOLD, None)]))
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()

    assert "root_pot" in captured, "never reached a flop decision"
    assert captured["street"] == 1                              # at the flop root
    assert captured["prefix"] == []                             # no flop actions yet
    assert abs(captured["root_pot"] - captured["live_pot_bb"]) < 1e-6  # exact
