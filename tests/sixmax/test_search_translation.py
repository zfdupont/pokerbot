"""Live (Action, amount) -> vocab-index translation used to replay the real
action line into the C++ search. Requires the gitignored HU checkpoint."""
import os

import pytest

from models.enums import Action

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "hu_blueprint.bin")
_TOML = os.path.join(_ROOT, "sixmax", "configs", "hu.toml")
pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT), reason="checkpoint gitignored")


def _agent():
    from agents.sixmax_agent import SixmaxAgent
    return SixmaxAgent(_CKPT, config_toml=_TOML, search_toml=_TOML)


def test_fold_check_call_map_to_fixed_indices():
    import sixmax
    a = _agent()
    cfg = sixmax.EngineConfig(num_players=2)
    st = sixmax.EngineGameState(cfg, 0, list(range(52)), a._deploy._vocab, [100.0, 100.0])
    assert a._action_to_vocab(st, Action.FOLD, None, 1.0) == 0
    assert a._action_to_vocab(st, Action.CHECK, None, 1.0) == 1
    assert a._action_to_vocab(st, Action.CALL, None, 1.0) == 2


def test_bet_maps_to_a_bet_vocab_index():
    import sixmax
    a = _agent()
    cfg = sixmax.EngineConfig(num_players=2)
    st = sixmax.EngineGameState(cfg, 0, list(range(52)), a._deploy._vocab, [100.0, 100.0])
    idx = a._action_to_vocab(st, Action.RAISE, 6, 1.0)   # raise-to 6bb
    assert str(a._deploy._vocab.at(idx).type) in ("ActionType.Bet", "ActionType.AllIn")
