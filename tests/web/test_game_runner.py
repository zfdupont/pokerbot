import pytest

from models.enums import Action
from web.config import load_config
from web.game_runner import advance, legal_actions
from web.session import SessionStore


class _FoldBot:
    """Folds when facing a bet, else checks."""

    def get_action(self, player, state):
        if state.current_bet - player.current_bet > 0:
            return Action.FOLD, None
        return Action.CHECK, None


def _store():
    return SessionStore(ttl_s=100, bot_factory=_FoldBot)


def _new_session(store):
    cfg = load_config(env={})
    s = store.create()
    s.small_blind = cfg.small_blind
    s.start_stack = cfg.stack
    s.hero_stack = cfg.stack
    s.bot_stack = cfg.stack
    s.start_hand(seed=1)
    return s


def test_initial_advance_pauses_at_first_decision_without_action():
    s = _new_session(_store())
    events, state = advance(s)
    assert state["hand_complete"] is False
    assert len(state["hero"]["hole_cards"]) == 2
    assert state["legal_actions"]          # hero is the SB and to act
    assert state["street"] == "preflop"
    assert state["seq"] == 0


def test_bot_hole_cards_never_leaked():
    s = _new_session(_store())
    events, state = advance(s)
    assert set(state["bot"].keys()) == {"stack", "current_bet"}
    assert "hole_cards" not in state["bot"]
    assert "showdown" not in {e["type"] for e in events}


def test_fold_ends_hand_and_conserves_chips():
    s = _new_session(_store())
    state = advance(s)[1]
    for _ in range(20):
        if state["hand_complete"]:
            break
        acts = {a["action"] for a in state["legal_actions"]}
        pick = Action.CHECK if "check" in acts else Action.FOLD
        state = advance(s, (pick, None))[1]
    assert state["hand_complete"] is True
    assert s.hero_stack + s.bot_stack == 2 * s.start_stack
    assert s.hands_played == 1


def test_legal_actions_facing_bet_has_fold_call_raise():
    from types import SimpleNamespace
    state = SimpleNamespace(current_bet=10, big_blind=2)
    player = SimpleNamespace(current_bet=0, stack=100)
    acts = {a["action"] for a in legal_actions(state, player)}
    assert {"fold", "call", "bet", "all_in"} <= acts


def test_legal_actions_unopened_has_check_not_fold():
    from types import SimpleNamespace
    state = SimpleNamespace(current_bet=0, big_blind=2)
    player = SimpleNamespace(current_bet=0, stack=100)
    acts = {a["action"] for a in legal_actions(state, player)}
    assert "check" in acts and "fold" not in acts


def test_validate_rejects_check_facing_bet():
    from types import SimpleNamespace
    from web.game_runner import IllegalAction, _validate
    state = SimpleNamespace(current_bet=10, big_blind=2)
    player = SimpleNamespace(current_bet=0, stack=100)
    with pytest.raises(IllegalAction):
        _validate(state, player, Action.CHECK, None)


def test_validate_rejects_below_min_raise():
    from types import SimpleNamespace
    from web.game_runner import IllegalAction, _validate
    state = SimpleNamespace(current_bet=10, big_blind=2)
    player = SimpleNamespace(current_bet=0, stack=100)
    with pytest.raises(IllegalAction):
        _validate(state, player, Action.BET, 5)   # min is 10 + max(10, 2) = 20


class _RaiseBot:
    """Raises once when it acts, so the hero is asked twice in one request."""

    def __init__(self):
        self._raised = False

    def get_action(self, player, state):
        to_call = state.current_bet - player.current_bet
        if not self._raised:
            self._raised = True
            return Action.BET, state.current_bet + 6
        return (Action.CALL, None) if to_call > 0 else (Action.CHECK, None)


def test_single_action_applied_once_then_pauses():
    # Regression: a re-raising bot must send the hero back to a FRESH decision,
    # not re-apply the previous action. History should hold exactly the hero's
    # call and the bot's raise.
    store = SessionStore(ttl_s=100, bot_factory=_RaiseBot)
    s = _new_session(store)
    state = advance(s)[1]                       # hero (SB) to act preflop
    assert state["hero"]["is_actor"]
    state = advance(s, (Action.CALL, None))[1]
    assert len(s.history) == 2                  # hero:call, bot:raise
    assert state["hero"]["is_actor"] is True
    assert state["current_bet"] == 8


def test_reload_returns_same_pause_state():
    store = _store()
    s = _new_session(store)
    _, state = advance(s)
    assert store.get(s.token) is s
    _, state2 = advance(s)
    assert state2["current_bet"] == state["current_bet"]
    assert state2["street"] == state["street"]
    assert state2["seq"] == state["seq"]
