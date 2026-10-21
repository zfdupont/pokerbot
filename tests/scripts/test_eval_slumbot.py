"""Tests for the Slumbot evaluator.

Everything here is offline: the HTTP transport is injected as a plain `post`
callable, and the agent is a stub that returns fixed actions. The one exception
is the `--summary` acceptance run, which reads the committed per-hand log.
"""
import json
import os

import pytest

from models.enums import Action

from scripts.eval_slumbot import (
    breakdown,
    build_state,
    login,
    main,
    parse_action,
    play_hand,
    run_hands,
    summarize,
    to_incr,
)


# ---------------------------------------------------------------------------
# Action-string parser (frozen to the spike)
# ---------------------------------------------------------------------------
def test_parse_preflop_unopened():
    assert parse_action("", 0) == ([0, 0], [100, 50], [0, 0, 0, 0])


def test_parse_flop_bet_after_check():
    assert parse_action("b200c/kb400", 1) == ([200, 200], [0, 400], [1, 1, 0, 0])


def test_parse_three_bet():
    assert parse_action("b300b900", 0) == ([0, 0], [900, 300], [2, 0, 0, 0])


def test_parse_new_street_and_check_through():
    assert parse_action("b200c/", 1) == ([200, 200], [0, 0], [1, 0, 0, 0])
    assert parse_action("b200c/kk/kk/", 3) == ([200, 200], [0, 0], [1, 0, 0, 0])


def test_parse_turn_reraise():
    assert parse_action("b200c/kk/b300b900", 2) == ([200, 200], [300, 900], [1, 0, 2, 0])


def test_parse_preflop_shove():
    assert parse_action("b20000", 0) == ([0, 0], [100, 20000], [1, 0, 0, 0])


# ---------------------------------------------------------------------------
# Inbound state construction
# ---------------------------------------------------------------------------
def test_build_state_bb_facing_open():
    gs, hero, facing = build_state({"action": "b250", "client_pos": 0,
                                    "hole_cards": ["Ac", "Kd"], "board": []})
    assert facing and gs.pot == 350 and gs.current_bet == 250
    assert gs.players[1] is hero and hero.current_bet == 100 and hero.stack == 19900
    assert gs.raises_per_street == [1, 0, 0, 0] and gs.button_pos == 0
    assert [str(c) for c in hero.hole_cards] == ["A\u2663", "K\u2666"]


# ---------------------------------------------------------------------------
# Outbound translation
# ---------------------------------------------------------------------------
def test_to_incr_maps_and_caps():
    hero = type("H", (), {"current_bet": 100, "stack": 19900})()
    assert to_incr(Action.FOLD, None, hero, facing=False) == "k"
    assert to_incr(Action.FOLD, None, hero, facing=True) == "f"
    assert to_incr(Action.CALL, None, hero, facing=True) == "c"
    assert to_incr(Action.CHECK, None, hero, facing=False) == "k"
    assert to_incr(Action.BET, 50000, hero, facing=True) == "b20000"


# ---------------------------------------------------------------------------
# One hand, with a fake transport
# ---------------------------------------------------------------------------
class FakePost:
    """Sequential fake of the module's `post`: pops one canned response per
    call and records every request body."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, ep, body):
        self.calls.append((ep, dict(body)))
        if not self.responses:
            raise AssertionError(f"unexpected request {ep} {body}")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class StubAgent:
    """Minimal stand-in for SixmaxAgent: fixed (Action, amount) replies."""

    def __init__(self, action=(Action.CHECK, None)):
        self._action = action
        self.calls = []

    def get_action(self, player, game_state):
        self.calls.append((player, game_state))
        return self._action


def _preflop_open(pos=0, token="t1"):
    return {"action": "b250", "client_pos": pos, "hole_cards": ["Ac", "Kd"],
            "board": [], "token": token}


def _terminal(winnings, pos=0, action="b250f", token="t2"):
    return {"token": token, "client_pos": pos, "action": action,
            "hole_cards": ["Ac", "Kd"], "board": [], "bot_hole_cards": ["7s", "2h"],
            "winnings": winnings, "baseline_winnings": winnings}


def test_play_hand_slumbot_folds_first():
    fake = FakePost([_terminal(50, action="f")])
    token, resp, acts = play_hand(fake, StubAgent(), None)
    assert token == "t2"
    assert resp["winnings"] == 50
    assert acts == []
    assert [ep for ep, _ in fake.calls] == ["new_hand"]


def test_play_hand_error_mid_hand_aborts():
    # Postflop, nobody facing a bet: our check is not a bet, so no fallback.
    state = {"action": "b200c/", "client_pos": 0, "hole_cards": ["Ac", "Kd"],
             "board": ["Ts", "7c", "2h"], "token": "t1"}
    fake = FakePost([state, {"error_msg": "server exploded"}])
    token, resp, acts = play_hand(fake, StubAgent((Action.CHECK, None)), None)
    assert token == "t1"
    assert resp is None
    assert acts == ["k"]
    assert len(fake.calls) == 2


def test_play_hand_transport_failure_aborts():
    fake = FakePost([RuntimeError("act failed 5 times")])
    token, resp, acts = play_hand(fake, StubAgent(), "t0")
    assert (token, resp, acts) == ("t0", None, [])


def test_agent_errors_are_not_swallowed():
    class ExplodingAgent:
        def get_action(self, player, game_state):
            raise RuntimeError("bug in the agent")

    with pytest.raises(RuntimeError, match="bug in the agent"):
        play_hand(FakePost([_preflop_open()]), ExplodingAgent(), None)


def test_bet_rejected_falls_back():
    fake = FakePost([_preflop_open(), {"error_msg": "bad size"},
                     _terminal(-250, action="b250c")])
    _, resp, acts = play_hand(fake, StubAgent((Action.RAISE, 800)), None)
    assert resp["winnings"] == -250
    assert acts == ["b800", "fallback:c"]
    assert fake.calls[1] == ("act", {"token": "t1", "incr": "b800"})
    assert fake.calls[2] == ("act", {"token": "t1", "incr": "c"})


# ---------------------------------------------------------------------------
# The CLI loop
# ---------------------------------------------------------------------------
def test_run_hands_writes_no_row_for_aborted_hand(tmp_path):
    log = tmp_path / "log.jsonl"
    # Hand 1 aborts mid-hand, hand 2 completes: one row, numbered 1.
    fake = FakePost([
        _preflop_open(),
        {"error_msg": "x"},
        _terminal(100, token="t9"),
    ])
    token = run_hands(fake, StubAgent((Action.RAISE, 800)), None, str(log), target=1)
    assert token == "t9"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["hand"] == 1 and rows[0]["winnings"] == 100


def test_run_hands_stops_after_five_consecutive_aborts(tmp_path):
    log = tmp_path / "log.jsonl"
    fake = FakePost([RuntimeError("down")] * 5)
    with pytest.raises(SystemExit) as exc:
        run_hands(fake, StubAgent(), None, str(log), target=2)
    assert "abort" in str(exc.value).lower()
    assert not log.exists() or log.read_text() == ""


# ---------------------------------------------------------------------------
# CLI guard rails: even hand counts
# ---------------------------------------------------------------------------
def test_odd_hands_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--hands", "3", "--log", str(tmp_path / "log.jsonl")])
    assert "even" in str(exc.value)


def test_resume_odd_log_refused(tmp_path):
    log = tmp_path / "log.jsonl"
    row = {"hand": 1, "pos": 0, "action": "f", "ours": [], "hole": [],
           "board": [], "bot_hole": [], "winnings": 100, "baseline": 100}
    log.write_text("".join(json.dumps(row) + "\n" for _ in range(3)))
    with pytest.raises(SystemExit) as exc:
        main(["--hands", "6", "--log", str(log)])
    assert "even" in str(exc.value)


# ---------------------------------------------------------------------------
# Login is opt-in and env-only
# ---------------------------------------------------------------------------
def test_login_skipped_without_env(monkeypatch):
    monkeypatch.delenv("SLUMBOT_USERNAME", raising=False)
    monkeypatch.delenv("SLUMBOT_PASSWORD", raising=False)
    called = []
    assert login(lambda ep, body: called.append(ep) or {}) is None
    assert called == []


def test_login_uses_env_and_returns_token(monkeypatch):
    monkeypatch.setenv("SLUMBOT_USERNAME", "u")
    monkeypatch.setenv("SLUMBOT_PASSWORD", "p")
    seen = {}

    def fetch(ep, body):
        seen[ep] = body
        return {"token": "secret"}

    assert login(fetch) == "secret"
    assert seen == {"login": {"username": "u", "password": "p"}}


# ---------------------------------------------------------------------------
# Summary + breakdown math
# ---------------------------------------------------------------------------
def test_summary_missing_log_is_an_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--summary", "--log", str(tmp_path / "absent.jsonl")])
    assert "no log" in str(exc.value)


def test_summarize_known_rows():
    rows = [{"pos": 0, "winnings": 100, "baseline": 200},
            {"pos": 1, "winnings": -300, "baseline": -200}]
    s = summarize(rows)
    assert s["hands"] == 2
    assert s["raw_bb100"] == pytest.approx(-100.0)
    assert s["baseline_bb100"] == pytest.approx(0.0)
    assert s["baseline_se"] > 0.0
    assert s["bb_baseline_bb100"] == pytest.approx(200.0)
    assert s["sb_baseline_bb100"] == pytest.approx(-200.0)


def test_summarize_empty_log():
    s = summarize([])
    assert s["hands"] == 0 and s["raw_bb100"] == 0.0 and s["raw_se"] == 0.0


def test_breakdown_bins():
    rows = [
        {"pos": 0, "winnings": 15000, "baseline": -4000, "action": "b300b900c///",
         "board": ["Ts", "7c", "2h", "4d", "9s"]},
        {"pos": 1, "winnings": -200, "baseline": 100, "action": "b200f", "board": []},
        {"pos": 0, "winnings": 2500, "baseline": 300, "action": "b200c/kb400f",
         "board": ["Ts", "7c", "2h"]},
    ]
    b = breakdown(rows)
    assert b["size_bins"]["100+"]["n"] == 1
    assert b["size_bins"]["100+"]["baseline_bb"] == pytest.approx(-40.0)
    assert b["size_bins"]["10-40"]["n"] == 1
    assert b["size_bins"]["<10"]["n"] == 1
    assert b["threebet_pots"]["n"] == 1
    assert b["showdown"]["n"] == 1  # river reached with no fold
    assert b["fold"]["n"] == 2


# ---------------------------------------------------------------------------
# Acceptance: the committed 3,000-hand baseline log
# ---------------------------------------------------------------------------
BEFORE_LOG = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "superpowers", "specs",
    "2026-10-21-hu-200bb-abstraction-roadmap", "slumbot_hu50m_3000.jsonl")


@pytest.mark.skipif(not os.path.exists(BEFORE_LOG), reason="baseline log absent")
def test_summary_on_committed_baseline_log(capsys):
    main(["--summary", "--log", os.path.normpath(BEFORE_LOG)])
    out = capsys.readouterr().out
    assert "baseline=-118.3" in out
    hundred = [ln for ln in out.splitlines() if ln.strip().startswith("100+")]
    assert hundred and "n=   34" in hundred[0]
