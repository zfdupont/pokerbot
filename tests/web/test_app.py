from fastapi.testclient import TestClient

from models.enums import Action
from web.app import build_app
from web.config import load_config


class _FoldBot:
    def get_action(self, player, state):
        if state.current_bet - player.current_bet > 0:
            return Action.FOLD, None
        return Action.CHECK, None


def _client():
    return TestClient(build_app(load_config(env={}), _FoldBot))


def test_session_lifecycle_and_health():
    c = _client()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/api/session")
    assert r.status_code == 200
    body = r.json()
    token = body["token"]
    assert len(body["state"]["hero"]["hole_cards"]) == 2
    assert "hole_cards" not in body["state"]["bot"]

    r2 = c.get(f"/api/session/{token}")
    assert r2.status_code == 200
    assert r2.json()["state"]["street"] == body["state"]["street"]


def test_unknown_token_404():
    c = _client()
    assert c.get("/api/session/nope").status_code == 404
    assert c.post("/api/session/nope/action",
                  json={"seq": 0, "action": "check"}).status_code == 404


def test_stale_seq_rejected():
    c = _client()
    body = c.post("/api/session").json()
    token, state = body["token"], body["state"]
    if not state["hero"]["is_actor"]:
        return
    r = c.post(f"/api/session/{token}/action",
               json={"seq": 999, "action": "check"})
    assert r.status_code == 409


def test_illegal_action_rejected_and_hand_unchanged():
    c = _client()
    body = c.post("/api/session").json()
    token, state = body["token"], body["state"]
    if not state["hero"]["is_actor"]:
        return
    to_call = state["current_bet"] - state["hero"]["current_bet"]
    if to_call <= 0:
        return
    r = c.post(f"/api/session/{token}/action",
               json={"seq": state["seq"], "action": "check"})
    assert r.status_code == 409
    state2 = c.get(f"/api/session/{token}").json()["state"]
    assert state2["seq"] == state["seq"]


def test_cors_header_present():
    c = _client()
    r = c.options("/api/session", headers={
        "Origin": "https://zfdupont.com",
        "Access-Control-Request-Method": "POST",
    })
    assert r.headers.get("access-control-allow-origin") == "https://zfdupont.com"
