"""FastAPI surface for the poker web service."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.enums import Action

from web.config import Config
from web.game_runner import IllegalAction, advance
from web.session import SessionStore


class _ActionBody(BaseModel):
    seq: int
    action: str
    amount: int | None = None


_CLIENT_ACTIONS = {"fold", "check", "call", "bet", "all_in"}


def _to_engine(state, body: _ActionBody):
    hero = state["hero"]
    if body.action == "fold":
        return Action.FOLD, None
    if body.action == "check":
        return Action.CHECK, None
    if body.action == "call":
        return Action.CALL, None
    if body.action == "bet":
        if body.amount is None:
            raise HTTPException(status_code=422, detail="bet requires amount")
        return Action.BET, body.amount
    if body.action == "all_in":
        return Action.BET, hero["current_bet"] + hero["stack"]
    raise HTTPException(status_code=422, detail="unknown action")


def build_app(config: Config, bot_factory) -> FastAPI:
    app = FastAPI(title="pokerbot-web")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = SessionStore(ttl_s=config.session_ttl_s, bot_factory=bot_factory)

    def _new_session():
        s = store.create()
        s.small_blind = config.small_blind
        s.start_stack = config.stack
        s.hero_stack = config.stack
        s.bot_stack = config.stack
        s.start_hand()
        return s

    def _get(token: str):
        s = store.get(token)
        if s is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return s

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/session")
    def create_session():
        s = _new_session()
        events, state = advance(s)
        return {"token": s.token, "events": events, "state": state}

    @app.get("/api/session/{token}")
    def get_session(token: str):
        s = _get(token)
        if s.last_state is None:          # freshly created, not yet advanced
            events, state = advance(s)
            return {"token": s.token, "events": events, "state": state}
        return {"token": s.token, "events": s.last_events, "state": s.last_state}

    @app.post("/api/session/{token}/action")
    def do_action(token: str, body: _ActionBody):
        s = _get(token)
        if body.action not in _CLIENT_ACTIONS:
            raise HTTPException(status_code=422, detail="unknown action")
        if not s.hand_active or s.last_state is None:
            raise HTTPException(status_code=409, detail="no active hand")
        current = s.last_state
        if body.seq != current["seq"]:
            raise HTTPException(status_code=409, detail="stale action")
        if not current["hero"]["is_actor"]:
            raise HTTPException(status_code=409, detail="not your turn")
        engine_action = _to_engine(current, body)
        try:
            events, state = advance(s, engine_action)
        except IllegalAction as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"events": events, "state": state}

    @app.post("/api/session/{token}/next-hand")
    def next_hand(token: str):
        s = _get(token)
        if s.hand_active:
            raise HTTPException(status_code=409, detail="hand in progress")
        s.start_hand()
        events, state = advance(s)
        return {"events": events, "state": state}

    @app.post("/api/session/{token}/rebuy")
    def rebuy(token: str):
        s = _get(token)
        if s.hand_active:
            raise HTTPException(status_code=409, detail="hand in progress")
        if s.hero_stack <= 0:
            s.hero_stack = config.stack
        if s.bot_stack <= 0:
            s.bot_stack = config.stack
        s.start_hand()
        events, state = advance(s)
        return {"events": events, "state": state}

    @app.delete("/api/session/{token}")
    def cash_out(token: str):
        s = _get(token)
        net = s.net
        store.delete(token)
        return {"cashed_out": True, "net": net}

    return app
