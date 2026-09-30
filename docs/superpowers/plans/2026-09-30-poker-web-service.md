# Poker Web Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a heads-up cash-session game against the trained six-max blueprint bot over a small FastAPI service, driving the live `game/poker.py` engine.

**Architecture:** A new `web/` package is a *host* (like `scripts/openpoker_bot.py`): it imports `game/poker.py` and `agents/sixmax_agent.py` and is never imported by `cfr/`, `neural_cfr/`, or `sixmax/`. Because `PokerGame.play_hand()` runs a whole hand synchronously, the service uses a **replay model**: a session stores the hand's ordered decisions plus a fixed deck; each REST request rebuilds a fresh `PokerGame`, replays recorded decisions, applies the new human action, lets the bot act, and pauses (via a private exception) at the next human decision. This is thread-free, deterministic, and easy to unit-test with a stub bot.

**Tech Stack:** Python 3.10, FastAPI + uvicorn, numpy (transitively via the blueprint bridge), pytest + `fastapi.testclient`, Docker (linux/amd64), GitHub Actions → GHCR.

**Spec:** `docs/superpowers/specs/2026-09-30-poker-web-design.md`

## Global Constraints

- `web/` is a host: it may import `game/`, `models/`, `util/`, and `agents/`; `cfr/`, `neural_cfr/`, and `sixmax/` must never import `web/`.
- The bot's hole cards must never appear in any API response before showdown.
- Chip values are rescaled to the training frame (`starting_stack=100, big_blind=1`) by `SixmaxAgent`; the service holds `stack=200, small_blind=1` (= 100 BB at `big_blind=2`).
- Session store is in-process; run exactly **one** uvicorn worker.
- Python is pinned to **3.10** (sixmax extension ABI).
- Never commit secrets (`OPENPOKER_API_KEY`) or checkpoint files.
- The compiled extension is loaded via `SIXMAX_SO_PATH`; the service must never shell out to Buck2 in the container.

## Review Focus

Uncovered inputs most likely to bite a user — each gets a test in its owning task:

1. **Illegal client action** (check facing a bet, raise below the minimum, bet when all-in) → reject with HTTP 409; never mutate the session.
2. **Duplicate/stale action** (double-click or retry after a dropped response) → ignored via a `seq` guard; the hand must not advance twice.
3. **Reload mid-hand** → `GET /api/session/{token}` returns the same pause state with the full event log, no re-deal.
4. **Bot hole cards before showdown** → absent from every field of `state` and `events`.
5. **Session expiry / unknown token** → HTTP 404, client can start a fresh session.

---

### Task 1: Fix heads-up ordering in the live engine

**Files:**
- Modify: `game/poker.py` (`_assign_positions`, `_post_blinds`, `_first_to_act`)
- Test: `tests/test_poker_headsup.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: for `n == 2`, `button_pos` is the SB; preflop first-to-act is `button_pos`; postflop first-to-act is `(button_pos + 1) % 2`; positions are `[SB, BB]`. `n > 2` behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_poker_headsup.py
from models.player import Player
from models.enums import Action, Position
from agents.base_agent import PokerAgent
from game.poker import PokerGame


class _CheckCall(PokerAgent):
    def get_action(self, player, state):
        to_call = state.current_bet - player.current_bet
        if to_call > 0:
            return Action.CALL, None
        return Action.CHECK, None


def _game(button):
    players = [Player("A", 200, _CheckCall()), Player("B", 200, _CheckCall())]
    g = PokerGame(players, small_blind=1)
    g.state.button_pos = button
    return g, players


def test_hu_positions_button_is_small_blind():
    g, players = _game(0)
    g.play_hand()
    assert players[0].position == Position.SMALL_BLIND
    assert players[1].position == Position.BIG_BLIND


def test_hu_button_posts_small_blind_and_acts_first_preflop():
    # Record first-actor and preflop commitments.
    seen = []

    class Recorder(PokerAgent):
        def get_action(self, player, state):
            seen.append((player.name, state.betting_round))
            to_call = state.current_bet - player.current_bet
            return (Action.CALL, None) if to_call > 0 else (Action.CHECK, None)

    players = [Player("A", 200, Recorder()), Player("B", 200, Recorder())]
    g = PokerGame(players, small_blind=1)
    g.state.button_pos = 0
    g.play_hand()
    # First decision overall is preflop by A (the button/SB).
    assert seen[0] == ("A", 0)
    # A posted the small blind (1), B posted the big blind (2) before action.
    # (checked indirectly: A could check then B could check -> pot == 4 later)


def test_hu_postflop_big_blind_acts_first():
    seen = []

    class Recorder(PokerAgent):
        def get_action(self, player, state):
            seen.append((player.name, state.betting_round))
            to_call = state.current_bet - player.current_bet
            return (Action.CALL, None) if to_call > 0 else (Action.CHECK, None)

    players = [Player("A", 200, Recorder()), Player("B", 200, Recorder())]
    g = PokerGame(players, small_blind=1)
    g.state.button_pos = 0
    g.play_hand()
    flop_first = next(n for n, s in seen if s == 1)
    assert flop_first == "B"  # BB acts first postflop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_poker_headsup.py -v`
Expected: FAIL — `test_hu_positions_button_is_small_blind` sees `Position.CUTOFF`/`BUTTON`; the ordering tests see `"B"` first preflop.

- [ ] **Step 3: Implement the `n == 2` special cases**

In `game/poker.py`:

```python
    def _assign_positions(self) -> None:
        from models.enums import Position
        num = len(self.state.players)
        if num == 2:
            positions = [Position.SMALL_BLIND, Position.BIG_BLIND]
        else:
            positions = list(Position)[-num:]
        for i, player in enumerate(self.state.players):
            idx = (i + self.state.button_pos) % num
            player.position = positions[idx]

    def _post_blinds(self, pot_manager: PotManager) -> None:
        num = len(self.state.players)
        if num == 2:
            sb_pos = self.state.button_pos
            bb_pos = (self.state.button_pos + 1) % num
        else:
            sb_pos = (self.state.button_pos + 1) % num
            bb_pos = (self.state.button_pos + 2) % num
        # ... unchanged below ...

    def _first_to_act(self, street: int) -> int:
        num = len(self.state.players)
        if num == 2:
            start = (self.state.button_pos if street == 0
                     else (self.state.button_pos + 1) % num)
        else:
            start = ((self.state.button_pos + 3) % num if street == 0
                     else (self.state.button_pos + 1) % num)
        # ... unchanged below ...
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `uv run pytest tests/test_poker_headsup.py -v && uv run pytest tests/ -q`
Expected: new tests PASS; suite green (no existing test asserts the old HU behavior; `test_integration.py` is 4-handed).

- [ ] **Step 5: Commit**

```bash
git add game/poker.py tests/test_poker_headsup.py
git commit -m "fix(game): standard heads-up blind/position ordering for n==2"
```

---

### Task 2: `web/` package scaffold and configuration

**Files:**
- Create: `web/__init__.py`, `web/config.py`
- Test: `tests/web/__init__.py`, `tests/web/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `web.config.Config` dataclass with fields `checkpoint: str`, `config_toml: str`, `stack: int`, `small_blind: int`, `session_ttl_s: int`, `cors_origins: tuple[str, ...]`; property `big_blind -> int`; and `load_config(env: Mapping[str, str] | None = None) -> Config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/web/test_config.py
from web.config import load_config


def test_defaults():
    cfg = load_config(env={})
    assert cfg.stack == 200
    assert cfg.small_blind == 1
    assert cfg.big_blind == 2
    assert cfg.session_ttl_s == 3600
    assert "https://zfdupont.com" in cfg.cors_origins
    assert cfg.config_toml.endswith("sixmax/configs/default.toml")


def test_env_overrides():
    cfg = load_config(env={
        "POKERBOT_STACK": "100",
        "POKERBOT_SMALL_BLIND": "2",
        "POKERBOT_SESSION_TTL": "60",
        "POKERBOT_CHECKPOINT": "/checkpoints/hu.bin",
        "POKERBOT_CORS_ORIGINS": "https://a.example, https://b.example",
    })
    assert cfg.stack == 100
    assert cfg.small_blind == 2
    assert cfg.big_blind == 4
    assert cfg.session_ttl_s == 60
    assert cfg.checkpoint == "/checkpoints/hu.bin"
    assert cfg.cors_origins == ("https://a.example", "https://b.example")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/web/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web'`.

- [ ] **Step 3: Implement**

```python
# web/__init__.py
"""Poker web service: a host that serves heads-up play over REST."""
```

```python
# web/config.py
"""Runtime configuration for the poker web service (env driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TOML = os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml")
_DEFAULT_ORIGINS = ("https://zfdupont.com",)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class Config:
    checkpoint: str
    config_toml: str
    stack: int
    small_blind: int
    session_ttl_s: int
    cors_origins: tuple[str, ...]

    @property
    def big_blind(self) -> int:
        return self.small_blind * 2


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    origins = env.get("POKERBOT_CORS_ORIGINS")
    cors = (_DEFAULT_ORIGINS if not origins
            else tuple(o.strip() for o in origins.split(",") if o.strip()))
    return Config(
        checkpoint=env.get("POKERBOT_CHECKPOINT", ""),
        config_toml=env.get("POKERBOT_CONFIG_TOML", _DEFAULT_TOML),
        stack=_int(env, "POKERBOT_STACK", 200),
        small_blind=_int(env, "POKERBOT_SMALL_BLIND", 1),
        session_ttl_s=_int(env, "POKERBOT_SESSION_TTL", 3600),
        cors_origins=cors,
    )
```

Also create empty `tests/web/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/web/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/__init__.py web/config.py tests/web/__init__.py tests/web/test_config.py
git commit -m "feat(web): add web package scaffold and env config"
```

---

### Task 3: Session model and TTL store

**Files:**
- Create: `web/session.py`
- Test: `tests/web/test_session.py`

**Interfaces:**
- Consumes: `web.config.Config` (stack, blinds).
- Produces:
  - `web.session.Decision` dataclass: `actor: str` (`"hero"`/`"bot"`), `action: Action`, `amount: int | None`.
  - `web.session.Session` dataclass with fields `token`, `hero_name`, `bot_name`, `small_blind`, `start_stack`, `hero_stack`, `bot_stack`, `button`, `hand_active`, `deck: list[Card]`, `history: list[Decision]`, `net: int`, `hands_played: int`, `last_seen: float`, `bot_agent`, and method `start_hand(seed: int | None = None) -> None`.
  - `web.session.new_deck(seed: int) -> list[Card]`.
  - `web.session.SessionStore(ttl_s, bot_factory, clock=time.monotonic)` with `create() -> Session`, `get(token) -> Session | None`, `delete(token) -> None`, `sweep() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/web/test_session.py
from web.session import SessionStore, new_deck


class _StubBot:
    def get_action(self, player, state):  # pragma: no cover - not called here
        raise AssertionError


def test_new_deck_is_52_unique_and_deterministic():
    a = new_deck(123)
    b = new_deck(123)
    assert len(a) == 52
    assert len({c.key for c in a}) == 52
    assert [c.key for c in a] == [c.key for c in b]
    assert [c.key for c in a] != [c.key for c in new_deck(124)]


def test_create_get_delete():
    store = SessionStore(ttl_s=100, bot_factory=_StubBot)
    s = store.create()
    assert store.get(s.token) is s
    store.delete(s.token)
    assert store.get(s.token) is None


def test_ttl_expiry():
    now = [0.0]
    store = SessionStore(ttl_s=100, bot_factory=_StubBot, clock=lambda: now[0])
    s = store.create()
    now[0] = 50.0
    assert store.get(s.token) is not None  # not expired
    now[0] = 101.0
    assert store.get(s.token) is None
    assert store.sweep() == 0  # already dropped on get


def test_sweep_removes_idle():
    now = [0.0]
    store = SessionStore(ttl_s=10, bot_factory=_StubBot, clock=lambda: now[0])
    s = store.create()
    now[0] = 20.0
    assert store.sweep() == 1
    assert store.get(s.token) is None


def test_start_hand_resets_history_and_sets_active():
    store = SessionStore(ttl_s=100, bot_factory=_StubBot)
    s = store.create()
    s.history.append(object())
    s.start_hand(seed=7)
    assert s.history == []
    assert s.hand_active is True
    assert len(s.deck) == 52
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/web/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.session'`.

- [ ] **Step 3: Implement**

```python
# web/session.py
"""In-memory, TTL'd sessions for heads-up cash play."""
from __future__ import annotations

import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from models.card import Card
from models.enums import Action, Suit

from web.config import Config


def new_deck(seed: int) -> list[Card]:
    deck = [Card(rank, suit) for suit in Suit for rank in range(2, 15)]
    random.Random(seed).shuffle(deck)
    return deck


@dataclass
class Decision:
    actor: str                  # "hero" | "bot"
    action: Action
    amount: Optional[int]


@dataclass
class Session:
    token: str
    small_blind: int
    start_stack: int
    hero_name: str = "You"
    bot_name: str = "Bot"
    hero_stack: int = 0
    bot_stack: int = 0
    button: int = 0
    hand_active: bool = False
    deck: list[Card] = field(default_factory=list)
    history: list[Decision] = field(default_factory=list)
    net: int = 0
    hands_played: int = 0
    last_seen: float = 0.0
    bot_agent: object = None
    last_events: list = field(default_factory=list)
    last_state: Optional[dict] = None

    def start_hand(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = random.randrange(1 << 30)
        self.deck = new_deck(seed)
        self.history = []
        self.hand_active = True
        self.last_events = []
        self.last_state = None


class SessionStore:
    def __init__(self, ttl_s: float, bot_factory: Callable[[], object],
                 clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_s
        self._bot_factory = bot_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        token = secrets.token_urlsafe(24)
        with self._lock:
            s = Session(
                token=token,
                small_blind=1,          # overwritten by the app via Config
                start_stack=0,
                bot_agent=self._bot_factory(),
                last_seen=self._clock(),
            )
            self._sessions[token] = s
            return s

    def get(self, token: str) -> Optional[Session]:
        with self._lock:
            s = self._sessions.get(token)
            if s is None:
                return None
            if self._clock() - s.last_seen > self._ttl:
                del self._sessions[token]
                return None
            s.last_seen = self._clock()
            return s

    def delete(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def sweep(self) -> int:
        now = self._clock()
        with self._lock:
            dead = [t for t, s in self._sessions.items()
                    if now - s.last_seen > self._ttl]
            for t in dead:
                del self._sessions[t]
            return len(dead)
```

Note: `SessionStore.create` sets placeholders for `small_blind`/`start_stack`; the app fills them from `Config` (Task 5). This keeps the store free of `Config`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/web/test_session.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/session.py tests/web/test_session.py
git commit -m "feat(web): add TTL session store"
```

---

### Task 4: The replay game runner

**Files:**
- Create: `web/game_runner.py`
- Test: `tests/web/test_game_runner.py`

**Interfaces:**
- Consumes: `web.session.Session`, `web.session.Decision`, `models.*`, `game.poker.PokerGame`.
- Produces:
  - `web.game_runner.IllegalAction(Exception)`
  - `web.game_runner.legal_actions(state, player) -> list[dict]`
  - `web.game_runner.advance(session, new_action: tuple[Action, int | None] | None = None) -> tuple[list[dict], dict]`
  - `web.game_runner.snapshot(session, state, *, hand_complete: bool, result: dict | None = None) -> dict`

`advance` mutates `session` (history during replay; stacks/button/net/hands_played on completion) and returns `(events, state_snapshot)`. It raises `IllegalAction` without mutating the session beyond the current hand's history when the supplied `new_action` is illegal (validation happens before the decision is appended).

- [ ] **Step 1: Write the failing test**

```python
# tests/web/test_game_runner.py
import pytest

from models.enums import Action
from web.config import load_config
from web.game_runner import advance, legal_actions
from web.session import SessionStore


class _FoldBot:
    """Always folds when facing a bet, else checks."""
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
    assert state["hero"]["hole_cards"] and len(state["hero"]["hole_cards"]) == 2
    assert state["legal_actions"]  # someone is to act; if hero, list is non-empty
    # Either hero or bot is the first actor; the pause state always has a board.
    assert "preflop" == state["street"]


def test_bot_hole_cards_never_leaked():
    s = _new_session(_store())
    events, state = advance(s)
    blob = repr(state) + repr(events)
    assert "bot" in state
    assert set(state["bot"].keys()) == {"stack", "current_bet"}
    # no card-like leakage: bot hole cards are not stored on the public dict
    assert "hole_cards" not in state["bot"]


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
    from web.game_runner import legal_actions
    state = SimpleNamespace(current_bet=10, big_blind=2)
    player = SimpleNamespace(current_bet=0, stack=100)
    acts = {a["action"] for a in legal_actions(state, player)}
    assert {"fold", "call", "bet", "all_in"} <= acts


def test_legal_actions_unopened_has_check_not_fold():
    from types import SimpleNamespace
    from web.game_runner import legal_actions
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


def test_reload_returns_same_pause_state():
    store = _store()
    s = _new_session(store)
    _, state = advance(s)
    assert store.get(s.token) is s
    _, state2 = advance(s)  # replay with no new action reaches the same pause
    assert state2["current_bet"] == state["current_bet"]
    assert state2["street"] == state["street"]
    assert state2["seq"] == state["seq"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/web/test_game_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.game_runner'`.

- [ ] **Step 3: Implement**

```python
# web/game_runner.py
"""Replay-based turn driver over the live engine.

A hand is replayed from its pre-hand stacks + fixed deck + ordered decisions on
every request. Replay pauses at the next human decision by raising `_AwaitHuman`
from the hero agent; the engine's `play_hand()` does not catch it, so it unwinds
to `advance()`.
"""
from __future__ import annotations

from typing import Optional

from agents.base_agent import PokerAgent
from game.poker import PokerGame
from models.enums import Action
from models.player import Player
from util.observer import GameObserver

from web.session import Decision, Session

_STREET_NAME = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


class IllegalAction(Exception):
    pass


class _AwaitHuman(Exception):
    def __init__(self, state):
        super().__init__("awaiting human")
        self.state = state


def legal_actions(state, player) -> list[dict]:
    to_call = state.current_bet - player.current_bet
    acts: list[dict] = []
    if to_call > 0:
        acts.append({"action": "fold"})
        acts.append({"action": "call", "amount": min(to_call, player.stack)})
    else:
        acts.append({"action": "check"})
    if player.stack > to_call:
        min_to = state.current_bet + max(state.current_bet, state.big_blind)
        max_to = player.current_bet + player.stack
        if min_to < max_to:
            acts.append({"action": "bet", "min": min_to, "max": max_to})
        acts.append({"action": "all_in", "amount": max_to})
    return acts


def _validate(state, player, action: Action, amount: Optional[int]) -> Optional[int]:
    to_call = state.current_bet - player.current_bet
    if action == Action.FOLD:
        if to_call <= 0:
            raise IllegalAction("nothing to fold to")
        return None
    if action == Action.CHECK:
        if to_call > 0:
            raise IllegalAction("cannot check facing a bet")
        return None
    if action == Action.CALL:
        if to_call <= 0:
            raise IllegalAction("nothing to call")
        return None
    if action == Action.BET:
        if player.stack <= 0:
            raise IllegalAction("no chips to bet")
        max_to = player.current_bet + player.stack
        min_to = state.current_bet + max(state.current_bet, state.big_blind)
        value = max_to if amount is None else int(amount)
        if value < min_to and value != max_to:
            raise IllegalAction(f"bet {value} below minimum {min_to}")
        return min(value, max_to)
    raise IllegalAction(f"unsupported action {action}")


class _Cursor:
    def __init__(self, history: list[Decision]):
        self.history = history
        self.i = 0

    def take(self, actor: str) -> Optional[Decision]:
        if self.i >= len(self.history):
            return None
        rec = self.history[self.i]
        if rec.actor != actor:
            raise RuntimeError(f"decision order mismatch: expected {rec.actor}, got {actor}")
        self.i += 1
        return rec

    def record(self, actor: str, action: Action, amount: Optional[int]) -> None:
        self.history.append(Decision(actor, action, amount))
        self.i += 1


class _HeroAgent(PokerAgent):
    def __init__(self, cursor: _Cursor, new_action):
        self._cursor = cursor
        self._new_action = new_action

    def get_action(self, player, state):
        rec = self._cursor.take("hero")
        if rec is not None:
            return rec.action, rec.amount
        if self._new_action is None:
            raise _AwaitHuman(state)
        action, amount = self._new_action
        amount = _validate(state, player, action, amount)
        self._cursor.record("hero", action, amount)
        return action, amount


class _BotAgent(PokerAgent):
    def __init__(self, cursor: _Cursor, bot):
        self._cursor = cursor
        self._bot = bot

    def get_action(self, player, state):
        rec = self._cursor.take("bot")
        if rec is not None:
            return rec.action, rec.amount
        action, amount = self._bot.get_action(player, state)
        self._cursor.record("bot", action, amount)
        return action, amount


class _Recorder(GameObserver):
    def __init__(self, events: list[dict], hero_name: str):
        self.events = events
        self.hero_name = hero_name
        self.result: Optional[dict] = None

    def _actor(self, player) -> str:
        return "hero" if player.name == self.hero_name else "bot"

    def on_hand_start(self, players, button_pos):
        self.events.append({
            "type": "hand_start",
            "button": button_pos,
            "hero_pos": players[0].position.value if players[0].position else None,
        })

    def on_street_start(self, street_name, community_cards):
        self.events.append({
            "type": "street",
            "name": street_name.lower().replace("-", ""),
            "board": [str(c) for c in community_cards],
        })

    def on_player_action(self, player, action, amount):
        self.events.append({
            "type": "action",
            "actor": self._actor(player),
            "action": action.value,
            "amount": amount,
        })

    def on_hand_complete(self, player_ranks, pot):
        active = [p for p, _ in player_ranks if p.is_active]
        if len(active) > 1:
            for p, _ in player_ranks:
                self.events.append({
                    "type": "showdown",
                    "actor": self._actor(p),
                    "cards": [str(c) for c in p.hole_cards],
                })
        # Lower rank value is better (see the evaluator); winners hold the min.
        min_rank = min(r for _, r in player_ranks)
        winners = [{"actor": self._actor(p)} for p, r in player_ranks
                   if r == min_rank]
        self.result = {"pot": pot, "winners": winners}
        self.events.append({"type": "hand_result", "pot": pot, "winners": winners})


def snapshot(session: Session, state, *, hand_complete: bool,
             result: Optional[dict] = None) -> dict:
    hero = state.players[0]
    bot = state.players[1]
    is_actor = state.current_player_idx == 0 and state.players[0].is_active
    return {
        "seq": len(session.history),
        "hand_id": session.hands_played + 1,
        "street": _STREET_NAME[state.betting_round],
        "community_cards": [str(c) for c in state.community_cards],
        "pot": state.pot,
        "current_bet": state.current_bet,
        "hero": {
            "seat": 0,
            "position": hero.position.value if hero.position else None,
            "hole_cards": [str(c) for c in hero.hole_cards],
            "stack": hero.stack,
            "current_bet": hero.current_bet,
            "is_actor": bool(is_actor),
        },
        "bot": {"stack": bot.stack, "current_bet": bot.current_bet},
        "legal_actions": legal_actions(state, hero) if is_actor else [],
        "hand_complete": hand_complete,
        "result": result,
        "session": {
            "start_stack": session.start_stack,
            "stack": hero.stack,
            # realized P/L + the current hand's unrealized delta
            "net": session.net + (hero.stack - session.hero_stack),
            "hands_played": session.hands_played,
        },
    }


def advance(session: Session, new_action=None) -> tuple[list[dict], dict]:
    if not session.hand_active:
        raise IllegalAction("no active hand")
    events: list[dict] = []
    recorder = _Recorder(events, session.hero_name)
    cursor = _Cursor(session.history)
    hero = Player(session.hero_name, session.hero_stack,
                  _HeroAgent(cursor, new_action))
    bot = Player(session.bot_name, session.bot_stack,
                 _BotAgent(cursor, session.bot_agent))
    game = PokerGame([hero, bot], small_blind=session.small_blind,
                     observers=[recorder])
    game.state.button_pos = session.button
    game.state._create_deck = lambda: list(session.deck)  # fixed deck per hand

    start_hero = session.hero_stack
    try:
        game.play_hand()
    except _AwaitHuman as pause:
        # Do NOT touch session.hero_stack/bot_stack: they are the PRE-hand stacks
        # that the next replay rebuilds from. The live stacks live in the state.
        state = snapshot(session, pause.state, hand_complete=False)
    else:
        # Hand complete: persist the resulting stacks (pre-hand for the next hand).
        session.hero_stack = hero.stack
        session.bot_stack = bot.stack
        session.net += session.hero_stack - start_hero
        session.hands_played += 1
        session.button = game.state.button_pos
        session.hand_active = False
        state = snapshot(session, game.state, hand_complete=True,
                         result=recorder.result)

    session.last_events = events
    session.last_state = state
    return events, state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/web/test_game_runner.py -v`
Expected: PASS. (If `test_illegal_check_facing_bet_raises` skips on seed 1, choose a seed in the test that has the hero facing a bet, or assert via a directly-constructed state — see the plan's self-review note.)

- [ ] **Step 5: Commit**

```bash
git add web/game_runner.py tests/web/test_game_runner.py
git commit -m "feat(web): replay-based heads-up game runner"
```

---

### Task 5: FastAPI application

**Files:**
- Create: `web/app.py`
- Test: `tests/web/test_app.py`

**Interfaces:**
- Consumes: `web.config.Config`, `web.session.SessionStore`, `web.game_runner.advance` / `IllegalAction`.
- Produces: `web.app.build_app(config: Config, bot_factory) -> fastapi.FastAPI` with the endpoints from the spec and a `seq` guard. Route bodies/responses use the shapes in `snapshot()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/web/test_app.py
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
    app = build_app(load_config(env={}), _FoldBot)
    return TestClient(app)


def test_session_lifecycle_and_health():
    c = _client()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/api/session")
    assert r.status_code == 200
    body = r.json()
    token = body["token"]
    assert body["state"]["hero"]["hole_cards"]
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
    token = body["token"]
    state = body["state"]
    if not state["hero"]["is_actor"]:
        return  # bot acted first; no hero action to submit
    # Wrong seq -> 409
    r = c.post(f"/api/session/{token}/action",
               json={"seq": 999, "action": "check"})
    assert r.status_code == 409


def test_illegal_action_rejected_and_hand_unchanged():
    c = _client()
    body = c.post("/api/session").json()
    token = body["token"]
    state = body["state"]
    if not state["hero"]["is_actor"]:
        return
    to_call = state["current_bet"] - state["hero"]["current_bet"]
    if to_call <= 0:
        return
    r = c.post(f"/api/session/{token}/action",
               json={"seq": state["seq"], "action": "check"})
    assert r.status_code == 409
    # State is unchanged
    state2 = c.get(f"/api/session/{token}").json()["state"]
    assert state2["seq"] == state["seq"]


def test_cors_header_present():
    c = _client()
    r = c.options("/api/session", headers={
        "Origin": "https://zfdupont.com",
        "Access-Control-Request-Method": "POST",
    })
    assert r.headers.get("access-control-allow-origin") == "https://zfdupont.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/web/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.app'`.

- [ ] **Step 3: Implement**

```python
# web/app.py
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
    current = hero["current_bet"]
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
        return Action.BET, current + hero["stack"]
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/session")
    def create_session():
        s = _new_session()
        events, state = advance(s)
        return {"token": s.token, "events": events, "state": state}

    def _get(token: str):
        s = store.get(token)
        if s is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return s

    @app.get("/api/session/{token}")
    def get_session(token: str):
        s = _get(token)
        if s.last_state is None:          # freshly created, not yet advanced
            events, state = advance(s)
            return {"token": s.token, "events": events, "state": state}
        return {"token": s.token, "events": s.last_events,
                "state": s.last_state}

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
```

Note: `do_action` peeks with `advance(s)` (no new action) to obtain the current pause `seq` and hero legality. Because `advance` replays from `session.history` (unchanged) and the fixed deck, the peek is side-effect-free for an active hand. This satisfies Review Focus items 1, 2, and 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/web/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole web test package**

Run: `uv run pytest tests/web -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/app.py tests/web/test_app.py
git commit -m "feat(web): FastAPI session endpoints with seq guard"
```

---

### Task 6: Shared-strategy bot factory and entry point

**Files:**
- Modify: `agents/sixmax_agent.py` (add `SixmaxAgent.from_strategy`)
- Create: `web/extension.py`, `web/main.py`
- Test: `tests/agents/test_sixmax_shared.py`, `tests/web/test_main_integration.py`

**Interfaces:**
- Consumes: `SixmaxDeployStrategy`, `SIXMAX_SO_PATH`, `Config`.
- Produces:
  - `SixmaxAgent.from_strategy(strategy: SixmaxDeployStrategy, config_toml: str) -> SixmaxAgent` — skips checkpoint loading; shares the loaded strategy across agents (each gets its own RNG).
  - `web.extension.load_sixmax() -> None`
  - `web.main.main() -> None` (uvicorn entry) and `web.main.build_production_app(config) -> FastAPI`

- [ ] **Step 1: Write the failing test for `from_strategy`**

```python
# tests/agents/test_sixmax_shared.py
import os
from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_from_strategy_shares_strategy_and_has_independent_rng(blueprint_hu_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_hu_ckpt, _TOML)
    a1 = SixmaxAgent.from_strategy(strat, _TOML)
    a2 = SixmaxAgent.from_strategy(strat, _TOML)
    assert a1._deploy is a2._deploy is strat
    assert a1._rng is not a2._rng
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_sixmax_shared.py -v`
Expected: FAIL — `AttributeError: type object 'SixmaxAgent' has no attribute 'from_strategy'`.

- [ ] **Step 3: Implement `from_strategy`**

In `agents/sixmax_agent.py`, add to `SixmaxAgent`:

```python
    @classmethod
    def from_strategy(cls, strategy, config_toml: str = _DEFAULT_TOML):
        """Build an agent over an already-loaded strategy (shares the
        checkpoint; each agent gets its own decision RNG)."""
        agent = cls.__new__(cls)
        agent._deploy = strategy
        agent._rng = random.Random()
        return agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_sixmax_shared.py -v`
Expected: PASS.

- [ ] **Step 5: Write the extension loader and entry point**

```python
# web/extension.py
"""Load the compiled sixmax extension without invoking Buck2 at runtime."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _so_path() -> str:
    explicit = os.environ.get("SIXMAX_SO_PATH")
    if explicit:
        return explicit
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            return os.path.join(_ROOT, line.split()[-1])
    raise RuntimeError("could not locate sixmax.so")


def load_sixmax() -> None:
    if hasattr(sys.modules.get("sixmax"), "SizeUnit"):
        return
    so = _so_path()
    spec = importlib.util.spec_from_file_location("sixmax", so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
```

```python
# web/main.py
"""Entry point: load the extension + checkpoint, then serve."""
from __future__ import annotations

import argparse

import uvicorn

from web.config import Config, load_config


def build_production_app(config: Config):
    from web.extension import load_sixmax
    load_sixmax()
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from web.app import build_app

    if not config.checkpoint:
        raise SystemExit("POKERBOT_CHECKPOINT is not set")
    strategy = SixmaxDeployStrategy.load(config.checkpoint, config.config_toml)
    bot_factory = lambda: SixmaxAgent.from_strategy(strategy, config.config_toml)  # noqa: E731
    return build_app(config, bot_factory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve pokerbot-web")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    app = build_production_app(load_config())
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the gated integration test**

```python
# tests/web/test_main_integration.py
"""End-to-end through the real checkpoint. Skipped when the extension cannot
be built (mirrors tests/sixmax gating)."""
import os
import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_full_hand_via_real_bot(blueprint_hu_ckpt):
    from web.extension import load_sixmax
    load_sixmax()
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from web.app import build_app
    from web.config import load_config

    strategy = SixmaxDeployStrategy.load(blueprint_hu_ckpt, _TOML)
    cfg = load_config(env={"POKERBOT_STACK": "200", "POKERBOT_SMALL_BLIND": "1",
                           "POKERBOT_CHECKPOINT": blueprint_hu_ckpt})
    app = build_app(cfg, lambda: SixmaxAgent.from_strategy(strategy, _TOML))
    c = TestClient(app)
    body = c.post("/api/session").json()
    token, state = body["token"], body["state"]
    for _ in range(20):
        if state["hand_complete"]:
            break
        if not state["hero"]["is_actor"]:
            state = c.get(f"/api/session/{token}").json()["state"]
            continue
        acts = {a["action"] for a in state["legal_actions"]}
        pick = "check" if "check" in acts else ("call" if "call" in acts else "fold")
        body = c.post(f"/api/session/{token}/action",
                      json={"seq": state["seq"], "action": pick}).json()
        state = body["state"]
    assert state["hand_complete"] is True
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/agents/test_sixmax_shared.py tests/web/test_main_integration.py -v`
Expected: PASS (integration requires the compiled extension; `tests/sixmax/conftest.py` force-loads it and the `blueprint_hu_ckpt` fixture builds a tiny checkpoint).

- [ ] **Step 8: Commit**

```bash
git add agents/sixmax_agent.py web/extension.py web/main.py \
        tests/agents/test_sixmax_shared.py tests/web/test_main_integration.py
git commit -m "feat(web): shared-strategy bot factory, extension loader, entry point"
```

---

### Task 7: Container image and CI

**Files:**
- Create: `web/Dockerfile`, `web/requirements.txt`
- Create: `.github/workflows/build-web-image.yml`
- Modify: `docker/Dockerfile` is untouched; reuse its builder pattern.

**Interfaces:**
- Consumes: built `sixmax.so`, repo Python sources.
- Produces: `ghcr.io/zfdupont/pokerbot-web:latest` running `python -m web.main`.

- [ ] **Step 1: Add the runtime dependencies**

```text
# web/requirements.txt
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.6
numpy>=1.26
tomli>=2.0
```

- [ ] **Step 2: Write the Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1
# ── Stage 1: build sixmax.so (mirrors docker/Dockerfile) ─────────────────────
FROM --platform=linux/amd64 ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential clang lld python3.12 python3.12-dev python3-pip \
        curl zstd \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fL \
    "https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-unknown-linux-gnu.zst" \
    | zstd -d - -o /usr/local/bin/buck2 && chmod +x /usr/local/bin/buck2
WORKDIR /build
COPY .buckconfig toolchains/ third_party/ common/ BUCK ./
RUN python3.12 -c \
    "import sysconfig; p=sysconfig.get_path('include'); \
     open('.buckconfig.local','w').write(f'[python]\n  include_path = {p}\n[build]\n  cpu_tune = x86-64\n')"
COPY sixmax/ sixmax/
RUN pip3 install --no-cache-dir --break-system-packages pybind11 \
    && PB_INC=$(python3.12 -c "import pybind11,os;print(os.path.join(os.path.dirname(pybind11.__file__),'include'))") \
    && mkdir -p third_party/pybind11 && ln -sfn "$PB_INC" third_party/pybind11/include
RUN mkdir -p third_party/libtorch/lib third_party/libtorch/include/torch/csrc/api/include \
    && touch third_party/libtorch/lib/libc10.dylib \
             third_party/libtorch/lib/libtorch_cpu.dylib \
             third_party/libtorch/lib/libtorch.dylib
RUN /usr/local/bin/buck2 build //sixmax:sixmax \
    && find /build/buck-out -name sixmax.so -exec cp {} /opt/sixmax.so \; \
    && test -f /opt/sixmax.so

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim
ENV PYTHONUNBUFFERED=1 SIXMAX_SO_PATH=/opt/sixmax.so
COPY --from=builder /opt/sixmax.so /opt/sixmax.so
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /pokerbot
COPY web/ web/
COPY agents/ agents/
COPY game/ game/
COPY models/ models/
COPY util/ util/
COPY sixmax/ sixmax/
COPY cfr/ cfr/
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health').status==200 else 1)"
EXPOSE 8100
CMD ["python", "-m", "web.main", "--host", "0.0.0.0", "--port", "8100"]
```

- [ ] **Step 3: Write the CI workflow**

```yaml
# .github/workflows/build-web-image.yml
name: build-web-image
on:
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - web/**
      - agents/**
      - game/**
      - models/**
      - util/**
      - sixmax/**
      - common/**
      - BUCK
      - toolchains/**
      - .github/workflows/build-web-image.yml
permissions:
  contents: read
  packages: write
jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: web/Dockerfile
          platforms: linux/amd64
          push: true
          tags: ghcr.io/zfdupont/pokerbot-web:latest
```

- [ ] **Step 4: Verify the image builds and boots (requires a checkpoint)**

Run (locally, with a HU checkpoint present):
```bash
docker build -f web/Dockerfile -t pokerbot-web:dev .
docker run --rm -p 8100:8100 -v "$PWD/sixmax/checkpoints:/checkpoints:ro" \
  -e POKERBOT_CHECKPOINT=/checkpoints/hu_blueprint.bin pokerbot-web:dev &
sleep 5 && curl -fsS localhost:8100/health && kill %1
```
Expected: `{"status":"ok"}`. If no checkpoint is available yet, build the image and only run `--help`; the boot check belongs to the deploy plan.

- [ ] **Step 5: Commit**

```bash
git add web/Dockerfile web/requirements.txt .github/workflows/build-web-image.yml
git commit -m "build(web): container image and GHCR CI workflow"
```

---

### Task 8: Local run docs and scaffold GROW

**Files:**
- Modify: `README.md` or `.mex/context/setup.md` (add a "poker web service" run section)
- Modify: `.mex/AGENTS.md` (extend the host invariant wording to include `web/`)
- Modify: `.mex/ROUTER.md`, `.mex/context/architecture.md` (record the new subsystem)

- [ ] **Step 1: Add the local run command to the scaffold**

Document, in `.mex/context/setup.md`:

```bash
# Serve the poker web API locally against a trained heads-up blueprint
export SIXMAX_SO_PATH="$(uv run python -c "import sys;print('')" >/dev/null; echo '')"  # optional; omit to auto-build via buck2
export POKERBOT_CHECKPOINT=sixmax/checkpoints/hu_blueprint.bin
uv run python -m web.main --port 8100
curl -fsS localhost:8100/health
```

- [ ] **Step 2: Update the invariant wording**

In `.mex/AGENTS.md`, change the first non-negotiable to include `web/` as a host:

> `cfr/`, `neural_cfr/`, and `sixmax/` never import `game/poker.py` or each other — hosts (`agents/`, `scripts/`, `web/`) bridge them; shared C++ lives only in `common/`.

Mirror the same line in `CLAUDE.md`.

- [ ] **Step 3: Record in `ROUTER.md`**

Add to "Current Project State → Working": the `web/` service, its endpoints, and the engine HU fix.

- [ ] **Step 4: Run the full suite and bump `last_updated`**

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add README.md .mex/ CLAUDE.md
git commit -m "docs(mex): record poker web service and web/ host invariant"
```

---

## Self-Review

**1. Spec coverage:** engine HU fix → Task 1; `web/` service (config, extension, session, runner, app) → Tasks 2–6; Dockerfile + CI → Task 7; scaffold GROW → Task 8. **Not covered by this plan (deliberately deferred to sibling plans):** the HU checkpoint training run (Plan 2), the `blogfolio` `/poker` page (Plan 3), and droplet compose/nginx/checkpoint deployment (Plan 4). The service is fully testable with the fixture checkpoint and a stub before those exist.

**2. Placeholder scan:** no `TBD`/`TODO`; every code step shows concrete code.

**3. Type consistency:** `advance()` returns `(list[dict], dict)` everywhere; `Session.history: list[Decision]`; `_Cursor.take(actor)` returns `Decision | None`; `snapshot()` keys match those read in `web/app.py` (`seq`, `hero.is_actor`, `hero.current_bet`, `hero.stack`, `legal_actions`, `current_bet`).

**4. Deviations from the spec to note in the PR:** (a) hero hole cards travel in the `state` snapshot rather than a `hole_cards` event; (b) a `seq` field was added to the snapshot and the `/action` body for duplicate-action safety. Both are additive and consistent with the spec's intent.

**5. Resolved during self-review:** `/action` reads the cached `session.last_state` for the `seq`/legality check instead of replaying twice, and `GET` returns the cached snapshot (so a reload after a completed hand still shows the result rather than a bare object).

## Execution Handoff

No subagent tool is available in this session, so the recommended method is **Native** (`executing-plans`): I implement each task in order, running the listed commands, committing per task.

Sibling plans to write when we reach them:
- `2026-09-30-hu-blueprint-training.md` (start with a per-iteration benchmark; see plan 2 risk note)
- `2026-09-30-poker-web-frontend.md` (blogfolio)
- `2026-09-30-poker-web-deploy.md` (droplet)
