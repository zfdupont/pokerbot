"""In-memory, TTL'd sessions for heads-up cash play.

The store is intentionally process-local: the service runs a single uvicorn
worker (see the spec). `last_seen` slides on access, so an open/polling tab
keeps its session alive while an abandoned one expires after `ttl_s`.
"""
from __future__ import annotations

import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from models.card import Card
from models.enums import Action, Suit


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
                small_blind=1,          # filled from Config by the app
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
