"""Replay-based turn driver over the live engine.

A hand is replayed from its pre-hand stacks + fixed deck + ordered decisions on
every request. Replay pauses at the next human decision by raising `_AwaitHuman`
from the hero agent; `play_hand()` does not catch it, so it unwinds to
`advance()`. This keeps the engine as the single source of game logic while
letting a synchronous hand be served one REST request at a time.
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
    """The requested action is not legal in the current state."""


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
    """Cursor over the hand's recorded decisions, in engine call order."""

    def __init__(self, history: list[Decision]):
        self.history = history
        self.i = 0

    def take(self, actor: str) -> Optional[Decision]:
        if self.i >= len(self.history):
            return None
        rec = self.history[self.i]
        if rec.actor != actor:
            raise RuntimeError(
                f"decision order mismatch: expected {rec.actor}, got {actor}")
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
        self._new_action = None  # consume once; later hero turns must pause
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
        # Lower rank value is better (see util.evaluator); winners hold the min.
        min_rank = min(r for _, r in player_ranks)
        winners = [{"actor": self._actor(p)} for p, r in player_ranks
                   if r == min_rank]
        self.result = {"pot": pot, "winners": winners}
        self.events.append({"type": "hand_result", "pot": pot, "winners": winners})


def snapshot(session: Session, state, *, hand_complete: bool,
             result: Optional[dict] = None) -> dict:
    hero = state.players[0]
    bot = state.players[1]
    is_actor = (not hand_complete) and state.current_player_idx == 0 \
        and hero.is_active
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
