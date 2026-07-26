#!/usr/bin/env python3
"""
Connect the CFR bot to openpoker.ai.

Supports both tabular CFR (.pkl) and neural CFR (.pt) checkpoints — detected
automatically from the file extension.

Usage:
    export OPENPOKER_API_KEY=your_key_here
    uv run python scripts/openpoker_bot.py

    uv run python scripts/openpoker_bot.py --checkpoint neural_cfr/checkpoints/checkpoint.pt
    uv run python scripts/openpoker_bot.py --log-level DEBUG   # verbose decision trace
"""
import argparse
import asyncio
import ctypes
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import uuid

import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr.abstraction import hand_to_bucket, board_to_bucket
from cfr.info_set import InfoSet, stack_bucket
from cfr.regret_table import RegretTable
from models.card import Card
from models.enums import Suit
from agents.sixmax_agent import SixmaxDeployStrategy, DreamDeployStrategy, canonical_live_after

log = logging.getLogger("openpoker")

WS_URL     = "wss://openpoker.ai/ws"
SUIT_MAP   = {'h': Suit.HEARTS, 'd': Suit.DIAMONDS, 'c': Suit.CLUBS, 's': Suit.SPADES}
RANK_MAP   = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
STREET_IDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}

# neural_cfr suit encoding: 0=clubs 1=diamonds 2=hearts 3=spades
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}


def _parse_card(s: str) -> Card:
    return Card(RANK_MAP[s[0].upper()], SUIT_MAP[s[1].lower()])


def _card_to_int(card: Card) -> int:
    """Convert a models.card.Card to neural_cfr integer encoding (0–51)."""
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


def _latest_checkpoint(directory: str, ext: str) -> str | None:
    try:
        files = [f for f in os.listdir(directory) if f.endswith(ext)]
    except FileNotFoundError:
        return None
    return os.path.join(directory, sorted(files)[-1]) if files else None


def _load_neural_strategy(checkpoint: str):
    """Build the C++ extension and load a neural_cfr.Strategy checkpoint."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(repo_root, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        path = os.path.join(lib_dir, lib)
        if os.path.exists(path):
            ctypes.CDLL(path)

    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//neural_cfr:neural_cfr", "--show-output"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = None
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            so_dir = os.path.join(repo_root, os.path.dirname(line.split()[-1]))
            break
    if so_dir is None:
        raise RuntimeError("Could not locate neural_cfr.so in buck2 output")
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)

    import neural_cfr  # noqa: E402
    return neural_cfr.Strategy(checkpoint)


class HandTracker:
    """Tracks per-hand state for CFR strategy lookup (tabular or neural)."""

    def __init__(self):
        self.hole_cards: list[Card] = []
        self.community_cards: list[Card] = []
        self.street: int = 0
        self.raises_per_street: list[int] = [0, 0, 0, 0]
        self.big_blind: float = 20.0
        self.my_seat: int | None = None
        self.my_stack: float = 0.0
        self.my_position: int = 0     # 0=SB, 1=BB
        self.my_committed: float = 0.0

    def on_hand_start(self, msg: dict) -> None:
        self.hole_cards = []
        self.community_cards = []
        self.street = 0
        self.raises_per_street = [0, 0, 0, 0]
        self.my_seat = msg["seat"]
        bb = msg["blinds"]["big_blind"]
        sb = msg["blinds"]["small_blind"]
        self.big_blind = bb

        dealer_seat = msg.get("dealer_seat")
        if dealer_seat == self.my_seat:
            self.my_committed = sb
            self.my_position  = 0   # SB / button
        else:
            self.my_committed = bb
            self.my_position  = 1   # BB

        log.info(
            f"Hand {msg['hand_id']} | seat={self.my_seat} pos={'SB' if self.my_position == 0 else 'BB'} "
            f"dealer={dealer_seat} | blinds {sb}/{bb} | committed={self.my_committed}"
        )

    def on_hole_cards(self, msg: dict) -> None:
        self.hole_cards = [_parse_card(c) for c in msg["cards"]]
        log.info(f"Hole cards: {msg['cards']}")

    def on_community_cards(self, msg: dict) -> None:
        self.community_cards = [_parse_card(c) for c in msg["cards"]]
        self.street = STREET_IDX.get(msg.get("street", "preflop"), self.street)
        self.my_committed = 0.0
        log.info(f"{msg.get('street', '?')}: {msg['cards']}")

    def on_player_action(self, msg: dict) -> None:
        if msg.get("action") in ("raise", "all_in"):
            s = STREET_IDX.get(msg.get("street", "preflop"), 0)
            self.raises_per_street[s] = min(self.raises_per_street[s] + 1, 2)

    def decide(self, msg: dict, strategy, buy_in: int) -> dict:
        pot       = float(msg.get("pot", 0.0))
        valid_raw = msg.get("valid_actions", [])
        valid     = {a["action"]: a for a in valid_raw}

        for p in msg.get("players", []):
            if p.get("seat") == self.my_seat:
                self.my_stack = float(p["stack"])
                break

        to_call_chips = float(valid.get("call", {}).get("amount") or 0.0)

        abstract_legal: list[str] = []
        if "fold"   in valid: abstract_legal.append("fold")
        if "check"  in valid: abstract_legal.append("check")
        if "call"   in valid: abstract_legal.append("call")
        if "raise"  in valid: abstract_legal.extend(["b0.5", "b1.0"])
        if "raise"  in valid or "all_in" in valid: abstract_legal.append("allin")

        if not abstract_legal:
            log.warning("No legal abstract actions derived — folding/checking")
            return {"action": "check"} if "check" in valid else {"action": "fold"}

        bb             = self.big_blind
        committed_bb   = self.my_committed / bb
        to_call_bb     = to_call_chips / bb
        current_bet_bb = committed_bb + to_call_bb

        if isinstance(strategy, RegretTable):
            probs = self._decide_tabular(strategy, to_call_bb, pot / bb,
                                         self.my_stack / bb, abstract_legal)
        else:
            probs = self._decide_neural(strategy, to_call_chips, pot,
                                        buy_in, abstract_legal)

        abstract = np.random.choice(abstract_legal, p=probs)
        log.info(
            f"{'Neural' if not isinstance(strategy, RegretTable) else 'Tabular'} CFR: "
            f"street={self.street} pos={self.my_position} stack={self.my_stack:.0f} raises={self.raises_per_street} "
            f"→ {abstract} | "
            + " ".join(f"{a}:{p:.2f}" for a, p in zip(abstract_legal, probs.tolist()))
        )

        action_dict = self._translate(abstract, valid, to_call_chips, pot, current_bet_bb * bb)
        self._update_committed(action_dict, to_call_chips)
        return action_dict

    def hu_position(self) -> int:
        """Binary 0=SB/1=BB position — a heads-up-only concept.

        Quarantined here per Phase 0 item 4: the sixmax path uses
        seat-relative-to-button positions and must never consume this.
        """
        return self.my_position

    def _decide_tabular(self, table: RegretTable, to_call_bb: float,
                        pot_bb: float, stack_bb: float,
                        abstract_legal: list[str]) -> np.ndarray:
        hbucket = hand_to_bucket(self.hole_cards, self.community_cards, self.street)
        bbucket = board_to_bucket(self.community_cards, self.street)
        infoset = InfoSet(
            player=0,
            hand_bucket=hbucket,
            street=self.street,
            board_bucket=bbucket,
            betting_history=tuple(self.raises_per_street),
            stack_bucket=stack_bucket(stack_bb),
        )
        return table.get_average_strategy(infoset, abstract_legal)

    def _decide_neural(self, strategy, to_call_chips: float, pot: float,
                       buy_in: int, abstract_legal: list[str]) -> np.ndarray:
        # Training frame is big_blind == 1: dividing by the table's big blind
        # is the whole rescale. (buy_in/100 was only correct at exactly 100BB.)
        scale      = float(self.big_blind)
        hole_ints  = [_card_to_int(c) for c in self.hole_cards]
        board_ints = [_card_to_int(c) for c in self.community_cards]

        probs_dict = strategy.get_action_probs(
            hole_ints,
            board_ints,
            self.street,
            pot          / scale,
            self.my_stack / scale,
            to_call_chips / scale,
            self.raises_per_street,
            self.hu_position(),
            my_street_bet  = self.my_committed / scale,
            opp_street_bet = (self.my_committed + to_call_chips) / scale,
        )

        # Mask to OpenPoker's legal actions and renormalize
        raw = np.array([probs_dict.get(a, 0.0) for a in abstract_legal], dtype=float)
        total = raw.sum()
        if total <= 0:
            return np.ones(len(abstract_legal)) / len(abstract_legal)
        probs = raw / total
        # Variance guard: don't realize low-probability branches. Equilibrium
        # mixing frequencies assume an equilibrium opponent; against a field
        # that calls them down, sampled tail all-ins are pure spew. With <= 6
        # actions the max prob is always >= 1/6 > 0.15, so this never empties.
        probs[probs < 0.15] = 0.0
        return probs / probs.sum()

    def _translate(self, abstract: str, valid: dict, to_call_chips: float,
                   pot: float, current_bet_chips: float) -> dict:
        if abstract == "fold":
            return {"action": "fold"} if "fold" in valid else (
                {"action": "check"} if "check" in valid else {"action": "call"}
            )
        if abstract == "check":
            return {"action": "check"} if "check" in valid else (
                {"action": "call"} if "call" in valid else {"action": "fold"}
            )
        if abstract == "call":
            return {"action": "call"} if "call" in valid else (
                {"action": "check"} if "check" in valid else {"action": "fold"}
            )
        if abstract == "allin":
            if "all_in" in valid:
                return {"action": "all_in"}
            if "raise" in valid:
                return {"action": "raise", "amount": float(valid["raise"]["max"])}
            return {"action": "call"} if "call" in valid else {"action": "check"}

        if abstract in ("b0.5", "b1.0") and "raise" in valid:
            size     = float(abstract[1:])
            eff_pot  = pot + to_call_chips * 2
            raise_to = round(current_bet_chips + size * eff_pot, 1)
            min_r    = float(valid["raise"].get("min", 0))
            max_r    = float(valid["raise"].get("max", self.my_stack))
            return {"action": "raise", "amount": max(min_r, min(max_r, raise_to))}

        return {"action": "call"} if "call" in valid else (
            {"action": "check"} if "check" in valid else {"action": "fold"}
        )

    def should_leave_table(self, buy_in: float) -> bool:
        """Decide whether to bank the current stack and re-sit fresh.

        The net was trained at exactly 100BB effective (= buy_in chips here);
        the further self.my_stack drifts from buy_in, the further its inputs go
        off-distribution. Above 2x buy_in the normalized stack feature exceeds
        the maximum value ever seen in training. Called once per hand_result.
        self.my_stack is the stack seen on our most recent turn — it may lag
        by one hand, and is 0.0 if we haven't acted since (re)joining.
        Leaving costs nothing score-wise (chips return to the balance), but
        each re-sit spends a few hands' worth of time in the lobby queue.
        """
        return self.my_stack > 5*buy_in

    def _update_committed(self, action: dict, to_call_chips: float) -> None:
        a = action.get("action")
        if a == "call":
            self.my_committed += to_call_chips
        elif a == "raise":
            self.my_committed = float(action.get("amount", self.my_committed))
        elif a == "all_in":
            self.my_committed += self.my_stack


async def graceful_shutdown(ws) -> None:
    """Bank the table stack before dying: leave_table, then close.

    Best-effort — errors are swallowed because this runs on the way out."""
    try:
        await ws.send(json.dumps({"type": "leave_table"}))
        await ws.close()
    except Exception:
        pass


async def _shutdown_and_exit(ws) -> None:
    log.info("Signal received — leaving table and shutting down")
    await graceful_shutdown(ws)
    sys.exit(0)


class SixmaxHandTracker:
    """Six-max blueprint deployment tracker. Maintains per-seat fold/all-in
    state and per-street raise counts (cap 3) so it can rebuild the abstraction
    infoset key that the trainer used."""

    def __init__(self):
        self.hole_cards = []
        self.community_cards = []
        self.street = 0
        self.raises_per_street = [0, 0, 0, 0]
        self.big_blind = 20.0
        self.my_seat = None
        self.my_stack = 0.0
        self.my_committed = 0.0
        self.seats = []          # sorted table seat numbers present
        self.button_seat = None
        self.folded = {}         # seat -> bool
        self.all_in = {}         # seat -> bool

    def on_hand_start(self, msg):
        self.hole_cards = []
        self.community_cards = []
        self.street = 0
        self.raises_per_street = [0, 0, 0, 0]
        self.my_seat = msg["seat"]
        self.big_blind = msg["blinds"]["big_blind"]
        self.button_seat = msg.get("dealer_seat")
        self.seats = sorted(p["seat"] for p in msg.get("players", []))
        self.folded = {s: False for s in self.seats}
        self.all_in = {s: False for s in self.seats}
        self.my_committed = 0.0

    def on_hole_cards(self, msg):
        self.hole_cards = [_parse_card(c) for c in msg["cards"]]

    def on_community_cards(self, msg):
        self.community_cards = [_parse_card(c) for c in msg["cards"]]
        self.street = STREET_IDX.get(msg.get("street", "preflop"), self.street)
        self.my_committed = 0.0

    def on_player_action(self, msg):
        seat = msg.get("seat")
        act = msg.get("action")
        if act == "fold" and seat in self.folded:
            self.folded[seat] = True
        if act == "all_in" and seat in self.all_in:
            self.all_in[seat] = True
        if act in ("raise", "all_in"):
            s = STREET_IDX.get(msg.get("street", "preflop"), self.street)
            self.raises_per_street[s] = min(self.raises_per_street[s] + 1, 3)

    def _positions(self):
        n = len(self.seats)
        idx = {s: i for i, s in enumerate(self.seats)}
        return (n, idx[self.button_seat], idx[self.my_seat],
                [self.folded[s] for s in self.seats],
                [self.all_in[s] for s in self.seats])

    def decide(self, msg, strategy, buy_in):
        bb = self.big_blind
        pot = float(msg.get("pot", 0.0))
        valid = {a["action"]: a for a in msg.get("valid_actions", [])}
        for p in msg.get("players", []):
            if p.get("seat") == self.my_seat:
                self.my_stack = float(p["stack"])
                break
        to_call_chips = float(valid.get("call", {}).get("amount") or 0.0)

        n, button, hero, folded, all_in = self._positions()
        live, after = canonical_live_after(n, button, hero, folded, all_in,
                                           self.street)

        current_bet_bb = (self.my_committed + to_call_chips) / bb
        stack_bb = (self.my_committed + self.my_stack) / bb
        legal = self._legal_mask(strategy, valid, current_bet_bb, pot / bb,
                                 to_call_chips / bb, stack_bb)

        import random as _random
        idx, raise_to_bb = strategy.decide(
            hole=[_card_to_int(c) for c in self.hole_cards],
            board=[_card_to_int(c) for c in self.community_cards],
            street=self.street, raises_per_street=self.raises_per_street,
            pot_bb=pot / bb, current_bet_bb=current_bet_bb,
            to_call_bb=to_call_chips / bb, stack_bb=stack_bb, live=live,
            after=after, legal=legal, rng=_random.Random())
        return self._translate(strategy, idx, valid, raise_to_bb, bb)

    def _legal_mask(self, strategy, valid, current_bet_bb, pot_bb, to_call_bb,
                    stack_bb):
        vocab = strategy._vocab
        facing = "call" in valid
        can_raise = "raise" in valid or "all_in" in valid
        unopened_preflop = self.street == 0 and current_bet_bb <= 1.0 + 1e-9
        min_raise_to_bb = float(valid.get("raise", {}).get("min", 0.0)) / self.big_blind
        ctx = __import__("sixmax").BetContext(
            pot=pot_bb, current_bet=current_bet_bb, to_call=to_call_bb,
            stack=stack_bb)
        sm = __import__("sixmax")
        mask = [0] * vocab.size()
        for i in range(vocab.size()):
            a = vocab.at(i)
            if a.type == sm.ActionType.Fold:
                mask[i] = 1 if facing else 0
            elif a.type == sm.ActionType.Check:
                mask[i] = 0 if facing else 1
            elif a.type == sm.ActionType.Call:
                mask[i] = 1 if facing else 0
            elif a.type == sm.ActionType.Bet:
                bb_unit = a.unit == sm.SizeUnit.BB
                if not can_raise or "raise" not in valid \
                        or bb_unit != unopened_preflop:
                    continue
                t = vocab.target_bb(i, ctx)
                mask[i] = 1 if (t >= min_raise_to_bb - 1e-9
                                and t < stack_bb - 1e-9) else 0
            elif a.type == sm.ActionType.AllIn:
                mask[i] = 1 if can_raise else 0
        return mask

    def _translate(self, strategy, idx, valid, raise_to_bb, bb):
        sm = __import__("sixmax")
        a = strategy._vocab.at(idx)
        if a.type == sm.ActionType.Fold:
            return {"action": "fold"} if "fold" in valid else (
                {"action": "check"} if "check" in valid else {"action": "call"})
        if a.type == sm.ActionType.Check:
            return {"action": "check"} if "check" in valid else (
                {"action": "call"} if "call" in valid else {"action": "fold"})
        if a.type == sm.ActionType.Call:
            return {"action": "call"} if "call" in valid else (
                {"action": "check"} if "check" in valid else {"action": "fold"})
        if a.type == sm.ActionType.AllIn:
            if "all_in" in valid:
                return {"action": "all_in"}
            if "raise" in valid:
                return {"action": "raise", "amount": float(valid["raise"]["max"])}
            return {"action": "call"} if "call" in valid else {"action": "check"}
        # Bet: raise-to in chips (openpoker uses the raise-to convention).
        raise_to = round(raise_to_bb * bb, 1)
        min_r = float(valid["raise"].get("min", 0))
        max_r = float(valid["raise"].get("max", self.my_stack))
        return {"action": "raise", "amount": max(min_r, min(max_r, raise_to))}


async def run(api_key: str, strategy, buy_in: int) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}

    # connect() as an async iterator yields a fresh connection whenever the
    # previous one drops, with exponential backoff between attempts.
    async for ws in websockets.connect(WS_URL, additional_headers=headers):
        try:
            log.info(f"Connected to {WS_URL}")
            tracker = (SixmaxHandTracker()
                       if isinstance(strategy, (SixmaxDeployStrategy, DreamDeployStrategy))
                       else HandTracker())

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda: asyncio.ensure_future(_shutdown_and_exit(ws)))

            await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
            log.info(f"Sent join_lobby (buy_in={buy_in})")
            await ws.send(json.dumps({"type": "set_auto_rebuy", "enabled": True}))

            async for raw in ws:
                msg   = json.loads(raw)
                mtype = msg.get("type", "")
                log.debug(f"← {mtype}: {json.dumps(msg)[:300]}")

                if mtype == "connected":
                    log.info(f"Authenticated: {msg}")
                elif mtype == "hand_start":
                    tracker.on_hand_start(msg)
                elif mtype == "hole_cards":
                    tracker.on_hole_cards(msg)
                elif mtype == "community_cards":
                    tracker.on_community_cards(msg)
                elif mtype == "player_action":
                    tracker.on_player_action(msg)
                elif mtype == "your_turn":
                    if not tracker.hole_cards:
                        log.warning("your_turn with no hole cards — folding")
                        action = {"action": "fold"}
                    else:
                        action = tracker.decide(msg, strategy, buy_in)

                    response = {
                        "type":             "action",
                        "hand_id":          msg.get("hand_id"),
                        "turn_token":       msg.get("turn_token"),
                        "client_action_id": f"cfr-{msg.get('turn_token') or uuid.uuid4().hex}",
                        **action,
                    }
                    log.info(f"→ {action}")
                    await ws.send(json.dumps(response))

                elif mtype == "action_rejected":
                    log.error(f"Action rejected: {msg}")
                elif mtype == "hand_result":
                    winners = [w.get("name") for w in msg.get("winners", [])]
                    log.info(f"Hand result: winners={winners} pot={msg.get('pot')}")
                    if tracker.should_leave_table(buy_in):
                        log.info(f"Stack {tracker.my_stack:.0f} outside comfort band "
                                 f"— banking it and re-sitting at {buy_in}")
                        await ws.send(json.dumps({"type": "leave_table"}))
                        tracker.my_stack = 0.0
                        await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
                elif mtype == "busted":
                    log.warning("Busted — re-entering lobby from balance")
                    await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
                elif mtype == "table_closed":
                    log.info("Table closed — rejoining lobby")
                    await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
                elif mtype == "error":
                    log.error(f"Server error: {msg}")
                elif mtype:
                    log.debug(f"Ignored: {mtype}")
        except websockets.ConnectionClosed:
            log.warning("Connection lost — reconnecting")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFR bot on openpoker.ai")
    parser.add_argument("--api-key", default=os.environ.get("OPENPOKER_API_KEY"),
                        help="API key (or set OPENPOKER_API_KEY env var)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pkl (tabular) or .pt (neural) checkpoint")
    parser.add_argument("--buy-in",    type=int, default=2000)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set OPENPOKER_API_KEY or pass --api-key")
        sys.exit(1)

    checkpoint = args.checkpoint
    if not checkpoint:
        # Prefer neural if available, fall back to tabular
        checkpoint = (_latest_checkpoint("neural_cfr/checkpoints", ".pt") or
                      _latest_checkpoint("cfr/checkpoints", ".pkl"))
    if not checkpoint or not os.path.exists(checkpoint):
        print("Error: no checkpoint found. Train a model first or pass --checkpoint.")
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    log.info(f"Loading {checkpoint} ...")
    _ckpt_path = pathlib.Path(checkpoint)
    if checkpoint.endswith(".bin"):
        toml = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "sixmax", "configs", "default.toml")
        strategy = SixmaxDeployStrategy.load(checkpoint, toml)
        log.info("Six-max blueprint strategy loaded.")
    elif _ckpt_path.suffix == ".pt" and _ckpt_path.stem.startswith("dream_"):
        toml = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "sixmax", "configs", "default.toml")
        strategy = DreamDeployStrategy.load(checkpoint, toml)
        log.info("DREAM neural blueprint strategy loaded.")
    elif checkpoint.endswith(".pt"):
        strategy = _load_neural_strategy(checkpoint)
        log.info("Neural CFR strategy loaded.")
    else:
        strategy = RegretTable()
        strategy.load(checkpoint)
        log.info("Tabular CFR strategy loaded.")

    log.info("Connecting...")
    asyncio.run(run(args.api_key, strategy, args.buy_in))


if __name__ == "__main__":
    main()
