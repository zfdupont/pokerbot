#!/usr/bin/env python3
"""
Connect the CFR bot to openpoker.ai.

Usage:
    export OPENPOKER_API_KEY=your_key_here
    uv run python scripts/openpoker_bot.py

    uv run python scripts/openpoker_bot.py --api-key YOUR_KEY --buy-in 2000
    uv run python scripts/openpoker_bot.py --log-level DEBUG   # verbose decision trace
"""
import argparse
import asyncio
import json
import logging
import os
import sys

import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr.abstraction import hand_to_bucket, board_to_bucket, legal_abstract_actions
from cfr.info_set import InfoSet, stack_bucket
from cfr.regret_table import RegretTable
from models.card import Card
from models.enums import Suit

log = logging.getLogger("openpoker")

WS_URL     = "wss://openpoker.ai/ws"
SUIT_MAP   = {'h': Suit.HEARTS, 'd': Suit.DIAMONDS, 'c': Suit.CLUBS, 's': Suit.SPADES}
RANK_MAP   = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
STREET_IDX = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}


def _parse_card(s: str) -> Card:
    return Card(RANK_MAP[s[0].upper()], SUIT_MAP[s[1].lower()])


def _latest_checkpoint(directory="cfr/checkpoints"):
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return None
    return os.path.join(directory, sorted(files)[-1]) if files else None


class HandTracker:
    """Tracks per-hand state needed to build InfoSets for the CFR strategy lookup."""

    def __init__(self):
        self.hole_cards: list[Card] = []
        self.community_cards: list[Card] = []
        self.street: int = 0
        self.raises_per_street: list[int] = [0, 0, 0, 0]
        self.big_blind: float = 20.0
        self.my_seat: int | None = None
        self.my_stack: float = 0.0
        # Chips committed by ME this street (tracks blind + bets so raise-to is correct)
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

        # Set opening commitment from blind position
        dealer_seat = msg.get("dealer_seat")
        if dealer_seat == self.my_seat:
            self.my_committed = sb   # we're the button/SB in HU (or dealer in ring)
        else:
            self.my_committed = bb   # we're BB

        log.info(
            f"Hand {msg['hand_id']} | seat={self.my_seat} "
            f"dealer={dealer_seat} | blinds {sb}/{bb} | committed={self.my_committed}"
        )

    def on_hole_cards(self, msg: dict) -> None:
        self.hole_cards = [_parse_card(c) for c in msg["cards"]]
        log.info(f"Hole cards: {msg['cards']}")

    def on_community_cards(self, msg: dict) -> None:
        self.community_cards = [_parse_card(c) for c in msg["cards"]]
        self.street = STREET_IDX.get(msg.get("street", "preflop"), self.street)
        self.my_committed = 0.0   # new street resets per-street commitment
        log.info(f"{msg.get('street', '?')}: {msg['cards']}")

    def on_player_action(self, msg: dict) -> None:
        """Track raise counts; own chip commitments are updated after we send."""
        if msg.get("action") in ("raise", "all_in"):
            s = STREET_IDX.get(msg.get("street", "preflop"), 0)
            self.raises_per_street[s] = min(self.raises_per_street[s] + 1, 2)

    def decide(self, msg: dict, table: RegretTable) -> dict:
        """
        Given a your_turn message and a loaded RegretTable, return the action dict
        to send back to the server (sans type/hand_id/turn_token).
        """
        pot           = float(msg.get("pot", 0.0))
        valid_raw     = msg.get("valid_actions", [])
        valid         = {a["action"]: a for a in valid_raw}

        # Update stack from players snapshot in the message
        for p in msg.get("players", []):
            if p.get("seat") == self.my_seat:
                self.my_stack = float(p["stack"])
                break

        # Additional chips needed to call
        to_call_chips = float(valid.get("call", {}).get("amount") or 0.0)

        # --- Build abstract legal action set ---
        # We map OpenPoker valid_actions directly rather than recomputing them,
        # so the CFR mask exactly matches what the server will accept.
        abstract_legal: list[str] = []
        if "fold"   in valid: abstract_legal.append("fold")
        if "check"  in valid: abstract_legal.append("check")
        if "call"   in valid: abstract_legal.append("call")
        if "raise"  in valid: abstract_legal.extend(["b0.5", "b1.0"])
        if "raise"  in valid or "all_in" in valid: abstract_legal.append("allin")

        if not abstract_legal:
            log.warning("No legal abstract actions derived — folding/checking")
            return {"action": "check"} if "check" in valid else {"action": "fold"}

        # --- CFR InfoSet lookup ---
        bb       = self.big_blind
        to_call_bb   = to_call_chips / bb
        pot_bb       = pot / bb
        stack_bb     = self.my_stack / bb
        committed_bb = self.my_committed / bb
        current_bet_bb = committed_bb + to_call_bb   # total bet level (in BB)

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

        probs   = table.get_average_strategy(infoset, abstract_legal)
        abstract = np.random.choice(abstract_legal, p=probs)

        log.info(
            f"CFR: street={self.street} hbucket={hbucket} "
            f"raises={self.raises_per_street} stack_bkt={stack_bucket(stack_bb)} "
            f"→ {abstract} | "
            + " ".join(f"{a}:{p:.2f}" for a, p in zip(abstract_legal, probs.tolist()))
        )

        action_dict = self._translate(abstract, valid, to_call_chips, pot, current_bet_bb * bb)
        self._update_committed(action_dict, to_call_chips)
        return action_dict

    def _translate(
        self,
        abstract: str,
        valid: dict,
        to_call_chips: float,
        pot: float,
        current_bet_chips: float,
    ) -> dict:
        """Map an abstract CFR action to an OpenPoker action dict."""

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

        # b0.5 or b1.0 — compute raise-to amount
        if abstract in ("b0.5", "b1.0") and "raise" in valid:
            size        = float(abstract[1:])
            eff_pot     = pot + to_call_chips * 2
            # raise-to = current_bet_chips + size * eff_pot
            # (matches CFRAgent convention: amount = to_call + size*eff = current_bet + size*eff)
            raise_to    = current_bet_chips + size * eff_pot
            raise_to    = round(raise_to, 1)
            min_r       = float(valid["raise"].get("min", 0))
            max_r       = float(valid["raise"].get("max", self.my_stack))
            raise_to    = max(min_r, min(max_r, raise_to))
            return {"action": "raise", "amount": raise_to}

        # Fallback when raise isn't available for a bet abstract action
        return {"action": "call"} if "call" in valid else (
            {"action": "check"} if "check" in valid else {"action": "fold"}
        )

    def _update_committed(self, action: dict, to_call_chips: float) -> None:
        """Update my_committed after deciding an action so future raise-to is correct."""
        a = action.get("action")
        if a == "call":
            self.my_committed += to_call_chips
        elif a == "raise":
            self.my_committed = float(action.get("amount", self.my_committed))
        elif a == "all_in":
            self.my_committed += self.my_stack   # stack goes to 0


async def run(api_key: str, table: RegretTable, buy_in: int) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        log.info(f"Connected to {WS_URL}")
        tracker = HandTracker()

        # Join matchmaking lobby immediately after connecting
        await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
        log.info(f"Sent join_lobby (buy_in={buy_in})")

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
                    action = tracker.decide(msg, table)

                response = {
                    "type":             "action",
                    "hand_id":          msg.get("hand_id"),
                    "turn_token":       msg.get("turn_token"),
                    "client_action_id": f"cfr-{id(msg)}",
                    **action,
                }
                log.info(f"→ {action}")
                await ws.send(json.dumps(response))

            elif mtype == "action_rejected":
                log.error(f"Action rejected: {msg}")

            elif mtype == "hand_result":
                winners = [w.get("name") for w in msg.get("winners", [])]
                log.info(f"Hand result: winners={winners} pot={msg.get('pot')}")

            elif mtype == "busted":
                log.warning("Busted! Waiting for auto-rebuy or table close.")

            elif mtype == "table_closed":
                log.info("Table closed — rejoining lobby")
                await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))

            elif mtype == "error":
                log.error(f"Server error: {msg}")

            elif mtype:
                log.debug(f"Ignored: {mtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFR bot on openpoker.ai")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENPOKER_API_KEY"),
        help="API key (or set OPENPOKER_API_KEY env var)",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to .pkl checkpoint")
    parser.add_argument("--buy-in",    type=int, default=2000)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set OPENPOKER_API_KEY or pass --api-key")
        sys.exit(1)

    checkpoint = args.checkpoint or _latest_checkpoint()
    if not checkpoint or not os.path.exists(checkpoint):
        print("Error: no checkpoint found. Train a model first or pass --checkpoint.")
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info(f"Loading {checkpoint} ...")
    table = RegretTable()
    table.load(checkpoint)
    log.info("Strategy loaded. Connecting...")

    asyncio.run(run(args.api_key, table, args.buy_in))


if __name__ == "__main__":
    main()
