#!/usr/bin/env python3
"""THROWAWAY SPIKE: play a sixmax HU checkpoint against Slumbot's public API.

Sequential, one request at a time. Logs one JSON line per hand.
Usage: slumbot_spike.py --hands N --out log.jsonl [--checkpoint ...] [--config ...]
"""
import argparse
import json
import re
import sys
import time
from types import SimpleNamespace

import urllib.error
import urllib.request

REPO = "/Users/zfdupont/pokerbot"
sys.path.insert(0, REPO)

from agents.sixmax_agent import SixmaxAgent  # noqa: E402
from models.card import Card  # noqa: E402
from models.enums import Action, Suit  # noqa: E402

HOST = "https://slumbot.com/slumbot/api"
SB, BB, STACK = 50, 100, 20000
RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
SUITS = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
TOKEN_RE = re.compile(r"[kcf]|b\d+")


def card(s):
    return Card(RANKS[s[0]], SUITS[s[1]])


def street_of(board):
    return {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]


def parse(action, street):
    """Replay a Slumbot action string. Returns per-street commitments for
    pos 0 (BB) / pos 1 (SB), prior-street totals, raises per street."""
    segs = action.split("/")
    segs += [""] * (street + 1 - len(segs))
    tot = [0, 0]
    raises = [0, 0, 0, 0]
    for st in range(street + 1):
        sb_ = [BB, SB] if st == 0 else [0, 0]
        cur = BB if st == 0 else 0
        actor = 1 if st == 0 else 0
        for tok in TOKEN_RE.findall(segs[st]):
            if tok[0] == "b":
                sb_[actor] = int(tok[1:])
                cur = sb_[actor]
                raises[st] += 1
            elif tok == "c":
                sb_[actor] = cur
            actor ^= 1
        if st < street:
            tot = [tot[0] + sb_[0], tot[1] + sb_[1]]
        else:
            street_bet = sb_
    return tot, street_bet, raises


def build_state(resp):
    board = resp.get("board", [])
    street = street_of(board)
    tot, sbet, raises = parse(resp["action"], street)
    hero_pos = resp["client_pos"]           # 0 = BB, 1 = SB
    # engine seats: 0 = button/SB, 1 = BB
    seat_of = {1: 0, 0: 1}
    players = [None, None]
    for pos in (0, 1):
        stack = STACK - tot[pos] - sbet[pos]
        players[seat_of[pos]] = SimpleNamespace(
            name=f"p{pos}",
            hole_cards=[card(c) for c in resp["hole_cards"]] if pos == hero_pos else [],
            is_active=True, is_all_in=stack == 0,
            current_bet=sbet[pos], stack=stack)
    gs = SimpleNamespace(
        big_blind=BB, players=players, betting_round=street,
        community_cards=[card(c) for c in board], button_pos=0,
        current_bet=max(sbet), pot=sum(tot) + sum(sbet),
        raises_per_street=raises)
    return gs, players[seat_of[hero_pos]], sbet[hero_pos] < max(sbet)


def to_incr(action, amount, hero, facing):
    if action == Action.FOLD:
        return "f" if facing else "k"
    if action == Action.CHECK:
        return "k" if not facing else "c"
    if action == Action.CALL:
        return "c" if facing else "k"
    cap = hero.current_bet + hero.stack
    return f"b{min(int(amount), cap)}"


def post(sess, ep, body):
    data = json.dumps(body).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{HOST}/{ep}", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())  # API reports error_msg in body
            except ValueError:
                time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{ep} failed 5 times")


def play_hand(sess, agent, token):
    resp = post(sess, "new_hand", {"token": token} if token else {})
    actions = []
    while True:
        if resp.get("error_msg"):
            raise RuntimeError(resp["error_msg"])
        token = resp.get("token", token)
        if "winnings" in resp:
            return token, resp, actions
        gs, hero, facing = build_state(resp)
        a, amt = agent.get_action(hero, gs)
        incr = to_incr(a, amt, hero, facing)
        actions.append(incr)
        nxt = post(sess, "act", {"token": token, "incr": incr})
        if nxt.get("error_msg") and incr.startswith("b"):
            # illegal size per Slumbot's min-raise rule: fall back to call/check
            incr = "c" if facing else "k"
            actions.append("fallback:" + incr)
            nxt = post(sess, "act", {"token": token, "incr": incr})
        resp = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=f"{REPO}/sixmax/checkpoints/hu_blueprint_50000000.bin")
    ap.add_argument("--config", default=f"{REPO}/sixmax/configs/hu.toml")
    args = ap.parse_args()

    agent = SixmaxAgent(args.checkpoint, config_toml=args.config)
    sess = None
    token = None
    won = base = 0
    with open(args.out, "a") as f:
        for h in range(1, args.hands + 1):
            token, resp, acts = play_hand(sess, agent, token)
            won += resp["winnings"]
            base += resp.get("baseline_winnings", 0)
            f.write(json.dumps({"hand": h, "pos": resp["client_pos"],
                                "action": resp["action"], "ours": acts,
                                "hole": resp["hole_cards"], "board": resp.get("board"),
                                "bot_hole": resp.get("bot_hole_cards"),
                                "winnings": resp["winnings"],
                                "baseline": resp.get("baseline_winnings")}) + "\n")
            f.flush()
            if h % 50 == 0 or h == args.hands:
                print(f"hands={h} raw={won / BB / h * 100:+.1f} bb/100 "
                      f"baseline={base / BB / h * 100:+.1f} bb/100", flush=True)


if __name__ == "__main__":
    main()
