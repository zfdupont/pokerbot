#!/usr/bin/env python3
"""Play a heads-up sixmax checkpoint against Slumbot's public API.

Slumbot speaks its own protocol: a hand is driven by POSTing one action at a
time, and the server replies with the whole hand so far as an ACPC action string
plus the current board and hole cards. This evaluator replays that string into
the stand-in state `SixmaxAgent.get_action` reads (a SimpleNamespace graph shaped
like game/poker.py's GameState), translates our chosen action back to Slumbot's
incremental tokens, and logs one JSON line per finished hand so a long run is
resumable and its BB/100 can be re-derived offline.

Behaviour is frozen to the proven spike at
docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/slumbot_spike.py:
same state construction, same outbound translation, same fallback to `c`/`k`
when Slumbot rejects a bet size. Every gate's number must stay comparable to the
-118 BB/100 baseline, so do not change parsing or translation without re-running
`hu_blueprint_50000000.bin` and agreeing with -118 within 2 SE.

Constraints this file keeps: strictly one request at a time; stdlib urllib only,
no new dependency; login only from SLUMBOT_USERNAME / SLUMBOT_PASSWORD, and the
token is never printed or logged; an aborted hand (server error mid-hand) is
dropped rather than logged half-played.

Usage:
    uv run python scripts/eval_slumbot.py --checkpoint sixmax/checkpoints/hu.bin \
        --hands 13000 --log slumbot_run.jsonl
    uv run python scripts/eval_slumbot.py --summary --log slumbot_run.jsonl
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from types import SimpleNamespace

import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.sixmax_agent import SixmaxAgent  # noqa: E402
from models.card import Card  # noqa: E402
from models.enums import Action, Suit  # noqa: E402

HOST = "https://slumbot.com/slumbot/api"
SB, BB, STACK = 50, 100, 20000
RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
SUITS = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
TOKEN_RE = re.compile(r"[kcf]|b\d+")

# Hands that abort in a row (server errors, repeated transport failures) mean the
# API is down or our requests are malformed: stop rather than spin forever.
MAX_CONSECUTIVE_ABORTS = 5
RETRIES = 5

SIZE_BINS = ("<10", "10-40", "40-100", "100+")


def card(s):
    return Card(RANKS[s[0]], SUITS[s[1]])


def street_of(board):
    return {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]


def parse_action(action, street):
    """Replay a Slumbot action string up to `street`.

    Returns (prior-street totals, this-street commitments, raises per street),
    each chip list indexed by Slumbot pos (0 = BB, 1 = SB)."""
    segs = action.split("/")
    segs += [""] * (street + 1 - len(segs))
    tot = [0, 0]
    raises = [0, 0, 0, 0]
    street_bet = [0, 0]
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
    """Shape one Slumbot response into the GameState stand-in the agent reads."""
    board = resp.get("board", [])
    street = street_of(board)
    tot, sbet, raises = parse_action(resp["action"], street)
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
    """Translate one agent decision into Slumbot's incremental token."""
    if action == Action.FOLD:
        return "f" if facing else "k"
    if action == Action.CHECK:
        return "k" if not facing else "c"
    if action == Action.CALL:
        return "c" if facing else "k"
    cap = hero.current_bet + hero.stack
    return f"b{min(int(amount), cap)}"


def post(ep, body):
    """POST one API call. Retries transport failures with exponential backoff;
    Slumbot reports protocol errors in a 200/4xx JSON body, which is returned
    as-is for the caller to inspect. Raises RuntimeError when retries run out."""
    data = json.dumps(body).encode()
    for attempt in range(RETRIES):
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
    raise RuntimeError(f"{ep} failed {RETRIES} times")


def login(fetch=None):
    """Log in only when both credentials are in the environment.

    The token is returned for the request bodies and never printed."""
    user = os.environ.get("SLUMBOT_USERNAME")
    password = os.environ.get("SLUMBOT_PASSWORD")
    if not user or not password:
        return None
    resp = (fetch or post)("login", {"username": user, "password": password})
    token = resp.get("token")
    if not token:
        raise SystemExit("Slumbot login failed; check SLUMBOT_USERNAME/SLUMBOT_PASSWORD")
    return token


def play_hand(fetch, agent, token):
    """Play one hand to completion.

    Returns (latest token, response, our actions). The response is None when the
    hand was dropped — a server error mid-hand or the transport giving up after
    its retries — so the caller never logs a half-played hand."""
    actions = []
    failed = object()   # sentinel: the transport raised, so the hand is dead

    def call(ep, body):
        try:
            return fetch(ep, body)
        except RuntimeError:
            return failed

    resp = call("new_hand", {"token": token} if token else {})
    while resp is not failed:
        if resp.get("error_msg"):
            return token, None, actions
        token = resp.get("token", token)
        if "winnings" in resp:
            return token, resp, actions
        gs, hero, facing = build_state(resp)
        action, amount = agent.get_action(hero, gs)
        incr = to_incr(action, amount, hero, facing)
        actions.append(incr)
        nxt = call("act", {"token": token, "incr": incr})
        if nxt is not failed and nxt.get("error_msg") and incr.startswith("b"):
            # Illegal size per Slumbot's min-raise rule: fall back to call/check.
            incr = "c" if facing else "k"
            actions.append("fallback:" + incr)
            nxt = call("act", {"token": token, "incr": incr})
        resp = nxt
    return token, None, actions


def _lookup_counts(agent):
    """The deploy strategy's lookup counters (hit/nearest/safe/uniform), or
    None for agents without them."""
    counts = getattr(getattr(agent, "_deploy", None), "lookups", None)
    return dict(counts) if isinstance(counts, dict) else None


def run_hands(fetch, agent, token, log_path, target, existing_rows=()):
    """Play until the log holds `target` rows, appending one JSONL row per
    finished hand. `existing_rows` are the rows already logged, so a resumed run
    continues the numbering and the running summary.

    Returns the latest server token."""
    won = sum(row["winnings"] for row in existing_rows)
    base = sum(row.get("baseline") or 0 for row in existing_rows)
    written = len(existing_rows)
    aborts = 0
    with open(log_path, "a") as f:
        while written < target:
            before = _lookup_counts(agent)
            token, resp, acts = play_hand(fetch, agent, token)
            if resp is None:
                aborts += 1
                print(f"hand {written + 1} aborted ({aborts} in a row)", flush=True)
                if aborts >= MAX_CONSECUTIVE_ABORTS:
                    raise SystemExit(
                        f"stopping: {aborts} consecutive aborted hands "
                        f"(server error or transport failure)")
                continue
            aborts = 0
            written += 1
            won += resp["winnings"]
            base += resp.get("baseline_winnings") or 0
            row = {"hand": written, "pos": resp["client_pos"],
                   "action": resp["action"], "ours": acts,
                   "hole": resp["hole_cards"], "board": resp.get("board"),
                   "bot_hole": resp.get("bot_hole_cards"),
                   "winnings": resp["winnings"],
                   "baseline": resp.get("baseline_winnings")}
            after = _lookup_counts(agent)
            if before is not None and after is not None:
                # How this hand's decisions found their strategy.
                row["lookups"] = {k: after[k] - before.get(k, 0) for k in after}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if written % 50 == 0 or written == target:
                print(f"hands={written} raw={won / BB / written * 100:+.1f} bb/100 "
                      f"baseline={base / BB / written * 100:+.1f} bb/100", flush=True)
    return token


def read_rows(path):
    """Read a JSONL log, tolerating a missing file and blank lines."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bb100(values):
    """Mean and standard error of per-hand chip values, in BB/100."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = statistics.fmean(values) * 100.0 / BB
    if n == 1:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(n) * 100.0 / BB


def summarize(rows):
    """Raw and Slumbot-baseline BB/100 with standard errors, overall and by seat."""
    raw_bb100, raw_se = _bb100([row["winnings"] for row in rows])
    baseline = [row.get("baseline") or 0 for row in rows]
    baseline_bb100, baseline_se = _bb100(baseline)
    sb = [row.get("baseline") or 0 for row in rows if row["pos"] == 1]
    bb = [row.get("baseline") or 0 for row in rows if row["pos"] == 0]
    return {
        "hands": len(rows),
        "hands_bb": sum(1 for row in rows if row["pos"] == 0),
        "hands_sb": sum(1 for row in rows if row["pos"] == 1),
        "raw_bb100": raw_bb100,
        "raw_se": raw_se,
        "baseline_bb100": baseline_bb100,
        "baseline_se": baseline_se,
        "sb_baseline_bb100": _bb100(sb)[0],
        "bb_baseline_bb100": _bb100(bb)[0],
        "lookups": _sum_lookups(rows),
    }


def _sum_lookups(rows):
    """Total lookup outcomes over the rows that recorded them, or None."""
    total = {}
    for row in rows:
        for k, v in (row.get("lookups") or {}).items():
            total[k] = total.get(k, 0) + v
    return total or None


def size_bin(winnings):
    """Hand-size bin from the absolute result, in BB."""
    bb = abs(winnings) / BB
    if bb < 10:
        return "<10"
    if bb < 40:
        return "10-40"
    if bb < 100:
        return "40-100"
    return "100+"


def _last_token(action):
    toks = TOKEN_RE.findall(action or "")
    return toks[-1] if toks else ""


def _bucket(rows):
    return {"n": len(rows), "baseline_bb": sum(r.get("baseline") or 0 for r in rows) / BB}


def breakdown(rows):
    """Where the score goes: by hand size, 3-bet pots, showdowns, fold-ended.

    Showdown = the river was dealt and the hand did not end in a fold. Fold =
    the last action of the hand was a fold (by either player)."""
    bins = {name: [] for name in SIZE_BINS}
    for row in rows:
        bins[size_bin(row["winnings"])].append(row)
    threebet = [r for r in rows
                if parse_action(r["action"], street_of(r.get("board") or []))[2][0] >= 2]
    showdown = [r for r in rows
                if len(r.get("board") or []) == 5 and _last_token(r["action"]) != "f"]
    fold = [r for r in rows if _last_token(r["action"]) == "f"]
    return {
        "size_bins": {name: _bucket(bins[name]) for name in SIZE_BINS},
        "threebet_pots": _bucket(threebet),
        "showdown": _bucket(showdown),
        "fold": _bucket(fold),
    }


def format_report(summary, parts):
    lines = [
        f"hands={summary['hands']}",
        f"raw={summary['raw_bb100']:+.1f} +/- {summary['raw_se']:.1f} bb/100",
        f"baseline={summary['baseline_bb100']:+.1f} +/- {summary['baseline_se']:.1f} bb/100",
        f"  bb={summary['bb_baseline_bb100']:+.1f} bb/100  "
        f"sb={summary['sb_baseline_bb100']:+.1f} bb/100 (baseline by seat)",
        f"seat balance: BB={summary['hands_bb']} SB={summary['hands_sb']}",
    ]
    lookups = summary.get("lookups")
    if lookups:
        n = sum(lookups.values())
        lines.append("lookups: " + "  ".join(
            f"{k}={v} ({100 * v / n:.1f}%)" for k, v in lookups.items()))
    lines += [
        "size bins (BB won or lost):",
    ]
    for name in SIZE_BINS:
        b = parts["size_bins"][name]
        lines.append(f"  {name:>7}: n={b['n']:5d} baseline={b['baseline_bb']:+.1f} bb")
    for name in ("threebet_pots", "showdown", "fold"):
        b = parts[name]
        lines.append(f"  {name}: n={b['n']:5d} baseline={b['baseline_bb']:+.1f} bb")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", help="sixmax checkpoint to play")
    ap.add_argument("--config", default=os.path.join(REPO, "sixmax/configs/hu.toml"),
                    help="training config the checkpoint was built with")
    ap.add_argument("--hands", type=int, default=20,
                    help="TOTAL target hand count including rows already in --log; must be even")
    ap.add_argument("--log", required=True, help="JSONL per-hand log; a run resumes from it")
    ap.add_argument("--summary", action="store_true",
                    help="print the summary and breakdown of --log and exit")
    args = ap.parse_args(argv)

    if args.summary:
        if not os.path.exists(args.log):
            raise SystemExit(f"no log at {args.log}")
        existing = read_rows(args.log)
        print(format_report(summarize(existing), breakdown(existing)))
        return 0

    existing = read_rows(args.log)

    # Seat balance: sessions start in the big blind, so an odd total would skew
    # the score toward one seat. Refuse rather than silently skew it.
    if args.hands % 2:
        raise SystemExit(f"--hands must be even (got {args.hands})")
    if len(existing) % 2:
        raise SystemExit(
            f"--log holds {len(existing)} hands, which is not an even number; "
            f"refusing to resume with unbalanced seats")
    if len(existing) >= args.hands:
        print(f"--log already holds {len(existing)} hands (target {args.hands}); "
              f"nothing to do", flush=True)
        return 0
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required to play hands")

    agent = SixmaxAgent(args.checkpoint, config_toml=args.config)
    token = login()
    run_hands(post, agent, token, args.log, args.hands, existing)
    final = read_rows(args.log)
    print(format_report(summarize(final), breakdown(final)))
    return 0


if __name__ == "__main__":
    main()
