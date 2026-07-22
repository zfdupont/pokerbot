#!/usr/bin/env python3
"""Six-max blueprint-vs-baseline eval harness.

Sits the six-max blueprint (via the SixmaxAgent deploy bridge) in one seat of a
full-ring table against n-1 fixed baseline agents (default PotOddsAgent),
seat-rotated over identical decks, and sweeps a series of checkpoints to draw
the "blueprint vs baseline, BB/100 per checkpoint" curve — with a standard error
so real training gains are distinguishable from variance.

Why the live engine: PotOddsAgent (and the other baselines) read
game/poker.py's GameState and return live Action enums, so they only run inside
that engine. This is a six-max generalization of eval_hu_sanity.py — the
blueprint plays through agents/sixmax_agent.py, the only sanctioned bridge.

Variance reduction: only the deck shuffle consumes the global `random` stream
(agents draw from their own RNG), so re-seeding deals a byte-identical deck. The
hero plays every rotation of that deck once, occupying all n positions relative
to a fixed button, which cancels "which cards did the hero get" luck — the
dominant noise source at a six-max table. Each deck is one independent sample
for the standard error.

Usage:
    uv run python scripts/eval_sixmax_baseline.py \
        --checkpoints "sixmax/checkpoints/blueprint_*.bin" \
        [--villain potodds] [--hands 500] [--seed 1] [--players 6] \
        [--bb 2] [--stack 200] [--csv curve.csv]
"""
import argparse
import glob
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player import Player  # noqa: E402
from game.poker import PokerGame  # noqa: E402

from agents.potodds_agent import PotOddsAgent  # noqa: E402
from agents.simple_agent import SimpleAgent  # noqa: E402
from agents.position_agent import PositionBasedAgent  # noqa: E402
from agents.hand_strength_agent import HandStrengthAgent  # noqa: E402


# Villain registry: name -> zero-arg factory building a fresh baseline agent.
# All are stateless (read-only over player + game_state), so a single instance
# can safely fill every villain seat and be reused across every checkpoint.
VILLAINS = {
    "potodds": PotOddsAgent,
    "simple": SimpleAgent,
    "position": PositionBasedAgent,
    "hand_strength": HandStrengthAgent,
}


def play_deck(hero_agent, villain_agent, seed, n, small_blind, stack):
    """Hero's total chip delta over one deck played n times, hero rotating
    through every seat.

    Re-seeding the global RNG before each rotation deals the identical deck
    (agents use their own RNG, so only the shuffle consumes global `random`).
    A fresh PokerGame resets button_pos to 0, so across the n rotations the
    hero occupies all n positions relative to a fixed button and sees each of
    the n dealt hands exactly once.
    """
    total = 0.0
    for hero_seat in range(n):
        random.seed(seed)
        players = [
            Player(f"p{i}", stack,
                   agent=(hero_agent if i == hero_seat else villain_agent))
            for i in range(n)
        ]
        PokerGame(players, small_blind=small_blind).play_hand()
        total += players[hero_seat].stack - stack
    return total


def block_stats(per_deck_bb_per_hand):
    """(bb100, stderr) from per-deck mean-BB/hand samples.

    Each deck is one independent sample (its n mirrored hands are correlated,
    so they are collapsed into a single block mean). bb100 is 100x the grand
    mean; stderr is the sample-std (ddof=1) of the per-deck means over sqrt of
    the deck count. A single deck has undefined sample std -> stderr 0.0.
    """
    m = len(per_deck_bb_per_hand)
    if m == 0:
        return 0.0, 0.0
    mean = sum(per_deck_bb_per_hand) / m
    bb100 = 100.0 * mean
    if m < 2:
        return bb100, 0.0
    var = sum((x - mean) ** 2 for x in per_deck_bb_per_hand) / (m - 1)
    stderr = 100.0 * math.sqrt(var) / math.sqrt(m)
    return bb100, stderr


def run_match(hero_agent, villain_agent, hands, seed, n, small_blind, stack):
    """Play `hands` decks, hero seat-rotated within each. Returns
    (bb100, stderr, n_hands). big_blind = 2*small_blind sets the BB frame."""
    big_blind = small_blind * 2
    samples = []
    for h in range(hands):
        s = seed * 1_000_003 + h
        delta = play_deck(hero_agent, villain_agent, s, n, small_blind, stack)
        samples.append(delta / (n * big_blind))  # mean BB/hand for this deck
    bb100, stderr = block_stats(samples)
    return bb100, stderr, hands * n


def parse_iters(path):
    """Iteration count from a checkpoint filename (last integer group), else
    None. e.g. blueprint_1000000.bin -> 1000000; best_checkpoint.bin -> None."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def expand_checkpoints(single, multi):
    """Flatten --checkpoint + --checkpoints into a de-duplicated, glob-expanded
    ordered list. Supports literal globs so quoted patterns still work when the
    shell does not expand them."""
    raw = ([single] if single else []) + list(multi or [])
    paths = []
    for p in raw:
        matched = sorted(glob.glob(p))
        paths.extend(matched if matched else [p])
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_villain(name):
    try:
        return VILLAINS[name]()
    except KeyError:
        raise SystemExit(
            f"unknown villain '{name}'; choose from "
            f"{', '.join(sorted(VILLAINS))}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Six-max blueprint vs baseline eval (BB/100 curve)")
    parser.add_argument("--checkpoint", default=None, help="single .bin path")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="multiple .bin paths (glob-friendly)")
    parser.add_argument("--villain", default="potodds",
                        help=f"baseline: {', '.join(sorted(VILLAINS))}")
    parser.add_argument("--hands", type=int, default=500,
                        help="number of decks (each played once per seat)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--bb", type=int, default=2,
                        help="big blind in chips (small_blind = bb // 2)")
    parser.add_argument("--stack", type=int, default=None,
                        help="starting stack (default 100*bb = 100 BB)")
    parser.add_argument("--config", default=os.path.join(
        root, "sixmax", "configs", "default.toml"))
    parser.add_argument("--csv", default=None, help="optional CSV output path")
    args = parser.parse_args()

    paths = expand_checkpoints(args.checkpoint, args.checkpoints)
    if not paths:
        raise SystemExit("provide --checkpoint or --checkpoints")
    if args.bb < 2:
        raise SystemExit("--bb must be >= 2 (small_blind = bb // 2 >= 1)")
    if args.players < 2:
        raise SystemExit("--players must be >= 2")

    n = args.players
    small_blind = args.bb // 2
    stack = args.stack if args.stack is not None else 100 * args.bb

    # Import here so the extension build only happens for a real run — the pure
    # functions above stay importable (and unit-testable) without buck2.
    from agents.sixmax_agent import SixmaxAgent

    villain = build_villain(args.villain)

    rows = []
    for path in paths:
        hero = SixmaxAgent(path, config_toml=args.config)
        bb100, stderr, n_hands = run_match(
            hero, villain, args.hands, args.seed, n, small_blind, stack)
        rows.append((parse_iters(path), path, bb100, stderr, n_hands))

    rows.sort(key=lambda r: (r[0] is None, r[0] or 0))

    print(f"\nBlueprint vs {args.villain} — {n}-max, {args.hands} decks/ckpt, "
          f"seed {args.seed}\n")
    print(f"{'iters':>12}  {'BB/100':>10}  {'95% CI':>9}  {'hands':>7}  "
          f"checkpoint")
    for iters, path, bb100, stderr, n_hands in rows:
        it = "-" if iters is None else str(iters)
        ci = f"±{1.96 * stderr:.2f}"
        print(f"{it:>12}  {bb100:>+10.2f}  {ci:>9}  {n_hands:>7}  "
              f"{os.path.basename(path)}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("iters,bb100,stderr,n_hands,checkpoint\n")
            for iters, path, bb100, stderr, n_hands in rows:
                f.write(f"{iters if iters is not None else ''},{bb100:.4f},"
                        f"{stderr:.4f},{n_hands},{os.path.basename(path)}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
