#!/usr/bin/env python3
"""LBR over every snapshot of one or more arms, for learning curves.

Each snapshot gets one LBR run (turn+river, the paper's "fcpa 3-4") on the
same decks: decks depend only on (seed, deck index), so every point in the
sweep is played on identical cards and any two can later be compared paired
from their saved per-deck values.

One JSON line per snapshot is appended to --out (resumable: a snapshot whose
checkpoint, decks and seed are already recorded is skipped). Per-deck values
(BB/100, float64) go to --per-deck-dir as <checkpoint stem>_d<decks>_s<seed>.npy;
they are regenerable from the checkpoint, so keep them beside the checkpoints
(gitignored) rather than in git.

Usage:
    uv run python scripts/lbr_sweep.py \\
        --arm c0=sixmax/configs/hu200.toml:'sixmax/checkpoints/hu200_c0_*.bin' \\
        --arm c1=sixmax/configs/hu200_c1.toml:'sixmax/checkpoints/hu200_c1_*.bin' \\
        --out docs/.../ablations/lbr_sweep.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.arm_plateau import _SNAPSHOT_RE  # noqa: E402


def parse_arm(text):
    """`name=config:glob` -> (name, config, glob)."""
    name, sep, rest = text.partition("=")
    config, sep2, pattern = rest.partition(":")
    if not (sep and sep2 and name and config and pattern):
        raise argparse.ArgumentTypeError(
            "arm must be name=config:glob, got %r" % text)
    return name, config, pattern


def snapshots(pattern):
    """(iterations, path) for `<base>_<digits>.bin` matches, oldest first."""
    out = []
    for path in glob.glob(pattern):
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        if m is not None:
            out.append((int(m.group(1)), path))
    return sorted(out)


def done_keys(out_path):
    """(checkpoint basename, decks, seed) already recorded in --out."""
    keys = set()
    if not os.path.exists(out_path):
        return keys
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                keys.add((r["checkpoint"], r["decks"], r["seed"]))
    return keys


def per_deck_path(per_deck_dir, checkpoint, decks, seed):
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    return os.path.join(per_deck_dir, "%s_d%d_s%d.npy" % (stem, decks, seed))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", type=parse_arm, action="append", required=True,
                   metavar="NAME=CONFIG:GLOB")
    p.add_argument("--decks", type=int, default=100000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", required=True, help="JSON-lines summary file")
    p.add_argument("--per-deck-dir",
                   default=os.path.join("sixmax", "checkpoints",
                                        "lbr_per_deck"))
    args = p.parse_args(argv)

    import numpy as np
    from scripts.eval_sixmax_lbr import run_lbr

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.per_deck_dir, exist_ok=True)
    done = done_keys(args.out)
    for name, config, pattern in args.arm:
        snaps = snapshots(pattern)
        if not snaps:
            print("warning: arm %s matched no snapshots (%s)" % (name, pattern),
                  file=sys.stderr)
        for iters, path in snaps:
            key = (os.path.basename(path), args.decks, args.seed)
            if key in done:
                continue
            start = time.time()
            rep = run_lbr(path, config, args.decks, args.seed,
                          threads=args.threads)
            seconds = time.time() - start
            deck_file = per_deck_path(args.per_deck_dir, path, args.decks,
                                      args.seed)
            np.save(deck_file, np.asarray(rep["per_deck"], dtype=np.float64))
            row = {
                "arm": name, "config": config,
                "checkpoint": os.path.basename(path), "iterations": iters,
                "decks": args.decks, "seed": args.seed,
                "act_streets": [2, 3],
                "bb100": float(rep["bb100"]), "stderr": float(rep["stderr"]),
                "seat0_bb100": float(rep["seat_bb100"][0]),
                "seat1_bb100": float(rep["seat_bb100"][1]),
                "stack_bb": rep["stack_bb"],
                "opp_unseen": int(rep["opp_unseen"]),
                "opp_decisions": int(rep["opp_decisions"]),
                "per_deck_file": deck_file, "seconds": round(seconds, 1),
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            done.add(key)
            print("%-4s %11d  %+8.2f +- %5.2f BB/100  (%.0f s)" % (
                name, iters, row["bb100"], 1.96 * row["stderr"], seconds),
                flush=True)


if __name__ == "__main__":
    main()
