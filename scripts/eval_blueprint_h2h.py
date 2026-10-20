#!/usr/bin/env python3
"""Direct heads-up duplicate match between two sixmax blueprint checkpoints.

Every RNG is pinned per hand (eval_hu_sanity._play_seeded), so each seed's
duplicate score is already the paired difference A - B: a checkpoint against
itself scores exactly 0, and anything nonzero comes from decisions where the
two strategies differ. Reports per-seed BB/100 for A, then mean/sd/se/t over
seeds and a VERDICT against the |t| >= 2 gate (patterns/eval-checkpoint-ab.md).

Usage:
    uv run python scripts/eval_blueprint_h2h.py \\
        --a sixmax/checkpoints/hu_blueprint.bin \\
        --b sixmax/checkpoints/blueprint.bin --hands 20000 --seeds 1,2,3,4,5,6,7,8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="candidate checkpoint")
    p.add_argument("--b", required=True, help="incumbent checkpoint")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--hands", type=int, default=20000)
    p.add_argument("--bb", type=int, default=2)
    p.add_argument("--stack", type=int, default=200)
    args = p.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    from scripts.eval_hu_sanity import run_duplicate_match
    from scripts.eval_search_ab import paired_stats, verdict

    seeds = [int(s) for s in args.seeds.split(",")]
    scores = []
    print(f"A = {args.a}\nB = {args.b}")
    print(f"{'seed':>4} {'A vs B':>10}")
    for seed in seeds:
        bb100 = run_duplicate_match(lambda _: SixmaxAgent(args.a),
                                    lambda _: SixmaxAgent(args.b),
                                    args.hands, seed, args.bb, args.stack)
        scores.append(bb100)
        print(f"{seed:>4} {bb100:>+10.2f}", flush=True)

    stats = paired_stats(scores)
    print(f"\nn={stats['n']} seeds x {args.hands} duplicate deals")
    print(f"A - B: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.1f}  t={stats['t']:.2f}")
    print(f"VERDICT: {'promote A' if verdict(stats) == 'promote' else 'keep B'}")


if __name__ == "__main__":
    main()
