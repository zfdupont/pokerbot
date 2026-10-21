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


def run_h2h(a, b, hands, seeds, bb, stack, on_seed=None):
    """A vs B duplicate match over `seeds`, summarised with paired stats.

    Returns mean/sd/se/t (one paired score per seed) plus per_seed. `on_seed`,
    if given, is called as on_seed(seed, bb100) after each seed so callers can
    stream progress during long runs. A single seed has no sample spread, so
    paired_stats already gives sd=se=t=0.0 rather than NaN.
    """
    from agents.sixmax_agent import SixmaxAgent
    from scripts.eval_hu_sanity import run_duplicate_match
    from scripts.eval_search_ab import paired_stats

    scores = []
    for seed in seeds:
        bb100 = run_duplicate_match(lambda _: SixmaxAgent(a),
                                    lambda _: SixmaxAgent(b),
                                    hands, seed, bb, stack)
        scores.append(bb100)
        if on_seed is not None:
            on_seed(seed, bb100)

    stats = paired_stats(scores)
    return {"mean": stats["mean"], "sd": stats["sd"], "se": stats["se"],
            "t": stats["t"], "per_seed": scores}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="candidate checkpoint")
    p.add_argument("--b", required=True, help="incumbent checkpoint")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--hands", type=int, default=20000)
    p.add_argument("--bb", type=int, default=2)
    p.add_argument("--stack", type=int, default=200)
    args = p.parse_args()

    from scripts.eval_search_ab import verdict

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"A = {args.a}\nB = {args.b}")
    print(f"{'seed':>4} {'A vs B':>10}")

    def show(seed, bb100):
        print(f"{seed:>4} {bb100:>+10.2f}", flush=True)

    stats = run_h2h(args.a, args.b, args.hands, seeds, args.bb, args.stack,
                    on_seed=show)
    print(f"\nn={len(stats['per_seed'])} seeds x {args.hands} duplicate deals")
    print(f"A - B: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.1f}  t={stats['t']:.2f}")
    print(f"VERDICT: {'promote A' if verdict(stats) == 'promote' else 'keep B'}")


if __name__ == "__main__":
    main()
