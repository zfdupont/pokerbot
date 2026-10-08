#!/usr/bin/env python3
"""Paired search-on vs search-off A/B in the HU engine.

Both arms play identical seeds; only the [search] section differs. Reports the
per-seed paired difference and the paired t statistic, then a VERDICT against
the deployment gate (mean > 0 AND |t| >= 2), matching
patterns/eval-checkpoint-ab.md.

The on-arm forces search on via SIXMAX_SEARCH_ENABLED so the comparison works
even when the checked-in [search].enabled is false (the deployment default).
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENV_KEY = "SIXMAX_SEARCH_ENABLED"


def paired_stats(diffs):
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / n**0.5 if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": (mean / se) if se > 0 else 0.0}


def verdict(stats, min_t: float = 2.0) -> str:
    return ("promote" if stats["mean"] > 0 and stats["t"] >= min_t
            else "keep-blueprint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="sixmax/checkpoints/blueprint.bin")
    p.add_argument("--tabular", default="cfr/checkpoints/checkpoint_09040000.pkl")
    p.add_argument("--search-toml", default="sixmax/configs/default.toml")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--hands", type=int, default=20000)
    p.add_argument("--bb", type=int, default=2)
    p.add_argument("--stack", type=int, default=200)
    args = p.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    from agents.cfr_agent import CFRAgent
    from scripts.eval_hu_sanity import run_duplicate_match

    seeds = [int(s) for s in args.seeds.split(",")]
    diffs = []
    prev = os.environ.get(_ENV_KEY)
    print(f"{'seed':>4} {'search':>10} {'blueprint':>10} {'diff':>10}")
    try:
        for seed in seeds:
            os.environ[_ENV_KEY] = "true"
            on = run_duplicate_match(
                lambda _: SixmaxAgent(args.checkpoint,
                                      search_toml=args.search_toml),
                lambda _: CFRAgent(args.tabular), args.hands, seed,
                args.bb, args.stack)
            if prev is None:
                os.environ.pop(_ENV_KEY, None)
            else:
                os.environ[_ENV_KEY] = prev
            off = run_duplicate_match(
                lambda _: SixmaxAgent(args.checkpoint),
                lambda _: CFRAgent(args.tabular), args.hands, seed,
                args.bb, args.stack)
            diffs.append(on - off)
            print(f"{seed:>4} {on:>+10.2f} {off:>+10.2f} {on - off:>+10.2f}")
    finally:
        if prev is None:
            os.environ.pop(_ENV_KEY, None)
        else:
            os.environ[_ENV_KEY] = prev

    stats = paired_stats(diffs)
    print(f"\nn={stats['n']} seeds x {args.hands} hands")
    print(f"paired diff: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.1f}  t={stats['t']:.2f}")
    print(f"VERDICT: {verdict(stats)}")


if __name__ == "__main__":
    main()
