"""
Profile external_sample (training hotpath) to find where time is spent.

Usage:
    uv run python scripts/profile_training.py [--iterations 500]

Runs external_sample for both players N times (no exploitability overhead).
"""
import sys
import os
import argparse
import cProfile
import pstats
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cfr.regret_table import RegretTable
from cfr.abstract_state import deal_heads_up
try:
    from cfr.mccfr_cy import external_sample_cy as _sample
    _impl = "Cython"
except ImportError:
    from cfr.mccfr import external_sample as _sample
    _impl = "Python"


def main():
    parser = argparse.ArgumentParser(description="Profile external_sample training loop")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of deal+_sample×2 iterations (default 500)")
    args = parser.parse_args()

    table = RegretTable()
    print(f"Profiling {args.iterations} training iterations ({_impl} external_sample × 2 per iter)...")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.iterations):
        state = deal_heads_up()
        _sample(state, traversing_player=0, table=table)
        _sample(state, traversing_player=1, table=table)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(25)
    print(stream.getvalue())

    stream2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=stream2)
    ps2.strip_dirs()
    ps2.sort_stats("tottime")
    ps2.print_stats(15)
    print("--- Sorted by tottime (self-time) ---")
    print(stream2.getvalue())


if __name__ == "__main__":
    main()
