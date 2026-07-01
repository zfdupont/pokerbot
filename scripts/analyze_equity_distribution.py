"""
Analyze preflop equity distribution across all 1,326 unique hole-card combinations.

Enumerates all C(52,2) combos, computes Monte Carlo equity for each, then shows:
  - Equity histogram vs. current 8-bucket boundaries
  - Bucket coverage under the current equal-width scheme
  - Suggested percentile-based bucket boundaries for 4, 6, and 8 buckets

Usage:
    uv run python scripts/analyze_equity_distribution.py [--samples 200] [--buckets 8]
"""
import sys
import os
import argparse
import random
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.card import Card
from models.enums import Suit
from cfr.abstraction import _equity, NUM_PREFLOP_BUCKETS, MONTE_CARLO_SAMPLES


def all_hole_combos():
    deck = [Card(r, s) for r in range(2, 15) for s in list(Suit)]
    return list(combinations(deck, 2))


def equity_histogram(equities, n_bins=20):
    width = 1.0 / n_bins
    bins = [0] * n_bins
    for eq in equities:
        idx = min(int(eq / width), n_bins - 1)
        bins[idx] += 1
    total = len(equities)
    print(f"\n  Equity distribution ({len(equities):,} combos, {n_bins} bins)")
    print(f"  {'Equity range':>16}  {'Count':>6}  {'Pct':>5}  {'Bar'}")
    print(f"  {'-'*60}")
    for i, count in enumerate(bins):
        lo = i * width
        hi = lo + width
        pct = count / total * 100
        bar = "#" * int(pct * 1.5)
        label = f"{lo:.2f}-{hi:.2f}"
        print(f"  {label:>16}  {count:>6,}  {pct:>4.1f}%  {bar}")


def current_bucket_coverage(equities, n_buckets):
    from collections import Counter
    bucket_counts = Counter()
    for eq in equities:
        b = min(int((1.0 - eq) * n_buckets), n_buckets - 1)
        bucket_counts[b] += 1
    total = len(equities)
    print(f"\n  Current equal-width bucket coverage ({n_buckets} buckets)")
    print(f"  bucket = int((1 - equity) * {n_buckets}), capped at {n_buckets - 1}")
    print(f"  {'Bucket':>8}  {'Equity range':>16}  {'Count':>6}  {'Pct':>5}  {'Bar'}")
    print(f"  {'-'*65}")
    for b in range(n_buckets):
        lo = 1.0 - (b + 1) / n_buckets
        hi = 1.0 - b / n_buckets
        count = bucket_counts[b]
        pct = count / total * 100
        bar = "#" * int(pct * 1.5)
        status = "" if count > 0 else "  ← EMPTY"
        print(f"  {b:>8}  {lo:.3f}-{hi:.3f}  {count:>6,}  {pct:>4.1f}%  {bar}{status}")


def percentile_boundaries(equities, n_buckets):
    sorted_eq = sorted(equities)
    n = len(sorted_eq)
    cuts = [sorted_eq[int(i * n / n_buckets)] for i in range(1, n_buckets)]
    print(f"\n  Suggested percentile-based boundaries ({n_buckets} equal-frequency buckets)")
    print(f"  Each bucket would cover ~{100/n_buckets:.1f}% of hands ({n // n_buckets} combos)")
    print(f"  Boundaries (equity thresholds): {[f'{c:.3f}' for c in cuts]}")
    print(f"\n  In code:")
    thresholds = [f"{c:.3f}" for c in reversed(cuts)]
    print(f"  PREFLOP_THRESHOLDS = [{', '.join(thresholds)}]  # descending")
    print(f"  bucket = sum(1 for t in PREFLOP_THRESHOLDS if equity < t)")
    return cuts


def main():
    parser = argparse.ArgumentParser(description="Analyze preflop equity distribution")
    parser.add_argument("--samples", type=int, default=MONTE_CARLO_SAMPLES,
                        help=f"MC rollouts per hand (default: {MONTE_CARLO_SAMPLES})")
    parser.add_argument("--buckets", type=int, default=NUM_PREFLOP_BUCKETS,
                        help=f"Number of buckets to analyze (default: {NUM_PREFLOP_BUCKETS})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    combos = all_hole_combos()
    print(f"Computing equity for {len(combos):,} unique hole-card combinations")
    print(f"MC samples per hand: {args.samples}  (modify MONTE_CARLO_SAMPLES in abstraction.py to change default)")

    # Temporarily override sample count if requested
    import cfr.abstraction as abst
    original_samples = abst.MONTE_CARLO_SAMPLES
    abst.MONTE_CARLO_SAMPLES = args.samples

    equities = []
    from tqdm import tqdm
    for c1, c2 in tqdm(combos, desc="Estimating equity", unit="combo"):
        eq = _equity([c1, c2], [])
        equities.append(eq)

    abst.MONTE_CARLO_SAMPLES = original_samples

    print(f"\n  Min equity : {min(equities):.4f}")
    print(f"  Max equity : {max(equities):.4f}")
    print(f"  Mean equity: {sum(equities)/len(equities):.4f}")

    equity_histogram(equities)
    current_bucket_coverage(equities, args.buckets)

    print("\n  --- Percentile boundary suggestions ---")
    for nb in [4, 6, args.buckets]:
        percentile_boundaries(equities, nb)


if __name__ == "__main__":
    main()
