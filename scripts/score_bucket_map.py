#!/usr/bin/env python3
"""Score candidate bucket maps offline, before any training (roadmap spec,
Phase 3 "3-pre").

Each map is scored per street by its within-bucket spread against fixed,
map-independent features from the equity store: river equity on the river,
and the earth mover's distance between each hand's next-street equity
histogram and its bucket's mean histogram on the flop and turn. Lower is
better; `share` is the spread left relative to a single bucket.

Each map is compared, street by street, with the candidate that has the next
smaller bucket count on that street. A step's efficiency is the share of the
street's one-bucket spread it removes, per extra infoset; extra infosets are
the street's betting contexts times the extra buckets (measured on the C0
checkpoint: flop 73, turn 242, river 753). Normalising by the one-bucket
spread puts river (equity deviation) and flop/turn (EMD) on one scale, so a
river bucket has to earn about 10x what a flop bucket does. A step is kept
when its efficiency is at least `--fraction` of the best step's; a map is
kept if any of its steps is.

Usage:
    uv run python scripts/score_bucket_map.py \\
        sixmax/tables/map_pct_50_50_20.bin sixmax/tables/map_pct_200_200_20.bin \\
        --json docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/map_scores.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_bucket_map import _force_load_sixmax, _repo_root  # noqa: E402

_STREETS = {1: "flop", 2: "turn", 3: "river"}


def keeps(efficiency, best_efficiency, fraction):
    """Is a step's spread removed per extra infoset close enough to the best
    step's to be worth training? A step that removes nothing never is."""
    return efficiency > 0.0 and efficiency >= fraction * best_efficiency


def steps(rows, contexts):
    """Every (map, street) where the map adds buckets, compared with the
    candidate holding the next smaller count on that street."""
    by_street = {}
    for r in rows:
        by_street.setdefault(r["street"], []).append(r)
    out = []
    for r in rows:
        smaller = [o for o in by_street[r["street"]]
                   if o["buckets"] < r["buckets"]]
        if not smaller:
            continue
        ref = max(smaller, key=lambda o: o["buckets"])
        extra = contexts[r["street"] - 1] * (r["buckets"] - ref["buckets"])
        removed = ((ref["error"] - r["error"]) / r["baseline"]
                   if r["baseline"] else 0.0)
        out.append({
            "map": r["map"], "street": _STREETS[r["street"]],
            "vs": ref["map"], "extra_infosets": extra,
            "gain": 1.0 - r["error"] / ref["error"] if ref["error"] else 0.0,
            "efficiency": removed / extra})
    return out


def verdicts(rows, contexts, fraction):
    """Per map: its steps, each with a keep/drop call against the best
    step's efficiency, and the map's call. Maps that add buckets on no street
    (the coarsest, or the reference) are always kept."""
    found = steps(rows, contexts)
    best = max((s["efficiency"] for s in found), default=0.0)
    result = {}
    for s in found:
        entry = result.setdefault(s["map"], {"streets": [], "keep": False})
        s = dict(s, keep=keeps(s["efficiency"], best, fraction),
                 relative=s["efficiency"] / best if best > 0 else 0.0)
        del s["map"]
        entry["streets"].append(s)
        entry["keep"] = entry["keep"] or s["keep"]
    for r in rows:
        result.setdefault(r["map"], {"streets": [], "keep": True})
    return result


def _contexts(text):
    parts = text.split(",")
    if len(parts) != 3 or not all(p.strip().isdigit() and int(p) > 0
                                  for p in parts):
        raise argparse.ArgumentTypeError(
            "contexts must be flop,turn,river (three positive integers)")
    return [int(p) for p in parts]


def parse_args(argv=None):
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="+", help="SIXBM001 maps to score")
    p.add_argument("--equity",
                   default=os.path.join(root, "sixmax", "tables", "equity_v1.bin"),
                   help="SIXEQ001 store (default: sixmax/tables/equity_v1.bin)")
    p.add_argument("--contexts", type=_contexts, default=[73, 242, 753],
                   help="betting contexts per street, flop,turn,river "
                        "(default: 73,242,753, C0's layout; re-measure on "
                        "the Phase 2 winner)")
    p.add_argument("--fraction", type=float, default=0.5,
                   help="share of the best step's efficiency a step must "
                        "reach to be kept (default: 0.5)")
    p.add_argument("--threads", type=int, default=0,
                   help="worker threads (0 = hardware concurrency)")
    p.add_argument("--json", help="also write rows and verdicts here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sixmax = _force_load_sixmax(_repo_root())
    started = time.time()
    rows = sixmax.score_bucket_maps(args.equity, list(args.maps),
                                    num_threads=args.threads)
    elapsed = time.time() - started
    calls = verdicts(rows, args.contexts, args.fraction)

    print(f"{'map':<40} {'street':<6} {'buckets':>7} {'spread':>9} {'share':>7}")
    for r in rows:
        share = r["error"] / r["baseline"] if r["baseline"] else 0.0
        print(f"{os.path.basename(r['map']):<40} {_STREETS[r['street']]:<6} "
              f"{r['buckets']:>7} {r['error']:>9.5f} {share:>7.1%}")
    print()
    for path, call in calls.items():
        detail = "; ".join(
            f"{s['street']} {s['gain']:+.1%} for {s['extra_infosets']:,} "
            f"infosets ({s['relative']:.0%} of best) vs "
            f"{os.path.basename(s['vs'])}"
            for s in call["streets"]) or "reference"
        print(f"{'KEEP' if call['keep'] else 'DROP'} "
              f"{os.path.basename(path)}: {detail}")
    print(f"\nscored {len(args.maps)} maps in {elapsed:.0f}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"contexts": args.contexts, "fraction": args.fraction,
                       "rows": rows,
                       "verdicts": calls}, f, indent=2)


if __name__ == "__main__":
    main()
