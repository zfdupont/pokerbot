#!/usr/bin/env python3
"""How much does the price faced vary inside one infoset key? ("leak 2")

The sixmax abstraction key (pack_abstract_key) carries card bucket, street,
raise counts, a 4-level pot bucket, live and after, but no bet size. So facing
a 0.33-pot bet and a 1.5-pot overbet can share a key and one strategy. This
plays blueprint self-play in the live engine, logs every postflop decision
that faces a bet with its key and required call equity
(to_call / (pot + to_call)), and reports how much of that price variation the
key explains. Keys are grouped with the card bits (0-7) masked off: price is
set by the betting state, and the card bucket would only split the same
betting state into many tiny groups that understate the spread.

Usage:
    uv run python scripts/diagnose_key_aliasing.py \\
        --checkpoint sixmax/checkpoints/hu_blueprint.bin --deals 25000
"""
import argparse
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def required_equity(to_call_bb, pot_bb):
    return to_call_bb / (pot_bb + to_call_bb)


def aliasing_report(records, wide=0.10):
    """records: [(key, req_equity)]. Returns decision-weighted stats on how
    much required equity varies within a key."""
    by_key = collections.defaultdict(list)
    for k, q in records:
        by_key[k].append(q)
    allq = [q for _, q in records]
    total_var = st.pvariance(allq) if len(allq) > 1 else 0.0
    within = sum(st.pvariance(v) * len(v) for v in by_key.values()
                 if len(v) > 1) / len(allq)
    in_wide = sum(len(v) for v in by_key.values()
                  if max(v) - min(v) > wide)
    spreads = sorted((max(v) - min(v), len(v), k) for k, v in by_key.items())
    return {
        "decisions": len(allq),
        "keys": len(by_key),
        "r2_key": 1.0 - within / total_var if total_var > 0 else 1.0,
        "share_in_wide_keys": in_wide / len(allq),
        "worst": spreads[-5:][::-1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="sixmax/checkpoints/hu_blueprint.bin")
    p.add_argument("--deals", type=int, default=25000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bb", type=int, default=2)
    p.add_argument("--stack", type=int, default=200)
    args = p.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    from scripts.eval_hu_sanity import run_duplicate_match
    import sixmax

    records = []

    def factory(_):
        agent = SixmaxAgent(args.checkpoint)
        deploy = agent._deploy
        inner = deploy.decide

        def logged(**kw):
            if kw["street"] > 0 and kw["to_call_bb"] > 0:
                card = deploy._card_bucket(kw["hole"], kw["board"], kw["street"])
                key = sixmax.pack_abstract_key(
                    card, kw["street"], list(kw["raises_per_street"]),
                    kw["pot_bb"], kw["live"], kw["after"])
                records.append((key & ~0xFF, required_equity(kw["to_call_bb"],
                                                     kw["pot_bb"])))
            return inner(**kw)

        deploy.decide = logged
        return agent

    run_duplicate_match(factory, factory, args.deals, args.seed,
                        args.bb, args.stack)
    r = aliasing_report(records)
    print(f"facing-bet postflop decisions: {r['decisions']}  betting-state keys: {r['keys']}")
    print(f"share of price variance explained by key (R^2): {r['r2_key']:.2f}")
    print(f"decisions in keys whose required equity spans >10 pts: "
          f"{r['share_in_wide_keys']:.0%}")
    print("widest keys (spread, decisions, key):")
    for spread, n, k in r["worst"]:
        print(f"  {spread:.3f}  {n:>6}  {k:#x}")


if __name__ == "__main__":
    main()
