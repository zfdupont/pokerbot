#!/usr/bin/env python3
"""How much does the price faced vary inside one infoset key? ("leak 2")

The sixmax abstraction key carries card bucket, street, raise counts, a pot
bucket (legacy 4-level, or the checkpoint's own PotLayout), live and after,
but no bet size. So facing a 0.33-pot bet and a 1.5-pot overbet can share a
key and one strategy. This plays blueprint self-play in the live engine, logs
every postflop decision that faces a bet with its key and required call equity
(to_call / (pot + to_call)), and reports how much of that price variation the
key explains. Keys come from the checkpoint's own abstraction
(`Abstraction.pack_key`), so a SIXBP003 checkpoint's pot layout is read
correctly. Keys are grouped with the card bits (0-7) masked off: price is
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


# A SIXBP004 key carries the last-raise class (bits 30-41), which this script
# cannot rebuild: it sees the decision but not the action line that produced
# it. Refuse rather than report aliasing over wrong keys.
BET_HISTORY_REFUSAL = (
    "diagnose_key_aliasing: refuses a bet_history checkpoint (this script "
    "rebuilds keys without the last-raise field, so they would not be the "
    "keys the agent queries)")


def decision_key(deploy, *, hole, board, street, raises_per_street, pot_bb,
                 stack_bb, eff_stack_bb, live, after, **_ignored):
    """The infoset key the checkpoint's own abstraction gives this decision.

    Mirrors `SixmaxDeployStrategy.decide` exactly (card bucket, then
    `Abstraction.pack_key`), so a SIXBP003 checkpoint's non-legacy pot index at
    key bits 26-29 is read the same way the strategy reads it. `eff_stack_bb`
    comes from the live-game bridge (`agents/sixmax_agent.py`); for the legacy
    layout it falls back to the hero's all-in stack, as the bridge does. The
    other `decide()` kwargs (`current_bet_bb`, `to_call_bb`, `legal`, `rng`, …)
    are accepted and ignored — the key does not depend on them.
    """
    from agents.sixmax_agent import _is_legacy_pot_layout

    abstraction = deploy._strategy.abstraction()
    if abstraction.bet_history() != "off":
        raise ValueError(BET_HISTORY_REFUSAL)
    if eff_stack_bb is None:
        if not _is_legacy_pot_layout(abstraction.pot_layout()):
            raise ValueError(
                "eff_stack_bb is required for non-legacy pot layouts")
        eff_stack_bb = stack_bb
    card = deploy._card_bucket(hole, board, street)
    return abstraction.pack_key(card, street, list(raises_per_street),
                                pot_bb, eff_stack_bb, live, after)


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
    p.add_argument("--stack", type=int, default=400)  # 200 BB at bb 2
    args = p.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    from scripts.eval_hu_sanity import run_duplicate_match

    records = []

    def factory(_):
        agent = SixmaxAgent(args.checkpoint)
        deploy = agent._deploy
        if deploy.bet_history != "off":
            raise SystemExit(BET_HISTORY_REFUSAL)
        inner = deploy.decide

        def logged(**kw):
            if kw["street"] > 0 and kw["to_call_bb"] > 0:
                key = decision_key(deploy, **kw)
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
