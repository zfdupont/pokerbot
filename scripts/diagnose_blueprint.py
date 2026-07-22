"""Blueprint plateau autopsy — offline diagnostic over checkpoint snapshots.

Reads the visit-weight (strategy_sum L1) and regret L1 that sixmax.dump_infosets
recovers, then decides whether the flat BB/100 curve is undertraining (scale) or
a structural ceiling (redesign). Pure metric functions below are stdlib-only and
unit-tested; checkpoint loading + the vocab-dependent probe resolution live in
main() so importing this module never triggers a Buck2 build.

See docs/superpowers/specs/2026-07-22-blueprint-plateau-autopsy-design.md.
"""
import math


def entropy_bits(probs):
    """Shannon entropy (bits) over the support of a probability vector."""
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def support_size(probs, eps=1e-12):
    """Number of actions with non-negligible probability mass."""
    return sum(1 for p in probs if p > eps)


def gini(weights):
    """Gini coefficient of a non-negative weight vector (0 = equal)."""
    xs = sorted(weights)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0.0:
        return 0.0
    s = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * s) / (n * total) - (n + 1.0) / n


def weighted_mean(values, weights):
    tw = sum(weights)
    if tw == 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / tw


def decode_key(key):
    """Unpack the abstract infoset key per abstract_key.h's documented layout:
    card 0-7, street 8-9, raises 10-17 (2 bits/street), pot 18-19,
    live 20-22, after 23-25."""
    return {
        "card": key & 0xFF,
        "street": (key >> 8) & 0x3,
        "raises": [(key >> (10 + 2 * st)) & 0x3 for st in range(4)],
        "pot": (key >> 18) & 0x3,
        "live": (key >> 20) & 0x7,
        "after": (key >> 23) & 0x7,
    }


def top_tier(records, frac):
    """The top `frac` fraction of records by strategy_sum mass (index 2),
    at least one element."""
    ordered = sorted(records, key=lambda r: r[2], reverse=True)
    k = max(1, int(len(ordered) * frac))
    return ordered[:k]


def probe_policy(records, card_id, street, total_raises):
    """Mass-weighted mean policy over every real infoset matching a probe
    predicate (card bucket, street, and total raises-so-far). Returns
    (mean_probs, total_mass); ([], 0.0) when no infoset matches."""
    acc = None
    total = 0.0
    for key, probs, mass, _reg in records:
        d = decode_key(key)
        if d["card"] != card_id or d["street"] != street:
            continue
        if total_raises is not None and sum(d["raises"]) != total_raises:
            continue
        if mass <= 0.0:
            continue
        if acc is None:
            acc = [0.0] * len(probs)
        for i, p in enumerate(probs):
            acc[i] += p * mass
        total += mass
    if acc is None or total == 0.0:
        return [], 0.0
    return [a / total for a in acc], total
