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


# ---------------------------------------------------------------------------
# verdict() — pure stdlib, no sixmax dependency
# ---------------------------------------------------------------------------

def verdict(reports):
    """Classify the plateau from the metric trend across ordered reports.

    reports: list of checkpoint_report dicts ordered by iterations ascending.
    Returns (label, rationale) where label is one of:
      "undertraining" — all three signals trend learning-way (more iters would help)
      "structural"    — no signals moving (model hit a true ceiling)
      "mixed"         — 1–2 signals moving (inconclusive; run more diagnostics)

    Decision rule (relative tolerance tol=0.05):
      d_entropy  = entropy_top[last]  - entropy_top[first]   -> learning if < -tol*scale
      d_regret   = avg_regret_top[last] - avg_regret_top[first] -> learning if < -tol*scale
      d_probe    = mean over probes of (expected_mass[last] - expected_mass[first])
                   -> learning if > +tol*scale
    """
    if len(reports) < 2:
        return ("mixed", "need >=2 checkpoints to assess a trend")

    _TOL = 0.05
    _EPS = 1e-9

    first, last = reports[0], reports[-1]

    def _rel_tol(a, b):
        return _TOL * max(abs(a), abs(b), _EPS)

    # Signal 1: entropy_top falling
    d_entropy = last["entropy_top"] - first["entropy_top"]
    tol_e = _rel_tol(first["entropy_top"], last["entropy_top"])
    entropy_learning = d_entropy < -tol_e

    # Signal 2: avg_regret_top shrinking
    d_regret = last["avg_regret_top"] - first["avg_regret_top"]
    tol_r = _rel_tol(first["avg_regret_top"], last["avg_regret_top"])
    regret_learning = d_regret < -tol_r

    # Signal 3: mean probe expected_mass rising
    probe_learning = False
    if first["probes"] and last["probes"]:
        n_probes = min(len(first["probes"]), len(last["probes"]))
        if n_probes > 0:
            d_probe = sum(
                last["probes"][i]["expected_mass"] - first["probes"][i]["expected_mass"]
                for i in range(n_probes)
            ) / n_probes
            f_probe = sum(first["probes"][i]["expected_mass"] for i in range(n_probes)) / n_probes
            l_probe = sum(last["probes"][i]["expected_mass"] for i in range(n_probes)) / n_probes
            tol_p = _rel_tol(f_probe, l_probe)
            probe_learning = d_probe > tol_p

    learning_count = sum([entropy_learning, regret_learning, probe_learning])

    if learning_count == 3:
        return (
            "undertraining",
            f"entropy_top falling ({d_entropy:+.3f}), avg_regret_top shrinking "
            f"({d_regret:+.3e}), probe expected-mass rising ({'+' if probe_learning else '-'}): "
            "all signals indicate the model is still learning; train longer.",
        )
    elif learning_count == 0:
        return (
            "structural",
            f"entropy_top flat ({d_entropy:+.3f}), avg_regret_top flat "
            f"({d_regret:+.3e}), probe masses flat: well-evidenced infosets not "
            "differentiating despite more iterations — likely a structural ceiling.",
        )
    else:
        moving = []
        not_moving = []
        if entropy_learning:
            moving.append(f"entropy_top falling ({d_entropy:+.3f})")
        else:
            not_moving.append(f"entropy_top flat ({d_entropy:+.3f})")
        if regret_learning:
            moving.append(f"avg_regret_top shrinking ({d_regret:+.3e})")
        else:
            not_moving.append(f"avg_regret_top flat ({d_regret:+.3e})")
        if probe_learning:
            moving.append("probe expected-mass rising")
        else:
            not_moving.append("probe expected-mass flat")
        return (
            "mixed",
            f"Learning signals: {', '.join(moving)}. "
            f"Flat signals: {', '.join(not_moving)}. "
            "Recommend paired-per-deck eval slope as tie-breaker.",
        )


# ---------------------------------------------------------------------------
# Vocab + probe resolution (lazy; requires sixmax extension)
# ---------------------------------------------------------------------------

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def _sixmax():
    """Lazy import so this module stays Buck2-free until main() runs."""
    import agents.sixmax_agent  # noqa: F401  (force-loads the .so)
    import sixmax
    return sixmax


def load_vocab_for(checkpoint_path):
    """Reconstruct the ActionVocab this checkpoint was trained with.

    Strategy: locate the resolved config.toml (sibling of the checkpoint, or
    blueprint.bin.config.toml in the same directory), extract its config_path
    field (the original training config that contains [actions.blueprint]),
    and load the vocab from there.  Falls back to the repo-default.toml when
    config_path is absent."""
    from agents.sixmax_agent import _load_vocab  # sanctioned bridge helper
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    resolved_cfg = checkpoint_path + ".config.toml"
    if not _os.path.exists(resolved_cfg):
        resolved_cfg = _os.path.join(_os.path.dirname(checkpoint_path),
                                     "blueprint.bin.config.toml")

    # The resolved config stores config_path pointing at the original TOML
    # that has [actions.blueprint].
    vocab_toml = None
    if _os.path.exists(resolved_cfg):
        with open(resolved_cfg, "rb") as f:
            resolved = tomllib.load(f)
        vocab_toml = resolved.get("resolved", {}).get("config_path")

    if not vocab_toml or not _os.path.exists(vocab_toml):
        # Fall back to the repo default config
        vocab_toml = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "sixmax", "configs", "default.toml")

    return _load_vocab(vocab_toml, "blueprint")


def resolve_action_roles(vocab):
    """Map vocab indices to fold / passive (check-call) / aggressive
    (bet-raise-allin) roles from AbstractAction.type — the vocab is
    config-defined so indices must never be assumed."""
    sixmax = _sixmax()
    T = sixmax.ActionType
    roles = {"fold": [], "passive": [], "aggressive": []}
    for i in range(vocab.size()):
        t = vocab.at(i).type
        if t == T.Fold:
            roles["fold"].append(i)
        elif t in (T.Check, T.Call):
            roles["passive"].append(i)
        else:  # Bet / AllIn
            roles["aggressive"].append(i)
    return roles


def _role_mass(probs, roles):
    """Fraction of a policy's mass on each role."""
    return {r: sum(probs[i] for i in idxs) for r, idxs in roles.items()}


def checkpoint_report(iterations, records, roles, probes, top_frac=0.05):
    """Compute the plateau-diagnosis metrics for one checkpoint.

    probes: list of (label, card_id, street, total_raises, expected_role)
    where expected_role in {"fold","aggressive"} names the mass that SHOULD
    grow across checkpoints for that spot.
    """
    masses = [r[2] for r in records]
    ent_all = weighted_mean([entropy_bits(r[1]) for r in records], masses)
    tier = top_tier(records, top_frac)
    tier_masses = [r[2] for r in tier]
    ent_top = weighted_mean([entropy_bits(r[1]) for r in tier], tier_masses)
    supp_top = weighted_mean([support_size(r[1]) for r in tier], tier_masses)
    total_mass = sum(masses) or 1.0
    avg_regret_top = weighted_mean([r[3] / iterations for r in tier], tier_masses)

    probe_rows = []
    for label, card_id, street, total_raises, expected in probes:
        probs, mass = probe_policy(records, card_id, street, total_raises)
        rm = _role_mass(probs, roles) if probs else {}
        probe_rows.append({
            "label": label, "matched_mass": mass,
            "expected_role": expected,
            "expected_mass": rm.get(expected, 0.0),
            "role_mass": rm,
        })
    return {
        "iterations": iterations,
        "num_infosets": len(records),
        "gini": gini(masses),
        "top1pct_share": sum(sorted(masses, reverse=True)[:max(1, len(masses)//100)]) / total_mass,
        "entropy_all": ent_all,
        "entropy_top": ent_top,
        "support_top": supp_top,
        "avg_regret_top": avg_regret_top,
        "probes": probe_rows,
    }


# ---------------------------------------------------------------------------
# Probe hands and card encoding
# ---------------------------------------------------------------------------

# Canonical probe spots resolved to card buckets in main(): premium hands should
# grow aggression, trash should grow folds, and the "uniform open-jam" pathology
# should shrink (falling entropy_top captures the last one).
PROBE_HANDS = [
    ("AA UTG unraised", ("A", "A", ""), 0, 0, "aggressive"),
    ("KK UTG unraised", ("K", "K", ""), 0, 0, "aggressive"),
    ("AKs UTG unraised", ("A", "K", "s"), 0, 0, "aggressive"),
    ("72o UTG unraised", ("7", "2", "o"), 0, 0, "fold"),
    ("83o UTG unraised", ("8", "3", "o"), 0, 0, "fold"),
]

_RANKS = "23456789TJQKA"


def _card_int(rank, suit):
    """A concrete card code for preflop_class lookup.
    card = rank_index * 4 + suit_index (rank_index 0='2'..12='A'),
    matching common/src/game/card.h exactly."""
    return _RANKS.index(rank) * 4 + suit


def _probe_card_id(hand):
    """Resolve a (rank, rank, 's'/'o'/'') hand label to its preflop_class id.
    Suited => same suit index; offsuit and pairs => different suit indices."""
    sixmax = _sixmax()
    r1, r2, suited = hand
    if suited == "s":
        c1, c2 = _card_int(r1, 0), _card_int(r2, 0)  # same suit
    else:
        c1, c2 = _card_int(r1, 0), _card_int(r2, 1)  # offsuit / pair
    return sixmax.preflop_class([c1, c2])


def main(argv=None):
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Blueprint plateau autopsy")
    ap.add_argument("--checkpoints",
                    default="sixmax/checkpoints/blueprint_0*.bin")
    ap.add_argument("--top-frac", type=float, default=0.05)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    sixmax = _sixmax()
    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints}")

    vocab = load_vocab_for(paths[0])
    roles = resolve_action_roles(vocab)
    probes = [(lbl, _probe_card_id(hand), st, tr, exp)
              for (lbl, hand, st, tr, exp) in PROBE_HANDS]

    reports = []
    for p in paths:
        iters, records = sixmax.dump_infosets(p)
        reports.append(checkpoint_report(iters, records, roles, probes,
                                         top_frac=args.top_frac))

    reports.sort(key=lambda r: r["iterations"])
    for r in reports:
        print(f"iters={r['iterations']:>9}  infosets={r['num_infosets']:>7}  "
              f"gini={r['gini']:.3f}  top1%={r['top1pct_share']:.3f}  "
              f"H_all={r['entropy_all']:.3f}  H_top={r['entropy_top']:.3f}  "
              f"supp_top={r['support_top']:.2f}  regret_top={r['avg_regret_top']:.3e}")
    print("\nProbe traces (expected-role mass across checkpoints):")
    for i, (lbl, *_rest) in enumerate(PROBE_HANDS):
        trace = " ".join(f"{r['probes'][i]['expected_mass']:.2f}" for r in reports)
        print(f"  {lbl:<20} [{PROBE_HANDS[i][4]:>10}]: {trace}")

    label, rationale = verdict(reports)
    print(f"\nVERDICT: {label}\n  {rationale}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iters", "num_infosets", "gini", "top1pct_share",
                        "entropy_all", "entropy_top", "support_top",
                        "avg_regret_top"])
            for r in reports:
                w.writerow([r["iterations"], r["num_infosets"], r["gini"],
                            r["top1pct_share"], r["entropy_all"],
                            r["entropy_top"], r["support_top"],
                            r["avg_regret_top"]])


if __name__ == "__main__":
    main()
