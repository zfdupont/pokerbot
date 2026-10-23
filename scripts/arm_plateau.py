#!/usr/bin/env python3
"""Decide when an abstraction arm has plateaued, and its iteration cap.

An arm is converged when LBR exploitability has stopped trending (spec 1d,
revised 2026-10-22): every snapshot faces local best response on the same decks
(turn+river, seed-shared), the per-deck LBR values of the last `--window`
snapshots are regressed on iterations deck by deck, and the window's slope
(BB/100 per 10M iterations) is t-tested against 0. The arm is converged when
the windows ending at the newest two snapshots both have |t| < 2: one flat
window can be a dip between two rises. Per-deck values are cached in
`--per-deck-dir` (shared with scripts/lbr_sweep.py), so a check after each sync
only runs LBR on new snapshots.

`--method h2h` keeps the earlier pair rule, N does not beat N-lag head-to-head
at |t| >= 2 (patterns/eval-checkpoint-ab.md); a lag that lands between
snapshots uses the newest one at or below N-lag. Snapshots are the trainer's
`<base>_<iters>.bin` files; any digit count is accepted.

The iteration cap is C0's plateau scaled by the arm's infoset count, clamped to
[20M, 150M] for Phase 2 arms and [20M, 500M] for Phase 3 arms (`--phase 3`).
`--cap` and `--infosets` print a bare integer so they are machine-readable.

Usage:
    uv run python scripts/arm_plateau.py \\
        --snapshots 'sixmax/checkpoints/hu200_c2_*.bin' \\
        --config sixmax/configs/hu200_c2.toml
    uv run python scripts/arm_plateau.py --infosets sixmax/checkpoints/hu200_c0.bin
    uv run python scripts/arm_plateau.py --cap 25000000 40000 120000
    uv run python scripts/arm_plateau.py --cap 60000000 33831 217000 --phase 3
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SNAPSHOT_RE = re.compile(r"_(\d+)\.bin$")
_MIN_CAP = 20_000_000
# Ceiling per roadmap phase (spec 1d, revised 2026-10-21): Phase 3 card maps
# reach about 8x C0's infosets, which a 150M ceiling would undertrain.
_MAX_CAP = {2: 150_000_000, 3: 500_000_000}
_DEFAULT_TOML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sixmax", "configs", "default.toml")


def snapshot_iters(path):
    """Iteration count from a snapshot name: `<base>_<digits>.bin`."""
    m = _SNAPSHOT_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"not a snapshot name: {path}")
    return int(m.group(1))


def pick_pair(paths, lag):
    """(newest snapshot, newest snapshot at or below its iters - lag).

    Paths whose name lacks the `_<digits>.bin` suffix (e.g. the unsuffixed base
    checkpoint) are ignored. Irregular spacing therefore pairs the newest
    snapshot with the most recent one that is at least `lag` behind it.
    """
    snaps = []
    for path in paths:
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        if m is not None:
            snaps.append((int(m.group(1)), path))
    if not snaps:
        raise ValueError("no snapshots")
    snaps.sort()
    newest_iters, newest = snaps[-1]
    target = newest_iters - lag
    older = [s for s in snaps[:-1] if s[0] <= target]
    if not older:
        raise ValueError(f"no snapshot at or below {target} iterations")
    return newest, older[-1][1]


def iteration_cap(c0_plateau, c0_infosets, arm_infosets, phase=2):
    """C0's plateau scaled by infoset count, clamped to [20M, the phase's
    ceiling]: 150M in Phase 2, 500M in Phase 3."""
    if c0_infosets <= 0:
        raise ValueError("c0_infosets must be positive")
    if phase not in _MAX_CAP:
        raise ValueError(f"phase must be one of {sorted(_MAX_CAP)}")
    scaled = round(c0_plateau * arm_infosets / c0_infosets)
    return int(min(max(scaled, _MIN_CAP), _MAX_CAP[phase]))


def converged(stats):
    """Converged unless the newer snapshot still beats the older one."""
    return not (stats["mean"] > 0 and stats["t"] >= 2.0)


_PER_10M = 1e7


def trend(iters, per_deck):
    """Paired LBR trend over one window of snapshots.

    `iters` holds each snapshot's iteration count; `per_deck[i]` its per-deck
    LBR values (BB/100) on the same decks. Each deck's values are regressed on
    iterations by least squares, and the slopes are averaged over decks, so
    deck luck cancels as in a paired test. Returns {slope, se, t, decks} with
    slope and se in BB/100 per 10M iterations; t is 0 when se is 0.
    """
    import numpy as np

    if len(iters) < 2 or len(iters) != len(per_deck):
        raise ValueError("trend needs >= 2 snapshots with per-deck values")
    x = np.asarray(iters, dtype=float) / _PER_10M
    dx = x - x.mean()
    sxx = float((dx * dx).sum())
    if sxx == 0.0:
        raise ValueError("trend needs snapshots at distinct iterations")
    y = np.stack([np.asarray(v, dtype=float) for v in per_deck])
    slopes = (dx[:, None] * (y - y.mean(axis=0))).sum(axis=0) / sxx
    n = slopes.size
    slope = float(slopes.mean())
    se = float(slopes.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"slope": slope, "se": se, "t": slope / se if se > 0 else 0.0,
            "decks": n}


def converged_trend(windows):
    """Converged when the last two windows' slopes are both within 2 se of
    0. Fewer than two windows is not enough evidence: keep training."""
    return len(windows) >= 2 and all(abs(w["t"]) < 2.0 for w in windows[-2:])


def _infosets(path):
    """Queryable infoset count of a checkpoint as deployment sees it."""
    from agents.sixmax_agent import SixmaxDeployStrategy

    return SixmaxDeployStrategy.load(path, _DEFAULT_TOML).num_infosets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots",
                   help="glob of <base>_<iters>.bin snapshots")
    p.add_argument("--lag", type=int, default=10_000_000,
                   help="h2h: iterations between the compared snapshots")
    p.add_argument("--method", choices=("lbr", "h2h"), default="lbr",
                   help="LBR trend over snapshots (default) or head-to-head")
    p.add_argument("--window", type=int, default=4,
                   help="lbr: snapshots per trend window (>= 3)")
    p.add_argument("--decks", type=int, default=100000,
                   help="lbr: duplicate decks shared by every snapshot "
                        "(100k: about 3.5 min per new snapshot on 8 cores)")
    p.add_argument("--per-deck-dir",
                   default=os.path.join("sixmax", "checkpoints",
                                        "lbr_per_deck"),
                   help="lbr: per-deck LBR cache (shared with lbr_sweep.py)")
    p.add_argument("--seed", type=int, default=1, help="lbr: deck seed")
    p.add_argument("--config", default=_DEFAULT_TOML,
                   help="lbr: config whose blueprint vocab the snapshots use")
    p.add_argument("--threads", type=int, default=0,
                   help="lbr: worker threads (0 = hardware concurrency)")
    p.add_argument("--hands", type=int, default=20000, help="h2h only")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8", help="h2h only")
    p.add_argument("--bb", type=int, default=2, help="h2h only")
    p.add_argument("--stack", type=int, default=400, help="h2h only")
    p.add_argument("--infosets", metavar="PATH",
                   help="print the checkpoint's queryable infoset count")
    p.add_argument("--cap", nargs=3, type=int,
                   metavar=("C0_PLATEAU", "C0_INFOSETS", "ARM_INFOSETS"),
                   help="print the arm's iteration cap")
    p.add_argument("--phase", type=int, choices=sorted(_MAX_CAP), default=2,
                   help="roadmap phase whose cap ceiling applies (with --cap)")
    args = p.parse_args()

    if args.cap is not None and args.infosets is not None:
        p.error("--cap and --infosets are mutually exclusive")
    if args.cap is not None:
        print(iteration_cap(*args.cap, phase=args.phase))
        return
    if args.infosets is not None:
        print(_infosets(args.infosets))
        return
    if args.snapshots is None:
        p.error("--snapshots is required unless --infosets or --cap is given")
    if args.method == "lbr":
        _plateau_trend(p, args)
        return
    if args.lag <= 0:
        p.error("--lag must be positive (a lag of 0 would compare a snapshot "
                "with itself)")

    try:
        newest, older = pick_pair(glob.glob(args.snapshots), args.lag)
    except ValueError as exc:
        p.error(str(exc))

    from scripts.eval_blueprint_h2h import run_h2h

    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < 2:
        # A t-test on one seed has no spread: t is 0, so converged() always
        # says "converged". Say so instead of letting the verdict mislead.
        print(f"warning: t-test needs >= 2 seeds; verdict will read converged "
              f"(got {len(seeds)})", file=sys.stderr)
    print(f"A = {newest} (iters={snapshot_iters(newest)})")
    print(f"B = {older} (iters={snapshot_iters(older)})")
    print(f"{'seed':>4} {'A vs B':>10}")

    def show(seed, bb100):
        print(f"{seed:>4} {bb100:>+10.2f}", flush=True)

    stats = run_h2h(newest, older, args.hands, seeds, args.bb, args.stack,
                    on_seed=show)
    print(f"\nn={len(stats['per_seed'])} seeds x {args.hands} duplicate deals")
    print(f"A - B: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.1f}  t={stats['t']:.2f}")
    print(f"VERDICT: {'converged' if converged(stats) else 'training'}")


def _per_deck_lbr(path, args):
    """Per-deck LBR values for one snapshot, from the cache or a fresh run."""
    import numpy as np
    from scripts.eval_sixmax_lbr import run_lbr
    from scripts.lbr_sweep import per_deck_path

    cached = per_deck_path(args.per_deck_dir, path, args.decks, args.seed)
    if os.path.exists(cached):
        return np.load(cached)
    rep = run_lbr(path, args.config, args.decks, args.seed,
                  threads=args.threads)
    values = np.asarray(rep["per_deck"], dtype=np.float64)
    os.makedirs(args.per_deck_dir, exist_ok=True)
    np.save(cached, values)
    return values


def _plateau_trend(parser, args):
    if args.decks < 2:
        parser.error("--decks must be at least 2 (the t-test needs a spread)")
    if args.window < 3:
        parser.error("--window must be at least 3 (a 2-point line has no "
                     "check against a dip)")
    snaps = []
    for path in glob.glob(args.snapshots):
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        if m is not None:
            snaps.append((int(m.group(1)), path))
    snaps.sort()
    if not snaps:
        parser.error("no snapshots")
    # The last two windows need window + 1 snapshots; older ones are unused.
    used = snaps[-(args.window + 1):]
    values = {}
    for iters, path in used:
        values[iters] = _per_deck_lbr(path, args)
        print(f"{iters:>12,}  LBR {values[iters].mean():+8.2f} BB/100",
              flush=True)
    windows = []
    for end in range(args.window, len(used) + 1):
        w = [i for i, _ in used[end - args.window:end]]
        stats = trend(w, [values[i] for i in w])
        windows.append(stats)
        print(f"window {w[0]:,}-{w[-1]:,}: slope {stats['slope']:+.2f} "
              f"BB/100 per 10M  se {stats['se']:.2f}  t={stats['t']:+.2f}")
    if len(windows) < 2:
        print(f"need {args.window + 1} snapshots for two windows "
              f"(have {len(used)})")
    print(f"VERDICT: {'converged' if converged_trend(windows) else 'training'}")


if __name__ == "__main__":
    main()
