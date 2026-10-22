#!/usr/bin/env python3
"""Decide when an abstraction arm has plateaued, and its iteration cap.

An arm is converged at snapshot N when N is not less exploitable than
snapshot N-lag by paired LBR at |t| >= 2 (spec 1d, revised 2026-10-22): both
snapshots face local best response on the same decks and the per-deck
differences are t-tested. `--method h2h` keeps the earlier rule, N does not
beat N-lag head-to-head at |t| >= 2 (patterns/eval-checkpoint-ab.md). Snapshots
are the trainer's `<base>_<iters>.bin` files; any digit count is accepted, and
a lag that lands between snapshots uses the newest one at or below N-lag.

The iteration cap is C0's plateau scaled by the arm's infoset count, clamped to
[20M, 150M] for Phase 2 arms and [20M, 500M] for Phase 3 arms (`--phase 3`).
`--cap` and `--infosets` print a bare integer so they are machine-readable.

Usage:
    uv run python scripts/arm_plateau.py \\
        --snapshots 'sixmax/checkpoints/hu200_c0_*.bin' --lag 10000000 \\
        --config sixmax/configs/hu200.toml
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


def converged_lbr(stats):
    """Converged unless the newer snapshot is still less exploitable: lower
    LBR is better, so `mean` (newer - older) must be <= -2 standard errors
    to keep training."""
    return not (stats["mean"] < 0 and stats["t"] <= -2.0)


def _infosets(path):
    """Queryable infoset count of a checkpoint as deployment sees it."""
    from agents.sixmax_agent import SixmaxDeployStrategy

    return SixmaxDeployStrategy.load(path, _DEFAULT_TOML).num_infosets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots",
                   help="glob of <base>_<iters>.bin snapshots")
    p.add_argument("--lag", type=int, default=10_000_000,
                   help="iterations between the compared snapshots")
    p.add_argument("--method", choices=("lbr", "h2h"), default="lbr",
                   help="paired LBR exploitability (default) or head-to-head")
    p.add_argument("--decks", type=int, default=100000,
                   help="lbr: duplicate decks shared by both snapshots "
                        "(100k gives se ~5 BB/100, about 7 min on 8 cores)")
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
    if args.lag <= 0:
        p.error("--lag must be positive (a lag of 0 would compare a snapshot "
                "with itself)")

    try:
        newest, older = pick_pair(glob.glob(args.snapshots), args.lag)
    except ValueError as exc:
        p.error(str(exc))
    if args.method == "lbr":
        _plateau_lbr(p, args, newest, older)
        return

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


def _plateau_lbr(parser, args, newest, older):
    from scripts.eval_sixmax_lbr import paired_lbr

    if args.decks < 2:
        parser.error("--decks must be at least 2 (the t-test needs a spread)")
    print(f"A = {newest} (iters={snapshot_iters(newest)})")
    print(f"B = {older} (iters={snapshot_iters(older)})")
    try:
        stats = paired_lbr(newest, older, args.config, args.decks, args.seed,
                           threads=args.threads)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"LBR(A) {stats['a_bb100']:+.2f}  LBR(B) {stats['b_bb100']:+.2f} "
          f"BB/100 (turn+river, seed {args.seed})")
    print(f"\nn={stats['n']} paired decks")
    print(f"A - B: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.2f}  t={stats['t']:.2f}")
    print(f"VERDICT: {'converged' if converged_lbr(stats) else 'training'}")


if __name__ == "__main__":
    main()
