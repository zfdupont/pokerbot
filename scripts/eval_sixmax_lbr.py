#!/usr/bin/env python3
"""Six-max LBR exploitability estimator.

Reports each seat's best-response gain (BB/100) against the blueprint's own
profile: Pass A plays the blueprint in the hero seat, Pass B plays the LBR
exploiter, both over identical decks (common random numbers). The gain is a
lower bound on the true best-response gain; a plateau at a positive value means
the abstraction, not training, is the limit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Six-max LBR exploitability")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=os.path.join(
        root, "sixmax", "configs", "default.toml"))
    p.add_argument("--decks", type=int, default=200)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--uniform-opponents", action="store_true")
    p.add_argument("--csv", default=None)
    return p.parse_args(argv)


def report_rows(path, report):
    return [(os.path.basename(path), seat, gain, se, report["decks"])
            for seat, (gain, se) in enumerate(zip(report["gains"],
                                                  report["stderrs"]))]


def main(argv=None):
    args = parse_args(argv)
    from agents.sixmax_agent import _ensure_sixmax_extension, _load_vocab
    _ensure_sixmax_extension()
    import sixmax

    vocab = _load_vocab(args.config, "blueprint")
    strategy = sixmax.BlueprintStrategy.load(args.checkpoint, vocab)
    cfg = sixmax.EngineConfig(num_players=strategy.num_players())
    lbr = sixmax.LbrConfig(samples=args.samples, seed=args.seed,
                           uniform_opponents=args.uniform_opponents)
    ev = sixmax.LbrEvaluator(strategy, vocab, cfg, lbr)
    report = ev.run(args.decks, args.seed)

    print(f"\nLBR exploitability — {os.path.basename(args.checkpoint)} "
          f"({strategy.iterations()} iters), {args.decks} decks, "
          f"samples={args.samples}, seed {args.seed}\n")
    print(f"{'seat':>4}  {'gain BB/100':>12}  {'95% CI':>9}")
    for seat, (gain, se) in enumerate(zip(report["gains"], report["stderrs"])):
        print(f"{seat:>4}  {gain:>+12.2f}  {'±' + format(1.96 * se, '.2f'):>9}")
    print(f"\nmax gain {report['max_gain']:+.2f}  "
          f"mean gain {report['mean_gain']:+.2f} BB/100")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("checkpoint,seat,gain_bb100,stderr,decks\n")
            for ckpt, seat, gain, se, decks in report_rows(
                    args.checkpoint, report):
                f.write(f"{ckpt},{seat},{gain:.4f},{se:.4f},{decks}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
