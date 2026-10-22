#!/usr/bin/env python3
"""Heads-up LBR exploitability estimator (Lisy & Bowling 2017).

Duplicate evaluation: every deck is played twice, with the local best response
(LBR) in seat 0 and in seat 1 and the blueprint in the other seat. The reported
BB/100 is LBR's winnings against the blueprint: a lower bound on the
blueprint's exploitability (not the best-response gain of the blueprint).
Chip values stay in the engine frame (big blinds); pots and stacks come from
the checkpoint's training configuration unless --stack-bb overrides it.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_COLUMNS = ("checkpoint,act_streets,fold_call_only,decks,bb100,stderr,"
               "seat0_bb100,seat1_bb100,stack_bb,seed,uniform_opponents,"
               "preflop_wp_samples,iterations,config,opp_unseen_rate")


def parse_act_streets(text):
    """Comma-separated street list (0 preflop .. 3 river); empty = no acting."""
    if text.strip() == "":
        return []
    streets = []
    for token in text.split(","):
        token = token.strip()
        try:
            street = int(token)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "act streets must be integers in 0-3, got %r" % token)
        if street < 0 or street > 3:
            raise argparse.ArgumentTypeError(
                "act streets must be integers in 0-3, got %r" % token)
        if street not in streets:
            streets.append(street)
    return streets


def parse_args(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description="Heads-up LBR exploitability (Lisy & Bowling 2017)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=os.path.join(
        root, "sixmax", "configs", "default.toml"))
    p.add_argument("--decks", type=int, default=2000,
                   help="duplicate deck pairs (each is played twice)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--act-streets", type=parse_act_streets, default=[2, 3],
                   metavar="LIST",
                   help="streets LBR may bet on, comma separated (0-3); "
                        "empty means check/call only")
    p.add_argument("--fold-call-only", action="store_true",
                   help="LBR considers only fold and check/call")
    p.add_argument("--preflop-wp-samples", type=int, default=500,
                   help="sampled boards behind a preflop win probability")
    p.add_argument("--threads", type=int, default=0,
                   help="worker threads (0 = hardware concurrency)")
    p.add_argument("--uniform-opponents", action="store_true",
                   help="test hook: opponent model is uniform over the mask")
    p.add_argument("--stack-bb", type=float, default=None,
                   help="starting stack in BB (default: the checkpoint's "
                        "training stack)")
    p.add_argument("--csv", default=None)
    return p.parse_args(argv)


def csv_header():
    return CSV_COLUMNS


def write_csv(path, checkpoint, args, report, stack_bb, iterations):
    """Write the header and the single data row through csv.writer.

    `act_streets` stays one field (e.g. "2,3"), so the writer quotes it and the
    row keeps exactly len(CSV_COLUMNS) fields.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header().split(","))
        writer.writerow(report_row(checkpoint, args, report, stack_bb,
                                   iterations))


def _opp_unseen_rate(report):
    total = int(report["opp_decisions"])
    return float(report["opp_unseen"]) / total if total else 0.0


def report_row(path, args, report, stack_bb, iterations):
    """One CSV row: checkpoint, config, and the duplicate + per-seat values."""
    return (os.path.basename(path),
            ",".join(str(s) for s in args.act_streets),
            bool(args.fold_call_only),
            int(report["decks"]),
            float(report["bb100"]),
            float(report["stderr"]),
            float(report["seat_bb100"][0]),
            float(report["seat_bb100"][1]),
            float(stack_bb),
            int(args.seed),
            bool(args.uniform_opponents),
            int(args.preflop_wp_samples),
            int(iterations),
            args.config,
            _opp_unseen_rate(report))


def _ci(std_err):
    return 1.96 * std_err


def main(argv=None):
    args = parse_args(argv)
    from agents.sixmax_agent import _ensure_sixmax_extension, _load_vocab
    _ensure_sixmax_extension()
    import sixmax

    vocab = _load_vocab(args.config, "blueprint")
    strategy = sixmax.BlueprintStrategy.load(args.checkpoint, vocab)
    # The checkpoint records its own training stack; --stack-bb overrides it.
    stack = (strategy.starting_stack() if args.stack_bb is None
             else args.stack_bb)
    cfg = sixmax.EngineConfig(num_players=strategy.num_players(),
                              starting_stack=stack)
    lbr = sixmax.LbrConfig(act_streets=args.act_streets,
                           fold_call_only=args.fold_call_only,
                           preflop_wp_samples=args.preflop_wp_samples,
                           uniform_opponents=args.uniform_opponents,
                           num_threads=args.threads)
    ev = sixmax.LbrEvaluator(strategy, vocab, cfg, lbr)
    report = ev.run(args.decks, args.seed)

    streets = ",".join(str(s) for s in args.act_streets) or "none"
    model = "uniform" if args.uniform_opponents else "blueprint"
    print("\nLBR duplicate evaluation - %s" % os.path.basename(args.checkpoint))
    print("config: %s; iterations: %d; stack: %g BB; decks: %d; seed: %d"
          % (args.config, strategy.iterations(), stack, args.decks, args.seed))
    print("preflop wp samples: %d; opponent model: %s; LBR acts on streets: %s"
          % (args.preflop_wp_samples, model, streets))
    print("opponent unseen-infoset rate: %.2f%% (%d/%d decisions)"
          % (100.0 * _opp_unseen_rate(report), report["opp_unseen"],
             report["opp_decisions"]))
    print("\n%22s  %12s  %9s" % ("seat", "LBR BB/100", "95% CI"))
    labels = ("seat 0 (button/SB)", "seat 1 (BB)")
    for label, value, se in zip(labels, report["seat_bb100"],
                                report["seat_stderr"]):
        print("%22s  %+12.2f  %9s" % (
            label, value, "+-" + format(_ci(se), ".2f")))
    print("\n%-22s  %+.2f BB/100  (+-%0.2f)  =  %+.1f mBB/h" % (
        "duplicate", report["bb100"], _ci(report["stderr"]),
        report["bb100"] * 10.0))

    if args.csv:
        write_csv(args.csv, args.checkpoint, args, report, stack,
                  strategy.iterations())
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()
