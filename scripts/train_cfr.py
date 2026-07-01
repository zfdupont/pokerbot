"""
Train a heads-up MCCFR poker agent.

Usage:
    uv run python scripts/train_cfr.py [--iterations 1000000] [--checkpoint-interval 100000] [--checkpoint-dir cfr/checkpoints]

The trained strategy is saved as checkpoints and can be loaded by CFRAgent.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cfr.regret_table import RegretTable
from cfr.trainer import Trainer
from cfr.mccfr import compute_exploitability


def main():
    parser = argparse.ArgumentParser(description="Train MCCFR heads-up poker agent")
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", type=str, default="cfr/checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (0 = all CPU cores)")
    parser.add_argument("--chunk-size", type=int, default=5_000,
                        help="Iterations per worker per round (parallel mode only)")
    args = parser.parse_args()

    table = RegretTable()
    trainer = Trainer(table)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print(f"Training for {args.iterations:,} iterations")
    print(f"Checkpointing every {args.checkpoint_interval:,} to {args.checkpoint_dir}/")

    if args.workers != 1:
        trainer.parallel_train(
            num_iterations=args.iterations,
            n_workers=args.workers,
            chunk_size=args.chunk_size,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        trainer.train(
            num_iterations=args.iterations,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )

    print("\nFinal exploitability estimate (10,000 samples)...")
    expl = compute_exploitability(table, num_samples=10_000, show_progress=True)
    print(f"Exploitability: {expl:.1f} mbb/h")
    print(f"Info sets visited: {len(table.regrets):,}")


if __name__ == "__main__":
    main()
