#!/usr/bin/env python3
"""
Neural CFR training launcher.

Builds the C++ extension (via Buck2), pre-loads libtorch dylibs (required on
macOS where SIP prevents DYLD_LIBRARY_PATH propagation), then imports the
pybind11 module and drives neural_cfr.Trainer.

Usage:
    uv run python scripts/train_neural.py [options]

Options:
    --iterations           CFR traversal iterations total     (default: 100_000)
    --checkpoint-interval  Save every N iterations            (default: 100_000, i.e. end only)
    --reservoir-size       Reservoir buffer capacity per net  (default: 2_000_000)
    --batch-size           SGD mini-batch size                (default: 4096)
    --lr                   Learning rate                      (default: 1e-4)
    --checkpoint           Output checkpoint path             (default: neural_cfr/checkpoints/checkpoint.pt)
    --resume               Resume from existing checkpoint    (default: None)
"""
import argparse
import ctypes
import os
import subprocess
import sys


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preload_libtorch(repo_root: str) -> None:
    """Pre-load libtorch shared libraries (macOS SIP ignores DYLD_LIBRARY_PATH)."""
    lib_dir = os.path.join(repo_root, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        ctypes.CDLL(os.path.join(lib_dir, lib))


def _build_and_get_so_dir(repo_root: str) -> str:
    """Run Buck2 build and return the directory containing neural_cfr.so."""
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//neural_cfr:neural_cfr", "--show-output"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(repo_root, os.path.dirname(rel_so))
    raise RuntimeError(
        f"Could not locate neural_cfr.so in buck2 output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def main() -> None:
    repo_root = _get_repo_root()

    # Build extension and configure import path before parsing args so that
    # --help works even if the build is stale.
    _preload_libtorch(repo_root)
    so_dir = _build_and_get_so_dir(repo_root)
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)

    import neural_cfr  # noqa: E402 — must follow path setup

    parser = argparse.ArgumentParser(
        description="Neural CFR training launcher (C++ core via pybind11)"
    )
    parser.add_argument("--iterations",          type=int,   default=100_000,
                        help="Total CFR traversal iterations (default: 100000)")
    parser.add_argument("--checkpoint-interval", type=int,   default=None,
                        help="Save checkpoint every N iterations (default: end only)")
    parser.add_argument("--reservoir-size",      type=int,   default=2_000_000,
                        help="Reservoir buffer capacity per network (default: 2000000)")
    parser.add_argument("--batch-size",          type=int,   default=4096,
                        help="SGD mini-batch size (default: 4096)")
    parser.add_argument("--lr",                  type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--checkpoint",          type=str,
                        default="neural_cfr/checkpoints/checkpoint.pt",
                        help="Output checkpoint path (default: neural_cfr/checkpoints/checkpoint.pt)")
    parser.add_argument("--resume",              type=str,   default=None,
                        help="Resume from an existing checkpoint file")
    parser.add_argument("--train-interval",       type=int,   default=10,
                        help="Train networks every N CFR rounds (default: 10)")
    parser.add_argument("--num-threads",          type=int,   default=0,
                        help="Traversal threads (default: 0 = hardware_concurrency)")
    parser.add_argument("--eval-interval",        type=int,   default=None,
                        help="Run win-rate eval every N iterations (default: off)")
    parser.add_argument("--eval-hands",           type=int,   default=500,
                        help="Hands per eval run (default: 500)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    trainer = neural_cfr.Trainer(
        reservoir_size=args.reservoir_size,
        batch_size=args.batch_size,
        lr=args.lr,
        train_interval=args.train_interval,
        num_threads=args.num_threads,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load(args.resume)

    ckpt_interval = args.checkpoint_interval or args.iterations
    eval_interval = args.eval_interval
    completed = 0
    print(f"Running {args.iterations:,} iterations …")
    while completed < args.iterations:
        chunk = min(ckpt_interval, args.iterations - completed)
        trainer.run(chunk)
        completed += chunk
        trainer.checkpoint(args.checkpoint)
        if completed < args.iterations:
            print(f"[{completed:,}/{args.iterations:,}] Checkpoint saved to {args.checkpoint}")
        if eval_interval and completed % eval_interval == 0:
            print(f"[{completed:,}] Running eval …")
            eval_script = os.path.join(repo_root, "scripts", "eval_openspiel_neural.py")
            result = subprocess.run(
                [sys.executable, eval_script,
                 "--checkpoint", args.checkpoint,
                 "--hands", str(args.eval_hands),
                 "--baseline", "random"],
                cwd=repo_root,
            )
            if result.returncode != 0:
                print(f"[{completed:,}] Eval failed (exit {result.returncode}), continuing …")


if __name__ == "__main__":
    main()
