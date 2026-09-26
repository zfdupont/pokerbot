#!/usr/bin/env python3
"""Train DREAM neural blueprint for sixmax/."""
import argparse
import importlib
import importlib.util
import os
import pathlib
import sys
import time


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Train DREAM neural blueprint for sixmax/",
    )
    parser.add_argument("--config",     required=True,
                        help="TOML config file (e.g. sixmax/configs/default.toml)")
    parser.add_argument("--checkpoint", required=True,
                        help="Output checkpoint path (e.g. /tmp/dream.pt)")
    parser.add_argument("--resume",     default=None,
                        help="Resume from existing checkpoint (not yet supported)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Total CFR iterations (overrides [train.dream].iterations)")
    parser.add_argument("--device",     default=None,
                        help="Torch device string, e.g. 'cpu', 'mps', 'cuda'")
    args = parser.parse_args()

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(args.config, "rb") as f:
        cfg_data = tomllib.load(f)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Pre-load libtorch shared libraries (macOS SIP ignores DYLD_LIBRARY_PATH)
    import ctypes
    lib_dir = os.path.join(repo_root, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        lib_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path)

    # Force-load sixmax extension — discover via Buck2 (same pattern as conftest.py)
    import subprocess
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        sys.exit(f"Buck2 build failed:\n{result.stderr}")
    so_path = None
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            rel_so = line.split()[-1]
            so_path = os.path.join(repo_root, rel_so)
            break
    if so_path is None:
        sys.exit(f"Could not locate sixmax.so in buck2 output.\n"
                 f"stdout: {result.stdout}\nstderr: {result.stderr}")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    sixmax = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = sixmax
    spec.loader.exec_module(sixmax)

    # Force-load sixmax_dream extension — sixmax must be loaded first (cross-module pybind types)
    result_dream = subprocess.run(
        [buck2, "build", "//sixmax:sixmax_dream", "--show-output"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result_dream.returncode != 0:
        sys.exit(f"Buck2 build failed for sixmax_dream:\n{result_dream.stderr}")
    so_path_dream = None
    for line in result_dream.stdout.splitlines():
        if "sixmax_dream.so" in line:
            rel_so = line.split()[-1]
            so_path_dream = os.path.join(repo_root, rel_so)
            break
    if so_path_dream is None:
        sys.exit(f"Could not locate sixmax_dream.so in buck2 output.\n"
                 f"stdout: {result_dream.stdout}\nstderr: {result_dream.stderr}")
    spec_dream = importlib.util.spec_from_file_location("sixmax_dream", so_path_dream)
    sixmax_dream = importlib.util.module_from_spec(spec_dream)
    sys.modules["sixmax_dream"] = sixmax_dream
    spec_dream.loader.exec_module(sixmax_dream)

    vc = load_module(os.path.join(repo_root, "sixmax", "vocab_config.py"), "vocab_config")
    vocab = vc.load_vocab(args.config, "blueprint")

    dc = cfg_data.get("train", {}).get("dream", {})
    dream_cfg = sixmax_dream.DreamConfig()
    dream_cfg.hidden_size    = dc.get("hidden_size",    256)
    dream_cfg.hidden_layers  = dc.get("hidden_layers",  3)
    dream_cfg.lr             = dc.get("lr",             1e-3)
    dream_cfg.batch_size     = dc.get("batch_size",     4096)
    dream_cfg.reservoir_size = dc.get("reservoir_size", 2_000_000)
    dream_cfg.train_interval = dc.get("train_interval", 10_000)
    dream_cfg.sgd_steps      = dc.get("sgd_steps",      2_000)
    dream_cfg.epsilon        = dc.get("epsilon",         0.06)
    dream_cfg.num_threads    = dc.get("num_threads",     0)
    dream_cfg.seed           = dc.get("seed",            42)
    dream_cfg.stack_min      = dc.get("stack_min",       20.0)
    dream_cfg.stack_max      = dc.get("stack_max",       250.0)
    dream_cfg.stack_log_mean = dc.get("stack_log_mean",  4.605)
    dream_cfg.stack_log_std  = dc.get("stack_log_std",   0.5)
    dream_cfg.players_min    = dc.get("players_min",     2)
    dream_cfg.players_max    = dc.get("players_max",     6)
    device_str = args.device or dc.get("device", "cpu")

    total_iters = args.iterations if args.iterations is not None else dc.get("iterations", 1_000_000)
    ckpt_interval = dc.get("checkpoint_interval", 0)
    ckpt_path = args.checkpoint

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    # Load abstraction
    abs_cfg = cfg_data.get("abstraction", {})
    abstraction = sixmax.Abstraction(
        flop_buckets=abs_cfg.get("flop_buckets",      50),
        turn_buckets=abs_cfg.get("turn_buckets",      50),
        river_buckets=abs_cfg.get("river_buckets",    20),
        equity_rollouts=abs_cfg.get("equity_rollouts", 100),
        quantile_samples=abs_cfg.get("quantile_samples", 10000),
        seed=abs_cfg.get("seed",                      20260719),
    )

    trainer = sixmax_dream.DreamTrainer(vocab.size(), vocab, abstraction,
                                       dream_cfg, device_str)

    if args.resume:
        print(f"Resume not yet supported for DreamTrainer (nets only). "
              f"Starting fresh.")

    print(f"Training DREAM blueprint: {total_iters} iterations, device={device_str}")
    start = time.time()

    if ckpt_interval > 0:
        done = 0
        while done < total_iters:
            chunk = min(ckpt_interval, total_iters - done)
            # external-sampling counts both player traversals; ds.iterations returns 2*total_iters
            trainer.train(chunk)
            done += chunk
            elapsed = time.time() - start
            print(f"  {done}/{total_iters} iters | {elapsed:.0f}s elapsed")
            trainer.save(ckpt_path, vocab.hash())
            print(f"  Saved checkpoint -> {ckpt_path}")
    else:
        # external-sampling counts both player traversals; ds.iterations returns 2*total_iters
        trainer.train(total_iters)
        trainer.save(ckpt_path, vocab.hash())
        print(f"Saved -> {ckpt_path} ({time.time()-start:.0f}s)")


if __name__ == "__main__":
    main()
