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
import json
import os
import subprocess
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


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


# Built-in defaults — lowest precedence. Keys match CLI flag names
# (underscored) and TOML keys in [training]/[trainer] sections.
BUILTIN_DEFAULTS = {
    "iterations":          100_000,
    "checkpoint_interval": None,
    "checkpoint":          "neural_cfr/checkpoints/checkpoint.pt",
    "reservoir_size":      2_000_000,
    "batch_size":          4096,
    "lr":                  1e-3,
    "train_interval":      10_000,
    "sgd_steps":           2_000,
    "reinit_adv":          True,
    "num_threads":         0,
    "epsilon":             0.06,
    "eval_interval":       None,
    "eval_hands":          500,
}


def resolve_config(args: "argparse.Namespace", repo_root: str) -> dict:
    """Merge config sources. Precedence: CLI flag > TOML file > builtin."""
    cfg = dict(BUILTIN_DEFAULTS)
    path = args.config or os.path.join(repo_root, "neural_cfr", "configs", "default.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for section in ("training", "trainer"):
            for key, value in data.get(section, {}).items():
                if key not in BUILTIN_DEFAULTS:
                    raise KeyError(f"Unknown config key [{section}].{key} in {path}")
                cfg[key] = value
    for key in BUILTIN_DEFAULTS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            cfg[key] = cli_value
    return cfg


def write_config_snapshot(cfg: dict, checkpoint_path: str) -> None:
    """Write the effective config next to the checkpoint for reproducibility."""
    lines = ["# Effective config — written by train_neural.py", "[resolved]"]
    for key, value in sorted(cfg.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = repr(value)
        lines.append(f"{key} = {rendered}")
    with open(checkpoint_path + ".config.toml", "w") as f:
        f.write("\n".join(lines) + "\n")


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
    parser.add_argument("--iterations",          type=int,   default=None,
                        help="Total CFR traversal iterations (default: 100000)")
    parser.add_argument("--checkpoint-interval", type=int,   default=None,
                        help="Save checkpoint every N iterations (default: end only)")
    parser.add_argument("--reservoir-size",      type=int,   default=None,
                        help="Reservoir buffer capacity per network (default: 2000000)")
    parser.add_argument("--batch-size",          type=int,   default=None,
                        help="SGD mini-batch size (default: 4096)")
    parser.add_argument("--lr",                  type=float, default=None,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--checkpoint",          type=str,   default=None,
                        help="Output checkpoint path (default: neural_cfr/checkpoints/checkpoint.pt)")
    parser.add_argument("--resume",              type=str,   default=None,
                        help="Resume from an existing checkpoint file")
    parser.add_argument("--train-interval",       type=int,   default=None,
                        help="Train networks every N CFR rounds (default: 10000)")
    parser.add_argument("--num-threads",          type=int,   default=None,
                        help="Traversal threads (default: 0 = hardware_concurrency)")
    parser.add_argument("--epsilon",              type=float, default=None,
                        help="ε-greedy exploration at opponent nodes (default: 0.06)")
    parser.add_argument("--eval-interval",        type=int,   default=None,
                        help="Run win-rate eval every N iterations (default: off)")
    parser.add_argument("--eval-hands",           type=int,   default=None,
                        help="Hands per eval run (default: 500)")
    parser.add_argument("--config",     type=str, default=None,
                        help="TOML config file (default: neural_cfr/configs/default.toml if present)")
    parser.add_argument("--sgd-steps",  type=int, default=None,
                        help="SGD mini-batches per training event (default: 2000)")
    parser.add_argument("--reinit-adv", action=argparse.BooleanOptionalAction, default=None,
                        help="Reinitialize advantage nets each training event (default: on)")
    args = parser.parse_args()

    cfg = resolve_config(args, repo_root)
    print("Effective config: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))

    os.makedirs(os.path.dirname(os.path.abspath(cfg["checkpoint"])), exist_ok=True)

    trainer = neural_cfr.Trainer(
        reservoir_size=cfg["reservoir_size"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        train_interval=cfg["train_interval"],
        num_threads=cfg["num_threads"],
        epsilon=cfg["epsilon"],
        sgd_steps=cfg["sgd_steps"],
        reinit_adv=cfg["reinit_adv"],
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load(args.resume)

    ckpt_interval = cfg["checkpoint_interval"] or cfg["iterations"]
    eval_interval = cfg["eval_interval"]
    completed = 0
    print(f"Running {cfg['iterations']:,} iterations …")
    while completed < cfg["iterations"]:
        chunk = min(ckpt_interval, cfg["iterations"] - completed)
        trainer.run(chunk)
        completed += chunk
        trainer.checkpoint(cfg["checkpoint"])
        write_config_snapshot(cfg, cfg["checkpoint"])
        if completed < cfg["iterations"]:
            print(f"[{completed:,}/{cfg['iterations']:,}] Checkpoint saved to {cfg['checkpoint']}")
        if eval_interval and completed % eval_interval == 0:
            print(f"[{completed:,}] Running eval …")
            eval_script = os.path.join(repo_root, "scripts", "eval_openspiel_neural.py")
            result = subprocess.run(
                [sys.executable, eval_script,
                 "--checkpoint", cfg["checkpoint"],
                 "--hands", str(cfg["eval_hands"]),
                 "--baseline", "random"],
                cwd=repo_root,
            )
            if result.returncode != 0:
                print(f"[{completed:,}] Eval failed (exit {result.returncode}), continuing …")


if __name__ == "__main__":
    main()
