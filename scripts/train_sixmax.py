#!/usr/bin/env python3
"""Six-max blueprint training launcher.

Builds //sixmax:sixmax via Buck2, force-loads the extension (the repo-root
`sixmax/` directory would otherwise shadow the .so as a namespace package),
then drives sixmax.BlueprintTrainer with chunked checkpointing.

Config precedence: CLI flag > TOML ([abstraction] + [train.blueprint]) >
builtin defaults. The effective config is snapshotted to
<checkpoint>.config.toml on every save (neural_cfr pattern).
"""
import argparse
import importlib.util
import os
import subprocess
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_and_get_so_dir(repo_root: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(repo_root, os.path.dirname(rel_so))
    raise RuntimeError(f"Could not locate sixmax.so in buck2 output.\n"
                       f"stdout: {result.stdout}\nstderr: {result.stderr}")


def _force_load_sixmax(repo_root: str):
    """Load the .so and register it as sys.modules['sixmax'] (the repo-root
    sixmax/ directory is a namespace package that would win otherwise)."""
    so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_vocab(repo_root: str, toml_path: str):
    vc_path = os.path.join(repo_root, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", vc_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(toml_path, "blueprint")


BUILTIN_DEFAULTS = {
    # [abstraction]
    "flop_buckets": 50, "turn_buckets": 50, "river_buckets": 20,
    "equity_rollouts": 100, "quantile_samples": 10000,
    "abstraction_seed": 20260719,
    # [train.blueprint]
    "num_players": 6, "starting_stack": 100.0,
    "iterations": 100_000, "num_threads": 0,
    "checkpoint_interval": 0,
    "checkpoint": "sixmax/checkpoints/blueprint.bin",
    "seed": 7,
    "selection_enabled": False, "selection_hands": 2000,
}

# TOML key -> flat config key (only where they differ)
_TOML_RENAME = {"seed": {"abstraction": "abstraction_seed",
                         "train.blueprint": "seed"}}


def resolve_config(args, repo_root: str) -> dict:
    """Merge config sources. Precedence: CLI flag > TOML > builtin."""
    cfg = dict(BUILTIN_DEFAULTS)
    path = args.config or os.path.join(repo_root, "sixmax", "configs",
                                       "default.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        sections = {"abstraction": data.get("abstraction", {}),
                    "train.blueprint": data.get("train", {}).get("blueprint", {})}
        for section, block in sections.items():
            for key, value in block.items():
                flat = _TOML_RENAME.get(key, {}).get(section, key)
                if flat not in BUILTIN_DEFAULTS:
                    raise KeyError(f"Unknown config key [{section}].{key} in {path}")
                cfg[flat] = value
    for key in BUILTIN_DEFAULTS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            cfg[key] = cli_value
    cfg["config_path"] = path
    return cfg


def write_config_snapshot(cfg: dict, checkpoint_path: str) -> None:
    import json
    lines = ["# Effective config — written by train_sixmax.py", "[resolved]"]
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
    sixmax = _force_load_sixmax(repo_root)

    parser = argparse.ArgumentParser(description="Six-max blueprint trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None,
                        dest="num_threads")
    parser.add_argument("--num-players", type=int, default=None,
                        dest="num_players")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        dest="checkpoint_interval")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = resolve_config(args, repo_root)
    print("Effective config: "
          + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))
    vocab = _load_vocab(repo_root, cfg["config_path"])
    engine_cfg = sixmax.EngineConfig(num_players=cfg["num_players"],
                                     starting_stack=cfg["starting_stack"])
    os.makedirs(os.path.dirname(os.path.abspath(cfg["checkpoint"])),
                exist_ok=True)

    if args.resume:
        print(f"Resuming from {args.resume}")
        abstraction = sixmax.load_abstraction(args.resume)
        trainer = sixmax.resume_blueprint(
            args.resume, engine_cfg, vocab, abstraction,
            num_threads=cfg["num_threads"], seed=cfg["seed"])
    else:
        print("Building abstraction (quantile edges) …")
        abstraction = sixmax.Abstraction(
            flop_buckets=cfg["flop_buckets"],
            turn_buckets=cfg["turn_buckets"],
            river_buckets=cfg["river_buckets"],
            equity_rollouts=cfg["equity_rollouts"],
            quantile_samples=cfg["quantile_samples"],
            seed=cfg["abstraction_seed"])
        trainer = sixmax.BlueprintTrainer(
            engine_cfg, vocab, abstraction,
            num_threads=cfg["num_threads"], seed=cfg["seed"])

    ckpt_interval = cfg["checkpoint_interval"] or cfg["iterations"]
    completed = 0
    print(f"Running {cfg['iterations']:,} iterations …")
    while completed < cfg["iterations"]:
        chunk = min(ckpt_interval, cfg["iterations"] - completed)
        trainer.train(chunk)
        completed += chunk
        trainer.save(cfg["checkpoint"], vocab, engine_cfg, abstraction)
        write_config_snapshot(cfg, cfg["checkpoint"])
        print(f"[{completed:,}/{cfg['iterations']:,}] "
              f"{trainer.num_infosets():,} infosets — saved {cfg['checkpoint']}")
    print(f"Done: {trainer.iterations():,} total iterations, "
          f"{trainer.num_infosets():,} infosets.")


if __name__ == "__main__":
    main()
