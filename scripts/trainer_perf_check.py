#!/usr/bin/env python3
"""Deterministic-trainer gate: golden digests and a throughput benchmark.

Every change under sixmax/src/{blueprint,engine} must leave a single-threaded
training run bit-identical. `digest` trains a fixed scenario, dumps the saved
checkpoint's infosets and hashes the values in key order, so the digest ignores
`unordered_map` layout but pins every floating-point value. `bench` measures
the trainer's own `[rate] it/s=` line through scripts/train_sixmax.py.

Usage:
  uv run python scripts/trainer_perf_check.py digest [--scenario NAME]
  uv run python scripts/trainer_perf_check.py bench --config PATH \\
      --iterations N [--threads T]

The module is importable: `from scripts.trainer_perf_check import digest`.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

C4_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]
C5_TOML = os.path.join("sixmax", "configs", "hu200_c5.toml")
DEFAULT_TOML = os.path.join("sixmax", "configs", "default.toml")


# ---------------------------------------------------------------------------
# Extension loading (mirrors tests/sixmax/conftest.py, so the script and the
# golden test exercise the same .so)
# ---------------------------------------------------------------------------
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preload_libtorch(repo_root: str) -> None:
    """Pre-load libtorch (macOS SIP ignores DYLD_LIBRARY_PATH)."""
    lib_dir = os.path.join(repo_root, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        lib_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path)


def _discover_so_dir(repo_root: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax", "--show-output"],
                            capture_output=True, text=True, cwd=repo_root)
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(repo_root, os.path.dirname(rel_so))
    raise RuntimeError(f"Could not locate sixmax.so\nstdout: {result.stdout}\n"
                       f"stderr: {result.stderr}")


def _ensure_sixmax():
    """Return the force-loaded extension, loading it unless a test's conftest
    already did (the repo-root sixmax/ namespace package has no bindings)."""
    mod = sys.modules.get("sixmax")
    if mod is not None and hasattr(mod, "BlueprintTrainer"):
        return mod
    repo_root = _repo_root()
    _preload_libtorch(repo_root)
    so_path = os.environ.get("SIXMAX_SO_PATH")
    if not so_path:
        so_path = os.path.join(_discover_so_dir(repo_root), "sixmax.so")
    so_dir = os.path.dirname(so_path)
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod  # register before exec so circular refs work
    spec.loader.exec_module(mod)
    return mod


def _load_vocab(toml_path: str):
    _ensure_sixmax()
    path = os.path.join(_repo_root(), "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_repo_root(), toml_path), "blueprint")


def _write_idx_mod_map(path: str, counts=(7, 5, 3)) -> str:
    """Write a SIXBM001 map whose bucket is `index % k` on each street."""
    import numpy as np

    sixmax = _ensure_sixmax()
    arrays = []
    for street, k in zip((1, 2, 3), counts):
        size = sixmax.hand_index_size(street)
        values = np.empty(size, dtype=np.uint8)
        chunk = 1 << 22
        for start in range(0, size, chunk):
            stop = min(size, start + chunk)
            values[start:stop] = (
                np.arange(start, stop, dtype=np.uint64) % k).astype(np.uint8)
        arrays.append(values)
    sixmax._write_bucket_map_for_test(str(path), *arrays, "percentile")
    return str(path)


def _scenario_dir(tmp_dir: str, name: str) -> str:
    directory = os.path.join(str(tmp_dir), name)
    os.makedirs(directory, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Scenarios: each trains from a fixed seed and returns the saved checkpoint.
# ---------------------------------------------------------------------------
def _hu_bet_history(tmp_dir: str, iterations: int = 3000,
                    name: str = "hu_bet_history") -> str:
    """HU 200 BB, C5 vocab, last-raise bet history on a synthetic map."""
    sixmax = _ensure_sixmax()
    directory = _scenario_dir(tmp_dir, name)
    map_path = _write_idx_mod_map(os.path.join(directory, "idxmod.bin"))
    abstraction = sixmax.Abstraction.from_bucket_map(map_path)
    abstraction.set_pot_layout("bb", C4_EDGES)
    abstraction.set_bet_history("last_raise")
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    vocab = _load_vocab(C5_TOML)
    trainer = sixmax.BlueprintTrainer(cfg, vocab, abstraction,
                                      num_threads=1, seed=13)
    trainer.train(iterations)
    path = os.path.join(directory, name + ".bin")
    trainer.save(path, vocab, cfg, abstraction)
    return path


def _hu_resume(tmp_dir: str) -> str:
    """hu_bet_history in two halves through the --resume binding path."""
    sixmax = _ensure_sixmax()
    directory = _scenario_dir(tmp_dir, "hu_resume")
    stage1 = _hu_bet_history(directory, iterations=1500, name="hu_resume_s1")
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    vocab = _load_vocab(C5_TOML)
    # Same binding calls scripts/train_sixmax.py --resume makes: the
    # checkpoint resolves its own map (kept beside it) and the loaded
    # abstraction is imported into a fresh trainer.
    abstraction = sixmax.load_abstraction(stage1)
    trainer = sixmax.resume_blueprint(stage1, cfg, vocab, abstraction,
                                      num_threads=1, seed=13)
    trainer.train(1500)
    # Keep the final checkpoint beside the synthetic map so it can also be
    # re-opened by a loader that resolves the map by its hint.
    path = os.path.join(os.path.dirname(stage1), "hu_resume.bin")
    trainer.save(path, vocab, cfg, abstraction)
    return path


def _sixmax_legacy(tmp_dir: str) -> str:
    """6-max legacy quantile abstraction: exercises multiway side pots."""
    sixmax = _ensure_sixmax()
    directory = _scenario_dir(tmp_dir, "sixmax_legacy")
    abstraction = sixmax.Abstraction(flop_buckets=10, turn_buckets=10,
                                     river_buckets=5, equity_rollouts=40,
                                     quantile_samples=300, seed=42)
    cfg = sixmax.EngineConfig(num_players=6)
    vocab = _load_vocab(DEFAULT_TOML)
    trainer = sixmax.BlueprintTrainer(cfg, vocab, abstraction,
                                      num_threads=1, seed=7)
    trainer.train(1500)
    path = os.path.join(directory, "sixmax_legacy.bin")
    trainer.save(path, vocab, cfg, abstraction)
    return path


SCENARIOS = {
    "hu_bet_history": _hu_bet_history,
    "sixmax_legacy": _sixmax_legacy,
    "hu_resume": _hu_resume,
}


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------
def digest(scenario: str, tmp_dir: str) -> str:
    """Train `scenario` and hash its saved table, key-sorted and bit-exact."""
    sixmax = _ensure_sixmax()
    path = SCENARIOS[scenario](str(tmp_dir))
    iterations, records = sixmax.dump_infosets(path)
    h = hashlib.sha256()
    h.update(f"{iterations}\n".encode("ascii"))
    for key, probs, mass, reg in sorted(records, key=lambda r: r[0]):
        fields = [str(key)] + [float(p).hex() for p in probs]
        fields += [float(mass).hex(), float(reg).hex()]
        h.update((",".join(fields) + "\n").encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
_RATE_RE = re.compile(r"^\[rate\] iters=\d+ it/s=([0-9.]+)", re.MULTILINE)


def bench(config: str, iterations: int, threads: "int | None" = None) -> float:
    """Train `iterations` in a temp checkpoint via scripts/train_sixmax.py and
    return the final `[rate] it/s=` value."""
    repo_root = _repo_root()
    with tempfile.TemporaryDirectory(prefix="trainer_bench_") as directory:
        checkpoint = os.path.join(directory, "bench.bin")
        cmd = [sys.executable,
               os.path.join(repo_root, "scripts", "train_sixmax.py"),
               "--config", config,
               "--iterations", str(iterations),
               "--checkpoint", checkpoint,
               "--no-snapshots"]  # keep the rate on training + checkpointing
        if threads is not None:
            cmd += ["--num-threads", str(threads)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=repo_root)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"bench failed (exit {result.returncode})")
    matches = _RATE_RE.findall(result.stderr)
    if not matches:
        sys.stderr.write(result.stderr)
        raise SystemExit("bench: no [rate] line in trainer stderr")
    return float(matches[-1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_digest = sub.add_parser("digest", help="print scenario digests")
    p_digest.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)

    p_bench = sub.add_parser("bench", help="measure trainer throughput")
    p_bench.add_argument("--config", required=True)
    p_bench.add_argument("--iterations", type=int, required=True)
    p_bench.add_argument("--threads", type=int, default=None)

    args = parser.parse_args()
    if args.command == "digest":
        names = [args.scenario] if args.scenario else sorted(SCENARIOS)
        with tempfile.TemporaryDirectory(prefix="trainer_digest_") as directory:
            for name in names:
                print(f"{name} {digest(name, directory)}")
    else:
        print(f"it/s={bench(args.config, args.iterations, args.threads):.1f}")


if __name__ == "__main__":
    main()
