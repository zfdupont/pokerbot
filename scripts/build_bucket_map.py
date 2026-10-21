#!/usr/bin/env python3
"""Build a sixmax bucket map from the exact equity feature store.

Writes a SIXBM001 file: a dense uint8 bucket id per suit-canonical (hole,
board), one array per street, plus a header carrying the hand-index version,
bucket counts, build method/params, the equity store hash and a content hash.
Tables are build artifacts and live under the gitignored sixmax/tables/.

Usage:
    uv run python scripts/build_bucket_map.py \\
        --equity sixmax/tables/equity_v1.bin --method percentile \\
        --counts 50,50,20 --out sixmax/tables/map_pct_50_50_20.bin
"""
import argparse
import ctypes
import importlib.util
import os
import subprocess
import sys
import time


def _repo_root() -> str:
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
    raise RuntimeError(
        f"Could not locate sixmax.so in buck2 output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}")


def _preload_libtorch(repo_root: str) -> None:
    """Pre-load libtorch (macOS SIP ignores DYLD_LIBRARY_PATH)."""
    lib_dir = os.path.join(repo_root, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        lib_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path)


def _force_load_sixmax(repo_root: str):
    """Force-load the extension (the repo-root sixmax/ dir shadows the .so)."""
    _preload_libtorch(repo_root)
    so_path = os.environ.get("SIXMAX_SO_PATH")
    if not so_path:
        so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


def _counts(text: str):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "counts must be flop,turn,river (three integers)")
    try:
        values = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "counts must be flop,turn,river (three integers)")
    for value in values:
        if not 1 <= value <= 255:
            raise argparse.ArgumentTypeError(
                "each bucket count must be in [1, 255]")
    return values


def parse_args(argv=None):
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Build a sixmax bucket map (SIXBM001)")
    parser.add_argument(
        "--equity",
        default=os.path.join(root, "sixmax", "tables", "equity_v1.bin"),
        help="input SIXEQ001 feature store "
             "(default: sixmax/tables/equity_v1.bin)")
    parser.add_argument(
        "--out",
        default=os.path.join(root, "sixmax", "tables", "bucket_map.bin"),
        help="output SIXBM001 path (default: sixmax/tables/bucket_map.bin)")
    parser.add_argument(
        "--method", choices=["percentile", "potential"], default="percentile",
        help="bucketing method (default: percentile)")
    parser.add_argument(
        "--counts", type=_counts, default=_counts("50,50,20"),
        help="flop,turn,river bucket counts (default: 50,50,20)")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="k-means seed for --method potential (default: 0)")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="worker threads (0 = hardware concurrency)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = _repo_root()
    sixmax = _force_load_sixmax(root)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"Building bucket map -> {out} "
          f"(method={args.method}, counts={args.counts}, seed={args.seed}, "
          f"threads={args.threads})", flush=True)
    started = time.time()
    sixmax.build_bucket_map(
        args.equity, out, args.method, args.counts,
        seed=args.seed, num_threads=args.threads)
    elapsed = time.time() - started
    bucket_map = sixmax.BucketMap.open(out)
    counts = [bucket_map.num_buckets(s) for s in (1, 2, 3)]
    print(f"Done in {elapsed:.1f}s: {out} "
          f"(buckets {counts}, content hash "
          f"{bucket_map.content_hash():#018x}, "
          f"{os.path.getsize(out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
