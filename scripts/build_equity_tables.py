#!/usr/bin/env python3
"""Build the exact sixmax equity feature store.

Writes a SIXEQ001 file with dense float32 equity tables for flop, turn and
river, addressed by the suit-canonical hand index. The river table is built by
exact enumeration; turn and flop are exact means over the next street. Tables
are build artifacts and live under the gitignored sixmax/tables/ directory.

Usage:
    uv run python scripts/build_equity_tables.py \\
        --out sixmax/tables/equity_v1.bin --threads 0
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
    """Pre-load libtorch shared libraries (macOS SIP ignores DYLD_LIBRARY_PATH)."""
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


def parse_args(argv=None):
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Build the exact sixmax equity feature store")
    parser.add_argument(
        "--out", default=os.path.join(root, "sixmax", "tables",
                                      "equity_v1.bin"),
        help="output SIXEQ001 path (default: sixmax/tables/equity_v1.bin)")
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

    print(f"Building equity store -> {out} (threads={args.threads})",
          flush=True)
    started = time.time()
    content_hash = sixmax.build_equity_store(out, num_threads=args.threads)
    elapsed = time.time() - started
    print(f"Done in {elapsed:.1f}s: {out} "
          f"(content hash {content_hash:#018x}, "
          f"{os.path.getsize(out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
