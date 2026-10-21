#!/usr/bin/env python3
"""Regenerate tests/fixtures/legacy_buckets.json from the legacy abstraction.

Captures, for the two real SIXBP001 checkpoints under sixmax/checkpoints/, the
legacy (Monte-Carlo percentile) `Abstraction::hash()` plus `bucket()` for a
fixed seeded sample of (hole, board) deals spread evenly across flop, turn and
river. The golden test recomputes those values against the current build, so any
accidental change to the legacy bucket path (edges, equity salt, cache
semantics) fails loudly.

`hash()` is stored as a decimal string: it is a uint64 and JSON numbers are
doubles, so a numeric form would not round-trip.

Usage:
    uv run python tests/fixtures/make_legacy_buckets.py
"""
import ctypes
import importlib.util
import json
import os
import random
import subprocess
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "legacy_buckets.json")
_CHECKPOINTS = ("blueprint.bin", "hu_blueprint.bin")
_SEED = 20261020
_DEALS_PER_STREET = 100


def _build_and_get_so_dir() -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError(
        f"Could not locate sixmax.so in buck2 output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}")


def _force_load_sixmax():
    # macOS SIP ignores DYLD_LIBRARY_PATH, so pre-load libtorch by hand.
    lib_dir = os.path.join(_REPO_ROOT, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        lib_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path)
    so_path = os.environ.get("SIXMAX_SO_PATH")
    if not so_path:
        so_path = os.path.join(_build_and_get_so_dir(), "sixmax.so")
    # The repo-root sixmax/ package dir shadows the extension module.
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


def _deals():
    rng = random.Random(_SEED)
    deals = []
    for board_size in (3, 4, 5):  # flop, turn, river
        for _ in range(_DEALS_PER_STREET):
            cards = rng.sample(range(52), 2 + board_size)
            deals.append({"hole": cards[:2], "board": cards[2:]})
    return deals


def main() -> None:
    sixmax = _force_load_sixmax()
    deals = _deals()
    out = {"seed": _SEED, "deals_per_street": _DEALS_PER_STREET,
           "deals": deals, "checkpoints": {}}
    for name in _CHECKPOINTS:
        path = os.path.join(_REPO_ROOT, "sixmax", "checkpoints", name)
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}")
        abs_ = sixmax.load_abstraction(path)
        out["checkpoints"][name] = {
            "hash": str(abs_.hash()),
            "buckets": [abs_.bucket(d["hole"], d["board"]) for d in deals],
        }
        print(f"{name}: hash={abs_.hash()}")
    with open(_FIXTURE, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {_FIXTURE} ({len(deals)} deals)")


if __name__ == "__main__":
    main()
