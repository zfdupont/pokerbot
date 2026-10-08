#!/usr/bin/env python3
"""Measure how often Phase-2 search fires, its fallback rate, and wall time per hand.

Portable companion to docs/superpowers/specs/2026-10-07-sixmax-phase2-search-ab-results-design.md
(Appendix A). Run from anywhere; the repo root is derived from this file's location.

Usage:
    uv run python scripts/probe_search.py [hands] [on|off]

Search knobs are read from the SIXMAX_SEARCH_* env vars by agents.sixmax_agent;
PROBE_CKPT / PROBE_TOML override the blueprint checkpoint and config.
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agents.sixmax_agent import SixmaxAgent
from agents.cfr_agent import CFRAgent
from scripts.eval_hu_sanity import _play_seeded

CKPT = os.environ.get("PROBE_CKPT",
                      os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin"))
TAB = os.path.join(_ROOT, "cfr", "checkpoints", "checkpoint_09040000.pkl")
TOML = os.environ.get("PROBE_TOML",
                      os.path.join(_ROOT, "sixmax", "configs", "default.toml"))

hands = int(sys.argv[1]) if len(sys.argv) > 1 else 50
on = sys.argv[2] != "off" if len(sys.argv) > 2 else True
os.environ["SIXMAX_SEARCH_ENABLED"] = "true" if on else "false"

hero = SixmaxAgent(CKPT, search_toml=TOML) if on else SixmaxAgent(CKPT)
villain = CFRAgent(TAB)

total = 0.0
t0 = time.perf_counter()
for h in range(hands):
    s = 999983 + h
    for seat in (0, 1):
        total += _play_seeded(hero, villain, seat, s, 2, 200)
dt = time.perf_counter() - t0

print(f"hands={hands} (x2 games)  on={on}  wall={dt:.2f}s  "
      f"per_hand={dt / (hands * 2) * 1000:.1f}ms")
print(f"hero stack delta total={total:+.1f} chips  "
      f"BB/100={100.0 * total / (hands * 2 * 2):+.2f}")
if on:
    impl = hero._search._impl
    print(f"searches={impl.searches()}  fallbacks={impl.fallbacks()}")
