#!/usr/bin/env python3
"""Render the six-max blueprint's BTN-open and SB-open preflop RFI ranges as
13x13 terminal range charts, with a range-width headline for eyeballing against
human-professional opening frequencies.

Read-only diagnostic. Backend: sixmax/checkpoints/blueprint*.bin (the only bot
with a real BTN). See docs/superpowers/specs/2026-07-22-blueprint-range-chart-design.md.

Usage:
    uv run python scripts/blueprint_range_chart.py
    uv run python scripts/blueprint_range_chart.py --checkpoint sixmax/checkpoints/blueprint_01500000.bin
    uv run python scripts/blueprint_range_chart.py --position SB
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_blueprint import (  # sanctioned reuse
    _sixmax, decode_key, load_vocab_for, resolve_action_roles,
)

RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
RANK_LABEL = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
              9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}

POSITIONS = {"BTN": (2, 2), "SB": (1, 1)}


def build_index(records):
    """Map (live, after, card_id) -> strategy probs for street-0 unopened infosets."""
    idx = {}
    for key, probs, _mass, _regret in records:
        k = decode_key(key)
        if k["street"] != 0 or any(k["raises"]):
            continue
        idx[(k["live"], k["after"], k["card"])] = list(probs)
    return idx


def sample_card_ids(i, j):
    """Two concrete card codes (rank_index*4 + suit) for grid cell (i, j).
    Grid: RANKS high->low; i<j suited, i==j pair, i>j offsuit."""
    ri, rj = RANKS[i] - 2, RANKS[j] - 2   # rank_index 0='2'..12='A'
    if i == j:                            # pair: same rank, two suits
        return ri * 4 + 0, ri * 4 + 1
    if i < j:                             # suited: same suit
        return ri * 4 + 0, rj * 4 + 0
    return ri * 4 + 0, rj * 4 + 1         # offsuit: different suits


def cell_probs(index, pos_key, i, j):
    """Strategy probs for grid cell (i, j) at POSITIONS[pos_key], or None."""
    sixmax = _sixmax()
    live, after = POSITIONS[pos_key]
    c1, c2 = sample_card_ids(i, j)
    card_id = sixmax.preflop_class([c1, c2])
    return index.get((live, after, card_id))


def role_width(index, pos_key, roles):
    """Mean aggressive-role mass over the 169 grid cells present for a position."""
    total, agg = 0, 0.0
    for i in range(13):
        for j in range(13):
            p = cell_probs(index, pos_key, i, j)
            if p is None:
                continue
            total += 1
            agg += sum(p[k] for k in roles["aggressive"])
    return agg / total if total else float("nan")
