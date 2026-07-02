#!/usr/bin/env python3
"""
Display the CFR bot's preflop strategy as a 13x13 range chart.

Usage:
    uv run python scripts/range_chart.py
    uv run python scripts/range_chart.py --checkpoint path.pkl
    uv run python scripts/range_chart.py --scenario bb-vs-raise
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr.abstraction import hand_to_bucket, legal_abstract_actions
from cfr.info_set import InfoSet, stack_bucket
from cfr.regret_table import RegretTable
from models.card import Card
from models.enums import Suit


RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
RANK_LABEL = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
              9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}

G  = '\033[42m\033[37m'   # green bg, white text  — strong raise
Y  = '\033[43m\033[30m'   # yellow bg, dark text  — mixed raise
B  = '\033[44m\033[37m'   # blue bg, white text   — call/limp
DM = '\033[2m'             # dim                   — fold
RS = '\033[0m'             # reset

SCENARIOS = {
    'sb-open': {
        'title':       'SB Opening Range',
        'desc':        'SB acts first preflop, no prior raises',
        'betting_history': (0, 0, 0, 0),
        'stack':       99.5,   # SB posted 0.5 from 100BB
        'to_call':      0.5,
        'pot':          1.5,
        'current_bet':  1.0,
        'player_bet':   0.5,
    },
    'bb-vs-raise': {
        'title':       'BB vs SB Raise',
        'desc':        'BB facing one SB raise (~3BB open)',
        'betting_history': (1, 0, 0, 0),
        'stack':       97.0,   # BB posted 1.0, facing 3BB open → to_call=2.0
        'to_call':      2.0,
        'pot':          4.0,
        'current_bet':  3.0,
        'player_bet':   1.0,
    },
}


def _latest_checkpoint(directory="cfr/checkpoints"):
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return None
    return os.path.join(directory, sorted(files)[-1]) if files else None


def _hand_label(i, j):
    hi = RANKS[min(i, j)]
    lo = RANKS[max(i, j)]
    if i == j:
        return RANK_LABEL[hi] * 2
    suf = 's' if i < j else 'o'
    return f"{RANK_LABEL[hi]}{RANK_LABEL[lo]}{suf}"


def _sample_cards(i, j):
    r1, r2 = RANKS[i], RANKS[j]
    if i == j:
        return [Card(r1, Suit.HEARTS), Card(r1, Suit.SPADES)]
    if i < j:   # suited: row=high, col=low
        return [Card(r1, Suit.HEARTS), Card(r2, Suit.HEARTS)]
    return [Card(r2, Suit.HEARTS), Card(r1, Suit.SPADES)]  # offsuit


def _get_strategy(table, cards, sc):
    infoset = InfoSet(
        player=0,
        hand_bucket=hand_to_bucket(cards, [], 0),
        street=0,
        board_bucket=0,
        betting_history=sc['betting_history'],
        stack_bucket=stack_bucket(sc['stack']),
    )
    legal = legal_abstract_actions(
        to_call=sc['to_call'], pot=sc['pot'], stack=sc['stack'],
        current_bet=sc['current_bet'], player_bet=sc['player_bet'],
    )
    probs = table.get_average_strategy(infoset, legal)
    return dict(zip(legal, probs.tolist()))


def _cell_color(strat):
    raise_pct = sum(v for k, v in strat.items() if k.startswith('b') or k == 'allin')
    call_pct  = strat.get('call', 0.0)
    if raise_pct >= 0.70:
        return G
    if raise_pct >= 0.35:
        return Y
    if call_pct  >= 0.40:
        return B
    return DM


def print_chart(table, scenario_key):
    sc = SCENARIOS[scenario_key]
    print(f"\n\033[1mPreflop Strategy — {sc['title']} (100BB)\033[0m")
    print(f"{DM}{sc['desc']}{RS}\n")
    print(f"  {G} raise >70% {RS}  {Y} raise 35-70% {RS}  {B} call >40% {RS}  {DM}fold{RS}")
    print(f"  upper-right = suited · diagonal = pairs · lower-left = offsuit\n")

    # Pre-compute the full grid (equity bucket cache makes this fast after the first call)
    grid = {}
    for i in range(13):
        for j in range(13):
            cards = _sample_cards(i, j)
            strat  = _get_strategy(table, cards, sc)
            bucket = hand_to_bucket(cards, [], 0)
            grid[(i, j)] = (strat, bucket)

    # Header row
    col_labels = "".join(f"  {RANK_LABEL[r]}  " for r in RANKS)
    print(f"       {col_labels}")

    for i in range(13):
        row = f"  {RANK_LABEL[RANKS[i]]}  "
        for j in range(13):
            strat, _ = grid[(i, j)]
            color = _cell_color(strat)
            label = _hand_label(i, j).ljust(3)
            row += f"{color} {label} {RS}"
        print(row)

    # Per-bucket action breakdown
    bucket_strats = {}
    for (i, j), (strat, bucket) in grid.items():
        if bucket not in bucket_strats:
            bucket_strats[bucket] = strat

    print(f"\n  Bucket breakdown (0 = strongest equity):")
    for b in sorted(bucket_strats):
        strat = bucket_strats[b]
        bar = "  ".join(f"{a}:{v:.0%}" for a, v in strat.items() if v > 0.01)
        print(f"    bucket {b}: {bar}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CFR bot preflop range chart")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="sb-open")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint()
    if not checkpoint or not os.path.exists(checkpoint):
        print("Error: no checkpoint found.")
        sys.exit(1)

    print(f"Loading {checkpoint}...")
    table = RegretTable()
    table.load(checkpoint)

    print_chart(table, args.scenario)


if __name__ == "__main__":
    main()
