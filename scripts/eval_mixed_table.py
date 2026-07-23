#!/usr/bin/env python3
"""Mixed-table multi-agent eval — seats all trained agents at one 6-handed
table and reports BB/100 per agent and a net chip matrix.

Usage:
    uv run python scripts/eval_mixed_table.py \
        --blueprint sixmax/checkpoints/blueprint.bin \
        --neural neural_cfr/checkpoints/checkpoint.pt \
        --tabular cfr/checkpoints/checkpoint_09040000.pkl \
        [--hands 10000] [--seed 1] [--csv out.csv]
"""
import argparse
import math
import os
import random
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player import Player          # noqa: E402
from game.poker import PokerGame          # noqa: E402
from util.observer import GameObserver    # noqa: E402


class MixedTableObserver(GameObserver):
    """Accumulates per-seat chip deltas and builds the BB/100 + chip matrix.

    Chip matrix attribution: when seat i gains delta[i] > 0 chips in a hand,
    distribute delta[i] / num_losers to M[i][j] for each losing seat j, and
    -delta[i] / num_losers to M[j][i]. The matrix is antisymmetric:
    M[i][j] = -M[j][i].
    """

    def __init__(self, seat_labels: List[str], bb: int) -> None:
        self._labels = seat_labels
        self._bb = bb
        self._n = len(seat_labels)
        self._hands = 0
        # per-hand chip delta list for each seat (for CI calculation)
        self._deltas: List[List[float]] = [[] for _ in range(self._n)]
        # n×n chip matrix: _matrix[i][j] = chips seat i won from seat j
        self._matrix: List[List[float]] = [
            [0.0] * self._n for _ in range(self._n)
        ]
        self._stack_before: List[float] = [0.0] * self._n
        self._players: List = []

    # ------------------------------------------------------------------
    # GameObserver hooks
    # ------------------------------------------------------------------

    def on_hand_start(self, players, button_pos: int) -> None:
        self._players = list(players)
        for i, p in enumerate(players):
            self._stack_before[i] = float(p.stack)

    def on_hand_complete(self, winners, pot: int) -> None:
        n = self._n
        deltas = [
            float(self._players[i].stack) - self._stack_before[i]
            for i in range(n)
        ]

        for i in range(n):
            self._deltas[i].append(deltas[i])

        # Chip matrix attribution:
        # For each gainer i and loser j, seat i won delta[i]/num_losers from j.
        gainers = [i for i in range(n) if deltas[i] > 1e-9]
        losers = [j for j in range(n) if deltas[j] < -1e-9]
        num_losers = len(losers) if losers else 1

        for i in gainers:
            share = deltas[i] / num_losers
            for j in losers:
                self._matrix[i][j] += share
                self._matrix[j][i] -= share

        self._hands += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bb100_ci(self, seat: int) -> Tuple[float, float]:
        """Return (BB/100, 95% half-width CI) for a seat.

        Returns (0.0, 0.0) if no hands played. CI uses ddof=1 std and the
        formula: 1.96 * std(per_hand_deltas) / sqrt(n), converted to BB/100.
        """
        ds = self._deltas[seat]
        n = len(ds)
        if n == 0:
            return 0.0, 0.0
        mean = sum(ds) / n
        bb100 = 100.0 * mean / self._bb
        if n < 2:
            return bb100, 0.0
        var = sum((d - mean) ** 2 for d in ds) / (n - 1)
        ci = 1.96 * 100.0 * math.sqrt(var) / (self._bb * math.sqrt(n))
        return bb100, ci

    @property
    def chip_matrix(self) -> List[List[float]]:
        return self._matrix

    @property
    def hands_played(self) -> int:
        return self._hands

    def report(self) -> None:
        try:
            from tabulate import tabulate
            _tabulate = tabulate
        except ImportError:
            _tabulate = None

        rows = []
        for seat in range(self._n):
            bb100, ci = self.bb100_ci(seat)
            rows.append([self._labels[seat], self._hands,
                         f"{bb100:+.1f}", f"±{ci:.1f}"])

        header = ["Agent", "Hands", "BB/100", "95% CI"]
        if _tabulate:
            print(_tabulate(rows, headers=header, tablefmt="simple"))
        else:
            print("  ".join(f"{h:>10}" for h in header))
            for r in rows:
                print("  ".join(f"{v:>10}" for v in r))

        print()
        col_labels = [""] + self._labels
        mat_rows = []
        for i, label in enumerate(self._labels):
            row = [label]
            for j in range(self._n):
                row.append("—" if i == j else f"{self._matrix[i][j]:+.2f}")
            mat_rows.append(row)

        if _tabulate:
            print(_tabulate(mat_rows, headers=col_labels, tablefmt="simple"))
        else:
            print("  ".join(f"{h:>10}" for h in col_labels))
            for r in mat_rows:
                print("  ".join(f"{str(v):>10}" for v in r))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-table multi-agent poker eval"
    )
    parser.add_argument("--blueprint", metavar="PATH",
                        help="Path to sixmax blueprint checkpoint (.bin)")
    parser.add_argument("--neural", metavar="PATH",
                        help="Path to neural CFR checkpoint (.pt)")
    parser.add_argument("--tabular", metavar="PATH",
                        help="Path to tabular CFR checkpoint (.pkl)")
    parser.add_argument("--hands", type=int, default=10000,
                        help="Number of hands to play (default: 10000)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--csv", metavar="PATH",
                        help="Write BB/100 results to CSV file")
    args = parser.parse_args()

    print(f"Mixed-table eval: {args.hands} hands, seed={args.seed}")
    print("Agent wiring not yet implemented — observer module ready.")


if __name__ == "__main__":
    main()
