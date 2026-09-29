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

    def paired_bb100_ci(self, seat_a: int, seat_b: int) -> Tuple[float, float]:
        """Return (BB/100 diff, 95% CI) for seat_a minus seat_b, paired per hand."""
        da = self._deltas[seat_a]
        db = self._deltas[seat_b]
        n = min(len(da), len(db))
        if n < 2:
            return 0.0, 0.0
        diffs = [da[i] - db[i] for i in range(n)]
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        bb100 = 100.0 * mean / self._bb
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

        # Paired CI: sixmax vs each potodds seat
        potodds_seats = [
            (i, lbl) for i, lbl in enumerate(self._labels) if lbl.startswith("potodds")
        ]
        if potodds_seats:
            sixmax_seat = self._labels.index("sixmax")
            paired_rows = []
            for seat_b, lbl_b in potodds_seats:
                diff, ci = self.paired_bb100_ci(sixmax_seat, seat_b)
                paired_rows.append([f"sixmax − {lbl_b}", f"{diff:+.1f}", f"±{ci:.1f}"])
            print()
            if _tabulate:
                print(_tabulate(paired_rows,
                                headers=["Matchup", "BB/100 diff", "95% CI (paired)"],
                                tablefmt="simple"))
            else:
                print(f"{'Matchup':>20}  {'BB/100 diff':>12}  {'95% CI (paired)':>16}")
                for r in paired_rows:
                    print(f"{r[0]:>20}  {r[1]:>12}  {r[2]:>16}")


SEAT_LABELS = ["sixmax", "neural", "tabular", "potodds_3", "potodds_4", "potodds_5"]
_BB = 2
_SB = 1
_STACK = 2000


def _build_agents(blueprint_path: str, neural_path: str, tabular_path: str):
    """Instantiate all six agents. Fails fast if any checkpoint is missing."""
    for label, path in [("--blueprint", blueprint_path),
                        ("--neural", neural_path),
                        ("--tabular", tabular_path)]:
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found for {label}: {path}")

    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _toml = os.path.join(_repo, "sixmax", "configs", "default.toml")

    from agents.sixmax_agent import SixmaxAgent
    from agents.neural_agent import NeuralAgent
    from agents.cfr_agent import CFRAgent
    from agents.potodds_agent import PotOddsAgent

    sixmax = SixmaxAgent(blueprint_path, config_toml=_toml)
    neural = NeuralAgent(neural_path)
    tabular = CFRAgent(tabular_path)
    po3 = PotOddsAgent()
    po4 = PotOddsAgent()
    po5 = PotOddsAgent()
    return [sixmax, neural, tabular, po3, po4, po5]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-table multi-agent eval (BB/100 + chip matrix)")
    parser.add_argument("--blueprint", required=True,
                        help="SixmaxAgent checkpoint (.bin)")
    parser.add_argument("--neural", required=True,
                        help="NeuralAgent checkpoint (.pt)")
    parser.add_argument("--tabular", required=True,
                        help="CFRAgent checkpoint (.pkl)")
    parser.add_argument("--hands", type=int, default=10000,
                        help="number of hands to play (default 10000)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--csv", default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    agents = _build_agents(args.blueprint, args.neural, args.tabular)
    players = [
        Player(SEAT_LABELS[i], _STACK, agent=agents[i])
        for i in range(6)
    ]

    obs = MixedTableObserver(SEAT_LABELS, bb=_BB)
    game = PokerGame(players, small_blind=_SB, observers=[obs])

    sixmax_player = players[SEAT_LABELS.index("sixmax")]
    hands_played = 0
    for i in range(args.hands):
        random.seed(args.seed * 1_000_003 + i)
        game.play_hand()
        hands_played += 1
        if sixmax_player.stack == 0:
            print(f"[stopped after {hands_played} hands — sixmax busted]")
            break
        for p in players:          # rebuy others to keep 6-handed table
            if p is not sixmax_player and p.stack == 0:
                p.stack = _STACK

    print(f"\nMixed-table eval — {hands_played} hands, seed {args.seed}\n")
    obs.report()

    if args.csv:
        _write_csv(obs, args.csv)
        print(f"\nwrote {args.csv}")


def _write_csv(obs: MixedTableObserver, path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "hands", "bb100", "ci_95"])
        for seat, label in enumerate(SEAT_LABELS):
            bb100, ci = obs.bb100_ci(seat)
            w.writerow([label, obs.hands_played, f"{bb100:.4f}", f"{ci:.4f}"])
        w.writerow([])
        w.writerow([""] + SEAT_LABELS)
        for i, row_label in enumerate(SEAT_LABELS):
            w.writerow([row_label] + [
                "—" if i == j else f"{obs.chip_matrix[i][j]:.4f}"
                for j in range(6)
            ])


if __name__ == "__main__":
    main()
