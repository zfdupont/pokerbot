# Mixed-Table Multi-Agent Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/eval_mixed_table.py` that seats all four trained agents at one 6-handed table, runs N hands, and reports BB/100 per agent and a net chip matrix.

**Architecture:** A `MixedTableObserver(GameObserver)` owns all metric state and is wired into `PokerGame(observers=[obs])`; the main loop is a thin seed-and-play loop with no metric logic. Results are printed as two tabulated tables and optionally written to CSV.

**Tech Stack:** Python stdlib (`argparse`, `math`, `random`, `csv`), `tabulate` (already used across the project), `game.poker.PokerGame`, `util.observer.GameObserver`, `agents.*`.

## Global Constraints

- `SixmaxAgent` must be seated at a 6-handed table (`num_players=6`); never fewer.
- All three checkpoint args (`--blueprint`, `--neural`, `--tabular`) are required; fail fast before any hands if a path is missing or does not exist.
- Agents instantiated once before the hand loop; no per-hand checkpoint reload.
- Deck seeding: `random.seed(seed * 1_000_003 + i)` before each hand; agents use their own `random.Random()` instances.
- Big blind = 2, small blind = 1, starting stack = 200 (training frame for SixmaxAgent).
- Never commit checkpoint files (`*.bin`, `*.pt`, `*.pkl`).
- No modifications to existing scripts, engine, or agents.

---

### Task 1: MixedTableObserver — metric state + chip matrix

**Files:**
- Create: `scripts/eval_mixed_table.py`

**Interfaces:**
- Produces: `MixedTableObserver` class with `__init__(seat_labels, bb)`, `on_hand_start(players, button_pos)`, `on_hand_complete(winners, pot)`, `bb100_ci(seat)` → `(float, float)`, `chip_matrix` property → `list[list[float]]`, `hands_played` property → `int`.

- [ ] **Step 1: Write failing tests for MixedTableObserver**

Create `tests/scripts/test_eval_mixed_table.py`:

```python
"""Tests for eval_mixed_table — observer math, chip conservation, CLI smoke."""
import math
import os
import sys

import pytest

from models.player import Player
from models.enums import Action
from agents.base_agent import PokerAgent
from game.poker import PokerGame

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FoldAgent(PokerAgent):
    def get_action(self, player, game_state):
        to_call = game_state.current_bet - player.current_bet
        return (Action.CHECK, None) if to_call <= 0 else (Action.FOLD, None)


def _make_players(n, stack, agents):
    return [Player(f"p{i}", stack, agent=agents[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Observer: chips conserved every hand
# ---------------------------------------------------------------------------
def test_chips_conserved_per_hand():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(20):
        random.seed(i)
        game.play_hand()

    assert obs.hands_played == 20
    # Row sums of chip_matrix must match per-agent cumulative deltas within 1e-6
    for seat in range(6):
        row_sum = sum(obs.chip_matrix[seat])
        bb100, _ = obs.bb100_ci(seat)
        agent_total = bb100 / 100.0 * 2 * obs.hands_played  # BB back to chips
        assert abs(row_sum - agent_total) < 1e-4


def test_bb100_is_finite_after_hands():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(10):
        random.seed(i)
        game.play_hand()

    for seat in range(6):
        bb100, ci = obs.bb100_ci(seat)
        assert math.isfinite(bb100)
        assert math.isfinite(ci)
        assert ci >= 0.0


def test_chip_matrix_row_sums_match_totals():
    from scripts.eval_mixed_table import MixedTableObserver
    import random

    labels = [f"s{i}" for i in range(6)]
    obs = MixedTableObserver(labels, bb=2)
    agents = [FoldAgent() for _ in range(6)]
    players = _make_players(6, 200, agents)
    game = PokerGame(players, small_blind=1, observers=[obs])
    for i in range(15):
        random.seed(i)
        game.play_hand()

    for seat in range(6):
        row_sum = sum(obs.chip_matrix[seat])
        col_sum = sum(obs.chip_matrix[j][seat] for j in range(6))
        # chip matrix is antisymmetric: M[i][j] = -M[j][i]
        assert abs(row_sum + col_sum) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError` or `ImportError` for `scripts.eval_mixed_table`.

- [ ] **Step 3: Create eval_mixed_table.py with MixedTableObserver**

Create `scripts/eval_mixed_table.py`:

```python
#!/usr/bin/env python3
"""Mixed-table multi-agent eval — seats all four trained agents at one 6-handed
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
    """Accumulates per-seat chip deltas and builds the BB/100 + chip matrix."""

    def __init__(self, seat_labels: List[str], bb: int):
        self._labels = seat_labels
        self._bb = bb
        self._n = len(seat_labels)
        self._hands = 0
        # per-hand chip delta list for each seat (for CI calculation)
        self._deltas: List[List[float]] = [[] for _ in range(self._n)]
        # 6×6 chip matrix: _matrix[i][j] = chips seat i won from seat j
        self._matrix: List[List[float]] = [
            [0.0] * self._n for _ in range(self._n)
        ]
        self._stack_before: List[float] = [0.0] * self._n

    # ------------------------------------------------------------------
    # GameObserver hooks
    # ------------------------------------------------------------------

    def on_hand_start(self, players, button_pos: int) -> None:
        for i, p in enumerate(players):
            self._stack_before[i] = p.stack

    def on_hand_complete(self, winners, pot: int) -> None:
        # players list not passed; reconstruct delta from stack snapshots by
        # reading current stacks via the same observer event sequence.
        # We need current stacks — PokerGame restores stacks by dealing
        # winnings before calling this hook, so we access via the winners arg.
        # Build delta by tracking who lost chips this hand.
        # winners: List[Tuple[Player, rank]] — each entry is (player, hand_rank)
        # We compute delta as (current_stack - stack_before) for each seat.
        # Since Player objects are passed through on_hand_start, we must
        # store the player list there to recover stacks here.
        pass  # filled in with player list approach below — see note

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bb100_ci(self, seat: int) -> Tuple[float, float]:
        """(BB/100, 95% half-width CI) for a seat. Returns (0.0, 0.0) if no
        hands played."""
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
        except ImportError:
            tabulate = None

        rows = []
        for seat in range(self._n):
            bb100, ci = self.bb100_ci(seat)
            rows.append([self._labels[seat], self._hands,
                         f"{bb100:+.1f}", f"±{ci:.1f}"])

        header = ["Agent", "Hands", "BB/100", "95% CI"]
        if tabulate:
            print(tabulate(rows, headers=header, tablefmt="simple"))
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

        if tabulate:
            print(tabulate(mat_rows, headers=col_labels, tablefmt="simple"))
        else:
            print("  ".join(f"{h:>10}" for h in col_labels))
            for r in mat_rows:
                print("  ".join(f"{str(v):>10}" for v in r))
```

> **NOTE:** The `on_hand_complete` stub above is intentional — it needs access to the player list captured in `on_hand_start`. Step 4 completes the observer by storing the player list.

- [ ] **Step 4: Complete on_hand_start/on_hand_complete with player tracking**

Replace the stub body in `scripts/eval_mixed_table.py` with the full implementation. The key insight is that `on_hand_start` receives the live `Player` objects, so we hold a reference to read their stacks after the hand settles in `on_hand_complete`:

```python
def on_hand_start(self, players, button_pos: int) -> None:
    self._players = list(players)
    for i, p in enumerate(players):
        self._stack_before[i] = float(p.stack)

def on_hand_complete(self, winners, pot: int) -> None:
    n = self._n
    deltas = [float(self._players[i].stack) - self._stack_before[i]
              for i in range(n)]

    for i in range(n):
        self._deltas[i].append(deltas[i])

    # Chip matrix attribution: each winner i gains delta[i] chips;
    # attribute evenly across all losers.
    losers = [j for j in range(n) if deltas[j] < -1e-9]
    gainers = [i for i in range(n) if deltas[i] > 1e-9]
    num_losers = len(losers) or 1
    for i in gainers:
        for j in losers:
            share = -deltas[i] / num_losers
            self._matrix[i][j] += -share
            self._matrix[j][i] += share

    self._hands += 1
```

Add `self._players: List = []` to `__init__`.

- [ ] **Step 5: Run observer tests**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py::test_chips_conserved_per_hand tests/scripts/test_eval_mixed_table.py::test_bb100_is_finite_after_hands tests/scripts/test_eval_mixed_table.py::test_chip_matrix_row_sums_match_totals -v
```
Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_mixed_table.py tests/scripts/test_eval_mixed_table.py
git commit -m "feat(eval): MixedTableObserver — per-seat BB/100 + chip matrix"
```

---

### Task 2: CLI, agent wiring, CSV output

**Files:**
- Modify: `scripts/eval_mixed_table.py` (add `main()`, `_build_agents()`, CSV output)

**Interfaces:**
- Consumes: `MixedTableObserver` from Task 1; `SixmaxAgent`, `NeuralAgent`, `CFRAgent`, `PotOddsAgent` from `agents/`.
- Produces: runnable `main()` with `--blueprint`, `--neural`, `--tabular`, `--hands`, `--seed`, `--csv` flags.

- [ ] **Step 1: Write failing CLI smoke test**

Add to `tests/scripts/test_eval_mixed_table.py`:

```python
def test_main_runs_without_checkpoints(tmp_path, monkeypatch):
    """main() fails fast with SystemExit when checkpoint paths are missing."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_mixed_table",
         "--blueprint", str(tmp_path / "missing.bin"),
         "--neural", str(tmp_path / "missing.pt"),
         "--tabular", str(tmp_path / "missing.pkl"),
         "--hands", "5"],
        capture_output=True, text=True,
        cwd=_ROOT,
    )
    assert result.returncode != 0
    assert "missing" in result.stderr.lower() or "not found" in result.stderr.lower() \
        or result.returncode == 1
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py::test_main_runs_without_checkpoints -v
```
Expected: FAIL (no `main()` yet).

- [ ] **Step 3: Add _build_agents and main() to eval_mixed_table.py**

Append to `scripts/eval_mixed_table.py` after the `MixedTableObserver` class:

```python
SEAT_LABELS = ["sixmax", "neural", "tabular", "potodds_3", "potodds_4", "potodds_5"]
_BB = 2
_SB = 1
_STACK = 200


def _build_agents(blueprint_path: str, neural_path: str, tabular_path: str):
    """Instantiate all six agents. Fails fast if any checkpoint is missing."""
    for label, path in [("--blueprint", blueprint_path),
                        ("--neural", neural_path),
                        ("--tabular", tabular_path)]:
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found for {label}: {path}")

    import importlib.util as _ilu
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

    for i in range(args.hands):
        random.seed(args.seed * 1_000_003 + i)
        game.play_hand()

    print(f"\nMixed-table eval — {args.hands} hands, seed {args.seed}\n")
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
```

- [ ] **Step 4: Run CLI test**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py::test_main_runs_without_checkpoints -v
```
Expected: PASS.

- [ ] **Step 5: Run full observer test suite**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_mixed_table.py tests/scripts/test_eval_mixed_table.py
git commit -m "feat(eval): mixed-table CLI, agent wiring, CSV output"
```

---

### Task 3: Integration smoke test with real blueprints

**Files:**
- Modify: `tests/scripts/test_eval_mixed_table.py` (add `test_smoke_with_fixtures`)

**Interfaces:**
- Consumes: `blueprint_6max_ckpt`, `blueprint_hu_ckpt` session fixtures (from `tests/fixtures/sixmax_checkpoints.py`); `PotOddsAgent` as neural substitute when no `.pt` exists.
- Produces: smoke test verifying chip conservation + finite BB/100 over 50 hands.

- [ ] **Step 1: Write the smoke test**

Add to `tests/scripts/test_eval_mixed_table.py`:

```python
def test_smoke_50_hands(blueprint_6max_ckpt, blueprint_hu_ckpt):
    """50-hand mixed-table smoke: chip conservation + finite BB/100.

    NeuralAgent is skipped (no .pt fixture); a third PotOddsAgent fills seat 1.
    blueprint_hu_ckpt is loaded as the SixmaxAgent even though it is a 2-player
    checkpoint — the fixture trains a valid .bin that loads without error.
    We use blueprint_6max_ckpt for SixmaxAgent (correct 6-player training frame).
    """
    import random as _random
    from scripts.eval_mixed_table import MixedTableObserver, SEAT_LABELS, _BB, _SB, _STACK
    from agents.sixmax_agent import SixmaxAgent
    from agents.cfr_agent import CFRAgent
    from agents.potodds_agent import PotOddsAgent
    from models.player import Player
    from game.poker import PokerGame

    toml = os.path.join(_ROOT, "sixmax", "configs", "default.toml")

    # Substitute PotOddsAgent for NeuralAgent (no .pt fixture available)
    agents_under_test = [
        SixmaxAgent(blueprint_6max_ckpt, config_toml=toml),
        PotOddsAgent(),   # neural substitute
        PotOddsAgent(),   # tabular substitute (CFRAgent needs real .pkl)
        PotOddsAgent(),
        PotOddsAgent(),
        PotOddsAgent(),
    ]
    players = [Player(SEAT_LABELS[i], _STACK, agent=agents_under_test[i])
               for i in range(6)]
    obs = MixedTableObserver(SEAT_LABELS, bb=_BB)
    game = PokerGame(players, small_blind=_SB, observers=[obs])

    for i in range(50):
        _random.seed(1 * 1_000_003 + i)
        game.play_hand()

    assert obs.hands_played == 50

    # Chip conservation: sum of all per-hand deltas across all seats = 0
    for hand_idx in range(50):
        hand_total = sum(obs._deltas[seat][hand_idx] for seat in range(6))
        assert abs(hand_total) < 1e-6, \
            f"chips not conserved in hand {hand_idx}: total={hand_total}"

    # All BB/100 finite
    for seat in range(6):
        bb100, ci = obs.bb100_ci(seat)
        assert math.isfinite(bb100), f"seat {seat} BB/100 is not finite"
        assert ci >= 0.0

    # Chip matrix row sums match cumulative deltas
    for seat in range(6):
        row_sum = sum(obs.chip_matrix[seat])
        total_delta = sum(obs._deltas[seat])
        assert abs(row_sum - total_delta) < 1e-4, \
            f"seat {seat} row sum {row_sum:.4f} != delta total {total_delta:.4f}"
```

- [ ] **Step 2: Run the smoke test**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/scripts/test_eval_mixed_table.py::test_smoke_50_hands -v
```
Expected: PASS (SixmaxAgent + 5× PotOddsAgent, 50 hands).

- [ ] **Step 3: Run the full test suite to catch regressions**

```bash
cd /Users/zfdupont/pokerbot && uv run pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all 223+ tests pass (or previously-skipped tests remain skipped).

- [ ] **Step 4: Commit**

```bash
git add tests/scripts/test_eval_mixed_table.py
git commit -m "test(eval): 50-hand mixed-table smoke with real blueprints"
```
