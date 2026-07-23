# Mixed-Table Multi-Agent Eval Design

**Date:** 2026-07-23
**Status:** Approved

## Overview

A new script `scripts/eval_mixed_table.py` seats all four trained agents (sixmax blueprint, neural CFR, tabular CFR, pot-odds) at a single 6-handed table and runs 10k hands, reporting BB/100 per agent and a net chip matrix.

## Motivation

The existing evals (`eval_sixmax_baseline.py`, `eval_hu_sanity.py`) measure one agent vs identical villains. A mixed table answers: _relative to each other_, how do the four agents rank, and who is exploiting whom?

## Seating

Six fixed seats, `small_blind=1, big_blind=2, stack=200` (training frame):

| Seat | Agent | Label |
|------|-------|-------|
| 0 | `SixmaxAgent` | `sixmax` |
| 1 | `NeuralAgent` | `neural` |
| 2 | `CFRAgent` | `tabular` |
| 3 | `PotOddsAgent` | `potodds_3` |
| 4 | `PotOddsAgent` (filler) | `potodds_4` |
| 5 | `PotOddsAgent` (filler) | `potodds_5` |

Agents are instantiated once before the hand loop and reused — no per-hand checkpoint reload. The three PotOdds seats serve as both a named agent under test and two symmetric fillers; their clustering validates variance estimates.

## CLI

```bash
uv run python scripts/eval_mixed_table.py \
    --blueprint sixmax/checkpoints/blueprint.bin \
    --neural neural_cfr/checkpoints/checkpoint.pt \
    --tabular cfr/checkpoints/checkpoint_09040000.pkl \
    [--hands 10000] [--seed 1] [--csv out.csv]
```

All three checkpoint args are required. Missing checkpoints fail fast before any hands are played.

## Architecture

### `MixedTableObserver(GameObserver)`

Implements the existing `util.observer.GameObserver` interface and is passed into `PokerGame(observers=[obs])`. Owns all metric state:

- `on_hand_start(players, button_pos)` — snapshots `stack_before[seat]` for all 6 seats.
- `on_hand_complete(winners, pot)` — computes `delta[seat] = stack_after - stack_before` for each seat; accumulates into per-agent BB/100 running totals and the 6×6 chip matrix.

The main loop is:

```python
obs = MixedTableObserver(seat_labels, bb=2)
game = PokerGame(players, small_blind=1, observers=[obs])
for i in range(hands):
    random.seed(seed * 1_000_003 + i)
    game.play_hand()
obs.report()
```

No stack-tracking in the main loop. A future `ConsoleObserver` can be appended to the same list without touching eval logic.

### Chip Matrix Attribution

When seat `i` gains `delta[i] > 0` and seats `j` lose chips in the same hand, `-delta[i] / num_losers` is distributed evenly across losing seats. This is an approximation for multi-way pots (chips don't have exact per-opponent attribution in a 3-way all-in), but it is symmetric and consistent across hands.

## Metrics Output

**Table 1 — BB/100 per agent:**
```
Agent        Hands    BB/100    95% CI
-----------  -----  --------  --------
sixmax       10000    +18.4     ±12.1
neural       10000     +6.2     ±11.8
tabular      10000     -4.7     ±11.9
potodds_3    10000    -12.3     ±10.4
potodds_4    10000    -11.8     ±10.6
potodds_5    10000    -11.9     ±10.5
```

CI = `1.96 * std(per_hand_deltas) / sqrt(n)`.

**Table 2 — Net chip matrix (chips won per hand, row wins from column):**
```
          sixmax  neural  tabular  po_3  po_4  po_5
sixmax      —      +1.2    +2.1   +4.3  +4.1  +4.4
neural      -1.2    —      +1.8   +3.1  +3.0  +3.2
...
```

Row sums match each agent's total chip delta (sanity check).

With `--csv`, writes both tables to CSV (two sections separated by a blank line).

## Testing

`tests/scripts/test_eval_mixed_table.py`:

- Uses `blueprint_6max_ckpt` and `blueprint_hu_ckpt` session fixtures (dev-scale, trained in `tmp_path`).
- Skips neural if no `.pt` checkpoint exists (substitutes a third PotOddsAgent).
- Runs 50 hands.
- Asserts:
  - Chips conserved across all seats every hand (sum of deltas == 0).
  - Final BB/100 is finite for all agents.
  - Chip matrix row sums match per-agent totals within 1e-6.

## Constraints

- `SixmaxAgent` was trained for 6 players — seating at a 6-handed table is correct. Running fewer seats would violate the training frame.
- `NeuralAgent` was trained heads-up; its `position` feature assumes seat 0 = button, seat 1 = BB. At a 6-handed table its position encoding is approximate — results should be interpreted accordingly.
- The global `random` stream seeds the deck only; agents use their own `random.Random()` instances so seeding the deck does not affect agent decisions.
- Never commit checkpoint files (`*.bin`, `*.pt`, `*.pkl`).

## Files

**Create:**
- `scripts/eval_mixed_table.py`
- `tests/scripts/test_eval_mixed_table.py`

**No modifications** to existing scripts or engine.
