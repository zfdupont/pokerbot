# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python Texas Hold'em poker simulator. Agents (decision-making policies) play hands against each other under a shared game engine. Includes a full Monte Carlo CFR training pipeline that learns a Nash-approximate heads-up strategy.

## Commands

- `uv run pytest tests/` — run the full test suite (~83 tests)
- `uv run pytest tests/path/test_file.py::ClassName::test_name -v` — run a single test
- `uv run python scripts/run_simulation.py [--hands 100] [--stack 1000] [--agent simple|cfr]` — run multi-hand simulation
- `uv run python scripts/train_cfr.py [--iterations 1000000] [--checkpoint-interval 10000] [--checkpoint-dir cfr/checkpoints] [--resume path.pkl]` — train CFR agent
- `python main.py` — run one hand with a default 4-player table

All commands run from the repo root. Use `uv` for package management (numpy, tqdm deps).

## Architecture

### Engine ↔ agent boundary

`game/poker.py:PokerGame` owns the entire hand lifecycle (deal, blinds, betting rounds, showdown, button rotation). It holds a `models/state.py:GameState` and delegates pot logic to `game/pot_manager.py:PotManager`, which handles side pots and award splits.

Agents never touch `GameState` directly. The engine calls `Player.make_decision(state)`, which delegates to the injected agent's `get_action(player, state) -> (Action, amount)`. New agents subclass `agents/base_agent.py:PokerAgent`.

### CFR training pipeline (cfr/)

Self-contained package — never imports `game/poker.py`. Has its own lightweight `AbstractState` for tree traversal.

- `cfr/abstraction.py` — equity bucketing (8 preflop / 5 postflop buckets via Monte Carlo rollouts, memoized with `lru_cache`), bet-size discretization (fold/check/call/b0.5/b1.0/allin)
- `cfr/info_set.py` — `InfoSet` frozen dataclass (player, hand_bucket, street, board_bucket, betting_history, stack_bucket); `betting_history` is a 4-tuple of per-street raise counts capped at 2
- `cfr/abstract_state.py` — immutable game state for tree traversal; `deal_heads_up()` factory
- `cfr/regret_table.py` — stores regrets/strategy over fixed 6-action vocabulary, masks illegal actions at query time; pickle save/load
- `cfr/mccfr.py` — External Sampling MCCFR, best-response computation, exploitability in mbb/h
- `cfr/trainer.py` — training loop with tqdm, periodic checkpoint saves and exploitability logging

After training, `agents/cfr_agent.py` loads a checkpoint and translates abstract actions to concrete `(Action, amount)` for live play.

### Hand evaluators

There are two independent hand-evaluation implementations:

1. **Live game path** — `models/hand.py:Hand` (pure-Python rank checks) invoked via `util/evaluator.py:HandEvaluator.evaluate_hands`, called by `PokerGame._determine_winners`.
2. **CFR equity path** — `util/util.py:hand_value` (fast bitwise/prime-product evaluator adapted from Alvin Liang's `pokerhand-eval`) used exclusively by `cfr/abstraction.py` for Monte Carlo equity estimation. Note: 7-card evaluation falls back to `_fallback_hand_value()` which uses coarse `HandRank` values without kicker discrimination — adequate for equity bucketing but not precise showdown resolution.

### Module map

- `models/` — data only: `card.py`, `enums.py` (`Action`, `Position`, `Suit`, `HandRank`), `player.py`, `state.py`, `hand.py`
- `agents/` — `base_agent.py` interface; `simple_agent.py`, `position_agent.py`, `hand_strength_agent.py`, `cfr_agent.py`
- `game/` — `poker.py` (orchestration), `pot_manager.py` (side pot logic)
- `cfr/` — full MCCFR training pipeline (see above)
- `util/` — `evaluator.py` (live showdown), `util.py` + `lookup_table.py` (fast evaluator for CFR equity)
- `scripts/` — `run_simulation.py`, `train_cfr.py`, `gen_missing_entries.py`
- `tests/` — mirrors source layout; `tests/conftest.py` patches `MONTE_CARLO_SAMPLES=10` globally for fast runs; `tests/cfr/conftest.py` additionally stubs out `compute_exploitability` in the trainer

### Known gaps

- `PokerGame._process_action` treats `amount` as the new total bet for BET/RAISE but adds the full amount to the pot without crediting the player's existing `current_bet` — a raise from a player who already has chips committed will overpay.
- `agents/cfr_agent.py` uses `betting_history=(0,0,0,0)` at inference time since live `GameState` doesn't track per-street raise counts — the agent falls back to uniform strategy for InfoSets with non-zero raise history.
