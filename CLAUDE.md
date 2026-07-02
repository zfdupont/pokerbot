# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python Texas Hold'em poker simulator. Agents (decision-making policies) play hands against each other under a shared game engine. Includes a full Monte Carlo CFR training pipeline that learns a Nash-approximate heads-up strategy, plus tooling to play, visualize, evaluate, and deploy the trained bot.

## Commands

### Testing
```bash
uv run pytest tests/                                          # full suite (~82 tests)
uv run pytest tests/path/test_file.py::ClassName::test_name -v  # single test
```

### Training
```bash
uv run python scripts/train_cfr.py \
    [--iterations 1000000] \
    [--checkpoint-interval 10000] \
    [--checkpoint-dir cfr/checkpoints] \
    [--resume cfr/checkpoints/checkpoint_XXXXXXXX.pkl]
```
Checkpoints are saved to `cfr/checkpoints/` (gitignored). Latest checkpoint is auto-detected by all other scripts.

### Evaluation
```bash
# Within-abstraction exploitability (primary convergence metric, in mbb/h)
uv run python -c "
from cfr.regret_table import RegretTable
from cfr.mccfr import compute_exploitability
t = RegretTable(); t.load('cfr/checkpoints/checkpoint_09040000.pkl')
print(compute_exploitability(t, num_samples=300))
"

# Head-to-head vs uniform random or OpenSpiel CFR (BB/100 win rate)
uv run python scripts/eval_openspiel.py [--hands 2000] [--checkpoint path.pkl]
uv run python scripts/eval_openspiel.py --hands 500 --baseline cfr --cfr-iters 500
```

### Play / Visualize
```bash
# Interactive heads-up CLI against the bot
uv run python scripts/play.py [--checkpoint path.pkl] [--stack 1000] [--big-blind 10]

# 13×13 ANSI preflop range chart
uv run python scripts/range_chart.py [--checkpoint path.pkl] [--scenario sb-open|bb-vs-raise]
```

### Deploy to openpoker.ai
```bash
export OPENPOKER_API_KEY=your_key_here
uv run python scripts/openpoker_bot.py [--buy-in 2000] [--log-level DEBUG]
```

### Simulation
```bash
uv run python scripts/run_simulation.py [--hands 100] [--stack 1000] [--agent simple|cfr]
python main.py   # one hand, default 4-player table
```

## Architecture

### Engine ↔ agent boundary

`game/poker.py:PokerGame` owns the entire hand lifecycle (deal, blinds, betting rounds, showdown, button rotation). It holds a `models/state.py:GameState` and delegates pot logic to `game/pot_manager.py:PotManager`, which handles side pots and award splits.

Agents never touch `GameState` directly. The engine calls `Player.make_decision(state)`, which delegates to the injected agent's `get_action(player, state) -> (Action, amount)`. New agents subclass `agents/base_agent.py:PokerAgent`.

### CFR training pipeline (cfr/)

Self-contained package — never imports `game/poker.py`. Has its own lightweight `AbstractState` for tree traversal.

- `cfr/abstraction.py` — equity bucketing (6 preflop / 5 postflop buckets via Monte Carlo rollouts, dict-memoized), bet-size discretization: `fold / check / call / b0.5 / b1.0 / allin`
- `cfr/info_set.py` — `InfoSet` frozen dataclass `(player, hand_bucket, street, board_bucket, betting_history, stack_bucket)`; `betting_history` is a 4-tuple of per-street raise counts capped at 2
- `cfr/abstract_state.py` — immutable game state for tree traversal; `deal_heads_up()` factory
- `cfr/regret_table.py` — stores regrets/strategy over fixed 6-action vocabulary, masks illegal actions at query time; pickle save/load
- `cfr/mccfr.py` — External Sampling MCCFR + two-pass within-abstraction best-response + `compute_exploitability()` in mbb/h
- `cfr/trainer.py` — training loop with tqdm, periodic checkpoint saves and exploitability logging

After training, `agents/cfr_agent.py` loads a checkpoint and translates abstract actions to concrete `(Action, amount)` for live play.

### Exploitability metric

`compute_exploitability()` in `cfr/mccfr.py` uses **within-abstraction** best-response: the BR player is constrained to one action per abstract infoset, removing abstraction loss from the measurement. This is the correct metric to track convergence. As of checkpoint_09040000: **574 mbb/h** (9.04M iterations, 7,877 infosets).

The old "exact-card BR" metric (~19,460 mbb/h) is dominated by abstraction loss and is not a useful convergence signal.

### Hand evaluators

Two independent implementations:

1. **Live game path** — `models/hand.py:Hand` (pure-Python rank checks) via `util/evaluator.py:HandEvaluator.evaluate_hands`, called by `PokerGame._determine_winners`.
2. **CFR equity path** — `util/util.py:hand_value` (fast bitwise/prime-product evaluator) used exclusively by `cfr/abstraction.py` for Monte Carlo equity estimation. 7-card evaluation falls back to `_fallback_hand_value()` which uses coarse `HandRank` values without kicker discrimination — adequate for equity bucketing but not precise showdown resolution.

### External evaluation (scripts/eval_openspiel.py)

Uses Google DeepMind's OpenSpiel `universal_poker` environment (FCPA action abstraction, 100BB HU NL) as an independent game engine. `CFRBotPolicy` wraps our `RegretTable` as an OpenSpiel `Policy`, parsing info state strings to build `InfoSet`s on the fly. Key detail: OpenSpiel action 1 means "check" (postflop free check) or "call" (facing a raise) — `_is_check_action()` disambiguates using the sequences string and street.

### WebSocket connector (scripts/openpoker_bot.py)

Connects to `wss://openpoker.ai/ws` with `Authorization: Bearer <api_key>`. `HandTracker` accumulates `hole_cards`, `community_cards`, and `raises_per_street` across message types, then calls `RegretTable.get_average_strategy()` on `your_turn`. Raise amounts use OpenPoker's **raise-to** convention (total chips committed, not increment).

### Module map

- `models/` — `card.py`, `enums.py` (`Action`, `Position`, `Suit`, `HandRank`), `player.py`, `state.py`, `hand.py`
- `agents/` — `base_agent.py` interface; `simple_agent.py`, `position_agent.py`, `hand_strength_agent.py`, `cfr_agent.py`
- `game/` — `poker.py` (orchestration), `pot_manager.py` (side pot logic)
- `cfr/` — full MCCFR training pipeline (see above)
- `util/` — `evaluator.py` (live showdown), `util.py` + `lookup_table.py` (fast evaluator for CFR equity)
- `scripts/` — `train_cfr.py`, `play.py`, `range_chart.py`, `eval_openspiel.py`, `openpoker_bot.py`, `run_simulation.py`
- `tests/` — mirrors source layout; `tests/conftest.py` patches `MONTE_CARLO_SAMPLES=10` globally; `tests/cfr/conftest.py` stubs `compute_exploitability` in the trainer

### Known gaps / next steps

- `_fallback_hand_value()` in `util/util.py` uses coarse `HandRank` without kicker discrimination — a full 7-card lookup table would improve postflop equity estimates.
- Board abstraction ignores suit texture (flush draws, monotone boards get same bucket as rainbow).
- 6 preflop equity buckets means hands like A9o and KTs share a strategy — expanding to 169 exact preflop hands would be the largest single quality improvement.
- Next major milestone: **Neural CFR** (Deep CFR) — replace `RegretTable` with two MLPs, eliminating the abstraction ceiling. The MCCFR traversal in `cfr/mccfr.py` carries over directly; new additions are a feature encoder, advantage/strategy memory buffers, and interleaved network training.
