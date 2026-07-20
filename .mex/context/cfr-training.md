---
name: cfr-training
description: The tabular MCCFR pipeline (cfr/) — abstraction, InfoSet, RegretTable, training loop, and the exploitability metric. Load when training, evaluating, or modifying the tabular strategy.
triggers:
  - "tabular CFR"
  - "MCCFR"
  - "abstraction"
  - "InfoSet"
  - "RegretTable"
  - "exploitability"
  - "bucket"
edges:
  - target: context/architecture.md
    condition: when placing the pipeline in the overall system
  - target: context/decisions.md
    condition: for why within-abstraction exploitability is the metric
  - target: context/neural-cfr.md
    condition: when comparing with or porting concepts to the neural pipeline
  - target: patterns/train-strategy.md
    condition: when actually running a training job
last_updated: 2026-07-19
---

# Tabular CFR Training (`cfr/`)

## Pipeline

Self-contained package — never imports `game/poker.py`. Traverses its own immutable `cfr/abstract_state.py:AbstractState` (heads-up only, `deal_heads_up()` factory).

- `cfr/abstraction.py` — equity bucketing: 6 preflop / 5 postflop buckets via Monte Carlo rollouts (dict-memoized). Bet sizes discretized to `fold / check / call / b0.5 / b1.0 / allin`.
- `cfr/info_set.py` — `InfoSet` frozen dataclass: `(player, hand_bucket, street, board_bucket, betting_history, stack_bucket)`. `betting_history` is a 4-tuple of per-street raise counts **capped at 2**.
- `cfr/regret_table.py` — regrets/strategy stored over the fixed 6-action vocabulary; illegal actions masked at query time; pickle save/load.
- `cfr/mccfr.py` — External Sampling MCCFR + two-pass within-abstraction best response + `compute_exploitability()` (mbb/h).
- `cfr/trainer.py` — training loop, tqdm, periodic checkpoints to `cfr/checkpoints/checkpoint_XXXXXXXX.pkl` and exploitability logging.
- `agents/cfr_agent.py` — loads a checkpoint for live play; translates abstract actions to concrete `(Action, amount)`.

## The metric that matters

`compute_exploitability()` uses **within-abstraction** best response — the BR player is constrained to one action per abstract infoset, removing abstraction loss from the measurement. As of checkpoint_09040000: **574 mbb/h** (9.04M iterations, 7,877 infosets). The exact-card BR number (~19,460 mbb/h) is abstraction loss, not a convergence signal — never report it as progress.

## Gotchas

- Hand equity for bucketing uses `util/util.py:hand_value`, whose 7-card `_fallback_hand_value()` lacks kicker discrimination — fine for buckets, wrong for showdowns.
- 6 preflop buckets means e.g. A9o and KTs share a strategy. Expanding to 169 exact preflop hands is the documented largest single quality win.
- Board abstraction ignores suit texture (monotone board == rainbow board).
- Exploitability numbers are only comparable within one abstraction; changing bucketing resets history.
- Tests stub the expensive parts: `tests/conftest.py` patches `MONTE_CARLO_SAMPLES=10`, `tests/cfr/conftest.py` stubs `compute_exploitability` in the trainer.
