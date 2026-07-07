---
name: train-strategy
description: Running and resuming training jobs — tabular MCCFR and neural Deep CFR.
triggers:
  - "train"
  - "training run"
  - "resume checkpoint"
  - "train_cfr"
  - "train_neural"
edges:
  - target: context/cfr-training.md
    condition: for tabular pipeline internals and the exploitability metric
  - target: context/neural-cfr.md
    condition: for neural training algorithm, constants, and checkpoint format
  - target: patterns/eval-checkpoint.md
    condition: after training, to measure the result
last_updated: 2026-07-07
---

# Train a Strategy

## Context

Two independent pipelines produce checkpoints. Tabular: `scripts/train_cfr.py` → `cfr/checkpoints/checkpoint_XXXXXXXX.pkl`. Neural: `scripts/train_neural.py` → `.pt` via the C++ extension (build it first if `neural_cfr/src/` changed). Both checkpoint dirs are gitignored.

## Task: Tabular MCCFR

### Steps

1. `uv run python scripts/train_cfr.py --iterations 1000000 [--checkpoint-interval 10000] [--checkpoint-dir cfr/checkpoints]`
2. To continue a run: `--resume cfr/checkpoints/checkpoint_XXXXXXXX.pkl` (other scripts auto-detect the latest checkpoint, but resume must be explicit).
3. Track convergence with within-abstraction exploitability (mbb/h), not head-to-head win rate:
   ```
   uv run python -c "
   from cfr.regret_table import RegretTable
   from cfr.mccfr import compute_exploitability
   t = RegretTable(); t.load('cfr/checkpoints/checkpoint_XXXXXXXX.pkl')
   print(compute_exploitability(t, num_samples=300))"
   ```

### Gotchas

- Never report the exact-card BR number (~19,460 mbb/h) as progress — it measures abstraction loss. Baseline: 574 mbb/h at 9.04M iterations.
- Changing anything in `cfr/abstraction.py` (bucket counts, bet sizes) invalidates all existing checkpoints and historical exploitability numbers.

### Verify

- [ ] Exploitability decreased vs. the previous checkpoint at comparable `num_samples`.
- [ ] `uv run python scripts/eval_openspiel.py --hands 2000` shows a sane BB/100 vs. random.

## Task: Neural Deep CFR

### Steps

1. If C++ changed: `~/bin/buck2 build //neural_cfr:neural_cfr`.
2. `mkdir -p neural_cfr/checkpoints` (not auto-created).
3. Fresh (paper regime — recommended):
   ```bash
   uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml \
       --checkpoint neural_cfr/checkpoints/checkpoint.pt
   ```
   CLI flags override config values. The effective config is snapshotted to `<checkpoint>.config.toml` on every save.
4. Resume: add `--resume neural_cfr/checkpoints/checkpoint.pt`. Prefer single uninterrupted runs — resumed runs restart reservoir buffers from scratch, so the strategy net only sees post-resume M_π.
5. Legacy continual regime (quick smoke test only): `--config neural_cfr/configs/smoke.toml`.

### Gotchas

- Resume restores **nets only** — reservoir buffers are never serialized, so the first iterations after resume refill buffers from scratch.
- Checkpoint format is named sub-archives (`adv0`/`adv1`/`strat`); anything that loads it as a flat `.pt` will fail.
- Traversal invariants (M_π only at opponent nodes, per-player advantage nets, offline strategy-net training) are paper-faithful — see `context/decisions.md` before "fixing" them.
- ε-greedy opponent exploration defaults to ε=0.06.

### Verify

- [ ] `uv run pytest tests/neural_cfr/ -v` passes (trainer, strategy-compat, eval-compat).
- [ ] `uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 2000 --baseline random` beats random.
- [ ] `neural_cfr.Strategy` can load the checkpoint (it reads only the `"strat"` sub-archive).

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
