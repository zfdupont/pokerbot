---
name: neural-cfr
description: The C++ Deep CFR subsystem (neural_cfr/) — card/feature encoding, evaluator invariants, network architecture, training algorithm, and checkpoint format. Load before touching anything under neural_cfr/ or its Python consumers.
triggers:
  - "neural CFR"
  - "Deep CFR"
  - "libtorch"
  - "buck2"
  - "feature encoding"
  - "advantage net"
  - "strategy net"
  - "checkpoint.pt"
edges:
  - target: context/architecture.md
    condition: when placing the subsystem in the overall system
  - target: context/decisions.md
    condition: for why C++ was chosen and which traversal rules are invariants
  - target: context/cfr-training.md
    condition: when comparing with the tabular pipeline
  - target: patterns/debug-bot-misplay.md
    condition: when the trained bot behaves nonsensically
  - target: patterns/train-strategy.md
    condition: when running a neural training job
last_updated: 2026-07-14
---

# Neural CFR (`neural_cfr/`)

C++ Deep CFR (Brown et al. 2019). Self-contained — never imports `game/poker.py` or `cfr/`. Built with libtorch + pybind11 via Buck2 (`~/bin/buck2 build //neural_cfr:neural_cfr`). Exposed to Python as `import neural_cfr`.

## Card representation

`card = rank_index * 4 + suit_index`, where `rank_index = face_value - 2` (0=2 … 12=A) and suit is `0=clubs 1=diamonds 2=hearts 3=spades`. Examples: `2c=0`, `As=51`, `Ah=50`, `Ks=47`.

## Hand evaluator (`src/game/card.cpp`) — CRITICAL INVARIANT

`evaluate_5card` returns `category << 20 | kicker_bits`, **lower = better**. Categories: 1=straight_flush … 9=high_card. Kickers are stored **inverted** (`12 - rank`) so Ace → 0 (best) within a category. Removing the inversion inverts learned hand strength (AA folds, 22 raises) — a confirmed training bug. `evaluate_7card` takes `min` over all 21 five-card combos. Any encoding change must preserve `12 - rank`.

## Feature encoding (`src/net/features.h/.cpp`)

134-dim float tensor. Constants: `FEATURE_DIM=134`, `NUM_ACTIONS=6`, `CHIP_NORM=200.0f`, `RAISE_NORM=2.0f`.

| Dims | Content |
|------|---------|
| 0–33 | 2 hole cards × 17 (13 rank one-hot + 4 suit one-hot) |
| 34–118 | 5 board cards × 17, zero-padded |
| 119–122 | street one-hot |
| 123 | pot / CHIP_NORM |
| 124 | stack[player] / CHIP_NORM |
| 125–128 | betting_history / RAISE_NORM |
| 129–130 | player_bets (self, opponent) / CHIP_NORM |
| 131–132 | reserved (0.0) |
| 133 | position (player index) |

**Chip-scaling rule:** training uses `starting_stack=100, big_blind=1`. Inference inputs from other engines must be divided by `(their_stack / 100)` before `get_action_probs`. At inference, pass `my_street_bet`/`opp_street_bet` for exact dim-129/130 parity; `pot` must include all street bets.

## Networks (`src/net/mlp.h`, constants in `trainer.h`)

MLP `134 → 256 → 256 → 256 → 6` (ReLU, raw logits). Per-player advantage nets `adv0_`/`adv1_` (weighted MSE on counterfactual advantages); one shared strategy net `strat_` (weighted cross-entropy on M_π), used at inference via `neural_cfr.Strategy`. Defaults: `BATCH_SIZE=4096`, `LR=1e-3`, `RESERVOIR_SIZE=2_000_000`, `TRAIN_INTERVAL=10_000`, `SGD_STEPS=2_000`, `REINIT_ADV=true` (grad-norm clip 1.0).

## Training algorithm (invariants — see decisions.md)

External Sampling MCCFR, faithful to Brown et al. 2019:
- **Traverser node:** regret-match `adv_net` → traverse ALL actions → store advantages in `M_v[p]` only.
- **Opponent node:** regret-match the opponent's adv net → store σ in `M_π` → sample ONE action (ε-greedy, default ε=0.06).
- `strat_net` is trained offline on `M_π`, never queried during traversal.
- Linear CFR weighting: `weight = iteration t`.
- Traversal is multithreaded (thread-safe RNG, buffer mutex, thread pool in `Trainer::run()`); `--num-threads` exposed in `scripts/train_neural.py`.
- **Training regime (paper-faithful, 2026-07):** every `train_interval` traversal-pairs = one CFR iteration → advantage nets reinitialized from scratch and trained `sgd_steps` mini-batches (the two events run on concurrent threads — they share nothing). Strategy net is trained from scratch on M_π only at checkpoint time (`Trainer.train_strategy`, called automatically by `checkpoint()`). Legacy continual regime: `neural_cfr/configs/smoke.toml`. With selection_enabled (on in default.toml), the driver evals each save vs the tabular baseline (selection_hands, default 10k) and keeps the best in best_checkpoint.pt + best_checkpoint.json (strict-improvement replacement).
- **Regret-matching fallback:** if all predicted advantages are ≤ 0, play argmax(advantage) as a pure strategy (not uniform).

## Checkpoint format

Named sub-archives, NOT flat: `root.write("adv0", ...)`, `("adv1", ...)`, `("strat", ...)`. `neural_cfr.Strategy` loads only `"strat"`. Reservoir buffers are never serialized — `--resume` restores nets only. `checkpoint()` retrains the strategy net on M_π before saving; because buffers are not serialized, prefer single uninterrupted runs (a resumed run's strategy net only sees post-resume M_π). Checkpoints also carry a "meta" sub-archive with the cumulative iteration counter (total_iterations()); legacy checkpoints load with counter 0. Linear-CFR weights are globally monotonic across chunks and resumes.

## Commands

```bash
# Build C++ extension (required after any src/ change)
~/bin/buck2 build //neural_cfr:neural_cfr

# Train — paper regime (default.toml); CLI flags override; effective config
# is snapshotted to <checkpoint>.config.toml on every save
uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml \
    --checkpoint neural_cfr/checkpoints/checkpoint.pt

# Legacy continual regime (quick smoke test)
uv run python scripts/train_neural.py --config neural_cfr/configs/smoke.toml \
    --checkpoint neural_cfr/checkpoints/smoke.pt

# Resume (nets only — buffers restart from scratch)
uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml \
    --resume neural_cfr/checkpoints/checkpoint.pt \
    --checkpoint neural_cfr/checkpoints/checkpoint.pt

# Evaluate
uv run python scripts/eval_openspiel_neural.py \
    --checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 2000 --baseline random
```

## Action vocabulary (index order matters)

`0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin` — identical to the tabular pipeline.
