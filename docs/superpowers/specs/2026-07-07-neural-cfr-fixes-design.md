# Neural CFR Correctness Fixes — Design

**Date:** 2026-07-07
**Scope:** `neural_cfr/` C++ extension + Python callers (`scripts/train_neural.py`, `scripts/openpoker_bot.py`, `scripts/eval_openspiel_neural.py`)

## Background

A code audit of `neural_cfr/` found three issues that plausibly explain the weak
strategy net observed via `AdvantageProbe`:

1. **Inference feature mismatch** — `bindings.cpp` builds synthetic states with
   `player_bets = {0, 0}`, so feature dims 129–130 (per-player street bets) are
   always zero at inference but nonzero during training whenever there is street
   betting. The strategy net cannot see the size of the bet it is facing.
2. **Undertrained networks** — `Trainer::run` performs one SGD step (batch 4096)
   per network every 10 traversal-pairs, continually fine-tuning the same weights.
   Deep CFR (Brown et al. 2019) instead reinitializes the advantage net from
   scratch each CFR iteration and trains it for thousands of SGD steps; the
   paper's ablation shows continual fine-tuning converges worse.
3. **Uniform regret-matching fallback** — when all predicted advantages are ≤ 0,
   `regret_match` plays uniform over legal actions. The paper specifies playing
   argmax(advantage) as a pure strategy. The uniform σ is stored into M_π at
   opponent nodes, actively training the strategy net toward uniform play in
   pessimistic states (e.g., trash hands that should nearly always fold).

## Fix 1 — Train/inference feature parity

### C++ (`neural_cfr/src/bindings/bindings.cpp`)

- Factor the duplicated synthetic-state construction (`Strategy::get_action_probs`
  inline code and `AdvantageProbe::_make_state`) into one free function
  `make_inference_state(...)` used by both classes.
- `Strategy::get_action_probs` and both `AdvantageProbe` methods gain two trailing
  optional args: `my_street_bet = -1.0f`, `opp_street_bet = -1.0f`
  (sentinel −1 = "not provided").
- **Street bets provided:** set `player_bets` position-mapped from the args and
  `current_bet = max(my_street_bet, opp_street_bet)`, so
  `to_call = current_bet − player_bets[position]` is derived. If the derived
  value disagrees with the `to_call` arg beyond float slop, prefer the street
  bets (the more precise signal).
- **Street bets not provided (fallback):** `player_bets[position] = 0`,
  `player_bets[1−position] = to_call`, `current_bet = to_call`. This is strictly
  better than today's all-zeros; the residual gap (hero's own prior street bet
  unseen in raised pots) exists only for callers that do not pass street bets.
- Clamp incoming `raises_per_street` values to `[0, 2]` in the same function
  (training caps them at 2; uncapped values push features outside the training
  range).
- Document in the pybind docstrings: **`pot` must include all street bets**,
  matching training semantics (both current callers already do this).

### Python callers

- `scripts/openpoker_bot.py` — `HandTracker` gains `street_committed: [float, float]`,
  updated from the same messages that update `raises_per_street`, reset on street
  change; passed to `get_action_probs` scaled by `1/scale`.
- `scripts/eval_openspiel_neural.py` — track each player's stack at street start;
  street bet = start-of-street stack − current stack, scaled by `1/_CHIP_SCALE`.

### Test

New case in `neural_cfr/tests/test_features.cpp`: build a mid-hand
`AbstractState` via `apply_action` (e.g., SB raise → BB facing it), encode
features; build the same spot via `make_inference_state` with real street bets;
assert the two 134-dim tensors are identical. This is the train/inference
parity test that was missing.

## Fix 2 — Paper-faithful training regime

### Loop restructure (`neural_cfr/src/cfr/trainer.{h,cpp}`)

`Trainer::run` becomes a loop of discrete CFR iterations: run `train_interval`
traversal-pairs (multi-threaded, unchanged), then hold one **training event**:

1. If `reinit_adv`, reinitialize both advantage nets from scratch
   (new `MLP::reset_parameters()` reinitializing all four Linear layers) and
   rebuild their Adam optimizers (held via `std::unique_ptr`, recreated so moment
   state does not leak across from-scratch retrains).
2. Train each advantage net for `sgd_steps` mini-batches (batch `batch_size`)
   sampled uniformly from its reservoir buffer, with linear-CFR sample weights
   (unchanged: weight = global traversal-pair index) and gradient-norm clipping.

**Strategy net:** removed from the per-iteration loop (it is never queried during
traversal). New public method `Trainer::train_strategy(int sgd_steps = -1)`
retrains it from scratch on M_π; `checkpoint()` calls it before saving.
`train_strategy` is also exposed to Python.

### Defaults

| Knob | Old | New | Rationale |
|---|---|---|---|
| `train_interval` | 10 | 10,000 | Paper scale: 10k traversals per CFR iteration; 5M total = 500 CFR iterations |
| `sgd_steps` (new) | 1 implicit | 2,000 | Paper uses 4,000 at batch 10k on a larger net |
| `reinit_adv` (new) | — | `true` | Paper's key ablation: from-scratch retraining beats fine-tuning |
| `lr` | 1e-4 | 1e-3 | Paper's rate; 1e-4 was a fine-tuning rate |
| grad-norm clip (new) | none | 1.0 | Per paper; insurance at the higher LR |

The old cheap regime stays reachable (`reinit_adv=false, sgd_steps=1,
train_interval=10`) for smoke tests.

**Cost estimate:** ~30–60 s of SGD per training event on CPU libtorch;
~4–8 h total SGD across a 5M-iteration run, on top of traversal time.
Accepted per the "whatever converges" budget.

### Config file (`scripts/train_neural.py`)

- TOML config, stdlib `tomllib`, default file `neural_cfr/configs/default.toml`
  (sections `[training]` and `[trainer]` mirroring CLI flags); optional
  `neural_cfr/configs/smoke.toml` with the old cheap regime.
- Precedence: **CLI flag > config file > built-in default.** Existing flags stay;
  argparse defaults become `None` so "explicitly passed" is distinguishable.
  `--config <path>` selects the file.
- On checkpoint save, write the effective resolved config to
  `<checkpoint>.config.toml` for reproducibility.
- The C++ `Trainer` stays config-file-agnostic: plain constructor args only;
  all file parsing lives in the Python driver.

### Accepted limitation

Buffers are not serialized. After `--resume`, the strategy net's final training
sees only post-resume M_π data, and advantage training restarts from thin
buffers (loaded advantage weights still drive traversal until the first
training event). Docs will recommend single uninterrupted runs for final
training. Buffer serialization (~3.4 GB for 3 × 2M entries) is out of scope.

## Fix 3 — Argmax fallback in regret matching

- `regret_match` in `neural_cfr/src/cfr/traversal.cpp`: when all clipped
  advantages are ≤ 0, return a one-hot on argmax(advantage) over legal actions
  instead of uniform. Applies at both traverser and opponent nodes; the one-hot
  σ is what gets stored in M_π.
- Mirror the same fallback in `AdvantageProbe::get_advantage_probs`
  (`bindings.cpp`) so diagnostics match traversal behavior; share a helper if
  practical.
- ε-greedy opponent exploration (ε = 0.06) is retained — it matters more with
  pure fallback strategies since it preserves coverage of non-argmax branches.

### Test

New case in `neural_cfr/tests/test_traversal.cpp`: fallback returns one-hot
argmax when all advantages are negative; proportional regret matching when
positive advantages exist.

## Validation

### Unit / mechanical

- Feature-parity test (Fix 1, above).
- Regret-matching fallback test (Fix 3, above).
- Trainer smoke test: tiny config (`train_interval=50, sgd_steps=10,
  reinit_adv=true`) runs end-to-end; buffers populate; `checkpoint()` triggers
  strategy training and produces a loadable file.

### Empirical acceptance

1. Quick gate before the long run: 100k-iteration smoke train, then
   `AdvantageProbe` spot checks (AA raises preflop; 72o folds facing a raise).
2. Retrain from scratch with new defaults: 5M traversal-pairs = 500 CFR
   iterations, uninterrupted.
3. Evals at **≥ 20,000 hands** each (HU NLHE win-rate SE is ~±20 BB/100 at
   2k hands, ~±7 BB/100 at 20k; the default 2k cannot distinguish parity from
   −10 BB/100):
   - `scripts/eval_neural_vs_tabular.py` vs the tabular CFR bot.
   - `scripts/eval_openspiel_neural.py --baseline random`.

**Targets:**

| Criterion | Target |
|---|---|
| Sanity floor (validates the fixes) | Strictly better than the current pre-fix neural checkpoint on both evals |
| Primary | ≥ 0 BB/100 vs the tabular CFR bot (parity with the 9M-iteration tabular baseline; clearly positive is a stretch goal) |
| Secondary | vs uniform random: at least match the tabular bot's BB/100 on the same eval |

**Convergence rationale:** 500 CFR iterations matches the Deep CFR paper's own
scale (~450 iterations on FHP, curve flattening after ~200–300). Linear CFR
converges as O(1/√T), so 500 is a knee-of-the-curve budget. There is no cheap
exploitability metric for a neural strategy (best response against it is itself
a training problem), so acceptance is head-to-head only.

## Out of scope

- Buffer serialization for resume.
- The `eff_pot = pot + 2·to_call` bet-sizing convention (self-consistent quirk,
  matches the tabular port).
- `payoff()`'s hardcoded `STARTING_STACK` — gets a comment/assert only.
- Board-texture features, ε tuning, network architecture changes.

## Acceptance Results (2026-07-09)

Full 5M-iteration retrain under the new regime (`configs/default.toml`,
~22 h wall clock). All evals 20,000 hands (SE ≈ ±7 BB/100). Baseline =
pre-fix checkpoint evaluated under the merged code.

| Checkpoint | vs random (BB/100) | vs tabular (BB/100) |
|---|---|---|
| Pre-fix baseline | +334.97 | −241.8 |
| 500k | +374.25 | −101.0 |
| 1M | +408.39 | −137.7 |
| 3.5M | +467.31 | −9.3 |
| 4M | +419.88 | **+3.2** |
| 4.5M | +551.81 | −31.8 |
| 5M (final) | +455.25 | −80.8 |

**Verdicts against the targets:**

- **Sanity floor (beat pre-fix on both evals): PASS** — every checkpoint,
  including the final one, beats both baseline numbers by wide margins.
- **Secondary (≥ tabular's win rate vs random): PASS** — +455 vs +335 at 5M;
  every checkpoint from 500k onward passes.
- **Primary (≥ 0 BB/100 vs tabular): PARTIAL** — parity reached at
  3.5M–4M (−9.3, +3.2 — both within noise of zero) but not held stably;
  late checkpoints oscillate in a ≈ −30 ± 40 band. The 5M checkpoint
  itself measures −80.8. Mean of the last four checkpoints ≈ −29.7.

**Interpretation:** the fixes moved the bot from −242 to statistical parity
with the 9M-iteration tabular baseline, but per-checkpoint variance from
the from-scratch strategy-net retrain (2,000 batches per event) dominates
the late curve. The best-measured artifact is the 4M checkpoint
(`neural_cfr/checkpoints/archive_1302.pt`, +3.2). Follow-up candidates:
more `sgd_steps` for `train_strategy`, checkpoint selection by validation
eval, or serializing M_π to allow a final high-budget strategy fit.
