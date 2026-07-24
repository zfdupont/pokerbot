# DREAM Blueprint Design

**Date:** 2026-07-24
**Branch:** feature/neural-cfr
**Status:** Approved

## Overview

Replace the tabular `BlueprintTrainer` in `sixmax/` with a DREAM-style neural blueprint (Steinberger et al. 2020) — advantage net + strategy net trained via outcome-sampling MCCFR. The goal is cash-game generalization across variable stack sizes (20–250 BB) and player counts (2–6), which the tabular approach cannot support because it is locked to `starting_stack=100, num_players=6`.

## Motivation

The tabular blueprint indexes strategy by a 64-bit abstract infoset key. At a cash game table where effective stacks are 73 BB and only 4 players are seated, the bot operates outside the key space it was trained on. Neural function approximation generalizes across configurations seen in training; a tabular table cannot.

## Module Structure

New `sixmax/src/dream/` directory, added alongside existing `src/blueprint/`, `src/abstraction/`, `src/engine/`:

```
sixmax/src/dream/
  features.h/.cpp    — infoset → 154-dim float tensor
  nets.h/.cpp        — libtorch MLP (advantage net + strategy net)
  reservoir.h/.cpp   — weighted reservoir buffers M_v and M_π
  trainer.h/.cpp     — DreamTrainer (outcome sampling traversal + retraining loop)
  checkpoint.h/.cpp  — SIXDM001 checkpoint format + DreamStrategy (inference)
```

`src/blueprint/` stays **frozen** during validation. Once DREAM passes the Kuhn poker convergence gate, a single cleanup commit removes `BlueprintTrainer`, `BlueprintStrategy`, and the `SIXBP001` loader. During the transition window, `bindings.cpp` exposes both `BlueprintStrategy` and `DreamStrategy`; both implement the same `get_probs()` interface so `SixmaxDeployStrategy` needs no changes.

## Feature Encoding

**154-dim float tensor.** Constants: `FEATURE_DIM=154`, `CHIP_NORM=100.0f`, `RAISE_NORM=5.0f`.

| Dims | Content |
|------|---------|
| 0–33 | 2 hole cards × 17 (13 rank one-hot + 4 suit one-hot) |
| 34–118 | 5 board cards × 17, zero-padded for unsealed streets |
| 119–122 | Street one-hot (preflop / flop / turn / river) |
| 123 | `pot / CHIP_NORM` |
| 124–129 | `stack[seat 0–5] / CHIP_NORM` — 0.0 if seat absent or all-in |
| 130–135 | `street_bet[seat 0–5] / CHIP_NORM` — amount committed this street |
| 136–141 | `live_mask[seat 0–5]` ∈ {0, 1} |
| 142–147 | Acting player one-hot (seat index 0–5) |
| 148 | `n_active / 6.0` |
| 149 | `to_call / CHIP_NORM` |
| 150–153 | `min(raises_per_street[0–3], 5) / RAISE_NORM` |

**CHIP_NORM=100.0** — equals the standard buy-in (full stack = 1.0). Stack sizes up to 1000 BB encode as ≤10.0, well within ReLU range.

**RAISE_NORM=5.0, clipped at 5** — replaces the engine's hard cap of 3. The cap was a tabular tree-pruning heuristic; the neural tree is bounded naturally by all-in. Sequences longer than 5 raises are essentially never rational (players jam), so clipping is cosmetic in practice.

**6-seat block (dims 124–141)** — raw per-seat stacks and bets rather than a single effective-stack scalar. The net learns effective stack, SPR, and side-pot dynamics implicitly from the joint distribution. Absent seats zero-pad cleanly.

**Acting player one-hot (dims 142–147)** — categorical encoding rather than a scalar seat index; positional semantics (BTN, BB, UTG) are categorical, not ordinal.

## Network Architecture

Two libtorch MLPs: `154 → 256 → 256 → 256 → N_ACTIONS`, ReLU activations, raw logit outputs.

| Net | Trains on | Purpose |
|-----|-----------|---------|
| **Advantage net** (1 shared) | M_v — IS-weighted instantaneous advantages | Regret matching during traversal |
| **Strategy net** (1 shared) | M_π — strategy samples | Inference (`DreamStrategy::get_probs`) |

**One shared advantage net** (not per-player) — acting player identity is encoded via one-hot in the feature vector. Six per-player nets would be impractical for 6-max.

**`N_ACTIONS` is a runtime parameter** set from `ActionVocab::size()` at construction. The vocab is config-defined (default: 10 actions — fold/check/call + 3 preflop opens + 3 postflop sizes + all-in), so the output layer width cannot be a compile-time constant. The checkpoint embeds the vocab hash; loading refuses mismatches.

**Training defaults (`[train.dream]` in `default.toml`):**

```toml
[train.dream]
hidden_size    = 256
hidden_layers  = 3
lr             = 1e-3
batch_size     = 4096
reservoir_size = 2_000_000
train_interval = 10_000
sgd_steps      = 2_000
epsilon        = 0.06
device         = "cpu"    # "mps" for MacBook, "cuda" for cloud
```

Grad norm clipped at 1.0. Advantage net reinitializes from scratch each retraining cycle (`REINIT_ADV=true`) — prevents stale regrets from prior iterations contaminating updates. Strategy net trains continuously.

## DREAM Training Algorithm

Outcome-sampling MCCFR with IS-weighted advantage estimation.

### Traversal (per iteration)

1. Deal a new hand from `EngineGame` with randomized configuration (see Training Distribution)
2. At every node, **all players sample one action** from their current advantage net (ε-greedy, ε=0.06)
3. Follow sampled actions to terminal
4. Walk back up the trajectory: for each player `p` at each visited infoset, compute IS-corrected instantaneous advantage for each action `a`:

```
adv(a) = (terminal_utility[p] / π(a*)) * I[a == a*]
```

where `a*` is the sampled action and `π(a*)` is its probability under the current strategy.

5. Store `(features, adv_vector, weight=t)` → **M_v**
6. Store `(features, strategy_vector, weight=t)` → **M_π**

### Retraining (every `train_interval` traversals)

- Reinitialize and retrain advantage net from scratch on M_v — weighted MSE loss
- Retrain strategy net on M_π — weighted cross-entropy (not reinitialized)

### IS Weight Invariant

ε-greedy exploration (ε=0.06) ensures `π(a*) ≥ ε/N_ACTIONS` at all times, bounding IS weights and preventing gradient explosion from near-zero sampling probabilities.

### Linear CFR Weighting

`weight = iteration t` — monotonically increasing, cumulative across `train()` calls and resumes. Consistent with `BlueprintTrainer` and `neural_cfr/`.

### Threading

Each worker thread owns its own `EngineGame` instance via `GameFactory` (same pattern as `BlueprintTrainer`). Reservoir buffers protected by mutex. Thread-local RNG.

## Training Distribution

Each new hand randomizes:

- `n_players` — uniform over `[2, 6]`
- `stack[i]` per seat — LogNormal(μ=ln(100), σ=0.5), clipped to `[20, 250]` BB, independent per seat

LogNormal with μ=ln(100) gives median=100 BB exactly. ~68% of samples fall in `[61, 165]` BB — clustered around the standard buy-in, with a long right tail for deep-stack spots. Clipping is rare (<3% above 250 BB, <0.1% below 20 BB).

## Checkpoint Format

Magic header `SIXDM001`. Same atomic write pattern as `SIXBP001` (tmp file + rename).

| Sub-archive | Content |
|-------------|---------|
| `"adv"` | Advantage net weights |
| `"strat"` | Strategy net weights |
| `"meta"` | Iteration counter, vocab hash, abstraction config + edges, training config snapshot |

`DreamStrategy` loads only `"strat"` for inference. Device is **not** stored — `DreamStrategy::load(path, device)` takes device at load time; `module.to(device)` handles CPU → MPS → CUDA portably with no checkpoint conversion.

Checkpoint refuses to load on vocab hash mismatch (action set changed) or abstraction config mismatch (bucket assignments would drift).

## Inference Interface

```cpp
class DreamStrategy {
public:
    static DreamStrategy load(const std::string& path, torch::Device device);
    // Softmax probabilities over vocab actions; illegal actions zeroed + renormalized
    std::vector<double> get_probs(const EngineGameState& state) const;
private:
    std::shared_ptr<MLP> strat_net_;
    ActionVocab vocab_;
    torch::Device device_;
};
```

Exposed to Python via `bindings.cpp` identically to `BlueprintStrategy`. `SixmaxDeployStrategy`, `SixmaxAgent`, and `openpoker_bot.py` need no changes — they call `get_probs()` on whatever strategy object they hold. The `.pt` extension (vs `.bin` for tabular) is the existing auto-detect signal in `openpoker_bot.py`.

## Scripts

New entry point:

```bash
uv run python scripts/train_dream.py \
    --config sixmax/configs/default.toml \
    --checkpoint sixmax/checkpoints/dream.pt
```

Resume support:

```bash
uv run python scripts/train_dream.py \
    --config sixmax/configs/default.toml \
    --resume sixmax/checkpoints/dream.pt \
    --checkpoint sixmax/checkpoints/dream.pt
```

## Testing Strategy

### Tier 1 — Unit tests (`tests/sixmax/test_dream_*.py`, fast)

- Feature encoding: verify 154-dim vector for known states (card encoding, zero-padding for absent seats, one-hot acting player, live mask)
- Net construction: verify MLP output shape equals `ActionVocab::size()` for both 6-action and 10-action vocabs
- Reservoir: verify size cap, weighted sampling, correct weight accumulation
- IS weights: verify `adv(a) = (utility / π(a*)) * I[a == a*]` on a trivial 2-action hand

### Tier 2 — Convergence gate (MacBook validation criterion)

Kuhn poker convergence to `−1/18` within 50K traversals, using the existing `kuhn_exact_value_lookup` fixture. **This is the go/no-go signal before moving to cloud compute.** If DREAM passes this gate, the IS weights, reservoir, and retraining loop are correct.

### Tier 3 — Smoke test (pre-cloud)

500-hand mixed-table eval (`scripts/eval_mixed_table.py`) against the tabular blueprint. Verify `DreamStrategy` produces non-degenerate output (no always-fold, no always-check) and BB/100 exceeds a random agent baseline.

## Transition Plan

1. Implement `src/dream/` — all five files
2. Add `DreamStrategy` to `bindings.cpp` (alongside `BlueprintStrategy`)
3. Add `scripts/train_dream.py`
4. Pass Tier 1 + Tier 2 tests on MacBook
5. Run Tier 3 smoke test
6. Move to cloud, run full training
7. Cleanup commit: remove `src/blueprint/BlueprintTrainer`, `BlueprintStrategy`, `SIXBP001` loader; remove `BlueprintStrategy` from bindings
