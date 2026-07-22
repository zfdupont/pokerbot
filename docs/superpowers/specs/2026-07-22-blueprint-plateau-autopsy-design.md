# Blueprint Plateau Autopsy — Design

**Date:** 2026-07-22
**Status:** Design (pending implementation plan)
**Author:** brainstormed with Claude

## Problem

The 6-max blueprint is flat at **~−485 ±50 BB/100 vs PotOdds** across **1.1M→2.0M total
iters** (`sixmax/checkpoints/curve.csv`, this-run increments 100k–1M), with no monotonic
trend — it loses ~5 BB/hand to a *passive* baseline. ROUTER.md records the root cause as
**severe undertraining** (~117k infosets at 2M iters ≈ ~17 visits/infoset). The 20k-deck
sweep makes the flat line statistically real, not eval noise.

But a *perfectly flat* curve is ambiguous. It is consistent with two opposite diagnoses:

- **Undertraining** — the policy is improving, but the per-iteration gain is below the ±50
  BB/100 CI. Fix: scale training (10–100×). Expensive (~90 iters/s → 20M iters ≈ days).
- **Structural ceiling** — the abstraction is too coarse (117k infosets is *tiny* for
  6-max), or exploration starves most nodes, so more iterations cannot help. Fix: redesign
  abstraction/exploration. Scaling would waste days of compute.

Same symptom, opposite fixes. **We must not commit compute until we know which it is.**

## Goal

Cheaply (offline, no retraining) determine whether the flat curve is undertraining or a
structural ceiling, and emit a falsifiable verdict that either justifies scale-up (with an
iteration estimate) or redirects effort to abstraction/exploration.

## Non-goals

- Fixing the plateau. This is diagnosis only; the fix is a separate spec chosen by the verdict.
- More/better BB/100 evaluation (paired per-deck slope is a separate, complementary lever).
- Any change to the trainer, abstraction, or checkpoint format.

## Key assets (already present)

- **10 checkpoint snapshots** `blueprint_00100000.bin … blueprint_01000000.bin`, each
  embedding the full regret/strategy table (`checkpoint.h`).
- `resume_blueprint(...)` → a `BlueprintTrainer` with `.keys()` + `.average_strategy(key)`:
  enumerates every infoset in a checkpoint without retraining.
- `pack_abstract_key`, `preflop_class`, `pot_bucket`, `hand_equity`, `Abstraction.bucket`,
  `EngineGameState.abstract_key()` / `probs_for`: build and query canonical probe spots.
- Abstract-key bit layout (`abstract_key.h`): card 0-7, street 8-9, raises 10-17,
  pot 18-19, live 20-22, after 23-25 — lets Python slice infosets by street/raises/live.

## The one gap → new binding

Every exposed accessor returns *normalized* probs; the `strategy_sum` mass (visit-weight
proxy) and `regret` are discarded. Without mass we cannot separate a **well-visited**
infoset from a tail one, which is the crux of the diagnosis.

**Add one read-only binding** in `sixmax/src/bindings/bindings.cpp`:

```
dump_infosets(checkpoint_path, vocab) -> list of (key: int, probs: list[float],
                                                   strategy_sum_l1: float, regret_l1: float)
```

- Loads via the existing `resume_blueprint` path (no retraining).
- Read-only; no change to `InfosetData`, the trainer, or the checkpoint format.
- Respects the invariant that bridging lives only in `bindings/` and `scripts/`.
- `strategy_sum_l1 = sum(strategy_sum)` — monotone proxy for cumulative reach-weighted
  visits. `regret_l1 = sum(|regret|)` — for the average-regret convergence signal.

## Metrics (per checkpoint, traced 100k→1M)

1. **Visit-weight distribution.** Sort infosets by `strategy_sum_l1`. Report: Gini,
   top-1% and top-100 weight share, and the count of infosets above a "matured" weight
   threshold `T` (default: mass ≥ some multiple of a single-visit unit; tuned once from the
   observed distribution). Tests whether the "17 avg visits" figure is a mirage hiding a
   power-law where most keys are effectively untrained.

2. **Visit-weighted policy entropy.** Shannon entropy of `probs` over legal actions,
   averaged with `strategy_sum_l1` weights — computed (a) over **all** infosets and
   (b) over the **top-weight tier only** (infosets above `T`). Traced across iterations.
   Falling entropy ⇒ policy is differentiating (learning). Flat-at-near-max ⇒ stuck.
   The top-tier restriction is the decisive cut: those infosets *have* enough evidence, so
   if even they stay near-uniform the problem is structural, not scale.

3. **Probe-set trace.** ~15 canonical spots with a known correct direction, queried via
   `probs_for` on constructed `EngineGameState`s and traced across snapshots. Examples:
   - Premium UTG opens (AA, KK, AKs): raise mass should rise, fold ≈ 0.
   - Trash UTG (72o, 83o): fold mass should rise; the "uniform open-jam" pathology (roughly
     equal mass across bet sizes + allin) should shrink.
   - A couple of clear postflop spots (nut hand on dry board should bet/raise).
   The probe set and each spot's expected direction are defined explicitly in the script.

## Decision rule (the payoff)

- **Undertraining** if: top-tier entropy is falling **and** probes move in the correct
  direction **and** average regret (`regret_l1 / iterations`) at top-tier infosets is
  shrinking. → Scale is justified. Extrapolate the entropy/probe trend to estimate the
  iteration budget, then proceed to the scale-up spec (fork 2).
- **Structural ceiling** if: top-tier infosets stay near-uniform **or** probes are flat
  **or** average regret is not shrinking. → Scaling wastes compute; redirect to an
  abstraction/exploration redesign spec.
- **Mixed** (e.g. probes move but entropy barely falls): report both signals and their
  extrapolations; recommend the paired-per-deck eval slope as a tie-breaker before
  committing compute.

## Deliverable

`scripts/diagnose_blueprint.py`:
- Sweeps a series of `blueprint_<iters>.bin` snapshots (glob or explicit list).
- Emits a per-checkpoint table (num_infosets, Gini, top-1% share, matured count, all-infoset
  entropy, top-tier entropy, mean average-regret) + a probe-trace table. Optional `--csv`.
- Prints the verdict per the decision rule at the end.

Pure metric functions (entropy, Gini, weighted mean, threshold counts, probe-direction
scoring) are buck2/checkpoint-free and **unit-tested**, mirroring the existing
`eval_sixmax_baseline.py` pure/bridge split. The binding-touching and checkpoint-loading
parts are thin and covered by a smoke test against one real snapshot.

## Testing

- Unit tests for each pure metric on hand-constructed inputs (known entropy/Gini values;
  a synthetic power-law weight vector; probe-direction scoring on synthetic probs).
- One smoke test: `dump_infosets` on `blueprint_00100000.bin` returns non-empty, probs sum
  to ~1 per infoset over legal actions, weights are non-negative.
- Full suite (`uv run pytest tests/`) stays green; C++ rebuilds via
  `~/bin/buck2 build //sixmax:sixmax`.

## Risks / open questions

- **Threshold `T` is data-driven.** Pick it once from the observed `strategy_sum_l1`
  distribution (e.g. a percentile), and report results at 2–3 thresholds so the verdict is
  not threshold-sensitive.
- **Probe keying must match the trainer.** Build probes through the same
  `abstract_key`/`probs_for` path the deployment bridge uses, so we cannot silently query
  keys the trainer never wrote.
- **Endianness / arch.** Checkpoints are native-endian; run the autopsy on the same machine
  that trained them (already the case).
