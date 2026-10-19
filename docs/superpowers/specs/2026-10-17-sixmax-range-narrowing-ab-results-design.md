# Phase 2 Search — Range-Narrowing Results (HU)

**Date:** 2026-10-17
**Status:** Interim — latency probe complete; full paired A/B pending (run-when-ready)
**Implements:** `docs/superpowers/plans/2026-10-17-sixmax-search-range-narrowing.md`
**Spec:** `docs/superpowers/specs/2026-10-17-sixmax-search-range-narrowing-design.md`
**Prior run:** `docs/superpowers/specs/2026-10-07-sixmax-phase2-search-ab-results-design.md`

## TL;DR

Preflop-inclusive range narrowing + mass pruning is implemented end-to-end
(Tasks 1-8, all unit-tested, full suite 368 passed) and is live behind
`[search].narrow_preflop` (default true, but `[search].enabled` still false).

A live latency probe against the 50M HU blueprint confirms the spec's central
speed/fidelity hypothesis: **narrowing is ~3x faster wall-time and falls back
far less often** than solving against the uniform ~1081-combo range. The full
statistical strength verdict (paired 2000×12 A/B) is a multi-hour run and has
not yet been executed; the command is below.

## What shipped (verified)

- Engine `action_history` log (voluntary actions only, blinds excluded).
- C++ `RangeTracker::set_prior` (degenerate-safe) and
  `SearchStrategy::probs(..., opp_prior=[])` (back-compatible).
- Agent: faithful street-root reconstruction (`_replay_to_street_root`, pot
  matches the live hand exactly on a no-quantization line), preflop opponent
  prior (`_opponent_prior` + per-combo `_combo_likelihood`), mass pruning
  (`_prune_prior`), and the wired `get_action` search branch with a
  blueprint fallback on any failure.
- Config knobs `narrow_preflop` (true) and `prune_keep_mass` (0.99) in both
  `sixmax/configs/default.toml` and `hu.toml`.
- Unit coverage: `test_action_history`, `test_range_tracker`,
  `test_search_strategy`, `test_search_config`, `test_search_translation`,
  `test_search_reconstruction`, `test_opponent_prior`,
  `test_search_agent_wiring` (incl. the disabled-path production regression).

## Latency probe (real, 2026-10-17)

Fast-knob probe (`budget_ms=15000, leaf_samples=1, max_nodes=16`), HU 50M
blueprint vs the frozen tabular HU opponent, 10 hands × 2 seats = 20 games:

```bash
SIXMAX_SEARCH_BUDGET_MS=15000 SIXMAX_SEARCH_LEAF_SAMPLES=1 SIXMAX_SEARCH_MAX_NODES=16 \
PROBE_CKPT=sixmax/checkpoints/hu_blueprint.bin PROBE_TOML=sixmax/configs/hu.toml \
  uv run python scripts/probe_search.py 10 on                       # narrowing ON
# + SIXMAX_SEARCH_NARROW_PREFLOP=false for the uniform baseline
```

| Arm | Wall (20 games) | Per-hand | searches | fallbacks |
|-----|-----------------|----------|----------|-----------|
| Narrowing ON (narrow_preflop=true) | 32.7s | 1633 ms | 2 | 2 |
| Narrowing OFF (uniform range)       | 98.5s | 4926 ms | 3 | 9 |

**Read:** at these knobs narrowing is ~3.0x faster wall-time and completes
within budget far more often (2 fallbacks vs 9). The fallback drop is the
load-bearing result: against the uniform range the subgame exhausts the
deadline and falls back to the blueprint; the pruned range lets search actually
act. The chip/BB-100 deltas in this probe are statistical noise (2-3 searches
total) and carry no strength signal.

The degenerate-knob caveat below is superseded by the production-knob
profile that follows — the ~3x is a fallback-rate effect at `leaf_samples=1`,
not a measurement of either the real leaf cost or the Python prior path.

## Production-knob profile (2026-10-17) — decisive

Per postflop search decision at the shipped HU knobs (`budget_ms=5000`,
`leaf_samples=256`, `max_nodes=20000`), timed phase-by-phase
(`scripts/_profile_search_decision.py`, throwaway, now deleted):

| Phase | Time | Share |
|-------|------|-------|
| `_replay_to_street_root` | 0.1 ms | ~0% |
| `_opponent_prior` (all ~1081 combos) | 11 ms | 0.2% |
| `_prune_prior` | 0.3 ms | ~0% |
| `SearchStrategy.probs` (C++ leaf eval + solve) | **3007 ms** | **~60%** |

3007 ms is exactly the leaf deadline (`budget_ms * 0.6`, `search_strategy.cpp`).
The C++ leaf evaluator burns the entire leaf budget and still cannot value all
leaves, so `Subgame::complete()` is false and `probs` falls back to the
blueprint **before even solving** (`searched=False` on every decision at these
knobs: 0 successful searches / 14 fallbacks over 20 hands).

**Two conclusions:**

1. **The new Python range path is negligible — 11.5 ms total, 0.2% of budget.**
   The reviewer's deferred concern about `_opponent_prior` per-combo cost is
   resolved; no memoization is needed. Narrowing is cheap and correct.
2. **The leaf evaluator is the sole bottleneck**, exactly as `.mex/ROUTER.md`'s
   Phase-2 note says. Narrowing cuts the opponent support `m` (the leaf matrix
   is `1 × m`, hero is a point mass) by ~2.5x on a raise line, but the
   evaluator needs ~100x: leaf cost ≈ (#flop/turn leaf nodes) × `m` ×
   `leaf_samples` × rollout_length, and at `leaf_samples=256` it gets ~25-40%
   through 3000 ms.

## Full paired A/B — DEFERRED (null by construction at production knobs)

Not run. At the shipped knobs search falls back 100% of the time, so a
2000×12 A/B (~8-9 h) would compare blueprint-vs-blueprint and reproduce the
2026-10-08 null. It is only worth running once searches actually *complete*
within budget. The command remains valid for when that holds:

```bash
nohup caffeinate -is env PYTHONUNBUFFERED=1 uv run python scripts/eval_search_ab.py \
  --checkpoint sixmax/checkpoints/hu_blueprint.bin \
  --search-toml sixmax/configs/hu.toml \
  --hands 2000 --seeds 1,2,3,4,5,6,7,8,9,10,11,12 \
  > docs/superpowers/specs/2026-10-17-range-narrowing-ab.log 2>&1 &
```

**Statistical-power caveat (for when it runs):** 2000 × 12 is an interim gate,
below the plan's `--hands 20000`. Treat the paired diff / `|t|` as a
sign/trend, not a verdict.

## Next step — parallelize the leaf evaluator (no value net required)

The profile reframes the roadmap: a value net is the ultimate O(1) leaf, but
it is no longer the *only* path to a non-null A/B. `LeafEval::matrix`
(`sixmax/src/search/leaf_eval.cpp`) is a serial loop over opponent combos whose
cells are independent (`out[i*m+j]`); the only shared state is one
`std::mt19937_64`. Parallelizing it (per-combo RNG, `std::thread` +
`hardware_concurrency` as in `sixmax/src/dream/trainer.cpp`) is a bounded C++
change worth ~Ncore with zero accuracy cost. Combined with a more aggressive
`prune_keep_mass` (0.90) on raise lines (~2.5x on `m`) and per-rollout
allocation cleanup (the `clone_engine()` + 52-entry pool rebuild in
`rollout()`), the compounded speedup (~20x on an 8-core host) could get flop
searches completing within a few seconds — enough to make this A/B measure a
real difference. That change gets its own brainstorm → spec → plan.

## Parallel leaf evaluator — SHIPPED (2026-10-19)

`LeafEval::matrix` now evaluates opponent combos across worker threads
(`[search].leaf_threads`, default `0` = all cores; `SIXMAX_SEARCH_LEAF_THREADS`
env). Each sampled cell seeds its own RNG from `(seed, street, our_combo,
opp_combo)` instead of one shared sequential stream, so the matrix is
**bit-identical for any thread count** (pinned by
`test_sampled_leaf_identical_across_thread_counts`) and robust to deadline
early-exit — which the old shared stream was not.

**Measured (warm, 8 logical cores, one flop leaf, 400 combos, samples=64):**
serial 280 ms → all cores 95 ms ≈ **3x**, identical output. Sublinear in cores
(allocator contention on the per-rollout `clone_engine()` + efficiency/HT
cores), but free and accuracy-neutral.

**Throughput reframed — the limiter moved from leaf eval to the solver.** With
threading, leaf eval finishes under the 3000 ms leaf deadline, so searches now
*complete* (`searched=True`) where before they fell back 100% of the time. But
per-decision wall is unchanged (~budget_ms): `VectorCfr::solve` has no
convergence early-exit (`vector_cfr.cpp:45-48`) — it runs to `max_iters` or
burns the leftover budget, and the orchestrator hands it all remaining time.
So a completed decision ≈ `budget_ms` (~5 s shipped), ~0.2 decisions/s. The
throughput lever is now the **solver** (`max_iters` / a convergence early-exit),
not leaf eval.

## max_iters sweep (2026-10-19) — no signal, underpowered

Paired search-on vs no-search blueprint, HU 50M checkpoint vs the frozen
tabular opponent, `budget_ms=30000` (so `max_iters` binds), all cores,
150 hands × seeds 1-4. Log: `2026-10-19-maxiters-sweep.log`.

| max_iters | paired diff (search − blueprint) | sd | se | t | verdict |
|-----------|----------------------------------|-----|-----|------|----------------|
| 100  | −32.4 BB/100 | 57.5  | 28.8  | −1.13 | keep-blueprint |
| 2000 | +31.4 BB/100 | 244.7 | 122.4 | +0.26 | keep-blueprint |

**Read:** no evidence search beats the blueprint at either setting (both
`|t| ≪ 2`); the sign flip is noise. More iters did not improve the mean — it
**inflated variance** (sd 57 → 245). The run validated that searches now
*complete* and emit real non-blueprint strategies (nonzero, volatile diffs),
but says nothing conclusive about strength.

**Power caveat / known confound:** n=4 × 150 hands is far too small. Worse, the
harness pins the *deals* (duplicate matches) but not each agent's
action-sampling RNG (`SixmaxAgent._rng` is entropy-seeded), so the off-arm
blueprint score itself varies run-to-run (seed 2 blueprint: +90.5 at
`max_iters=100`, −158.0 at `2000`). The pairing cancels card variance, not
sampling variance. **Before any larger A/B: pin `agent._rng` in the harness**
(as `test_search_agent_wiring.py` already does) and raise hands/seeds — else the
residual sampling noise swamps any real edge.
