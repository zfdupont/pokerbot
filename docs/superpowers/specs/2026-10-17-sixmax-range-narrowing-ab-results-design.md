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

**What this probe does and does NOT measure.** `leaf_samples=1, max_nodes=16`
is a *degenerate* search (tiny tree, one leaf sample) chosen to isolate
fire/fallback behavior cheaply. The ~3x number is therefore a leaf-eval +
fallback effect; it does **not** measure the cost of the new Python
prior-construction path (`_opponent_prior` rebuilds a per-combo engine and
replays the preflop line for each of ~1081 unblocked combos, every postflop
decision). That cost is on the critical path and is invisible at these knobs.
**Before the full A/B, profile `_opponent_prior` at production `budget_ms` /
`leaf_samples`** — if per-decision prior construction is material, memoize the
per-combo likelihood replay. The full A/B below runs at production knobs and
will surface it.

## Full paired A/B — PENDING (run-when-ready, not CI)

Multi-hour unattended run; launch deliberately (detach per the repo's
long-job convention):

```bash
nohup caffeinate -is env PYTHONUNBUFFERED=1 uv run python scripts/eval_search_ab.py \
  --checkpoint sixmax/checkpoints/hu_blueprint.bin \
  --search-toml sixmax/configs/hu.toml \
  --hands 2000 --seeds 1,2,3,4,5,6,7,8,9,10,11,12 \
  > docs/superpowers/specs/2026-10-17-range-narrowing-ab.log 2>&1 &
```

**Statistical-power caveat:** 2000 hands × 12 seeds is an interim gate, below
the plan's full `--hands 20000` gate (which stays infeasible until the Phase-3
value net lands). Treat the paired diff / `|t|` as a sign/trend, not a verdict.
Success = search non-worse than the blueprint with latency low enough to scale
the gate up; stretch = positive paired diff with `|t|` trending up.

When run, append: per-seat BB/100 table, paired diff, `|t|`, and a VERDICT line,
mirroring the 2026-10-07 results doc.
