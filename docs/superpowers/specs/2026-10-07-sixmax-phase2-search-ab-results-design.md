# Phase 2 Search (HU) — A/B Results & Reproduce Spec

**Date:** 2026-10-07
**Status:** For review — exploratory result; **no deployment change**
**Path:** measurement / evaluation report (no code change)
**Related:** `docs/superpowers/plans/2026-10-06-sixmax-phase2-search-hu.md`,
`docs/superpowers/specs/2026-09-30-sixmax-phase2-search-hu-design.md`,
`.mex/patterns/eval-checkpoint-ab.md`,
`scripts/eval_search_ab.py`

## TL;DR

We pitted the Phase-2 depth-limited search bot against the identical bot with search off,
both facing the frozen tabular CFR opponent on paired seeds, to test whether search provides
an advantage.

**The decision-grade finding is feasibility, not win-rate.** The leaf evaluator cannot finish
within any affordable per-decision budget, so search is non-deployable (7–12 s/hand vs 1.6 ms).
It was also *unsafe* when deadline-truncated — silently breaking the plan's "never worse than the
blueprint by construction" invariant — but that half is now **fixed** (commit `b18b607`; see
**Safety invariant violated**): a starved decision falls back to the blueprint instead of acting
on a zero-valued subgame. Feasibility (latency) is what still keeps `enabled = false`; the
win-rate A/B is secondary and, as run here, statistically null.

- **Both A/B runs are statistically null** (CIs span zero); treat them as directional only:
  - Run A — plan-realistic budget (`budget_ms=600`, `leaf_samples=16`, `max_nodes=5000`):
    paired diff **−42.76 BB/100**, `t = −1.38` (p ≈ 0.2, not significant), 10 seeds × 250 hands.
  - Run B — completes-but-shallow (`budget_ms=15000`, `leaf_samples=1`, `max_nodes=16`):
    paired diff **+58.54 BB/100**, `t = +0.58`, CI `[−182, +299]`, 8 seeds × 30 hands.
- **Neither passes the deployment gate** (`mean > 0 AND |t| ≥ 2`). Search is not promoted;
  `enabled = false` stays.

The two point estimates differ in sign, but neither is distinguishable from zero and the runs
differ in five knobs at once (budget, samples, nodes, seeds, hands), so the sign flip is **not**
evidence of anything on its own. What *is* solid — because it is a deterministic measurement, not
a high-variance draw — is the latency sweep: the current sampler is the wall. Even a 16-node tree
costs ~7–12 s per decision, and at any realistic budget the leaf matrix is deadline-truncated to
zeros. The hypothesis that truncation (not search) is what hurts is plausible and consistent with
the latency data, but is not established by these two runs.

## Background & the gate

The Phase-2 plan defines the deployment gate for search-on vs search-off as a paired A/B with a
`promote` verdict when `mean(diff) > 0 AND |t| ≥ 2` (`.mex/patterns/eval-checkpoint-ab.md`).
`SearchStrategy` is wired into `SixmaxAgent` behind an opt-in flag; both configs ship
`[search].enabled = false` pending that gate.

The plan's reference invocation is `--hands 20000 --seeds 1,2,...`. That is infeasible with the
current leaf evaluator (see Feasibility), so this run trades hands for completed (unstarved)
search and documents the trade explicitly.

## Setup

| Item | Value |
|------|-------|
| Search bot | `SixmaxAgent`, checkpoint `sixmax/checkpoints/hu_blueprint.bin` (2-player, 50M iters), config `sixmax/configs/hu.toml` |
| Opponent (both arms) | `CFRAgent`, `cfr/checkpoints/checkpoint_09040000.pkl` (frozen tabular HU bot) |
| Harness | `scripts/eval_search_ab.py` (paired: search-on vs search-off, same seeds/opponent) |
| Engine frame | `bb=2`, `stack=200` (rescaled to the `starting_stack=100, big_blind=1` training frame) |
| Search-on forcing | harness sets `SIXMAX_SEARCH_ENABLED=true` for the on-arm only; deployment configs are untouched |

Search knob overrides are read by `_load_search_config` via `SIXMAX_SEARCH_<KEY>` env vars
(`agents/sixmax_agent.py`), which is how each run was parameterised without editing TOML.

## Commands run

All commands from the repo root. The harness forces `SIXMAX_SEARCH_ENABLED` itself; the
`SIXMAX_SEARCH_*` overrides set the search budget/tree/sample knobs.

### Latency probe (see Appendix A for the script)

```bash
# Fire rate, fallback rate, and wall time per hand
SIXMAX_SEARCH_BUDGET_MS=600 SIXMAX_SEARCH_LEAF_SAMPLES=16 \
SIXMAX_SEARCH_MAX_NODES=5000 SIXMAX_SEARCH_MAX_ITERS=500 \
PROBE_CKPT=sixmax/checkpoints/hu_blueprint.bin PROBE_TOML=sixmax/configs/hu.toml \
  uv run python scripts/probe_search.py 20 on

# Does leaf evaluation ever complete? (tiny tree, minimal samples, growing budget)
for b in 5000 15000 30000; do
  SIXMAX_SEARCH_BUDGET_MS=$b SIXMAX_SEARCH_MAX_NODES=16 SIXMAX_SEARCH_LEAF_SAMPLES=1 \
  SIXMAX_SEARCH_MAX_ITERS=5000 \
  PROBE_CKPT=sixmax/checkpoints/hu_blueprint.bin PROBE_TOML=sixmax/configs/hu.toml \
    uv run python scripts/probe_search.py 6 on
done
```

### Run A — plan-realistic budget (starved)

```bash
nohup env SIXMAX_SEARCH_BUDGET_MS=600 SIXMAX_SEARCH_LEAF_SAMPLES=16 \
  SIXMAX_SEARCH_MAX_NODES=5000 SIXMAX_SEARCH_MAX_ITERS=500 PYTHONUNBUFFERED=1 \
  uv run python scripts/eval_search_ab.py \
    --checkpoint sixmax/checkpoints/hu_blueprint.bin \
    --search-toml sixmax/configs/hu.toml \
    --hands 250 --seeds 1,2,3,4,5,6,7,8,9,10 \
    > /tmp/search_ab_run.txt 2>&1 &
```

### Run B — unstarved (large budget / tiny tree)

```bash
nohup env SIXMAX_SEARCH_BUDGET_MS=15000 SIXMAX_SEARCH_LEAF_SAMPLES=1 \
  SIXMAX_SEARCH_MAX_NODES=16 SIXMAX_SEARCH_MAX_ITERS=5000 PYTHONUNBUFFERED=1 \
  uv run python scripts/eval_search_ab.py \
    --checkpoint sixmax/checkpoints/hu_blueprint.bin \
    --search-toml sixmax/configs/hu.toml \
    --hands 30 --seeds 1,2,3,4,5,6,7,8 \
    > /tmp/search_ab_bigtree.txt 2>&1 &
```

`--hands N` = N duplicate deals × 2 seats per seed (`scripts/eval_hu_sanity.py::run_duplicate_match`).

## Results

### Run A — `budget_ms=600, leaf_samples=16, max_nodes=5000` (starved), 10 seeds × 250 hands

```
seed     search  blueprint       diff
   1    -114.20     -69.20     -45.00
   2     +11.10     -16.40     +27.50
   3     -11.90     -59.50     +47.60
   4    +191.40     +76.80    +114.60
   5     +58.80    +160.40    -101.60
   6     -89.50     -34.50     -55.00
   7    -143.70     +36.20    -179.90
   8    -123.20     +37.30    -160.50
   9     +33.80      -8.20     +42.00
  10     -83.50     +33.80    -117.30

n=10 seeds x 250 hands
paired diff: -42.76 BB/100  sd 98.3  se 31.1  t=-1.38
VERDICT: keep-blueprint
```

Median diff = **−50.0**. 6 of 10 seeds negative.

### Run B — `budget_ms=15000, leaf_samples=1, max_nodes=16` (unstarved), 8 seeds × 30 hands

```
seed     search  blueprint       diff
   1    -423.33    -245.83    -177.50
   2    +242.50    +222.50     +20.00
   3    -447.50    -195.83    -251.67
   4      -0.83    +190.00    -190.83
   5    +440.83    -179.17    +620.00
   6     +10.00    -141.67    +151.67
   7    -265.00    -304.17     +39.17
   8    -311.67    -569.17    +257.50

n=8 seeds x 30 hands
paired diff: +58.54 BB/100  sd 287.7  se 101.7  t=0.58
VERDICT: keep-blueprint
```

Median diff = **+29.6**; the mean is dominated by one `+620` seed. 5 of 8 seeds positive.
95% CI ≈ `[−182, +299]` — indistinguishable from zero.

### Latency / feasibility sweep

Per-decision wall time = total / `searches` (`searches` from the probe, 0 fallbacks in all runs):

| tree `max_nodes` | `leaf_samples` | `budget_ms` | wall / search | budget-bound? |
|---|---|---|---|---|
| 5000 | 16 | 600 (Run A config) | 0.79 s | yes |
| 24 | 1 | 2000 | 2.01 s | yes |
| 24 | 1 | 6000 | 5.78 s | yes |
| 60 | 1 | 2000 | 2.01 s | yes |
| 60 | 1 | 6000 | 6.02 s | yes |
| 16 | 1 | 5000 | 5.02 s | yes |
| 16 | 1 | 15000 | 11.48 s | borderline (completes) |
| 16 | 1 | 30000 | **7.08 s** | **no — completes early** |

Interpretation: when time/search grows ~linearly with `budget_ms`, the leaf deadline is truncating
work. Only a 16-node tree with a single sample stops scaling (≈7 s at a 30 s budget), i.e. the
leaf matrix finally fills before the deadline. Everything larger/sampled remains starved at any
budget we can afford.

### Feasibility of the plan's gate

At ~0.42 searches/game and ~7–12 s/search (Run B), the ON arm costs roughly:

- 20 000 hands/seed ⇒ ~16 800 searches ⇒ ~33 h/seed (≈ 11 days for 8 seeds).

The plan's `--hands 20000 --seeds 1..8` gate is therefore ~2 orders of magnitude out of reach with
the current sampled leaf continuation.

## Root cause (measured, not inferred)

1. **Leaf evaluation dominates and is unbounded by design.**
   `sixmax/src/search/search_strategy.cpp` sets the leaf deadline to `0.6 × budget_ms`, then
   `Subgame::build` fills a `1 × opp_hands × leaf_samples` value matrix per leaf by calling
   `LeafEval::matrix`. At a flop/turn leaf, `LeafEval::rollout`
   (`sixmax/src/search/leaf_eval.cpp`) clones the engine and plays the blueprint to showdown once
   per (opponent hand × sample).
2. **The opponent range is uniform over ~1000 combos.**
   `RangeTracker::reset` (`sixmax/src/search/range_tracker.cpp`) assigns a uniform prior over every
   combo not blocked by the board/hero cards — `C(47,2)=1081` on the flop, `C(46,2)=1035` on the
   turn, `C(45,2)=990` on the river (5 board + 2 hero cards removed). The agent calls
   `probs(engine_state, 0, [])` with an **empty observed prefix** (`agents/sixmax_agent.py`,
   postflop branch), so the preflop line is never used to narrow the range. Cost therefore carries
   a hard ~1000× factor per leaf.
3. **The deadline truncates to zeros, not to fewer samples.**
   `LeafEval::matrix` returns as soon as the deadline passes, leaving unfinished cells at `0.0`.
   CFR then treats those showdowns as worth nothing, distorting the subgame — which is the most
   plausible reason the starved Run A is *worse* than the blueprint.
4. **The state passed in is approximate.**
   `SixmaxAgent._reconstruct_search_state` rebuilds the decision as a canonical check-down pot and
   ignores in-street betting (`observed_actions=[]`; its own comment calls the chips "approximate").
   Even a fully evaluated search solves a subgame that does not match the live one.

## Safety invariant violated

The Phase-2 plan's central promise (plan §Goal) is that search is **"never worse than the
blueprint by construction"** — the blueprint fallback was supposed to guarantee it. That guarantee
is **false as implemented**, and this run has the evidence:

- The fallback in `SearchStrategy::probs` fires only on an **empty or throwing** solve. A solve
  that *completes on a corrupted subgame* returns a non-empty, normalised distribution, so
  `last_decision_searched()` is true and the agent acts on it.
- When `LeafEval::matrix` passes its deadline it `return`s early, leaving unevaluated cells at
  `0.0` (`leaf_eval.cpp`). CFR then solves a subgame in which many real showdowns are valued at
  zero — a confident-but-wrong strategy, not a degraded-but-safe one.
- The latency table reports **0 fallbacks in every run**. Combined with Run A's negative point
  estimate, that is the tell: during the arm that underperformed, the safety net *never engaged*.
  A fallback that cannot fire on a degraded solve protects nothing.

This reframes Run A from "search tuned poorly" to "search can return actively harmful actions with
no guard." It also yields a fix that is **independent of the value net**: on deadline, `LeafEval`
(and `SearchStrategy`) must **fall back** — return empty so the blueprint is used — rather than
emit a partially-zero matrix.

**Update — fixed (commit `b18b607`).** `Subgame` now exposes `complete()`, which is false whenever
a leaf could not be fully valued (deadline-truncated cells, or a sampled leaf with no blueprint);
`SearchStrategy::probs` checks it immediately after construction and falls back to the blueprint
(empty result, `last_decision_searched()` stays false) before spending budget on a solve it would
discard. Regression-pinned by `tests/sixmax/test_subgame.py::test_subgame_completeness_reflects_leaf_evaluation`
and `tests/sixmax/test_search_strategy.py::test_incomplete_search_falls_back_to_blueprint`. The
"never worse than the blueprint by construction" invariant now holds for the starvation case: a
decision that cannot finish search degrades to the blueprint rather than acting on a zero-valued
subgame. (This only restores *safety* — it does not make search *stronger*; a starved decision is
still just the blueprint. Strength still waits on range narrowing + the value net.)

## Interpretation

- The latency sweep is the load-bearing evidence (it is deterministic and repeatable); the A/B
  win-rate is not. Giving search enough budget to **complete** leaf evaluation moved the paired
  point estimate from −42.8 to +58.5 BB/100, but across five simultaneously-changed knobs and with
  both CIs spanning zero. Read as: *consistent with "truncation hurts," not a demonstration of it.*
- The positive Run B result is **not** a demonstrated advantage: `t = 0.58`, one large outlier,
  only 30 hands/seed, and a CI that spans zero. Neither is Run A a demonstrated loss (`t = −1.38`).
- Both configurations are **far slower** than the blueprint (~7–12 s vs ~1.6 ms/hand), so neither is
  deployable regardless of the sign.

**Conclusion:** the Phase-2 search, as currently implemented and parameterised, does not provide a
measurable advantage over the blueprint. `enabled = false` is correct. The result does not reject
the search idea; it rejects *this leaf evaluator + range prior* as the way to realise it.

## Limitations

- **Sample size.** Run A: 2 500 hands/arm; Run B: 240 hands/arm. Both are below the pattern's
  ≥ 8 seeds × ≥ 20 000 hands bar; this was a feasibility study, not a gate-qualifying run.
- **Tuned for completion, not strength.** Run B's 16-node tree is very shallow and `leaf_samples=1`
  makes each leaf a single noisy playout — the search is unstarved but low-fidelity.
- **One opponent, one frame.** Both arms face the frozen tabular bot at `bb=2`, `stack=200`.
- **Exploratory peeking.** Settings were chosen adaptively (A → diagnose → B). Treat signs, not
  magnitudes, as the signal.
- No code was changed to produce either run; both used the existing harness and env overrides.

## Requested review decisions

1. **Accept the "leaf evaluator is the wall" diagnosis** as the blocker for the Phase-2 gate, and
   redirect the next iteration to the value net (the plan's own Phase-3 replacement)?
2. **Prioritise range narrowing** — thread the observed preflop/in-street prefix into `RangeTracker`
   so `opp_hands` is a real range rather than ~1225 uniform combos — as a cheap standalone
   follow-up? This shrinks leaf cost ~4–10× and improves fidelity even before the value net.
3. **Confirm no deployment change**: keep `[search].enabled = false` in both
   `sixmax/configs/default.toml` and `sixmax/configs/hu.toml`, and record "search not yet better"
   in the scaffold?
4. **Re-run policy**: should the reference gate stay `hands=20000` (only viable post-value-net), or
   should a documented reduced-size interim gate (e.g. `hands=2000, seeds=12`) be adopted with an
   explicit power caveat?

## Recommended next steps (in order)

Ordered cheapest-and-safety-first; the value net is what finally makes the *full* gate affordable,
but it is not the first thing to do.

1. ~~**Deadline → fallback (safety, ~trivial).**~~ **DONE (commit `b18b607`).** `Subgame::complete()`
   + a guard in `SearchStrategy::probs` fall back to the blueprint when leaves are unevaluated,
   restoring the "never worse than blueprint" invariant for the starvation case (see **Safety
   invariant violated**).
2. **Narrow the opponent range (cost + fidelity, standalone).** Thread the observed preflop/in-street
   prefix into `RangeTracker` and pass real `observed_actions` through the agent's search branch.
   This shrinks the per-leaf `opp_hands` factor ~4–10×, improves fidelity, and re-activates the
   Bayes tracker that is currently dead code — valuable with or without the value net.
3. **Value net leaf (gating dependency for the real gate).** Replace the blueprint-continuation
   leaf with the Phase-3 value net, removing the `opp_hands × samples × rollout` cost entirely.
4. **Only then** run the plan's gate (`--hands 20000 --seeds 1..8`) and update the scaffold.

## Appendix A — latency probe script

Committed at `scripts/probe_search.py` (portable: derives the repo root from its own location,
no hardcoded paths), so this report reproduces from a clean checkout. Invoke as in *Commands run*;
`PROBE_CKPT`/`PROBE_TOML` and the `SIXMAX_SEARCH_*` env vars set the checkpoint, config, and knobs.

## Appendix B — raw outputs

The per-seed tables and `VERDICT` lines in *Results* are the full harness output for each run (the
harness prints nothing else). The original stdout captures were written to ephemeral `/tmp` files
(`/tmp/search_ab_run.txt`, `/tmp/search_ab_bigtree.txt`) and are not retained; reproduce with the
commands in *Commands run* (each run is ~20–40 min). If a future run is meant to be cited as a
gate, redirect its output into a committed `docs/.../<date>-search-ab-<run>.log` rather than `/tmp`.
