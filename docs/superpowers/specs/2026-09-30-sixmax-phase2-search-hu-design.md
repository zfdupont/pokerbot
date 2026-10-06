# Six-Max Phase 2 — Depth-Limited Search (HU-First) — Design

**Date:** 2026-09-30
**Status:** Approved (brainstorming complete)
**Parent spec:** `docs/superpowers/specs/2026-07-17-sixmax-search-design.md` (Approach A: Pluribus-pattern
blueprint + depth-limited search). This document *scopes Phase 2* for the deployed heads-up bot and
adds the detail the parent spec left at phase level. It does not replace it.

## Motivation

The deployed bot (`poker.zfdupont.com`, and the sixmax CLI/web paths) plays a **bucketed** average
strategy: 169 preflop classes, and postflop equity-percentile buckets (50 flop / 50 turn / 20 river).
Every postflop decision is therefore made at bucket granularity — many distinct hands and boards share
one strategy. Search is the mechanism that removes that error at play time: at each postflop decision,
solve the *current street's* subgame with the player's **exact** cards and a tracked opponent range,
instead of reading a bucket.

We build this **heads-up first** because re-solving with a tracked range is well-defined for two
players and the subgames are small, then extend the same unit to 6-max later. The parent spec's
non-goal (safe re-solving is not well-defined multiway) stays in force.

## Decisions taken (with rationale)

1. **Depth-limited, not solve-to-showdown.** `[actions.search]` is 3 pot-fractions + all-in, so the
   public postflop tree is ~10^5 nodes with ~1,326-combo range vectors; one vector-CFR+ iteration is
   then tens of milliseconds, i.e. only tens of iterations inside a 5 s budget — too few to converge.
   v1 therefore limits depth to the **end of the current street** and evaluates leaves with the
   blueprint's continuation. (River subgames have no leaves — they solve to showdown exactly.)
2. **Search from the deployed 6-max blueprint** (`sixmax/checkpoints/blueprint.bin`), because that is
   the actual product. Search un-buckets, so it should help either checkpoint; if the HU training run
   ever beats it in the A/B, the pointer moves and nothing else changes.
3. **Preflop search is deferred to v2.** Preflop is the deepest tree and the highest risk. v1 searches
   flop/turn/river decisions only; preflop stays blueprint.

## Goals

- Add an optional **depth-limited search** to the HU deployment path, off by default and enabled by
  config/flag.
- Keep the search unit **player-count-parameterized** from day one (2–6 seats in the interfaces), so
  the 6-max extension is growth, not a rewrite. Only 2-player is built and tested in v1.
- Hard per-decision **time budget (~5 s default)**, node cap, and a **blueprint fallback on any
  failure** — search must never make the bot worse by being slow or crashy.
- **No new external dependencies.** Reuse the existing C++ engine/vocab/blueprint/abstraction and
  Python bridging. No libtorch (that is Phase 3).

## Non-goals

- Safe re-solving. We ship **unsafe re-solving knowingly** (parent spec), with the observed-action
  entry beliefs.
- Opponent exploitation/adaptation. The tracked range uses the blueprint as the opponent model.
- 6-max enablement (interfaces only), preflop search, and the Phase-3 value net.

## Scope of the v1 search point

| Street | Behaviour |
|---|---|
| Preflop | Blueprint (v1). Search only on an off-tree opponent bet in v2. |
| Flop | Search; leaves at end-of-flop evaluated by blueprint continuation. |
| Turn | Search; leaves at end-of-turn evaluated by blueprint continuation. |
| River | Search; solved to showdown (terminal, exact). |

Search runs only when it is the hero's turn on a searched street. The hero's action is emitted as a
`[actions.search]` vocab index and translated by the existing `target_bb` path — no change to how the
live agent converts an index into chips.

## Architecture

New module `sixmax/src/search/`, compiled into the existing `core` Buck target (no new target, no new
dependency):

```
sixmax/src/search/
  range_tracker.{h,cpp}   # HU opponent range over combos, Bayes via the blueprint
  subgame.{h,cpp}         # public betting tree from the street root + node cap
  leaf_eval.{h,cpp}       # end-of-street leaf values (blueprint continuation)
  vector_cfr.{h,cpp}      # CFR+ over range vectors, deadline-driven
  search_strategy.{h,cpp} # orchestration: tracker -> subgame -> solver -> root action
```

Reused unchanged: `HandState`, `EngineConfig`, `EngineGameState`, `ActionVocab`, `BlueprintStrategy`,
`Abstraction`, and the `policy.h` sampling helpers. `EngineGameState::bet_context()` (currently
annotated "exposed for tests and Phase 2 search") becomes a real consumer.

### `RangeTracker`

- One **dense vector over the 1,326 two-card combos** per tracked player, canonically indexed. Our own
  hand is a point mass; the opponent's is the distribution being tracked.
- `observe(blueprint, vocab, public_state, actor, action)`: reweight each combo `c` by the blueprint's
  probability of the observed action at the infoset keyed by (public state, `c`), then renormalize.
  Off-tree observed sizes are keyed via `ActionVocab::nearest`.
- **Invariants:** exact card-removal against the board and our hand; posterior always normalized (an
  all-zero posterior degrades to uniform-over-legal); deterministic given the action history and seed.
- Test seam: expose the raw posterior.

### `Subgame`

- Builds the **public** betting tree rooted at the **start of the current street**, carrying board,
  pot, stacks, street, and the entry ranges.
- Our actions come from `[actions.search]` (`ActionVocab`, masked by legality); the opponent's come
  from the **observed** bet sizes, so the tree includes off-tree sizes rather than rounding them.
- `max_nodes` budget: on exhaustion, remaining nodes become **leaves** (evaluated, not expanded) so
  time and memory stay bounded.
- Legality mirrors `EngineGameState::legal_mask` exactly. **Vocab order is never reordered or
  filtered** — masking only, per the repo invariant.
- Test seam: the node list and per-node legal masks.

### `VectorCfr`

- **CFR+ over range vectors** on the public tree (regret-matching+ with linear averaging). No
  abstraction inside the subgame — this is exactly where search buys strength back from the buckets.
- Input: a built `Subgame`, entry ranges, and a leaf-value function.
- Output: average strategy per (node, player, combo, action). Runs until the caller's **deadline**
  (checking `steady_clock` between iterations) or an iteration cap, and returns whatever it has — a
  partially solved subgame is still a usable answer.
- **Correctness gate:** must reproduce the closed-form Kuhn equilibrium (existing `kuhn.h` /
  `kuhn_exact_value` infrastructure) before it is allowed near hold'em; a Leduc-style toy is added if
  Kuhn is too weak a discriminator.

### `LeafEval`

- End-of-street leaves (flop→turn, turn→river) are valued by **blueprint continuation**: both players
  follow the blueprint for the rest of the hand, board runouts sampled with the shared evaluator, and
  results returned per-combo. Deterministic seeding; cacheable per (leaf, street).
- River nodes are terminal and settled exactly by `HandState::payoffs()`.
- This interface is precisely what the parent spec's Phase 3 replaces with a value net.

### `SearchStrategy`

- Orchestrates: **tracker → subgame → leaf values → CFR+ until deadline → root average strategy →
  action for our exact hand**.
- Owns `SearchConfig` and reads it from a `[search]` TOML section (`enabled`, `budget_ms`,
  `max_nodes`, `max_iters`, `max_rollouts`, `seed`), following the existing config precedence:
  CLI/env > TOML > built-in default (the `resolve_config` pattern). Unknown keys are rejected.
- **Deadline is enforced between solver iterations**; on timeout, exception, node-cap abort with no
  usable result, or an empty root strategy it returns "no answer".
- Bindings: `SearchConfig` and `SearchStrategy` with a `decide(initial, hero, actors, actions, rng)`
  mirroring `LbrEvaluator::decide`, plus a debug seam returning the root action distribution (for
  tests and A/B).

## Data flow at a decision

`HandState` + observed action history → `RangeTracker` → root range vectors → `Subgame` → `LeafEval`
→ `VectorCfr` (until deadline) → root average strategy for our exact hand → `[actions.search]` index →
`SixmaxAgent` translates to `target_bb` → live action.

## Error handling

- Search **never throws into the agent**: every failure path returns "no answer".
- Fallback is the existing `blueprint_probs` sample — the same code path used today, so a fallback
  decision is byte-identical to the current bot.
- Telemetry counters: decisions searched, fallbacks (and why), mean/max solver ms, mean iterations,
  node counts. Emitted through the existing logging, not a new framework.

## Integration

- `agents/sixmax_agent.py`: a `SearchStrategy` wrapper beside `SixmaxDeployStrategy`; `SixmaxAgent`
  consults search only when enabled and the street qualifies, else the blueprint — one branch, not a
  parallel agent.
- Web service: `POKERBOT_SEARCH` env (default off). **The service is a single uvicorn worker with a
  thread-free, replay-based turn driver**, so the fallback path must be genuinely exercised in tests,
  not assumed.
- CLI/openpoker: same flag, same default.

## Testing

- **Solver correctness (hard gate):** Kuhn closed-form equilibrium reproduction; a Leduc-style toy if
  needed. Mirrors the parent spec's "must reproduce closed-form equilibria before touching holdem".
- **RangeTracker:** Bayes normalization, exact card removal, and closed-form toy posteriors.
- **Subgame:** tree invariants; legal masks match `EngineGameState`; node-cap behaviour degrades to
  leaves rather than throwing.
- **LeafEval:** determinism under a fixed seed.
- **SearchStrategy:** deadline respected; every failure path falls back; fallback output equals the
  blueprint distribution.
- **Latency:** p95 per-decision time under the budget on the dev machine (asserted in tests at a
  reduced budget so CI is fast).
- **A/B (deployment gate):** same blueprint, **search on vs off**, paired seeds at ≥ 20,000 hands per
  seed (the sample size that reversed our earlier HU-blueprint conclusion), reporting the paired mean,
  sd, and t-statistic.

## Validation and deployment gate

Search is enabled in production **only if** the HU A/B shows a positive paired mean AND
`|t| ≥ 2` (paired t-test over seeds, ≥ 20,000 hands per seed, same blueprint, same seeds for both
arms). A null or negative result means it ships disabled behind the flag and we keep the blueprint — a
null result is an acceptable outcome, not a failure. Before the A/B, run the existing
`scripts/eval_sixmax_lbr.py` to bound how much room the deployed blueprint leaves at all: a nearly
unexploitable blueprint caps what search can win.

## Risks

| Risk | Mitigation |
|---|---|
| Unsafe re-solving is genuinely exploitable | Accepted (parent spec); A/B gated; flag-off by default |
| Blueprint-continuation leaves bias the solve | Explicit interface for the Phase-3 value net to replace |
| Vector CFR+ subtly wrong | Kuhn/Leduc exactness gate before hold'em |
| Latency on the single-worker web service | Hard deadline + guaranteed fallback; latency test |
| An un-bucketed solve is still not better | A/B gate; keep the blueprint if not |

## Phasing within v1

| Step | Deliverable |
|---|---|
| 1 | `VectorCfr` + `Subgame` on Kuhn/Leduc; exactness tests green |
| 2 | `RangeTracker` + `LeafEval`; unit invariants green |
| 3 | HU flop/turn/river wiring + bindings + `SearchStrategy` deadline/fallback |
| 4 | A/B harness + latency; LBR pre-check |
| 5 | Conditional: enable in the web service on positive lift (else stay disabled) |

## Phasing summary (relative to the parent spec)

| Phase | Deliverable | Status |
|---|---|---|
| 0 | De-quirked foundations | done |
| 1 | Engine + MCCFR blueprint (+ LBR exploitability) | done |
| **2 (this spec)** | **Belief tracking + depth-limited search, HU-first** | **in progress** |
| 2.5 | 6-max enablement of the same unit | future |
| 3 | Value-net leaves (batched ReBeL loop) | future |
