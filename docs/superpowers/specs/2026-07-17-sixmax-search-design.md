# Six-Max Blueprint + Search System — Design

**Date:** 2026-07-17
**Status:** Approved (brainstorming complete)
**Approach:** "A" — Pluribus-pattern blueprint + depth-limited search, with ReBeL-flavored belief tracking and a batched-ReBeL value-net loop.

## Motivation

The deployed HU Deep CFR bot (`neural_cfr/`, +36.4 BB/100 vs the tabular baseline heads-up) loses at 6-max openpoker tables against fields that punish its heads-up-shaped strategy. Season data attributes roughly two-thirds of the loss to the game mismatch (HU ranges played multiway via a pseudo-heads-up bridge) and the rest to network strength and non-adaptive equilibrium play. This program replaces the bridge hack with a system trained and searched on the actual game.

## Goals and constraints (from scoping)

- **Goal:** strongest 6-max bot for the openpoker leaderboard. Not a faithful ReBeL reproduction; borrow from ReBeL and Pluribus per component.
- **Timeline:** 1–3 months, long-term only. No season patches; the HU pipeline stays frozen as a baseline and eval opponent.
- **Compute:** M-series Mac for development and play; ~$100–500 total cloud (CPU batches for blueprint/subgame solving, short GPU rentals for net training).
- **Stack:** extend the C++ subsystem (libtorch + pybind11 + Buck2). Python only for tooling and bridges, as today.
- **Adaptation:** equilibrium first. Interfaces must leave room for an exploitation layer to bias search later; no opponent modeling in scope now.

## Non-goals

- Faithful public-belief-state ReBeL (untested at 6-max; belief space over 5 opponents exceeds budget).
- Safe re-solving (not well-defined multiway; we ship unsafe re-solving knowingly).
- Exploitability measurement (intractable at 6-max; relative strength is the only metric).
- Opponent exploitation/adaptation (deferred; seams only).
- Any change to `neural_cfr/` training or its checkpoints.

## Phase 0 — De-quirking (before any sixmax code)

1. **Safe evaluator API.** Extract the 7-card evaluator into a shared Buck2 target (`//common:evaluator`), byte-identical internals (the `12 - rank` inversion is preserved — trained checkpoints depend on it). New consumers get only `beats(a, b)` / normalized-strength functions; raw inverted scores are not exposed outside `neural_cfr/`.
2. **BB-relative chips everywhere in sixmax.** The engine and all new features take chips denominated in big blinds; no absolute-chip frame, no caller-side rescaling contract. Fix the live `scale = buy_in / 100` line in `scripts/openpoker_bot.py` to derive from `big_blind`.
3. **C++ owns card abstraction.** Bucketing for sixmax lives in C++, exposed to Python via bindings. The tabular pipeline's Python abstraction stays as-is (frozen benchmark); no parallel implementations going forward.
4. **Real position encoding.** Seat-relative-to-button (0–5) natively; the HU bridge's binary SB/BB special-casing is isolated from the new path.
5. **Scripted, documented dev setup.** One script handles the `.venv` Python 3.10 ABI pin and `third_party/libtorch` symlinks so worktrees, clean clones, and cloud machines set up identically.
6. **Bot-runner hygiene.** Unbuffered logging; SIGTERM handler sends `leave_table` before exit; process management targets the whole process chain, not the `caffeinate` wrapper.
7. **Action vocabulary as a first-class parameter** (see next section). No hardcoded action enum anywhere in `sixmax/`.

## Action vocabulary

Config-defined, per run, self-describing in artifacts.

```toml
[actions.blueprint]
preflop_opens = [ { size = 2.5, unit = "bb" },
                  { size = 3.5, unit = "bb" },
                  { size = 5.0, unit = "bb" } ]
# Every raise after the open (3-bet+, and ALL postflop bets and raises)
# shares one pot-fraction grid. Raise-to = current_bet + size * (pot + 2*to_call).
bet_sizes     = [ { size = 0.33, unit = "pot" },
                  { size = 0.75, unit = "pot" },
                  { size = 1.5,  unit = "pot" } ]   # includes overbet
include_allin = true   # legal jam always in the action set, incl. preflop

[actions.search]       # may differ from blueprint; v1 starts identical.
bet_sizes     = [ { size = 0.33, unit = "pot" },   # enriching this later is a
                  { size = 0.75, unit = "pot" },   # config edit + A/B, not an
                  { size = 1.5,  unit = "pot" } ]  # architecture change
include_allin = true   # a search grid without the jam cannot express stack-offs
```

- **C++ descriptor:** `ActionVocab` built once from config; immutable per run; canonical order = config order; strategy/regret vectors sized `vocab.size()`. Betting history is encoded as vocab indices.
- **Legality by masking**, never reordering or filtering storage (generalizes the proven HU invariant).
- **Artifact contract:** every checkpoint embeds the vocab descriptor + hash; loaders refuse mismatches. Any parameter that changes the meaning of stored data travels inside the artifact.
- **One translation layer:** `to_chips()` / `nearest()` in a single C++ unit used by blueprint training, search, and the openpoker bridge. `nearest()` uses randomized pseudo-harmonic mapping.
- Grid density and search intensity are substitutes (DeepStack sparse-grid/heavy-search vs Pluribus rich-grid/light-search). Phase 1 ships blueprint-only, so the v1 grid carries full load — hence the overbet and three preflop opens now; search later relaxes grid pressure.

## Architecture

```
common/            # shared C++: evaluator (extracted, wrapped), cards
sixmax/            # new C++ subsystem (Buck2 target + pybind11 module)
  engine/          # 2–6 player NLHE, side pots, BB-denominated, 100BB frame
  vocab/           # ActionVocab, translation
  abstraction/     # 169 preflop classes; equity-percentile postflop buckets
  blueprint/       # external-sampling MCCFR, linear weighting, sparse tables
  search/          # RangeTracker, subgame builder, vectorized CFR+ solver
  value_net/       # phase 3: leaf value net (libtorch)
agents/, scripts/  # bridging only, as today
```

`cfr/`, `neural_cfr/`, and `sixmax/` never import each other; `common/` is the only shared C++ code. Player count (2–6) is a runtime parameter: HU mode for validation, short-handed for realism, one engine for blueprint and search.

## Phase 1 — Engine + MCCFR blueprint

- **Engine:** 6-player NLHE in C++; seat rotation, blinds, side pots (ported behavior validated against the Python engine's side-pot tests); canonical `starting_stack=100, big_blind=1` frame.
- **Card abstraction:** lossless 169 preflop; postflop equity-percentile buckets via Monte Carlo rollouts with the shared evaluator (initial sizes ~50 flop / 50 turn / 20 river — final numbers set after measuring infoset counts and memory).
- **History abstraction:** per-street raise count capped at 3, plus pot-size bucket (initially 4 buckets; revisited with infoset-count measurements).
- **Trainer:** external-sampling MCCFR, linear weighting (global iteration counter pattern carried over), sparse hashmap regret/strategy tables, multithreaded. Config TOML, checkpointing, and best-checkpoint selection reuse the validated `neural_cfr` patterns.
- **Compute:** development runs on the Mac; final blueprint as a cloud CPU batch (~64 cores × days, ~$100–200).
- **Deliverable:** a deployable 6-max bot (blueprint-only) that fixes the multiway preflop leak; deployed via the hardened `openpoker_bot.py` with a new strategy loader.

## Phase 2 — Depth-limited search

- **When:** blueprint plays preflop on-tree; search runs from the flop onward and preflop on off-tree opponent bets (nested re-solving from day one — subgame trees include observed exact bet sizes rather than rounding).
- **RangeTracker:** per-opponent distribution over 1,326 combos; Bayes updates after each observed action using the blueprint as opponent model; exact card-removal on board cards. Input to subgames now, to the value net in Phase 3.
- **Subgame:** rooted at start of current street with tracked ranges as entry beliefs (unsafe re-solving, acknowledged); our actions from `[actions.search]`, opponents' at observed sizes; solved by vectorized CFR+ over range-vectors. Expected tree sizes 10³–10⁵ nodes; seconds on 8 cores.
- **Leaves:** end-of-street depth limit; leaf values via Pluribus's k=4 continuation strategies (blueprint, fold-biased, call-biased, raise-biased), estimated by evaluator rollouts. This interface is exactly what Phase 3 replaces.
- **Budget & fallback:** configurable per-decision time budget (default ~5s; tuned to openpoker's turn clock); node cap for pathological multiway trees; any failure or timeout falls back to the blueprint action. Search can never make the bot worse than Phase 1.
- **Validation before deployment:** same blueprint, search on vs off, A/B in the 6-max harness and HU mode. Deploy only on a clearly positive BB/100 lift.

## Phase 3 — Value net (batched ReBeL loop)

Search enters the training loop — the ReBeL component of the design:

1. Phase-2 self-play on the Mac logs every search invocation's (public state, tracked ranges).
2. A cloud CPU batch re-solves the logged subgames deeper (to or near showdown) for high-quality value targets.
3. A rented GPU trains the value net (hours); the net replaces rollout leaf evaluation.
4. Better leaves → better search → better logged targets: repeat per budget. Each generation is a discrete, A/B-testable checkpoint (batched rather than online bootstrapping).

- **Net:** input = public state (board, pot, stacks, street) + per-player range *summaries* (compressed statistics of the 1,326-vectors — equity buckets, range-vs-range equity moments), not raw beliefs. Output = per-player counterfactual values at the subgame root. Modest MLP on the existing libtorch stack. Checkpoints self-describe vocab hash and encoding version.
- The range-summary compression is the deliberate concession that makes 6-max belief inputs tractable where faithful PBS would not be.

## Evaluation program

In order of authority:

1. **In-house A/B:** every change measured as BB/100 vs the previous best configuration in the 6-max harness (blueprint-vs-blueprint, search on/off, net-vs-rollout leaves); best-checkpoint selection promotes winners.
2. **HU-mode sanity:** the sixmax system in 2-player mode vs the frozen tabular and neural bots; regression alarm if 6-max "improvements" lose ground heads-up.
3. **Openpoker:** live final exam; season API score trend as the field metric.

## Testing

- **Unit:** side-pot payouts property-tested against the Python engine on identical deals; RangeTracker Bayes correctness (posteriors normalize, card removal exact, closed-form toy cases); vocab translation round-trips; subgame-builder tree invariants.
- **Integration:** toy games with known CFR solutions (Kuhn/Leduc-style) — blueprint trainer and subgame solver must reproduce closed-form equilibria before touching full holdem.
- **Ops:** the Phase 0 runner hygiene (graceful leave, unbuffered logs, whole-chain process management) verified before any long deployment.

## Phasing summary

| Phase | Deliverable | Ships alone? |
|---|---|---|
| 0 | De-quirked foundations (evaluator wrap, vocab system, setup script, runner hygiene) | n/a |
| 1 | 6-max engine + MCCFR blueprint bot | Yes — deployable |
| 2 | Belief tracking + depth-limited search + nested re-solving | Yes — A/B-gated |
| 3 | Value-net leaves via batched ReBeL self-play loop | Yes — per generation |
