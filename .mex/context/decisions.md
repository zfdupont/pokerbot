---
name: decisions
description: Key architectural and technical decisions with reasoning. Load when making design choices or understanding why something is built a certain way.
triggers:
  - "why do we"
  - "why is it"
  - "decision"
  - "alternative"
  - "we chose"
edges:
  - target: context/architecture.md
    condition: when a decision relates to system structure
  - target: context/stack.md
    condition: when a decision relates to technology choice
  - target: context/neural-cfr.md
    condition: when a decision concerns the C++ Deep CFR subsystem
  - target: context/cfr-training.md
    condition: when a decision concerns the tabular pipeline or its metrics
last_updated: 2026-07-18
---

# Decisions

## Decision Log

### Within-abstraction exploitability as the convergence metric
**Date:** 2026-07-02
**Status:** Active
**Decision:** Track convergence with `compute_exploitability()` in `cfr/mccfr.py`, which constrains the best-response player to one action per abstract infoset (mbb/h).
**Reasoning:** The exact-card best-response metric (~19,460 mbb/h) is dominated by abstraction loss and never improves meaningfully with training; the within-abstraction number (574 mbb/h at checkpoint_09040000) actually measures learning.
**Alternatives considered:** Exact-card BR (rejected — measures the abstraction, not the strategy); head-to-head win rate only (kept as a secondary signal via OpenSpiel, but too noisy as a primary metric).
**Consequences:** Reported exploitability numbers are only comparable within the same abstraction. Changing bucketing invalidates historical numbers.

### Neural CFR implemented in C++ (libtorch + pybind11 + Buck2), not Python
**Date:** 2026-07-02
**Status:** Active
**Decision:** Deep CFR (Brown et al. 2019) lives in `neural_cfr/` as a C++ extension built by Buck2 with vendored libtorch, exposed as `import neural_cfr`.
**Reasoning:** External-sampling traversal is the bottleneck; C++ with a thread pool (thread-safe RNG, buffer mutex) gives orders-of-magnitude more traversals than Python.
**Alternatives considered:** PyTorch-Python training loop (rejected — traversal throughput too low); extending the tabular `cfr/` package (rejected — abstraction quality ceiling).
**Consequences:** Two build systems in one repo (uv + Buck2). Python only launches (`scripts/train_neural.py`) and consumes (`neural_cfr.Strategy`) the extension.

### Kicker bits are inverted (`12 - rank`) in the C++ evaluator
**Date:** 2026-07-02
**Status:** Active — critical invariant
**Decision:** `evaluate_5card` in `neural_cfr/src/game/card.cpp` encodes `category << 20 | kicker_bits` with kickers stored as `12 - rank`, so lower = better holds within categories.
**Reasoning:** Without the inversion, Ace kickers rank worse than 2s within a category (AA loses to 22), which inverts learned hand strength — this was a confirmed training bug (commit `1c606c5`): the bot folded AA and raised 22.
**Alternatives considered:** higher = better encoding throughout (rejected — would require touching every comparison site and the 21-combo `min` in `evaluate_7card`).
**Consequences:** Any change to the encoding must preserve the inversion. Verify with the C++ evaluator tests before trusting any training run. The 2026-10-22 rewrite (direct best-hand evaluation instead of the 21-combo `min`) kept every score bit-identical, checked exhaustively over all 7-card hands; `//common/tests:test_evaluator` re-checks every 5-card hand and 300k random 7-card hands against the original code.

### Faithful Brown et al. 2019 traversal (M_π only at opponent nodes)
**Date:** 2026-07-02
**Status:** Active
**Decision:** At traverser nodes: regret-match the traverser's advantage net, traverse all actions, store advantages in `M_v[p]` only. At opponent nodes: regret-match the opponent's net, store σ in `M_π`, sample one action. The strategy net is trained offline on `M_π` and never queried during traversal.
**Reasoning:** Earlier code stored strategy samples at traverser nodes and used the wrong net at opponent nodes (fixed in `580e5b7`, `d7532ff`); deviating from the paper corrupted the average-strategy target.
**Alternatives considered:** Single shared advantage net (rejected — paper uses per-player nets `adv0_`/`adv1_`).
**Consequences:** Changes to `neural_cfr/src/cfr/` traversal must be checked against the paper, not intuition. ε-greedy exploration at opponent nodes (ε=0.06, commit `d137bda`) is the one sanctioned deviation.

### Two (three) independent hand evaluators, kept separate on purpose
**Date:** 2026-06-25
**Status:** Active
**Decision:** Live showdowns use `models/hand.py` via `util/evaluator.py`; CFR equity bucketing uses the fast bitwise/prime-product `util/util.py:hand_value`; `neural_cfr` has its own C++ evaluator.
**Reasoning:** The live path needs exact, readable showdown resolution; the equity path needs raw speed for Monte Carlo rollouts and tolerates the coarse `_fallback_hand_value()` (no kicker discrimination) for 7 cards.
**Alternatives considered:** One shared evaluator (rejected — the fast path's fallback is not precise enough for showdowns; unifying would couple sealed subsystems).
**Consequences:** A bug fix in one evaluator does not fix the others. Precision-sensitive changes must state which path they target.

### Hand evaluator extracted to `common/`; `safe_eval` API hides inverted scores
**Date:** 2026-07-18
**Status:** Active
**Decision:** The 7-card hand evaluator (formerly only in `neural_cfr/src/game/`) was moved byte-identical to `common/src/game/` as Buck2 `//common:evaluator`. `sixmax` (and any future subsystem) depends on `//common:evaluator`, never on `//neural_cfr`. The opaque `safe_eval::HandRank` type exposes only `beats(other)` and `ties(other)` — raw inverted scores never cross the `common/` boundary.
**Reasoning:** Copying the evaluator into each subsystem would create divergence; importing across subsystem boundaries would break the no-cross-import rule. `common/` is the only sanctioned sharing point. The opaque type prevents callers from accidentally comparing inverted scores directly (the bug that caused the `12 - rank` inversion fix in the first place).
**Alternatives considered:** Duplicating into `sixmax/src/game/` (rejected — drift risk); exposing raw score as `int` (rejected — callers would inevitably write `a.score < b.score`, which is semantically wrong without knowing the inversion).
**Consequences:** Any change to the evaluator logic lives in `common/src/game/card.cpp` and is picked up by both `neural_cfr` and `sixmax` on the next `buck2 build`. The `12 - rank` inversion invariant is now documented in `safe_eval.h` and must be preserved there.

### Neural checkpoints use named sub-archives; buffers are not serialized
**Date:** 2026-07-02
**Status:** Active
**Decision:** Checkpoints are written as `root.write("adv0", ...)`, `("adv1", ...)`, `("strat", ...)` — not a flat archive. Reservoir buffers are never saved.
**Reasoning:** `neural_cfr.Strategy` needs to load only `"strat"` for inference; buffers (up to 2M entries each) would bloat checkpoints for little resume value.
**Alternatives considered:** Flat archive (rejected — Strategy would drag in both advantage nets); serializing buffers (rejected — size).
**Consequences:** `--resume` restores nets only — the first `train_interval` iterations after resume refill buffers from scratch. Loaders that expect a flat `.pt` will fail.


---
**Decision:** Corrected stale test count in ROUTER.md (2026-07-18)
**Context:** ROUTER.md said ~116 tests; CLAUDE.md said ~119 tests. Verified with `uv run pytest tests/ --collect-only -q`: **119 tests collected**.
**Consequences:** ROUTER.md line 39 updated to `~119-test`. CLAUDE.md was already correct.

---
**Decision:** Corrected stale test count in stack.md (2026-07-18)
**Context:** stack.md still said `~116 tests` after the ROUTER.md fix. Sweep found it; corrected to `~119 tests`.
**Consequences:** `.mex/context/stack.md` now matches ROUTER.md and CLAUDE.md. All three scaffold locations agree: 119 tests.

## Action vocabulary is config-defined in sixmax (2026-07-18)

The fixed 6-action vocabulary is scoped to the heads-up pipelines. `sixmax/` treats the action set as a first-class run parameter (TOML `[actions.*]`, BB/pot units, canonical order = config order) because vocabulary changes invalidate all trained artifacts — so artifacts embed a vocab hash and loaders refuse mismatches. Growing the grid later is config + retrain, not a rewrite.

## Boundary rescale rule corrected to big-blind division (2026-07-18)

The documented `their_stack/100` rescale was only correct at exactly 100BB and misplayed at other buy-ins; the rule is: divide every chip input by the table's big blind (training frame has bb=1). Fixed in `scripts/openpoker_bot.py`; scaffold references corrected in conventions.md, neural-cfr.md, patterns/eval-checkpoint.md.

## Missing-infoset fallback is play-time only: nearest pot bucket, then safe (2026-10-23)

A deployed blueprint misses its table when an opponent's off-tree bet size lands a pot bucket training never reached (C2 vs Slumbot: 41% of preflop decisions, the BB facing a 2 BB min-open; fine pot edges sit below our smallest trained pots). The miss used to play uniformly over the legal mask, CFR's own default for an unvisited infoset, which at 200 BB means a 1-in-6 jam: 1,177 of C2's 1,262 Slumbot all-ins. `SixmaxDeployStrategy(fallback="nearest")` (default) now retries the key with pot buckets up to 2 away (lower first) and, failing that, checks / calls at most half the pot / folds, never bets; `fallback="uniform"` keeps the old behaviour; `deploy.lookups` counts hit/nearest/safe/uniform. Only the deploy/eval bridge changed: the trainer never uses this class (training exploration is unchanged), and the C++ `exploit/policy` fallback that LBR's opponent model uses is left uniform (LBR plays our own sizes, 0% misses).

## Bet history in the key: last raise class per street, deploy keys from a hand replay (2026-10-23)

The key could not tell bet sizes apart: after a 2.5/3.5/5 BB open the pot is 3.5/4.5/6 BB, one pot bucket with the same raise count, so the big blind answered all three opens with one strategy (C4 folded 58/56/56%). Published bots key on the full action sequence (Pluribus: 664,845,654 sequences); C5's opt-in field is the cheap partial version: the last raise's size class on each street (every street, owner's call, so the open size stays visible postflop). Trainer keys read the class from the vocab index applied; the deployed `SixmaxAgent` replays the live hand into an `EngineGameState` (`_replay_to_now`, mapping each live action to the closest legal menu entry in the state it faced via pseudo-harmonic `nearest`) and reads the same classes, so trainer and deploy share one key path; an unreplayable line plays the safe action and counts `lookups["desync"]`. The Slumbot stand-in state carries `action_history` / `total_contributed` for the replay; `openpoker_bot.py` and `diagnose_key_aliasing.py` refuse bet-history checkpoints (no hand history / no classes). The closest-legal `_action_to_vocab` also changes the disabled search path's translation for legacy checkpoints. Found while building it and fixed 2026-10-23 (`engine_bet_context` in agents/sixmax_agent.py): the deploy `BetContext` used the gross live pot where the engine uses pot minus the amount owed, so every deployed pot-fraction raise facing a bet had been `fraction * owe` larger than trained (all arms C0-C4, every Slumbot and head-to-head run before the fix; e.g. a 0.33-pot raise facing a 3.5 BB open targeted 6.635 BB instead of 5.81). All five deploy BetContext sites (blueprint, DREAM, search, both legal masks) now use it; keys still use the gross pot, as the engine does.

## Buck headers are exported so header edits rebuild dependants (2026-10-23)

`sixmax/BUCK` declared private `headers`, so a header that changed a type's layout did not rebuild bindings or the DREAM targets, giving heap corruption until `buck2 clean`. The sixmax targets now use `exported_headers` (header_namespace ""); `common/BUCK` and `neural_cfr/BUCK` still use private headers and have the same hazard: after changing a layout header there, run `~/bin/buck2 clean`.

