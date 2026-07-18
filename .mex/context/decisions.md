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
**Consequences:** Any change to the encoding must preserve the inversion. Verify with the C++ evaluator tests before trusting any training run.

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
