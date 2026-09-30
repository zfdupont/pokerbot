---
name: architecture
description: How the major pieces of this project connect and flow. Load when working on system design, integrations, or understanding how components interact.
triggers:
  - "architecture"
  - "system design"
  - "how does X connect to Y"
  - "integration"
  - "flow"
edges:
  - target: context/stack.md
    condition: when specific technology details are needed
  - target: context/decisions.md
    condition: when understanding why the architecture is structured this way
  - target: context/cfr-training.md
    condition: when working on the tabular MCCFR pipeline, abstraction, or exploitability
  - target: context/neural-cfr.md
    condition: when working on the C++ Deep CFR subsystem (neural_cfr/)
last_updated: 2026-09-30
---

# Architecture

## System Overview

Four loosely coupled subsystems never import each other's internals (the two heads-up CFR pipelines share one fixed action vocabulary; `sixmax/` defines its own via config):

1. **Live game engine** — `game/poker.py:PokerGame` owns the hand lifecycle (deal, blinds, betting rounds, showdown, button rotation). It holds a `models/state.py:GameState`, delegates side-pot math to `game/pot_manager.py:PotManager`, and calls `Player.make_decision(state)`, which delegates to the injected agent's `get_action(player, state) -> (Action, amount)`.
2. **Tabular CFR training** (`cfr/`) — self-contained External Sampling MCCFR over an abstracted heads-up game (`cfr/abstract_state.py`). Produces pickle checkpoints in `cfr/checkpoints/` that `agents/cfr_agent.py` loads to translate abstract actions into concrete `(Action, amount)` for live play.
3. **Neural CFR** (`neural_cfr/`) — C++ Deep CFR (Brown et al. 2019) built with libtorch + pybind11 via Buck2, imported as `import neural_cfr`. Produces `.pt` checkpoints; `neural_cfr.Strategy` serves inference.

Trained strategies flow outward to consumers: `scripts/play.py` (interactive CLI), `scripts/eval_openspiel.py` / `eval_openspiel_neural.py` (independent OpenSpiel evaluation), `scripts/eval_hu_sanity.py` (duplicate-deal HU-mode eval of the six-max blueprint vs the frozen tabular/neural bots in the live engine), `scripts/eval_sixmax_baseline.py` (6-max BB/100 curve vs PotOdds), `scripts/eval_mixed_table.py` (all four trained agents at one 6-handed table → per-seat BB/100 + chip matrix), and `scripts/openpoker_bot.py` (WebSocket deployment to openpoker.ai, auto-detecting `.pt` vs `.pkl` vs `.bin` → six-max blueprint).

## Key Components

- **`game/poker.py:PokerGame`** — full hand orchestration; only place showdown winners are determined (via `util/evaluator.py:HandEvaluator`).
- **`agents/base_agent.py:PokerAgent`** — the agent interface. All decision policies subclass it; agents never touch `GameState` mutation, only read it.
- **`cfr/` package** — tabular MCCFR pipeline (abstraction, `InfoSet`, `RegretTable`, trainer). Never imports `game/poker.py`; has its own `AbstractState`.
- **`neural_cfr/` package** — C++ Deep CFR: `src/game/` (state + hand evaluator), `src/net/` (features + MLP), `src/cfr/` (traversal + trainer), `src/bindings/` (pybind11). Never imports `game/` or `cfr/`.
- **`agents/cfr_agent.py:CFRAgent`**, **`agents/sixmax_agent.py:{SixmaxDeployStrategy,SixmaxAgent}`**, **`agents/neural_agent.py:NeuralAgent`** — the sanctioned cross-subsystem bridges from a trained strategy to live `(Action, amount)` play. `SixmaxDeployStrategy` is host-agnostic (BB units in, `(vocab_index, raise_to_bb)` out); `SixmaxAgent`/`NeuralAgent`/`HandTracker` are its hosts, each owning legality + chip rescale.
- **`scripts/openpoker_bot.py`** — WebSocket connector; `HandTracker` (tabular/neural) and `SixmaxHandTracker` (six-max blueprint) accumulate hole/community cards, per-seat fold/all-in, and `raises_per_street` across messages, then query the strategy on `your_turn`.
- **Two hand evaluators** — `models/hand.py` + `util/evaluator.py` for live showdowns; `util/util.py:hand_value` (bitwise/prime-product, fast) exclusively for CFR equity bucketing. A third, independent C++ evaluator lives in `common/src/game/card.cpp` (Buck2 `//common:evaluator`; consumed by `neural_cfr` and, via the opaque `safe_eval` API, by `sixmax`).

## Module Map

- `models/` — `card.py`, `enums.py` (`Action`, `Position`, `Suit`, `HandRank`), `player.py`, `state.py`, `hand.py`
- `agents/` — `base_agent.py` interface; `simple_agent.py`, `position_agent.py`, `hand_strength_agent.py`, `cfr_agent.py`
- `game/` — `poker.py` (orchestration), `pot_manager.py` (side pots)
- `cfr/` — tabular MCCFR pipeline (see `context/cfr-training.md`)
- `neural_cfr/` — C++ Deep CFR (see `context/neural-cfr.md`)
- `common/` — shared C++ (Buck2 `//common:evaluator`): the 7-card evaluator (moved byte-identical from `neural_cfr`, `12 - rank` inversion intact) plus the opaque `safe_eval::HandRank` API (`beats`/`ties` only — raw inverted scores never leave `common/`)
- `sixmax/` — six-max blueprint + search subsystem (spec: `docs/superpowers/specs/2026-07-17-sixmax-search-design.md`). Never imports `cfr/` or `neural_cfr/`; shares C++ only via `common/`. Layout after Phases 1a+1b (merged 2026-07-19):
  - `src/vocab/` — config-defined `ActionVocab` (BB/pot units, pseudo-harmonic translation, FNV-1a hash); TOML loader in `sixmax/vocab_config.py`
  - `src/abstraction/` — lossless 169-class preflop index; deterministic MC `hand_equity` (RNG seeded from sorted hole+board+salt, so bucket assignments are stable with no cache or shared state); `Abstraction` = per-street equity-percentile quantile edges (50/50/20 default), bucket lookups MUST reuse the edge-sampling salt; `abstract_key.{h,cpp}` — the single source of truth for the infoset-key bit layout (`pack_abstract_key` + `pot_bucket`), used by both `EngineGameState::abstract_key` (trainer) and the Python deployment bridge (via bindings) so keys can never drift
  - `src/blueprint/` — `game.h` abstract `GameState`/`Game` solver interface; `kuhn.{h,cpp}` frozen Kuhn validation fixture; `mccfr.{h,cpp}` single-threaded reference trainer + shared `regret_matched`/`kuhn_exact_value_lookup`; `trainer.{h,cpp}` multithreaded `BlueprintTrainer` (GameFactory-based so the Kuhn −1/18 gate covers the concurrent path; 64-shard mutexed table; atomic global linear-CFR counter; per-thread Game instances — `EngineGame::new_hand` mutates `button_`); `checkpoint.{h,cpp}` binary artifacts (SIXBP001, embeds vocab hash + abstraction config+edges, atomic tmp+rename, loaders refuse hash mismatches) + read-only `BlueprintStrategy`; `engine_game.{h,cpp}` vocab-masked bridge — owns all action legality; `infoset_key()` = bit-packed abstraction key (card/street/raises≤3/pot-bucket/live_opps/after — table-size-agnostic) when an `Abstraction` is attached, naive exact hash otherwise
  - `src/engine/engine.{h,cpp}` — 2–6 player NLHE engine in the BB chip frame (doubles, SB 0.5, BB 1.0, stack 100, eps 1e-9); contribution-level side-pot settlement (`settle_pots`) property-tested against the Python `PotManager` oracle; the engine trusts callers on action legality (debug assert only — legality lives in the bridge mask)
  - `src/bindings/bindings.cpp` — pybind11 surface; `tests/sixmax/conftest.py` buck2-builds and force-loads the `.so` as `sys.modules["sixmax"]` (scripts `train_sixmax.py`/`eval_sixmax.py`/`diagnose_blueprint.py` do the same force-load). Includes read-only `dump_infosets(path)` → `(iterations, [(key, probs, strategy_sum_l1, regret_l1)])` for offline checkpoint analysis (consumed by `scripts/diagnose_blueprint.py`, the blueprint plateau autopsy)
  - `src/dream/` — DREAM-style neural blueprint (Steinberger et al. 2020; branch `feature/neural-cfr`, not yet merged): `features.{h,cpp}` (154-dim infoset tensor), `nets.{h,cpp}` (libtorch MLP adv + strategy nets), `reservoir.{h,cpp}` (thread-safe weighted M_v/M_π), `trainer.{h,cpp}` (`DreamTrainer`, outcome-sampling MCCFR + IS-weighted advantage targets), `checkpoint.{h,cpp}` (SIXDM001 magic + `DreamStrategy` inference). Generalizes over variable stacks (20–250 BB) and player counts (2–6) — the intended replacement for the tabular blueprint, which is locked to `starting_stack=100, num_players=6`. `guidelines`: `docs/superpowers/specs/2026-07-24-dream-blueprint-design.md`, ledger `.superpowers/sdd/dream-progress.md`
- `util/` — `evaluator.py` (live showdown), `util.py` + `lookup_table.py` (fast evaluator for CFR equity)
- `scripts/` — all runnable entry points (training, play, eval, deploy)
- `web/` — heads-up **web service** (a host, not a training subsystem): FastAPI + REST, a replay-based turn driver over `game/poker.py`, an in-memory TTL session store, and `SixmaxAgent` loaded once and shared per session. Imports `game/`/`agents/`; never imported by `cfr/`, `neural_cfr/`, or `sixmax/`. Design: `docs/superpowers/specs/2026-09-30-poker-web-design.md`.
- `tests/` — mirrors source layout; conftests make tests fast (see `context/conventions.md`)

## External Dependencies

- **OpenSpiel (`universal_poker`)** — Google DeepMind's game engine used as an independent referee for head-to-head evaluation (FCPA abstraction, 100BB HU NL). Not a runtime dependency of training or live play.
- **openpoker.ai** — deployment target; `wss://openpoker.ai/ws` with `Authorization: Bearer $OPENPOKER_API_KEY`. Raise amounts use the **raise-to** convention (total committed, not increment).
- **libtorch** (vendored under `third_party/`) — C++ tensor/NN library backing `neural_cfr`; built by Buck2 (`~/bin/buck2`), not pip.
- **pokerhand-eval (Alvin Liang)** — origin of the fast lookup-table evaluator in `util/util.py` + `util/lookup_table.py`.

## What Does NOT Exist Here

- No multiplayer (>2) CFR training — both CFR pipelines are strictly heads-up; only the live engine supports N-player tables.
- No shared state between the three subsystems — `cfr/` and `neural_cfr/` deliberately do not import `game/poker.py`; each has its own game-state implementation.
- No web **UI**, and no HTTP server beyond the new `web/` API host (which has no browser UI — that lives in the separate `blogfolio` site). Interaction is otherwise CLI scripts and the outbound WebSocket connector.
- No GPU training assumptions — training runs on CPU (threaded traversal in C++); no CUDA configuration exists.
