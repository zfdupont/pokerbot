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
last_updated: 2026-07-07
---

# Architecture

## System Overview

Three loosely coupled subsystems share one action vocabulary but never import each other's internals:

1. **Live game engine** — `game/poker.py:PokerGame` owns the hand lifecycle (deal, blinds, betting rounds, showdown, button rotation). It holds a `models/state.py:GameState`, delegates side-pot math to `game/pot_manager.py:PotManager`, and calls `Player.make_decision(state)`, which delegates to the injected agent's `get_action(player, state) -> (Action, amount)`.
2. **Tabular CFR training** (`cfr/`) — self-contained External Sampling MCCFR over an abstracted heads-up game (`cfr/abstract_state.py`). Produces pickle checkpoints in `cfr/checkpoints/` that `agents/cfr_agent.py` loads to translate abstract actions into concrete `(Action, amount)` for live play.
3. **Neural CFR** (`neural_cfr/`) — C++ Deep CFR (Brown et al. 2019) built with libtorch + pybind11 via Buck2, imported as `import neural_cfr`. Produces `.pt` checkpoints; `neural_cfr.Strategy` serves inference.

Trained strategies flow outward to three consumers: `scripts/play.py` (interactive CLI), `scripts/eval_openspiel.py` / `eval_openspiel_neural.py` (independent OpenSpiel evaluation), and `scripts/openpoker_bot.py` (WebSocket deployment to openpoker.ai, auto-detecting `.pt` vs `.pkl`).

## Key Components

- **`game/poker.py:PokerGame`** — full hand orchestration; only place showdown winners are determined (via `util/evaluator.py:HandEvaluator`).
- **`agents/base_agent.py:PokerAgent`** — the agent interface. All decision policies subclass it; agents never touch `GameState` mutation, only read it.
- **`cfr/` package** — tabular MCCFR pipeline (abstraction, `InfoSet`, `RegretTable`, trainer). Never imports `game/poker.py`; has its own `AbstractState`.
- **`neural_cfr/` package** — C++ Deep CFR: `src/game/` (state + hand evaluator), `src/net/` (features + MLP), `src/cfr/` (traversal + trainer), `src/bindings/` (pybind11). Never imports `game/` or `cfr/`.
- **`agents/cfr_agent.py:CFRAgent`** — bridge from abstract strategy to live play; loads a `RegretTable` checkpoint and maps abstract actions to legal chip amounts.
- **`scripts/openpoker_bot.py`** — WebSocket connector; `HandTracker` accumulates hole/community cards and `raises_per_street` across messages, then queries the strategy on `your_turn`.
- **Two hand evaluators** — `models/hand.py` + `util/evaluator.py` for live showdowns; `util/util.py:hand_value` (bitwise/prime-product, fast) exclusively for CFR equity bucketing. A third, independent C++ evaluator lives in `neural_cfr/src/game/card.cpp`.

## Module Map

- `models/` — `card.py`, `enums.py` (`Action`, `Position`, `Suit`, `HandRank`), `player.py`, `state.py`, `hand.py`
- `agents/` — `base_agent.py` interface; `simple_agent.py`, `position_agent.py`, `hand_strength_agent.py`, `cfr_agent.py`
- `game/` — `poker.py` (orchestration), `pot_manager.py` (side pots)
- `cfr/` — tabular MCCFR pipeline (see `context/cfr-training.md`)
- `neural_cfr/` — C++ Deep CFR (see `context/neural-cfr.md`)
- `common/` — shared C++ (Buck2 `//common:evaluator`): the 7-card evaluator (moved byte-identical from `neural_cfr`, `12 - rank` inversion intact) plus the opaque `safe_eval::HandRank` API (`beats`/`ties` only — raw inverted scores never leave `common/`)
- `sixmax/` — six-max blueprint + search subsystem (Phase 0 so far: pybind11 module skeleton, config-defined `ActionVocab` with BB/pot units + pseudo-harmonic translation, TOML vocab loader; spec: `docs/superpowers/specs/2026-07-17-sixmax-search-design.md`). Never imports `cfr/` or `neural_cfr/`; shares C++ only via `common/`
- `util/` — `evaluator.py` (live showdown), `util.py` + `lookup_table.py` (fast evaluator for CFR equity)
- `scripts/` — all runnable entry points (training, play, eval, deploy)
- `tests/` — mirrors source layout; conftests make tests fast (see `context/conventions.md`)

## External Dependencies

- **OpenSpiel (`universal_poker`)** — Google DeepMind's game engine used as an independent referee for head-to-head evaluation (FCPA abstraction, 100BB HU NL). Not a runtime dependency of training or live play.
- **openpoker.ai** — deployment target; `wss://openpoker.ai/ws` with `Authorization: Bearer $OPENPOKER_API_KEY`. Raise amounts use the **raise-to** convention (total committed, not increment).
- **libtorch** (vendored under `third_party/`) — C++ tensor/NN library backing `neural_cfr`; built by Buck2 (`~/bin/buck2`), not pip.
- **pokerhand-eval (Alvin Liang)** — origin of the fast lookup-table evaluator in `util/util.py` + `util/lookup_table.py`.

## What Does NOT Exist Here

- No multiplayer (>2) CFR training — both CFR pipelines are strictly heads-up; only the live engine supports N-player tables.
- No shared state between the three subsystems — `cfr/` and `neural_cfr/` deliberately do not import `game/poker.py`; each has its own game-state implementation.
- No web UI or HTTP server — interaction is CLI scripts and the outbound WebSocket connector.
- No GPU training assumptions — training runs on CPU (threaded traversal in C++); no CUDA configuration exists.
