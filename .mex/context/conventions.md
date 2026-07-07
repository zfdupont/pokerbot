---
name: conventions
description: How code is written in this project — naming, structure, patterns, and style. Load when writing new code or reviewing existing code.
triggers:
  - "convention"
  - "pattern"
  - "naming"
  - "style"
  - "how should I"
  - "what's the right way"
edges:
  - target: context/architecture.md
    condition: when a convention depends on understanding the subsystem boundaries
  - target: context/decisions.md
    condition: when a convention exists because of a past bug or design decision
  - target: patterns/add-agent.md
    condition: when applying the engine↔agent boundary to a new agent
last_updated: 2026-07-07
---

# Conventions

## Naming

- Python files/modules: snake_case (`pot_manager.py`, `abstract_state.py`); classes PascalCase (`PokerGame`, `RegretTable`, `HandTracker`).
- Agents are named `<Strategy>Agent` and live in `agents/` (`SimpleAgent`, `PositionAgent`, `CFRAgent`).
- Enums are centralized in `models/enums.py` (`Action`, `Position`, `Suit`, `HandRank`) — never define parallel action/position enums elsewhere.
- Test files mirror source names: `tests/test_pot_manager.py` for `game/pot_manager.py`; C++-facing tests under `tests/neural_cfr/`.
- Checkpoints: tabular `checkpoint_XXXXXXXX.pkl` (zero-padded iteration count) in `cfr/checkpoints/`; neural `checkpoint.pt` in `neural_cfr/checkpoints/`. Both dirs are gitignored.

## Structure

- Decision logic lives in agents, never in the engine: `PokerGame` calls `Player.make_decision(state)` → injected agent's `get_action(player, state) -> (Action, amount)`. New policies subclass `agents/base_agent.py:PokerAgent`.
- `cfr/` and `neural_cfr/` are sealed packages — they must never import `game/poker.py` or each other. Bridging happens only in `agents/cfr_agent.py` and `scripts/`.
- Runnable entry points live in `scripts/` with argparse CLIs; library packages contain no `__main__` logic (exception: `main.py` runs one demo hand).
- C++ code is split by concern: `neural_cfr/src/game/` (state, cards), `src/net/` (features, MLP), `src/cfr/` (traversal, trainer), `src/bindings/` (pybind11). Named constants live in headers (`features.h`, `trainer.h`) — no magic numbers in `.cpp`.

## Patterns

**Fixed 6-action vocabulary, mask at query time.** Both CFR pipelines use index order `0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`. Storage is always over all 6 actions; illegal actions are masked when querying (`RegretTable` does this at query time). Never reorder or filter the vocabulary at storage time.

**Checkpoint-first tooling.** Every consumer script takes `--checkpoint` and auto-detects the latest tabular checkpoint when omitted; `scripts/openpoker_bot.py` dispatches on extension (`.pt` → neural, `.pkl` → tabular). New tooling should follow this.

**Chip normalization at boundaries.** Neural nets were trained at `starting_stack=100, big_blind=1`. Any adapter feeding another engine's chip counts (OpenSpiel, openpoker.ai) must divide by `(their_stack / 100)` before `get_action_probs`. See the fixes in `scripts/eval_openspiel_neural.py` and `scripts/openpoker_bot.py`.

**Test-time cheapening via conftest.** `tests/conftest.py` patches `MONTE_CARLO_SAMPLES=10` globally and `tests/cfr/conftest.py` stubs `compute_exploitability` — tests must stay fast; don't add tests that run real equity rollouts at full sample counts.

## Verify Checklist

Before presenting any code:
- [ ] `uv run pytest tests/` passes (use a targeted subset first, then the full suite).
- [ ] No new import crosses a subsystem boundary (`cfr/` ↛ `game/`, `neural_cfr/` ↛ anything Python-side, `game/` ↛ `cfr/`).
- [ ] Action indices respect the fixed vocabulary order `fold/check/call/b0.5/b1.0/allin`; illegal actions masked, not removed.
- [ ] Chip values crossing an engine boundary are rescaled to the 100BB training frame.
- [ ] C++ changes: `~/bin/buck2 build //neural_cfr:neural_cfr` succeeds and any kicker/encoding change preserves the `12 - rank` inversion (lower = better).
- [ ] No secrets committed; `OPENPOKER_API_KEY` stays in the environment, never in source.
