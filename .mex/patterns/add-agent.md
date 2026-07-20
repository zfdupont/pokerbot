---
name: add-agent
description: Adding a new decision-making agent (policy) to the live game engine.
triggers:
  - "add agent"
  - "new agent"
  - "new policy"
  - "custom strategy"
edges:
  - target: context/conventions.md
    condition: for the engine↔agent boundary rules and naming conventions
  - target: context/architecture.md
    condition: to understand how PokerGame drives agents
last_updated: 2026-07-19
---

# Add a New Agent

## Context

Agents are pluggable decision policies. The engine (`game/poker.py:PokerGame`) calls `Player.make_decision(state)`, which delegates to the injected agent. Existing examples: `agents/simple_agent.py:SimpleAgent`, `agents/position_agent.py:PositionBasedAgent`, `agents/hand_strength_agent.py:HandStrengthAgent`, `agents/cfr_agent.py:CFRAgent`.

## Steps

1. Create `agents/<name>_agent.py` with a class `<Name>Agent(PokerAgent)` (from `agents/base_agent.py`).
2. Implement `get_action(self, player: Player, game_state: GameState) -> Tuple[Action, Optional[int]]` using the `Action` enum from `models/enums.py`.
3. Read state only — never mutate `GameState`; the engine owns all state transitions and pot math (`game/pot_manager.py`).
4. Wire it into `scripts/run_simulation.py`: import it and add its key to the `--agent` choices list.
5. Add `tests/test_<name>_agent.py` mirroring the style of `tests/test_cfr_agent.py`.

## Gotchas

- The amount in `(Action, amount)` is chips for the current action; check how `CFRAgent` clamps to legal bet sizes before inventing your own clamping.
- If the agent needs raise history, it's threaded through `GameState.raises_per_street` (added specifically because agents can't recompute it).
- Don't import anything from `cfr/` or `neural_cfr/` unless you are writing a checkpoint-backed agent — and then only load checkpoints, never training code.

## Verify

- [ ] `uv run pytest tests/ -v` passes, including the new test file.
- [ ] `uv run python scripts/run_simulation.py --hands 20 --agent <key>` completes without error.
- [ ] Agent never mutates `GameState` (read-only access).

## Debug

- Illegal action errors: compare the returned `(Action, amount)` against the engine's legality checks in `game/poker.py`; ensure call amounts match the outstanding bet.
- Agent never gets asked to act: check it was injected into `Player` in the simulation setup.

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
