---
name: debug-bot-misplay
description: Diagnosing a bot that plays nonsensically (folds strong hands, raises trash, loses badly to random) — the encoding/scaling checklist, in order of likelihood.
triggers:
  - "bot plays badly"
  - "folds aces"
  - "loses to random"
  - "nonsense actions"
  - "misplay"
  - "debug strategy"
edges:
  - target: context/neural-cfr.md
    condition: for the evaluator invariant and feature encoding being checked
  - target: patterns/eval-checkpoint.md
    condition: to reproduce the misplay in a controlled OpenSpiel match
  - target: patterns/deploy-openpoker.md
    condition: when the misplay is observed on openpoker.ai specifically
last_updated: 2026-07-19
---

# Debug Bot Misplay

## Context

Every confirmed "the bot is stupid" incident in this repo was an encoding or boundary bug, not a training failure. Check boundaries before retraining. Historical culprits: kicker inversion missing (`1c606c5` — AA folds, 22 raises), OpenSpiel chips not rescaled (`87317c5`), inverted `to_call` (`2706697`), wrong net at opponent nodes (`d7532ff`).

## Steps

1. **Reproduce cheaply:** run the checkpoint vs. random via the matching eval script (`eval_openspiel.py` or `eval_openspiel_neural.py`, ~2000 hands). Losing to random = boundary bug almost certainly.
2. **Check chip scaling (neural):** print the feature vector at a decision point; pot/stack/bet dims (123–124, 129–130) must be in the `starting_stack=100` frame after `CHIP_NORM=200` division. If the source engine uses different stacks, confirm the divide-by-`(stack/100)` rescale runs.
3. **Check the evaluator invariant (C++):** in `neural_cfr/src/game/card.cpp`, kickers must be `12 - rank` (lower = better). Spot-check: `evaluate_7card` of an AA hand must be `<` the same board with 22.
4. **Check action mapping:** vocabulary order is `0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`; for OpenSpiel, verify the action-1 check/call disambiguation (`_is_check_action()`).
5. **Check state accumulation (openpoker):** confirm `HandTracker` has the right hole cards, board, position, and `raises_per_street` at the moment of decision (DEBUG log).
6. Only if 2–5 all pass: suspect training (traversal invariants in `context/decisions.md`, buffer sizes, iteration count).

## Gotchas

- A bot that folds ~everything usually has inverted hand strength or an always-huge `to_call` — both look like "bad strategy" but are deterministic bugs.
- Tabular and neural share the action vocabulary but nothing else; don't apply neural chip-scaling fixes to the tabular path.
- Small eval samples mislead: 200 hands of HU NL can show a losing record for a winning bot.

## Verify

- [ ] After the fix, checkpoint beats uniform random clearly over ≥2000 hands.
- [ ] Feature-vector spot check shows normalized values in expected ranges (cards one-hot, chips ≲ 1.0).
- [ ] `uv run pytest tests/neural_cfr/ tests/test_cfr_agent.py -v` passes.

## Debug

If all checks pass and play is still weak (but not absurd), that's a strategy-quality issue, not a bug — see the known abstraction gaps in `context/cfr-training.md` or train longer.

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If a new failure mode was found, add it to this pattern's Context list
