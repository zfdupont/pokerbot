---
name: deploy-openpoker
description: Deploying a trained bot to openpoker.ai over WebSocket — auth, conventions, and message-tracking gotchas.
triggers:
  - "openpoker"
  - "deploy"
  - "websocket"
  - "live play"
edges:
  - target: context/neural-cfr.md
    condition: when deploying a .pt checkpoint (chip scaling applies)
  - target: context/setup.md
    condition: for the OPENPOKER_API_KEY environment variable
  - target: patterns/debug-bot-misplay.md
    condition: when the deployed bot makes nonsensical decisions
last_updated: 2026-07-07
---

# Deploy to openpoker.ai

## Context

`scripts/openpoker_bot.py` connects to `wss://openpoker.ai/ws` with `Authorization: Bearer <api_key>`. A `HandTracker` accumulates `hole_cards`, `community_cards`, and `raises_per_street` across message types and queries the strategy on `your_turn`. The script auto-detects checkpoint type: `.pt` → neural `Strategy`, `.pkl` → tabular `RegretTable.get_average_strategy()`.

## Steps

1. `export OPENPOKER_API_KEY=...` (or pass `--api-key`; never hardcode it).
2. `uv run python scripts/openpoker_bot.py [--checkpoint path] [--buy-in 2000] [--log-level DEBUG]`.
3. Start with `--log-level DEBUG` for a new checkpoint and watch a few decisions before leaving it running.

## Gotchas

- **Raise-to convention:** OpenPoker raise amounts are the TOTAL chips committed, not the increment. Sending increment-style amounts produces wrong sizings.
- **Chip scaling (neural):** table stacks differ from the training frame; inputs must be rescaled to `starting_stack=100, big_blind=1` (handled in the connector — commit `53074bc` — preserve it when editing).
- **Position tracking matters:** the neural feature vector includes player position (dim 133); the connector tracks it per hand — breaking that silently degrades play.
- State is accumulated across messages: `raises_per_street` builds up from action broadcasts, not from `your_turn` alone. Missing a message type leaves the tracker stale.

## Verify

- [ ] Bot connects and buys in without auth errors.
- [ ] DEBUG log shows sensible actions for obvious spots (raises premium hands, folds trash to big bets).
- [ ] Raise amounts in the log match raise-to semantics (total committed).
- [ ] No API key appears in any committed file or log output.

## Debug

- Auth failure → check `OPENPOKER_API_KEY` is exported in the running shell.
- Absurd play with a `.pt` checkpoint → chip scaling first: confirm the divide-by-`(stack/100)` path executes for the table's actual stack size.
- Wrong bet sizes → raise-to vs. increment confusion.

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
