# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python Texas Hold'em poker simulator. Agents (decision-making policies) play hands against each other under a shared game engine. Two self-contained CFR training pipelines learn Nash-approximate heads-up strategies — tabular MCCFR (`cfr/`) and C++ Deep CFR (`neural_cfr/`, libtorch + pybind11 + Buck2) — plus tooling to play, visualize, evaluate, and deploy the trained bot.

## Context Scaffold — read this first

Detailed project knowledge lives in the `.mex/` scaffold, which is the **source of truth**. At the start of every session:

1. Read `.mex/AGENTS.md` (identity, non-negotiables, commands).
2. Read `.mex/ROUTER.md` (current project state, routing table, behavioural contract) and follow its CONTEXT → BUILD → VERIFY → DEBUG → GROW loop for every task.
3. Before any task, check `.mex/patterns/INDEX.md` for a matching pattern and follow it.

Routing summary (full table in `.mex/ROUTER.md`):

| Topic | File |
|-------|------|
| System structure, module map | `.mex/context/architecture.md` |
| Tech stack, build tools | `.mex/context/stack.md` |
| Code conventions + verify checklist | `.mex/context/conventions.md` |
| Why things are built this way | `.mex/context/decisions.md` |
| Setup, commands, common issues | `.mex/context/setup.md` |
| Tabular CFR (abstraction, exploitability) | `.mex/context/cfr-training.md` |
| Neural CFR (encodings, training, checkpoints) | `.mex/context/neural-cfr.md` |

After meaningful work, update the scaffold (GROW step in `.mex/ROUTER.md`) — keep it current instead of growing this file.

## Hard Invariants

Always edit  Non-Negotiables after modifying any of these. Condensed from `.mex/AGENTS.md` — never violate these:

- `cfr/`, `neural_cfr/`, and `sixmax/` never import `game/poker.py` or each other; bridging lives only in `agents/cfr_agent.py` and `scripts/`; shared C++ only via `common/`.
- In `cfr/`/`neural_cfr/` the 6-action vocabulary order is fixed (`0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`); in `sixmax/` the vocab is config-defined and checkpoints embed its hash. Everywhere: mask illegal actions, never reorder or filter storage.
- The C++ evaluator's (`common/`) `12 - rank` kicker inversion (lower = better) must be preserved — removing it inverts learned hand strength; new consumers use the opaque `safe_eval` API only.
- Chip values crossing an engine boundary must be rescaled to the `starting_stack=100, big_blind=1` training frame.
- Never commit secrets (`OPENPOKER_API_KEY`) or checkpoint files.

## Quick Commands

```bash
uv run pytest tests/                                    # full suite (~119 tests)
uv run python scripts/train_cfr.py                      # tabular training
uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml --checkpoint neural_cfr/checkpoints/checkpoint.pt
uv run python scripts/eval_openspiel.py --hands 2000    # head-to-head eval (BB/100)
uv run python scripts/play.py                           # interactive CLI vs the bot
~/bin/buck2 build //neural_cfr:neural_cfr               # rebuild C++ extension
```

Full command reference, flags, and troubleshooting: `.mex/context/setup.md`.
