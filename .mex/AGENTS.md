---
name: agents
description: Always-loaded project anchor. Read this first. Contains project identity, non-negotiables, commands, and pointer to ROUTER.md for full context.
last_updated: 2026-07-19
---

# pokerbot

## What This Is
A Python Texas Hold'em simulator with pluggable agents, plus self-contained CFR training pipelines — tabular MCCFR (`cfr/`) and C++ Deep CFR (`neural_cfr/`), both heads-up — and a six-max blueprint+search subsystem (`sixmax/`, Phases 1a–1c merged; tabular blueprint now learning post-deal-loop-fix, +32.5 BB/100 vs PotOdds at 5M iters; a DREAM neural blueprint in `sixmax/src/dream/` is built and ready to merge on `feature/neural-cfr`). Strategies deploy to openpoker.ai.

## Non-Negotiables
- `cfr/`, `neural_cfr/`, and `sixmax/` never import `game/poker.py` or each other — bridging lives only in `agents/cfr_agent.py` and `scripts/`; shared C++ lives only in `common/`.
- In `cfr/` and `neural_cfr/`, the 6-action vocabulary order is fixed: `0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`. In `sixmax/`, the vocabulary is config-defined and every checkpoint embeds its vocab hash (loaders refuse mismatches). Everywhere: mask illegal actions; never reorder or filter storage.
- The C++ evaluator (`common/src/game/card.cpp`) keeps its `12 - rank` kicker inversion (lower = better) — removing it inverts learned hand strength. New consumers use only the opaque `safe_eval::HandRank` API; raw scores never leave `common/`.
- Chip values crossing an engine boundary must be rescaled to the `starting_stack=100, big_blind=1` training frame (divide by the table's big blind). `sixmax/` is BB-denominated natively — no absolute-chip constants.
- Never commit secrets (`OPENPOKER_API_KEY`) or checkpoint files.

## Commands
- Test: `uv run pytest tests/`
- Train (tabular): `uv run python scripts/train_cfr.py`
- Train (neural): `uv run python scripts/train_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt`
- Train (six-max blueprint): `uv run python scripts/train_sixmax.py --iterations 5000000 --checkpoint sixmax/checkpoints/blueprint.bin`
- Train (DREAM neural blueprint): `uv run python scripts/train_dream.py`
- Eval: `uv run python scripts/eval_openspiel.py --hands 2000`
- Build C++: `~/bin/buck2 build //neural_cfr:neural_cfr //sixmax:sixmax`
- Setup (fresh clone/worktree): `./scripts/setup_dev.sh`
- Deploy: `./scripts/run_openpoker.sh {start|stop|status}`
- Play: `uv run python scripts/play.py`

## Scaffold Growth
After meaningful work, run GROW:
- Ground: what changed in reality?
- Record: update `ROUTER.md` and relevant `context/` files
- Orient: create or update a `patterns/` runbook if this can recur
- Write: bump `last_updated` on changed scaffold files and run `mex log` when rationale matters

The scaffold grows from real work, not just setup. See the GROW step in `ROUTER.md` for details.

## Navigation
At the start of every session, read `ROUTER.md` before doing anything else.
For full project context, patterns, and task guidance — everything is there.
