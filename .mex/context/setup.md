---
name: setup
description: Dev environment setup and commands. Load when setting up the project for the first time or when environment issues arise.
triggers:
  - "setup"
  - "install"
  - "environment"
  - "getting started"
  - "how do I run"
  - "local development"
edges:
  - target: context/stack.md
    condition: when specific technology versions or library details are needed
  - target: context/neural-cfr.md
    condition: when building or troubleshooting the C++ extension
  - target: context/cfr-training.md
    condition: when running or resuming tabular training
  - target: patterns/debug-bot-misplay.md
    condition: when a trained bot behaves nonsensically after setup
last_updated: 2026-07-19
---

# Setup

## Prerequisites

- Python ≥3.10 with **uv** installed (all commands run through `uv run`).
- **Buck2** at `~/bin/buck2` (user-local binary, not on PATH) — needed for both C++ extensions: `//neural_cfr:neural_cfr` and `//sixmax:sixmax`.
- macOS (primary dev platform); libtorch and other C++ deps are vendored in `third_party/`.
- Optional: OpenSpiel installed in the environment for `scripts/eval_openspiel*.py`.

## First-time Setup

1. `./scripts/setup_dev.sh` — idempotent one-shot setup for clones, worktrees, and cloud machines: pins/verifies Python 3.10 (extension ABI), `uv sync`, creates missing `third_party/` symlinks (worktrees don't inherit the untracked vendored deps), writes `.buckconfig.local` with the python include path, builds `//neural_cfr:neural_cfr` + `//sixmax:sixmax`, and runs the full suite. Ends with `SETUP OK`.
2. `mkdir -p neural_cfr/checkpoints` — checkpoint dirs are gitignored and not auto-created.

Manual equivalents, if you need one piece: `uv sync`; `uv run pytest tests/`; `~/bin/buck2 build //neural_cfr:neural_cfr //sixmax:sixmax`.

## Environment Variables

- `OPENPOKER_API_KEY` (required only for `scripts/openpoker_bot.py`) — bearer token for `wss://openpoker.ai/ws`. Never commit it.
- No other environment variables are used; training and evaluation are configured via CLI flags.

## Common Commands

- `uv run pytest tests/` — full test suite; single test: `uv run pytest tests/path/test_file.py::ClassName::test_name -v`.
- `uv run python scripts/train_cfr.py --iterations 1000000` — tabular training; `--resume cfr/checkpoints/checkpoint_XXXXXXXX.pkl` to continue.
- `uv run python scripts/train_neural.py --iterations 5_000_000 --checkpoint neural_cfr/checkpoints/checkpoint.pt` — neural training (add `--resume` to continue).
- `uv run python scripts/eval_openspiel.py --hands 2000` — head-to-head eval (BB/100); `eval_openspiel_neural.py` for `.pt` checkpoints.
- `uv run python scripts/play.py [--stack 1000] [--big-blind 10]` — interactive heads-up CLI vs the bot.
- `uv run python scripts/range_chart.py [--scenario sb-open|bb-vs-raise]` — 13×13 ANSI preflop range chart.
- `uv run python scripts/run_simulation.py [--hands 100] [--agent simple|position|cfr]` — agent-vs-agent simulation; `python main.py` runs one demo hand at a 4-player table.
- `./scripts/run_openpoker.sh {start|stop|status}` — managed openpoker deployment: tracks the python PID (pidfile `.openpoker.pid`), `caffeinate -w` alongside, SIGTERM sends `leave_table` before exit (banks the table stack). Env overrides: `CHECKPOINT`, `BUY_IN`. Needs `OPENPOKER_API_KEY` in `.env`.
- `uv run python scripts/openpoker_bot.py --buy-in 2000` — raw deployment (prefer the run script).
- `~/bin/buck2 build //neural_cfr:neural_cfr` — rebuild the C++ extension after any `neural_cfr/src/` or `common/src/` change; `//sixmax:sixmax` for the six-max module.

## Common Issues

**OpenSpiel prints noise on import:** the `pokerkit_wrapper` print is suppressed in `scripts/eval_openspiel.py` (commit `83d9653`) — route new OpenSpiel imports through the same suppression.

**Neural bot plays absurdly (folds strong hands):** almost always a chip-scaling or encoding problem, not a training problem — inputs must be rescaled to the `starting_stack=100, big_blind=1` training frame, and the evaluator's `12 - rank` kicker inversion must be intact. See `patterns/debug-bot-misplay.md`.

**Slow tests:** you're bypassing the conftests. `tests/conftest.py` sets `MONTE_CARLO_SAMPLES=10`; run tests via pytest from the repo root so `pythonpath = ["."]` and the patches apply.

**`buck2: command not found`:** it's at `~/bin/buck2`, not on PATH.

**`pybind11/pybind11.h not found` when building `//sixmax` or `//neural_cfr`:** `third_party/pybind11/include` is a machine-local symlink that rots when the Python env it points at is rebuilt. Repoint it at the pyenv 3.10 site-packages copy (currently `~/.pyenv/versions/3.10.10/lib/python3.10/site-packages/pybind11/include`). `setup_dev.sh` does not yet create/verify this symlink — hardening deferred to Phase 1b.
