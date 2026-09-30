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
last_updated: 2026-07-22
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
- `uv run python scripts/train_sixmax.py` — six-max blueprint training (`[abstraction]` + `[train.blueprint]` in `sixmax/configs/default.toml`; CLI overrides; `--resume PATH` restores table + iteration counter; `--checkpoint-interval N` saves every N iters (`0` = save-at-end-only; add `--snapshots` to keep each save as `<base>_<iters>.bin`); `--report-interval S` logs a live `[done/total] rate it/s ETA infosets` heartbeat every S seconds (`0` = off, default 60); `--selection-enabled` maintains `best_checkpoint.bin` via A/B vs previous best). All progress goes to stderr via `logging` (byte-identical message text, so log greps/watcher regexes are unaffected).
- `uv run python scripts/eval_sixmax.py --a CKPT [--b CKPT|uniform] --hands 500` — duplicate-deal seat-rotated A/B eval; prints `Blueprint A win rate: ±X.XX BB/100`.
- `uv run python scripts/diagnose_blueprint.py [--checkpoints "sixmax/checkpoints/blueprint_0*.bin"] [--top-frac 0.05] [--csv PATH]` — offline blueprint plateau autopsy over a checkpoint series; prints per-checkpoint visit-weight/entropy/probe metrics + a trend-based `VERDICT` (undertraining | structural | mixed). Reads `sixmax.dump_infosets`; run from repo root.
- `uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/blueprint_hu.bin --tabular cfr/checkpoints/checkpoint_09040000.pkl --neural neural_cfr/checkpoints/checkpoint.pt` — HU-mode sanity eval: the six-max blueprint (2-player) vs the frozen tabular/neural bots in the live engine (BB/100). `--tabular`/`--neural` optional.
- `uv run python scripts/openpoker_bot.py --checkpoint sixmax/checkpoints/blueprint.bin` — deploy the six-max blueprint (`.bin` auto-detected → `SixmaxHandTracker`).
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

**`pybind11/pybind11.h not found` when building `//sixmax` or `//neural_cfr`:** `third_party/pybind11/include` is a machine-local symlink that rots when the Python env it points at is rebuilt. Run `./scripts/setup_dev.sh` — since Phase 1b it detects dead third_party symlinks and repoints `pybind11/include` at the venv's pybind11 package automatically.
