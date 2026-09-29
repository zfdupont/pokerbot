---
name: stack
description: Technology stack, library choices, and the reasoning behind them. Load when working with specific technologies or making decisions about libraries and tools.
triggers:
  - "library"
  - "package"
  - "dependency"
  - "which tool"
  - "technology"
edges:
  - target: context/decisions.md
    condition: when the reasoning behind a tech choice is needed
  - target: context/conventions.md
    condition: when understanding how to use a technology in this codebase
  - target: context/setup.md
    condition: when installing or building the stack (uv, Buck2, libtorch)
last_updated: 2026-07-25
---

# Stack

## Core Technologies

- **Python ≥3.10** — primary language for the engine, tabular CFR, agents, and all scripts (`pyproject.toml`).
- **uv** — package manager and runner; every command is `uv run ...`. Do not use pip/venv directly.
- **C++ (libtorch + pybind11)** — the `neural_cfr/` Deep CFR subsystem (exposed as `import neural_cfr`) and the `sixmax/` blueprint subsystem (exposed as `import sixmax`); shared C++ utilities live in `common/` (Buck2 `//common:evaluator`).
- **Buck2** (`~/bin/buck2`, not on PATH) — build system for all C++ extensions; deps vendored in `third_party/` (2000+ files, do not touch). Three targets: `//neural_cfr:neural_cfr`, `//sixmax:sixmax`, `//common:evaluator` (library only, not imported directly by Python).

## Key Libraries

- **numpy** — regret matching and strategy math in `cfr/regret_table.py`; vectorized equity estimation.
- **pytest** (not unittest) — full suite in `tests/`, 270 tests; `pythonpath = ["."]` set in `pyproject.toml` so imports are repo-root-relative. The full suite runs green in a single process; C++ bindings that call autograd release the GIL (see `context/conventions.md`).
- **websockets** — `scripts/openpoker_bot.py` connector to openpoker.ai.
- **tqdm** — progress bars in the tabular trainer; the C++ trainer uses p-ranav/indicators instead.
- **cython** — hot-path acceleration for tabular MCCFR traversal (added in the parallel-training perf pass).
- **OpenSpiel** (optional, eval-only) — `universal_poker` environment for independent head-to-head evaluation; only imported by `scripts/eval_openspiel*.py`.
- **libtorch** (C++, not PyTorch-Python) — MLPs, Adam, serialization in `neural_cfr/`; checkpoints are `.pt` archives written from C++.

## What We Deliberately Do NOT Use

- **No Python deep-learning framework for training** — Deep CFR is pure C++ for traversal throughput; Python only launches and consumes it (`scripts/train_neural.py`).
- **No cross-imports between subsystems** — `cfr/` and `neural_cfr/` never import `game/poker.py`; each carries its own lightweight game state. Don't "DRY" them together.
- **No linter/formatter config** — there is no ruff/black/flake8 setup; match existing file style by hand.
- **No requirements.txt** — dependencies live in `pyproject.toml` only, managed by uv.

## Version Constraints

- `requires-python = ">=3.10"`.
- Buck2 is invoked as `~/bin/buck2` — it is a user-local binary, not installed system-wide.
- Neural training assumes `starting_stack=100, big_blind=1`; any consumer feeding chips from another engine must rescale (see `context/neural-cfr.md`).
