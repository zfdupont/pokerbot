---
name: pre-commit
description: Run this repo's quality gates before committing — tests, any configured linter/typecheck, and a secret scan. Use before every commit.
---

# Pre-Commit Gates

Run all gates and only commit when they pass. Do not claim success without showing the
command output (see `/skill:verification-before-completion`).

## Gates

1. **Tests pass.**
   - Full suite: `uv run pytest tests/` (223 tests).
   - Or the targeted subset for the change: `uv run pytest tests/<path> -q`.
   - If C++ (`neural_cfr/`, `common/`) changed, rebuild first:
     `~/bin/buck2 build //neural_cfr:neural_cfr`.
2. **Lint / typecheck (if configured).** This repo has no mandated lint/typecheck script;
   if `pyproject.toml` defines ruff/mypy, run them (`uv run ruff check .`,
   `uv run mypy ...`). Otherwise follow the verify checklist in
   `.mex/context/conventions.md`.
3. **No secrets staged.** Run `/skill:security-review`.
4. **Hard Invariants intact.** Re-read `.mex/AGENTS.md`; if the change touches vocabulary
   order, action masking, the C++ kicker inversion, chip rescaling, or module boundaries,
   confirm none were violated and update Non-Negotiables if required.

## Result

Report each gate's actual output. Commit only after all pass. Keep the commit message
brief; do not add Co-Authored-By / "Generated with" trailers (per global CLAUDE.md).
