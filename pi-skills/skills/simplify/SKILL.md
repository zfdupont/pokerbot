---
name: simplify
description: Simplify and refine recently changed code for clarity, consistency, and maintainability WITHOUT changing behavior. Use after writing/modifying code, or when asked to clean up a change. Quality only, not bug hunting.
---

# Simplify

Improve the clarity and consistency of recently modified code while preserving exact
functionality. Ported from the Claude Code `code-simplifier` agent, retargeted to this
repo's Python/C++ stack. Prefer readable, explicit code over clever compactness.

## Rules

1. **Preserve functionality.** Change only *how* the code works, never *what* it does.
   All outputs, behaviors, and public interfaces stay identical.
2. **Apply project standards.** Follow `.mex/context/conventions.md` and the root
   `CLAUDE.md`. Do not invent style rules; use the ones this repo actually documents.
3. **Respect the Hard Invariants** in `.mex/AGENTS.md` — never "simplify" away the fixed
   6-action vocabulary order, action masking, the C++ `12 - rank` kicker inversion, chip
   rescaling, or module-boundary rules (`cfr/`, `neural_cfr/`, `sixmax/` isolation).
4. **Enhance clarity.** Reduce needless nesting and complexity, remove dead code and
   redundant abstractions, improve names, consolidate related logic, delete comments that
   restate obvious code. Avoid nested ternaries / dense one-liners — prefer explicit
   `if`/`else`.
5. **Keep balance.** Do not over-simplify: don't merge unrelated concerns, strip helpful
   abstractions, trade readability for fewer lines, or make debugging harder.
6. **Focus scope.** Only touch code modified in the current change/session unless told
   otherwise. Review the diff first (`git diff`) to find that scope.

## Process

1. Identify recently modified sections (from the diff).
2. Spot clarity/consistency improvements that keep behavior identical.
3. Apply repo standards.
4. Verify behavior is unchanged — run the relevant tests (`uv run pytest tests/...`), and
   rebuild the C++ extension if `neural_cfr/`/`common/` changed
   (`~/bin/buck2 build //neural_cfr:neural_cfr`).
5. Summarize only the significant changes.

This skill does not hunt for bugs — use `/skill:code-review` for that.
