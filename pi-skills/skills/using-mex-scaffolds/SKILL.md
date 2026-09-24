---
name: using-mex-scaffolds
description: Use when working in a repository containing a .mex/ directory (mex project-memory scaffold) — especially when wrapping up a session, finishing a task, or when scaffold/anchor documentation disagrees with the codebase.
---

# Using MEX Scaffolds

## Overview

`.mex/` is the project's persistent memory (ROUTER.md routing table, `context/` docs, `patterns/`, decision log). It only works if it stays true — and it stays true only if sessions that change the project also change the scaffold. You are its maintainer, not just its reader.

Reading/routing is usually handled by the repo's anchor file (CLAUDE.md/AGENTS.md); follow it. This skill governs the two moments agents reliably miss: **wrap-up** and **drift**.

## Session wrap-up in a mex repo

A wrap-up consists of, in order:

1. Tests / git mechanics as usual.
2. **GROW pass** — for each meaningful change this session, update the matching scaffold file:
   - new/moved module or dependency → `context/architecture.md`
   - new command, script, or setup step → `context/setup.md`
   - milestone reached, state changed → ROUTER.md project-state section
   - repeatable procedure learned → `patterns/`
3. `mex log "<one-line decision or outcome>"` for each decision worth remembering.
4. `mex check` — if your session worsened the drift score, fix what it broke.

A wrap-up containing only a summary and git handling is incomplete in a mex repo.

## Scaffold/code mismatch (drift)

When any scaffold or anchor claim contradicts reality, the full move is:

1. Fix the claim where you found it.
2. Sweep: `grep -rn "<the stale value>" .mex/ CLAUDE.md AGENTS.md` — search for the stale number, name, or path itself, not your phrasing of it — and fix every hit. (`mex check` automates this when installed.) Drift rarely lives in one spot.
3. `mex log` the correction.

## Quick reference

| Command | Purpose |
|---|---|
| `mex check` | Score scaffold drift 0–100 (no AI tokens) |
| `mex sync` | Generate targeted prompts to fix detected drift |
| `mex log <msg>` | Append to `.mex/events/decisions.jsonl` |
| `mex watch` | Install post-commit drift hook |

The CLI is often installed via npx rather than on PATH: `which mex` failing is not evidence of absence — try `npx -y mex-agent <cmd>` first. Only if that also fails, every step still happens — it just degrades to direct edits:
- `mex log` → append the decision to `context/decisions.md` (or the scaffold's decision file)
- `mex check` → grep the scaffold for the facts your session changed
- always: bump `last_updated` in every scaffold file you edited

## Common mistakes

- Treating "wrap up" as git-only → scaffold silently rots (the failure this skill exists for).
- Fixing drift at the first stale spot only → the same fact stays wrong elsewhere; sweep.
- Writing session narrative into the scaffold → it stores durable facts and procedures, not stories.
