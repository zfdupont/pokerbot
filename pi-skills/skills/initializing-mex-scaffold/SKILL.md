---
name: initializing-mex-scaffold
description: Use when a repository has NO .mex/ directory yet and you need to bootstrap the mex project-memory scaffold from scratch — e.g. the user asks to "init mex", "set up mex", "add mex", or scaffold project memory. NOT for maintaining an existing .mex/ (that is using-mex-scaffolds).
---

# Initializing a mex Scaffold

## Overview

`.mex/` is a project-memory scaffold (ROUTER.md, `context/`, `patterns/`,
decision log). This skill covers **creating** it in a repo that has none.
Maintaining an existing scaffold is a different skill (`using-mex-scaffolds`).

**The one trap that matters:** the scaffold is created by **`mex setup`**,
NOT `mex init`. The name `init` is a decoy — `mex init` only scans the code
and prints an AI pre-analysis brief; it creates nothing. Reach for `setup`.

Repo: https://github.com/mex-memory/mex

## When to Use

- Repo has no `.mex/` directory (verify: `ls -la .mex/` fails) AND the user
  wants project-memory scaffolding.
- User says "init/set up/add mex", "bootstrap mex", "scaffold project memory".

**When NOT to use:**
- `.mex/` already exists → this is maintenance/drift work → `using-mex-scaffolds`.
- User only wants a one-off note remembered → use your normal memory, not a scaffold.

## Preconditions

- Must be a git repository (`git rev-parse --git-dir`). Offer `git init` if not.
- The CLI is usually run via `npx -y mex-agent <cmd>` (rarely on PATH — `which
  mex` failing is not evidence of absence).

## Workflow

1. **Confirm no scaffold exists** — `ls -la .mex/` should fail. If it exists, stop
   and switch to `using-mex-scaffolds`.

2. **Run setup non-interactively.** `mex setup` prompts "Which AI tool do you
   use?" (default `1` = Claude Code) and will hang an agent. Pipe the answer:
   ```bash
   printf '1\n' | npx -y mex-agent setup           # code-repo (default)
   printf '1\n' | npx -y mex-agent setup --mode agent-memory   # notes/infra memory repo
   ```
   Preview first with `--dry-run` if unsure. Setup inspects the repo, builds
   the code graph, copies the scaffold (ROUTER.md, AGENTS.md, SETUP.md,
   SYNC.md, `context/{architecture,stack,conventions,decisions,setup}.md`,
   `patterns/{README,INDEX}.md`), installs the anchor (appends CLAUDE.md), and
   validates. It then asks the coding agent (you) to populate the wiki.

3. **Populate the scaffold — this is your job, not the CLI's.** Setup leaves
   skeleton context files. Fill them with real facts:
   - `npx -y mex-agent init` → get the pre-analysis brief (codebase scan).
   - Read the codebase, then write real content into `.mex/context/*.md`
     (architecture, stack, conventions, setup) and ROUTER.md's project-state.
   - Capture any reusable procedure in `.mex/patterns/`.
   Populate from evidence in the code — do not invent structure that isn't there.

4. **Verify.** `npx -y mex-agent doctor` (health) and `npx -y mex-agent check`
   (drift score 0–100). Fix what they flag before claiming done.

5. **Log it.** `npx -y mex-agent log "Initialized mex scaffold"`.

## Quick Reference

| Command | Purpose |
|---|---|
| `mex setup` | **Create** the `.mex/` scaffold (the init command) |
| `mex setup --mode agent-memory` | Scaffold for a notes/infra memory repo |
| `mex setup --dry-run` | Preview without writing files |
| `mex init` | Scan code, print AI brief — does NOT create scaffold |
| `mex doctor` / `mex check` | Verify scaffold health / drift after setup |
| `mex log <msg>` | Record the setup decision |

## Common Mistakes

- **Running `mex init` to create the scaffold.** It only prints a brief. Use `mex setup`.
- **Letting `mex setup` prompt interactively** and hanging. Pipe `printf '1\n'`.
- **Stopping after `setup`.** The scaffold ships as skeletons; populating
  `context/*` and ROUTER.md from the codebase is the actual work.
- **Scaffolding over an existing `.mex/`.** Check first; if present, it's a
  maintenance task, not init.
- **Assuming `mex` is on PATH.** Default to `npx -y mex-agent`.
