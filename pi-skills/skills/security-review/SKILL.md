---
name: security-review
description: Scan pending changes for leaked secrets and security issues before committing or pushing. Use before any commit/push, when preparing a PR, or when asked for a security check.
---

# Security Review

Author's global security gatekeeper, as a pi skill. These rules are absolute — no
exceptions. The goal is to guarantee nothing sensitive leaves the repo.

## Never publish sensitive data

- No passwords, API keys, or tokens committed to git.
- No hardcoded credentials in source.
- No secrets in error messages or logs.
- Repo-specific: **never commit `OPENPOKER_API_KEY`** or any checkpoint files (see
  `.mex/AGENTS.md` Hard Invariants).

## Never commit .env

- Verify `.env` is git-ignored; only `.env.example` (placeholders) is committed.

## Checklist (run before commit/push)

1. **Diff the staged/pending change:** `git diff --cached` and `git status`.
2. **Secret scan** the diff for: high-entropy strings, `API_KEY`/`SECRET`/`TOKEN`/
   `PASSWORD` assignments with real values, private keys (`BEGIN ... PRIVATE KEY`),
   `OPENPOKER_API_KEY`, `.pt`/checkpoint blobs, and `.env` files.
   - `git diff --cached | grep -nEi 'api[_-]?key|secret|token|password|BEGIN [A-Z ]*PRIVATE KEY|OPENPOKER_API_KEY'`
3. **Confirm ignores:** `.env`, `.env.*` (except `.env.example`), and checkpoint dirs are
   in `.gitignore`; nothing under `~/.pi/agent/` (holds the DeepSeek key) is referenced
   from the repo.
4. **No secrets in new logging/print statements** added by the change.

## Output

Report PASS only after every check passes. On any finding: stop, name the file:line and
the secret type, and recommend unstaging + rotating the exposed credential. Never print
the secret value itself.
