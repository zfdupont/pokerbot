---
name: code-review
description: Review a diff or pull request for real correctness bugs and adherence to this repo's CLAUDE.md and .mex conventions. Use when completing a change, before merging, or when asked to review a PR.
---

# Code Review

Review a change for high-confidence, real issues only. This is a pi port of the Claude
Code `/code-review` command: pi has no Haiku/Sonnet sub-agent fan-out, so do the passes
below yourself, in sequence, keeping the same confidence discipline.

## Scope

- Local branch: review `git diff` against the base branch (`git merge-base` with `main`).
- A PR: pass the number; use `gh pr diff <n>` / `gh pr view <n>`. Never web-fetch GitHub.
- First check eligibility: skip if the PR is closed, a draft, an automated/trivial PR, or
  already reviewed by you.

## Review passes (run each, collect candidate issues)

1. **Convention adherence.** Read the root `CLAUDE.md`, `.mex/AGENTS.md` (Hard
   Invariants), `.mex/context/conventions.md`, and any `CLAUDE.md` in touched dirs. Flag
   violations — especially the pokerbot non-negotiables: no cross-module imports between
   `cfr/`, `neural_cfr/`, `sixmax/`, `game/poker.py`; fixed 6-action vocabulary order;
   the C++ `12 - rank` kicker inversion; chip rescaling to `starting_stack=100,
   big_blind=1`; never commit secrets or checkpoints.
2. **Shallow bug scan.** Read only the changed lines. Look for real bugs (logic errors,
   off-by-one, wrong masks, unhandled None, resource leaks). Ignore nitpicks.
3. **Historical context.** `git blame` / `git log` the touched code for regressions the
   change reintroduces or context it ignores.
4. **Comment/guidance compliance.** Check that changes respect existing code comments and
   docstrings in the modified files.

## Confidence filter (keep the rubric)

Score every candidate 0-100 for "is this a real, impactful issue":

- **0** false positive / pre-existing / doesn't survive light scrutiny
- **25** maybe real, unverified, or stylistic and not called out in CLAUDE.md
- **50** verified real but minor / rare / low importance
- **75** verified, likely hit in practice, or explicitly required by CLAUDE.md
- **100** certain, frequent, evidence directly confirms it

**Report only issues scoring >= 80.** If none, say so plainly.

## Not issues (drop these)

Pre-existing problems; things a linter/typechecker/CI would catch (imports, types, format,
broken tests); general "add more tests/docs" unless CLAUDE.md requires it; nitpicks a
senior engineer would not raise; issues on lines the change did not touch; intentional
functional changes related to the PR's purpose.

## Output

Brief, no emojis, cite each issue with `file:line`. For a PR, post with `gh pr comment`
using permalinks (full sha + `#Lstart-Lend`, at least one line of context each). Format:

```
### Code review

Found N issues:

1. <brief description> (CLAUDE.md says "<quote>")
   <file:line or permalink>
```

Or, if clean: `No issues found. Checked for bugs and CLAUDE.md / .mex convention compliance.`
