---
name: sdd-pi-implementer
description: In subagent-driven development, run implementers as pi + DeepSeek V4 Flash via scripts/pi_run.sh instead of Claude subagents; reviewers stay Claude.
triggers:
  - "subagent-driven-development"
  - "SDD"
  - "dispatch implementer"
  - "pi"
  - "deepseek"
edges:
  - target: context/setup.md
    condition: for build and test commands the implementer brief must name
last_updated: 2026-10-21
---

# SDD Implementer = pi + DeepSeek V4 Flash

## Context

When executing a plan with `superpowers:subagent-driven-development`, the **implementer** role runs as the `pi` CLI (0.87.1) on `deepseek/deepseek-flash`, thinking `high`, through `scripts/pi_run.sh`. It does not run as a Claude `Agent` dispatch. Everything else in the SDD skill is unchanged:
- task briefs, dispatch files and report contract;
- Claude reviewers: Sonnet for task review, Haiku for scoped re-review, the most capable model for the final whole-branch review;
- the ledger and rulings.

**Why:** on the equity-lookup-tables plan (Tasks 1-5, 2026-10-20), Flash cost **$1.37** for 465 model calls (0.52M input, 0.47M output, 37.4M cache-read tokens). The same tokens at Sonnet 5.5 list prices come to about $13.30. Quality held: every task passed Sonnet review in at most one fix round. Sonnet would likely take fewer turns, so the real gap is closer to 5-7x than 10x.

## Steps

1. **Dispatch file.** Write the dispatch exactly as the SDD skill says (brief path, interfaces, rulings, report path, "do not spawn subagents"). Save it as `<workspace>/task-N-dispatch.md`. pi reads it as an attachment. Add "never run a command that reads from stdin" and "run long commands with output redirected to a file".
2. **Launch detached:**
   ```bash
   W=.superpowers/sdd/<plan-workspace>
   nohup caffeinate -is scripts/pi_run.sh sdd-<plan>-tN $W/task-N-dispatch.md $W/task-N </dev/null >/dev/null 2>&1 &
   ```
   Session ids must be unique per task: `sdd-<plan>-tN`.
3. **Watch progress.** `$W/task-N-progress.txt` gets one line per tool call. For the completion signal, use a background `until` loop that exits when the pi process for that session id is gone, or when `$W/task-N-events.jsonl` has had no mtime change for 25 min (a stall).
4. **Read the result.** `$W/task-N-stdout.txt` has the short status reply. The full report is in the report file named in the dispatch. Then review per the SDD skill (`review-package` → Sonnet task reviewer).
5. **Fix rounds 1-3** resume the same pi session. Write `$W/task-N-fixR.md` with the findings, then run `scripts/pi_run.sh sdd-<plan>-tN $W/task-N-fixR.md $W/task-N-fixR`, reusing the same session id.
6. **Rounds 4-5 escalate** per the skill: `PI_MODEL=deepseek/deepseek-v4-pro` with a fresh session id, or a Claude Sonnet `Agent` dispatch.

## Gotchas

- **`</dev/null` is mandatory.** pi blocks at startup until stdin hits EOF. A Claude Code Bash tool's stdin is a socket that never closes, so without the redirect pi sits at 0% CPU forever. That caused an 8 h hang on 2026-10-20. `nohup` does not fix it. The wrapper redirects stdin; keep the redirect on the launch line too.
- **Killing the shell or caffeinate does not kill pi.** The node process survives, resumes, and races any relaunch. Kill by `pgrep -f <session-id>` and verify it is gone before relaunching.
- **Text mode prints nothing until the end.** The wrapper uses `--mode json`, which streams live. Don't drop that flag.
- **Don't run two implementers at once.** They conflict in the worktree, which is the same rule as the SDD skill.
- **Credentials:** pi's DeepSeek key lives in pi's own config. Never put it in the repo, `.env.example`, a brief, or a log.
- **Cost check:** `jq` over `~/.pi/agent/sessions/--Users-zfdupont-pokerbot--/*<session-id>*.jsonl`, summing `.message.usage.cost.total` on assistant messages.

## Verify

- `$W/task-N-stdout.txt` ends with the report contract (Status, commits, tests).
- `git log` shows the commit the report names.
- No pi process is left running for the session: `pgrep -f sdd-<plan>-tN` is empty.

## Update Scaffold

- [ ] If model ids, the wrapper or the pi version change, update this file and `scripts/pi_run.sh` together.
