# pokerbot-pi-skills

A local [pi](https://github.com/earendil-works/pi) package that ports this repo's most
used Claude Code / superpowers / MEX skills into pi's `SKILL.md` format.

## Install (project-local)

```bash
cd /Users/zfdupont/pokerbot
pi install --local ./pi-skills   # writes the declaration to .pi/settings.json
pi list                          # confirm "pokerbot-pi-skills" is listed
```

Skills are then auto-discovered (name + description added to the system prompt) and
loadable on demand with `/skill:<name>` — e.g. `/skill:test-driven-development`.

## What's included (21 skills)

- **Core dev process:** brainstorming, test-driven-development, systematic-debugging,
  writing-plans, executing-plans, verification-before-completion.
- **Code review & quality:** code-review, simplify, security-review, pre-commit,
  requesting-code-review, receiving-code-review.
- **MEX workflow:** using-mex-scaffolds, initializing-mex-scaffold, mex-inbox, mex-relay.
- **Git & parallelism:** using-git-worktrees, finishing-a-development-branch,
  dispatching-parallel-agents, subagent-driven-development.
- **Meta:** writing-skills.

## Pi adaptations (differences from the Claude Code originals)

- Cross-skill references had the `superpowers:` plugin prefix stripped (pi uses bare names
  / `/skill:<name>`).
- `brainstorming`: the browser-based visual companion (a Claude Code helper server) was
  removed — pi cannot run it.
- `dispatching-parallel-agents`, `subagent-driven-development`, `requesting-code-review`:
  pi has no built-in Task/Agent sub-agent tool. Each carries a "Pi note" explaining how
  to approximate it (SDK/RPC child sessions or a separate `pi` invocation) or fall back to
  sequential execution.
- `code-review`/`simplify` were converted from a Claude Code command/agent; `security-review`
  and `pre-commit` were written fresh from the user's global quality gates and this repo's
  `.mex` conventions.
- `mex-inbox`/`mex-relay` are thin wrappers over the `mex` CLI + `.mex/` scaffold (their
  Claude Code source SKILL.md files were not available on disk). They require the `mex` CLI
  on PATH.

## Not ported

`using-superpowers` and `diagnosing-superpowers` (Claude-Code-harness-only; pi has native
skill discovery).
