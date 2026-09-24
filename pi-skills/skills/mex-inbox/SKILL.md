---
name: mex-inbox
description: Draft and review contributions to existing MEX project knowledge. Use when capturing a discussion or decision into project knowledge, or proposing an addition/correction for team review. Not for brainstorming, routine GROW upkeep, or session logs.
---

# MEX Inbox

Pi wrapper for this repo's MEX Inbox workflow (mirrors the Claude Code `/mex-inbox`
skill). Use it to turn a decision or discovery into a reviewable contribution to the
`.mex/` project knowledge, rather than editing canonical context files directly.

## When to use

- Capturing a concrete decision, correction, or durable discovery for the team.
- Proposing an addition to `.mex/context/*` or a Spec proposal.
- NOT for: brainstorming alone, ordinary GROW upkeep, email inboxes, session logs, or
  handoffs (use `/skill:mex-relay` for handoffs).

## How

1. Read `.mex/AGENTS.md` and `.mex/ROUTER.md` first; follow the GROW step guidance.
2. Check `mex logging --json` for the advisory mode before optional writes.
3. Draft the contribution as an Inbox entry using the `mex` CLI (e.g. `mex inbox ...`;
   run `mex --help` / `mex inbox --help` to confirm exact subcommands in the installed
   version). If the CLI is unavailable, draft the entry as a checkout-local note under
   `.mex/` and clearly mark it as a proposal for review.
4. Keep it evidence-backed and scoped to one decision/correction.

## After writing

State exactly what changed and its sharing boundary: a local draft is checkout-only
(nothing shared); a canonical artifact written to the working tree requires commit/push
to share. Skill activation is not approval for canonical actions — get the user's go-ahead
before promoting a draft.
