---
name: mex-relay
description: Prepare and manage durable MEX team handoffs. Use when handing work to a person/team, preparing an end-of-session handoff, or saving what the next engineer needs. Not for routine status messages.
---

# MEX Relay

Pi wrapper for this repo's MEX Relay workflow (mirrors the Claude Code `/mex-relay`
skill). A Relay is a durable memory and context baton for the next engineer — not chat,
notifications, task assignment, or an issue tracker.

## When to use

- Handing work off to a person or team.
- End-of-session handoff: what was done, what's in flight, what's next, and the traps.
- Taking or closing an existing Relay.
- NOT for: ordinary status updates that are not durable handoffs.

## How

1. Read `.mex/AGENTS.md`, `.mex/ROUTER.md`, and `.mex/SYNC.md` for the handoff conventions.
2. Use the `mex` CLI to create/take/close the Relay (run `mex relay --help` to confirm
   exact subcommands in the installed version). If unavailable, draft the handoff as a
   checkout-local note under `.mex/` marked as a Relay draft.
3. Capture: current state, decisions made and why, blockers/risks, next steps, and the
   exact files/commands the next person needs. Convert relative dates to absolute.

## After writing

State the sharing boundary: a local draft is checkout-only; a canonical Relay written to
the working tree requires commit/push to reach the team. Skill activation is not approval
for canonical actions.
