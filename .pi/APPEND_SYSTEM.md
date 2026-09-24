# pokerbot — pi project instructions

Pi already ingests this repo's root `CLAUDE.md`; this file adds pi-specific bootstrap and
translates the MEX workflow into pi terms. The `.mex/` scaffold is the source of truth.

## Session start (every session)

1. Read `.mex/AGENTS.md` (identity, non-negotiables, commands).
2. Read `.mex/ROUTER.md` and follow its CONTEXT -> BUILD -> VERIFY -> DEBUG -> GROW loop
   for every task.
3. Before any task, check `.mex/patterns/INDEX.md` for a matching pattern and follow it.

Routing (full table in `.mex/ROUTER.md`): architecture -> `.mex/context/architecture.md`;
stack -> `stack.md`; conventions + verify checklist -> `conventions.md`; rationale ->
`decisions.md`; setup/commands -> `setup.md`; tabular CFR -> `cfr-training.md`; neural CFR
-> `neural-cfr.md`.

## Hard Invariants (never violate — see `.mex/AGENTS.md`)

- `cfr/`, `neural_cfr/`, `sixmax/` never import `game/poker.py` or each other; bridging
  lives only in `agents/cfr_agent.py` and `scripts/`; shared C++ only via `common/`.
- Fixed 6-action vocabulary order (`0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`) in
  `cfr/`/`neural_cfr/`; config-defined in `sixmax/` (checkpoints embed its hash).
  Everywhere: mask illegal actions; never reorder or filter storage.
- Preserve the C++ evaluator's `12 - rank` kicker inversion; new consumers use `safe_eval`.
- Rescale chip values crossing an engine boundary to `starting_stack=100, big_blind=1`.
- Never commit secrets (`OPENPOKER_API_KEY`) or checkpoint files.

## Skills

This project's ported skills are installed via the local `pokerbot-pi-skills` package.
Load them with `/skill:<name>`. Notably:

- Start creative/feature work with `/skill:brainstorming`; plan with `/skill:writing-plans`.
- Implement with `/skill:test-driven-development`; debug with `/skill:systematic-debugging`.
- Before finishing: `/skill:verification-before-completion`, `/skill:pre-commit`, and
  `/skill:security-review`.
- Contribute knowledge with `/skill:mex-inbox`; hand off with `/skill:mex-relay`
  (both require the `mex` CLI on PATH).

## Commands

`uv run pytest tests/` (223 tests); `uv run python scripts/train_cfr.py`;
`uv run python scripts/train_neural.py ...`; `uv run python scripts/eval_openspiel.py`;
`~/bin/buck2 build //neural_cfr:neural_cfr`. Full reference: `.mex/context/setup.md`.
