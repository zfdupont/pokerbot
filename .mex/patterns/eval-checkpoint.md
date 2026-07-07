---
name: eval-checkpoint
description: Evaluating a trained checkpoint against OpenSpiel or other baselines — with the action-mapping and chip-scaling gotchas.
triggers:
  - "evaluate"
  - "eval"
  - "openspiel"
  - "win rate"
  - "BB/100"
edges:
  - target: context/cfr-training.md
    condition: for what the exploitability metric does and does not measure
  - target: context/neural-cfr.md
    condition: for the chip-scaling rule when evaluating .pt checkpoints
  - target: patterns/debug-bot-misplay.md
    condition: when eval results look absurd (large negative win rate)
last_updated: 2026-07-07
---

# Evaluate a Checkpoint

## Context

`scripts/eval_openspiel.py` (tabular `.pkl`) and `scripts/eval_openspiel_neural.py` (neural `.pt`) run head-to-head matches in Google DeepMind's OpenSpiel `universal_poker` (FCPA abstraction, 100BB HU NL) — an independent engine, so it also catches encoding bugs our own engine would mask. `scripts/eval_neural_vs_tabular.py` pits the two pipelines against each other.

## Steps

1. Tabular vs. random: `uv run python scripts/eval_openspiel.py --hands 2000 [--checkpoint path.pkl]` (latest checkpoint auto-detected).
2. Tabular vs. OpenSpiel CFR baseline: `uv run python scripts/eval_openspiel.py --hands 500 --baseline cfr --cfr-iters 500`.
3. Neural: `uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 2000 --baseline random`.
4. Interpret in BB/100; run enough hands (≥2000) — variance in HU NL is large.

## Gotchas

- **OpenSpiel action 1 is ambiguous:** it means "check" (postflop, no outstanding bet) or "call" (facing a raise). `CFRBotPolicy._is_check_action()` disambiguates using the sequences string and street — reuse it, don't re-derive.
- **Chip scaling (neural):** OpenSpiel chip counts must be divided by `(their_stack / 100)` before hitting the net — this was a confirmed bug (commit `87317c5`), as was an inverted `to_call` computation (`2706697`).
- `CFRBotPolicy` parses OpenSpiel info-state strings to rebuild our `InfoSet` on the fly — format changes on OpenSpiel's side break parsing silently.
- OpenSpiel's `pokerkit_wrapper` prints on import; the suppression lives in `eval_openspiel.py` — keep new imports behind it.

## Verify

- [ ] Win rate vs. uniform random is strongly positive (a competent strategy crushes random).
- [ ] No fold-dominant behavior in the printed action distribution (if available) — near-100% folding signals an encoding/scaling bug, not a bad strategy.
- [ ] Same checkpoint gives consistent sign of result across two seeds/runs.

## Debug

- Large negative BB/100 → go to `patterns/debug-bot-misplay.md`; suspect chip scaling or the check/call mapping before suspecting training.
- Crash while parsing info states → dump the raw info-state string and compare with `CFRBotPolicy`'s parsing assumptions.

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
