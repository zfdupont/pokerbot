# Heads-Up 200 BB Abstraction Roadmap (Leaks A-D) - Design

**Date:** 2026-10-21
**Status:** Draft, awaiting approval
**Scope:** `sixmax/` heads-up blueprint only
**Predecessor:** `docs/superpowers/specs/2026-10-20-sixmax-equity-lookup-tables-design.md` (leak A tooling, built on branch `feature/equity-lookup-tables`)

## Intent

Make the heads-up blueprint strong enough that its owner cannot beat it and it holds its own against a recognised strong bot. The yardstick is **Slumbot** (ACPC 2018 winner, public API, heads-up no-limit, 200 BB stacks).

- **First target:** Slumbot baseline score **at or above -20 BB/100**.
- **Starting point:** **-118 BB/100, ±42 (1 SE)**. Measured 2026-10-21 over 3,000 hands with `hu_blueprint_50000000.bin`, which was trained at 100 BB and played at 200 BB.

This document fixes the order in which the four abstraction leaks are addressed, how each is decided, and the gates between them.

## The leaks

| | Leak | Today |
|---|---|---|
| A | Card-bucket noise | Monte Carlo rollouts (100 per hand) put only 16% of flop/turn hands in the correct bucket. **Fix already built:** exact equity table plus percentile map (`equity_v1.bin`, `map_pct_50_50_20.bin`, SIXBP002). |
| B | Bet-size aliasing | The key has no facing-bet size or pot odds (measured R²=0.32). The raise count is capped at 3 per street. |
| C | Coarse pot buckets | 2 bits: ≤7, ≤15, ≤40, >40 BB (`abstract_key.cpp` `pot_bucket`). At 200 BB, every pot from 40 to 400 BB is one bucket. |
| D | Too few card buckets | 50/50/20 (flop/turn/river). The 8-bit field allows up to 255. |

## Evidence that sets the order

From the 3,000-hand Slumbot run (baseline score, BB, summed):

| Hand size (BB won or lost) | Hands | Result |
|---|---|---|
| < 10 | 2,692 | +789 |
| 10-40 | 243 | +57 |
| 40-100 | 31 | -405 |
| 100+ | 34 | -3,989 |

- 3-bet-or-more pots: 404 hands, -2,361.
- Showdowns: 290 hands, -4,438. Every fold-ended category was positive.
- The big blind folds 48% of the time to an open.

**Reading:** the deficit lives in big and re-raised pots, which is where 200 BB stacks, C and B matter. Small pots are already at or above break-even.

**Variance:** 23 BB per hand standard deviation. Slumbot's `baseline_winnings` did not reduce it (22.6 raw vs 23.1 baseline), because our play does not resemble Slumbot's yet. That works out to:

- ±20 BB/100 at about 13k hands (about 3.5 h at the measured 0.9 s per hand);
- ±10 BB/100 at about 50k hands.

## Decisions made by the owner

1. Train at **200 BB** from now on. Slumbot is the gate.
2. The first 200 BB runs change stack depth **and** pot buckets (C) together, using the leak A map.
3. **Ablate C and D** to find good values, rather than picking them once.
4. Cloud training runs **serially on one Hetzner ccx33.** The account cap is 8 dedicated cores, and it will not be raised. Measured: about 850 it/s with the legacy trainer, about $0.21-0.24/h.
5. **Finish leak A's branch first.**

## Goals

- A repeatable path from today's -118 to ≥ -20 BB/100 against Slumbot, with every step measured.
- Ablation winners chosen on a low-noise local metric. Slumbot only confirms.
- Checkpoints never silently invalidated: every key-layout change is versioned.

## Non-goals

- Six-max, search (stays `enabled=false`), DREAM.
- Bet-size vocabulary changes (G).
- Perfect recall (F).
- Promoting a checkpoint to the web service.
- Changing existing 100 BB checkpoints. They stay loadable as they are.

## Anti-goals (a failure even if it technically works)

- **Hammering Slumbot:** parallel requests, or unattended runs beyond the hand counts below. One request at a time, always.
- **Choosing an ablation winner from a Slumbot number** whose error bar overlaps the runner-up.
- **Comparing arms of different abstraction size at equal iterations.** The bigger arm loses because it is undertrained, not because it is worse.
- A key-layout or map change that loads old checkpoints without error but reads them wrong.
- Committing tables, checkpoints, `.env` or Slumbot credentials.
- Cloud spend beyond the budget below without asking.

## Order and gates

### Phase 0: Finish leak A (branch `feature/equity-lookup-tables`)

- Tasks 1-5 of the equity plan are done.
- **Task 6** (a 100 BB training run gated on HU 20M) is **superseded** by this roadmap. Record that in the SDD ledger and the plan; do not train it.
- Run the final whole-branch review and one fix wave, then integrate to `main`. The owner confirms the merge.
- On merge, CI rebuilds `ghcr.io/zfdupont/pokerbot-trainer:latest` with map support.
- The equity table stays local. Maps are built locally from it and uploaded per arm.

### Phase 1: Infrastructure (no training yet)

**1a. Slumbot evaluator.** `scripts/eval_slumbot.py`, promoted from the spike at `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/slumbot_spike.py`. The baseline run's per-hand log sits beside it as `slumbot_hu50m_3000.jsonl`. The spike's design is proven: it replays the action string into the stand-in state `SixmaxAgent.get_action` reads, translates our actions exactly, and falls back when Slumbot rejects a size.
- **No inbound bet translation exists, by design.** The agent reads Slumbot's real chip amounts. Slumbot's off-tree sizes reach the key only through the pot bucket and raise count, which is leak B and is fixed in Phase 4, not in the evaluator. Our outbound bets are sent as exact chip amounts (`bN` = street commitment).
- **Freeze this behaviour (my call).** Every gate's number must be comparable to the -118 baseline. Any change to parsing or translation must re-run `hu_blueprint_50000000.bin` and agree with -118 within 2 SE. The evaluator:
- Plays strictly sequentially and plays an even number of hands. Sessions start in the big blind.
- Is resumable: it appends JSONL per hand and continues where a log ends.
- Reports raw and baseline BB/100 with SE, by position.
- Runs the breakdown tables above (hand-size bins, 3-bet pots, showdown vs fold).
- Takes optional login only from env vars (`SLUMBOT_USERNAME`, `SLUMBOT_PASSWORD`). Login is only needed for the leaderboard. Never logged, never committed.
- Uses the stdlib `urllib` only, with no new dependency.
- Ports the spike's behaviour to the cases it does not handle: Slumbot folding before we act, and a 'b' that is an all-in for less than a min-raise.
- Gets unit tests for the action-string parser (including all-in `b20000c///`, check-through streets, multi-raise) and the action translation, using no network.

**1b. 200 BB training.**
- `starting_stack = 200.0` in a new `sixmax/configs/hu200.toml`.
- Verify that every path that reads the stack (trainer, engine bet grid and all-in, `SixmaxAgent` legal mask and all-in target, `eval_blueprint_h2h.py` `--stack`/`--bb`, the Slumbot evaluator) honours 200 BB. Fixing any hard-coded 100 found here is in scope.
- Amend the invariant "rescaled to the `starting_stack=100, big_blind=1` training frame". The BB unit stays. The stack is whatever the checkpoint's config says, and `sixmax/` hosts must read it from there. Update `.mex/AGENTS.md` and `CLAUDE.md` together.

**1c. Cloud arm runner.**
- Extend `scripts/cloud_train.sh` to upload a named bucket map with the run and set `SIXMAX_BUCKET_MAP` in the container. SIXBP002 stores the map basename as its hint, and resolution already supports env, checkpoint dir and tables dir.
- Synced checkpoints must come back next to a copy of the map, or the map must be resolvable locally by hash.
- Log it/s at intervals across the whole run, not just as an average. That guards against a repeat of the bucket-cache throughput cliff (`2026-09-29-bounded-bucket-cache-design.md`).
- First use: a 10-minute ccx33 benchmark of the map trainer at 200 BB, which sets every arm's time and cost estimate.

**1d. Local ranking and plateau check.**
- Use `scripts/eval_blueprint_h2h.py` (duplicate deals, shared random numbers, `|t| >= 2` gate per `.mex/patterns/eval-checkpoint-ab.md`) at 200 BB.
- **Plateau rule (my call):**
  - Snapshots every 5M iterations.
  - An arm is converged at snapshot N when N does not beat snapshot N-10M head-to-head at `|t| >= 2`. Check after each sync, locally, while training continues.
  - **Iteration caps:** C0 is capped at 60M. Every other arm's cap is `C0_plateau × (arm_infosets / C0_infosets)`, at least 20M and at most 150M.
  - Worked example: if C0 plateaus at 25M with 40k infosets, an arm with 120k infosets is capped at 75M.
  - Infoset counts come from the trained checkpoint (C0) or a 2M-iteration probe run (other arms).
  - An arm that hits its cap is compared at the cap and flagged as possibly undertrained.
  - Arms are always compared at their own plateau or cap, never at a shared iteration count.
- **Budget checkpoint:** after the ccx33 benchmark and C0's plateau are known, re-estimate the whole schedule before launching C1. If it exceeds the $40 budget, cut arms in this order: C3, D2, the C-D interaction sanity arm.

### Phase 2: C (pot buckets), with A in, at 200 BB

**Key change and versioning (my call).**
- The pot field moves to free key bits (26-63 are unused) and widens to 4 bits.
- Boundaries become config-defined: `[abstraction] pot_mode = "bb" | "spr"` and `pot_edges = [...]` (BB upper bounds, or stack-to-pot ratio upper bounds).
- **A new checkpoint version, SIXBP003,** stores the key layout version, `pot_mode` and `pot_edges` alongside the map hash. They are hashed into the abstraction identity.
- **Consumers read them from the checkpoint, not from a TOML:** `SixmaxDeployStrategy`, `eval_blueprint_h2h.py`, the Slumbot evaluator and resume. A mismatch on resume is refused with a clear error, like "map hash mismatch".
- SIXBP001/002 files load explicitly as the legacy layout (pot at bits 18-19, edges 7/15/40), because those formats predate the field. The version is implied by the magic, not guessed.
- The legacy layout must stay byte-identical for old checkpoints. Golden test, as in Task 4.
- New configs are not bound by the `hu.toml` comment about matching `default.toml`. That compatibility applies to the 100 BB deploy path only.

**Arms** (all with `map_pct_50_50_20.bin`, 200 BB, current vocab, same seeds; boundaries are my call):

| Arm | Pot edges (BB, upper bounds) | Buckets |
|---|---|---|
| C0 | 7, 15, 40 | 4 (today's layout at 200 BB; also the "stack change alone" reference) |
| C1 | 4, 8, 16, 40, 100 | 6 |
| C2 | 3, 6, 12, 24, 48, 96, 192 | 8 |
| C3 | `pot_mode="spr"`, edges 0.5, 1, 2, 4, 8 (effective stack behind / pot, at the decision) | 6 |

**Gate 2:** see "Gate policy" below. Expected: a large move from -118.

### Phase 3: D (card buckets), at the Phase 2 winner

Build the maps locally from `equity_v1.bin` (percentile about 30 s, potential-aware about 5 min) and upload them.

| Arm | Map |
|---|---|
| D0 | pct 50/50/20 (the Phase 2 winner, no retrain) |
| D1 | pct 100/100/50 |
| D2 | pct 200/200/100 |
| D3 | potential-aware at the better of D1/D2 |

**Gate 3:** the gate policy below. Add one sanity arm at the runner-up C with the D winner, to check that C and D don't interact strongly.

### Phase 4: B (facing-bet size and raise depth)

- Add a facing-bet-size or pot-odds field to the key, and raise the per-street raise cap above 3 (more bits).
- Version-bump as in Phase 2.
- One arm against the Phase 3 winner, under the gate policy below.
- The field's bucketing is left to the plan, informed by the R²=0.32 measurement method from the leak-2 analysis.

### Gate policy (applies to Phases 2-4; my call)

- **Local head-to-head picks the winner.** The challenger must beat the phase's control (C0, D0, or the Phase 3 winner for B) at `|t| >= 2`, each at its plateau or cap. If none does, the control wins. That is a legitimate result, recorded, and the roadmap moves on.
- **Slumbot is a regression check, not a tie-breaker.** At 13k hands (±20 BB/100) it cannot separate arms. It answers "did this phase help against a strong outside opponent?":
  - **Pass:** the phase winner's Slumbot score is not worse than the previous phase's confirmed score by more than 2 combined SE.
  - **Block:** if it is worse by more than 2 combined SE, local head-to-head and Slumbot disagree. Stop, run the evaluator's breakdown on both logs, and bring it to the owner before the next phase.
- **No-gain phase:** if a phase's control wins and Slumbot is flat, continue to the next phase. If two phases in a row give no gain, stop and bring the evidence to the owner. The leak ranking may be wrong.
- **Phase 2 extra stop:** if the confirmed number has not improved by at least 2 SE over -118, diagnose before Phase 3.

### Phase 5: Final measurement

- Slumbot at 50k hands (about 14 h, sequential, ±10 BB/100) on the best checkpoint.
- **Pass:** at or above -20 BB/100.
- If the gap is under 1 SE and we're short, an AIVAT-style variance reducer is the next tool, not more hands.

## Budget

- **Training:** about 115 ccx33 machine-hours estimated (C about 40, D about 55, evals about 20). That's **about $25-30** at the legacy rate, likely less at map speed.
  - Ask before exceeding **$40 total**, or before any single arm exceeds 2x its benchmark-based estimate.
- **Slumbot:** about 13k hands per gate × 3-4 gates, plus 50k final, all local and free. At most one evaluator process at a time.
- **Wall time:** serial, so roughly 4-6 days of machine time across the phases.

## Left to the builder

- Exact key bit positions and the version mechanism, consistent with the SIXBP002 pattern.
- The B field's bucketing.
- How `cloud_train.sh` carries the map: an upload flag, or baking it into a volume.
- Head-to-head hands and seeds per comparison: enough for SE ≤ 5 BB/100, measured from the first comparison.
- Whether C3's stack-to-pot ratio uses effective stack at the street start or at the decision.
