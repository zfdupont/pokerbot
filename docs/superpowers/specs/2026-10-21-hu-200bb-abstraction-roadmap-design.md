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
  - An arm is converged when its **LBR trend** has stopped (revised again 2026-10-22, evening). `scripts/arm_plateau.py --config <arm config>` runs LBR (turn and river, the paper's "fcpa 3-4") on every snapshot on the same 100k decks (seed 1), regresses each deck's LBR on iterations over the last 4 snapshots, and t-tests the mean slope (BB/100 per 10M iterations). Converged when the windows ending at the newest two snapshots both have |t| < 2. Check after each sync, locally.
    - Why not a pair of snapshots: LBR moves 10-18 BB/100 between neighbouring snapshots beyond deck noise (C3: 25M vs 15M t=-2.69, 35M vs 25M t=+3.82), so any single pair can read flat on a dip. On the sweep data the one-sided pair rule called C3 converged at 15M despite a significant rise (+22, t=+3.84).
    - Why two windows: one flat window can be a dip between rises (C3's 10-25M window, t=-0.7, sits between two rises). The second window costs one extra 5M snapshot, about $0.03.
    - Rules compared on the sweep (slope over 4 snapshots): residual-error OLS on snapshot means (2 df) missed C0's early rise (t=3.71 < 4.30); an equivalence test (90% CI within +-10 per 10M) never converged; the paired per-deck slope flagged every early rise and called the flat late windows converged (my call).
    - Both directions count: LBR rises early in training for every arm (+16 to +28 BB/100 by 10-15M), so "converged" means stopped changing, not stopped improving.
    - LBR decides when an arm has stopped changing. It does not pick winners: the gates below stay head-to-head plus Slumbot.
    - The pair rules stay available: `--method h2h` (head-to-head, N vs N-lag) and `paired_lbr` in `scripts/eval_sixmax_lbr.py`.
  - **Iteration caps:** C0 is capped at 60M. Every other arm's cap is `C0_plateau × (arm_infosets / C0_infosets)`, at least 20M. The ceiling is 150M for Phase 2 arms and 500M for Phase 3 arms (revised 2026-10-21: the 255/255/255 maximum is about 8.1x C0, and a 150M ceiling would undertrain every map above about 2.5x C0). The budget rules below still apply to every arm.
  - Worked example: if C0 plateaus at 25M with 40k infosets, an arm with 120k infosets is capped at 75M.
  - Infoset counts come from the trained checkpoint (C0) or a 2M-iteration probe run (other arms).
  - An arm that hits its cap is compared at the cap and flagged as possibly undertrained.
  - Arms are always compared at their own plateau or cap, never at a shared iteration count.
- **Budget checkpoint:** after the ccx33 benchmark and C0's plateau are known, re-estimate the whole schedule before launching C1. If it exceeds the $40 budget, cut arms in this order: C3, D1, the C-D interaction sanity arm, then step 3b down to its smaller variant.

### Phase 1.5: P0 (card-map method check), inserted 2026-10-21 (owner-approved)

**Why.** Offline map scoring (`results.md`, "Map scoring") found that equity-percentile flop and turn buckets leave about 15% / 21% of the next-street spread whatever their count (50 to 200 buckets moves it under 1%), while the potential-aware map at the same 50/50/20 counts leaves 12.1% / 12.6%. Phase 2 was going to measure pot layouts on the percentile map, which Phase 3 would likely replace, and pot and card buckets interact in the key.

**Arm.** P0 = C0's config (`hu200.toml`, pot edges 7/15/40) with `map_pot_50_50_20.bin` (`build_bucket_map.py --method potential --counts 50,50,20`, seed 0). Same infosets as C0 (33.8k).
- Trained to 25M with 5M snapshots, the iteration of C0's comparison snapshot (my call: the formula cap is 20M, but comparing at C0's 25M snapshot removes an iteration mismatch for about $0.10). Plateau checks as in 1d.
- **Gate:** local head-to-head, P0 vs C0 at 25M, `|t| >= 2` (gate policy below). No Slumbot run.

**Outcomes.**
- **P0 wins:** the potential map becomes Phase 2's base. C1-C3 use `map_pot_50_50_20.bin`, P0 replaces C0 as the Phase 2 control, and Phase 3's D0 is the potential map. Phase 3's 3a arms are re-planned from the map scores (proposal: drop DF and DR; P1 = potential 50/50/50), and 3c folds into 3a.
- **P0 does not win:** Phase 2 runs as written on the percentile map, and the map-score evidence is recorded as not translating into play at these counts.

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

**Revised 2026-10-21 (owner-approved), replacing the D1/D2/D3 ladder.** The ladder scaled every street together, so it could not say which street's buckets are worth their cost.

**Evidence behind the revision.**
- **Infosets are betting contexts × card buckets, per street.** Measured on the C0 5M snapshot (`hu200_c0_05000000.bin`, `sixmax.dump_infosets`): preflop 3,042 infosets (18 contexts × 169), flop 73 contexts, turn 242, river 753. So at C0's pot layout `infosets ≈ 3,042 + 73·F + 242·T + 753·R` (predicts 33,852 for 50/50/20; measured 33,831). A river bucket costs about 10x a flop bucket.
- **Rare river infosets are already undertrained:** the river's 10th-percentile visit mass is about 10x below the other streets'.
- **Hard ceiling: 255 buckets per street.** Bucket ids are `uint8` in SIXBM001 (`kMaxBucketCount`) and the key's card field is bits 0-7. Libratus-scale maps need a new map format, a new key layout and about 10^9 infosets; out of scope.
- **Published blueprints** (`docs/papers/`): Libratus used no card abstraction preflop or on the flop, 2.5M turn and 1.25M river buckets (Science 2018). Pluribus used lossless preflop and 200 k-means buckets per later street, and 500 per street inside search (Science 2019 supplementary). Pluribus scale is reachable here; Libratus scale is not.
- **Equity percentiles ignore potential.** On the flop and turn, a made hand and a draw with equal equity share a bucket. The river has no cards to come, so river equity is close to a sufficient statistic and extra river buckets have the weakest case on paper.

**Cost model.** Re-measure the context counts on the Phase 2 winner's checkpoint before building maps: a finer pot layout multiplies contexts (C2's 8 buckets roughly doubles them) and every number below scales with it. Estimates at C0's layout, assuming C0 plateaus at 60M and iterations scale with infosets:

| Map (F/T/R) | Infosets | × C0 | Iterations | About |
|---|---:|---:|---:|---:|
| 50/50/20 | 33.8k | 1.0 | 60M | $1 |
| 100/100/50 | 72k | 2.1 | 126M | $2.3 |
| 200/200/20 | 81k | 2.4 | 144M | $2.6 |
| 50/50/100 | 94k | 2.8 | 168M | $3.0 |
| 200/200/200 | 217k | 6.4 | 384M | $7 |
| 255/255/255 | 275k | 8.1 | 486M | $9 |

**3-pre. Offline map scoring (free, before any D training; my call on the metric and threshold).**
- Score each candidate map by its multiplicity-weighted within-bucket spread: equity variance on the river, and the earth mover's distance between each hand's next-street equity histogram and its bucket's mean histogram on the flop and turn.
- Each candidate is compared, per street, with the next-coarser candidate on that street. A step's efficiency is the share of the street's one-bucket spread it removes per extra infoset (contexts × extra buckets). A step below half the best step's efficiency is dropped, and a map with no kept step is dropped without training. Revised 2026-10-21 from a flat 10% relative gain: river percentile buckets halve their spread whenever their count doubles, so a relative rule never drops a river candidate, while each river bucket costs about 10x a flop bucket. Tool: `scripts/score_bucket_map.py`. Record every score in `results.md`.

**3a. Where to spend buckets** (percentile maps, at the Phase 2 winner):

| Arm | Map | Question |
|---|---|---|
| D0 | pct 50/50/20 (the Phase 2 winner, no retrain) | control |
| DF | pct 200/200/20 | do flop and turn buckets pay? (cheap direction, Libratus-like) |
| DR | pct 50/50/100 | do river buckets pay? (where the 100+ BB pots resolve) |
| D1 | pct 100/100/50 | balanced reference |

**3b. Scale the 3a winner** (one arm):
- DF wins: 255/255/20. DR wins: 50/50/255. D1 wins: 200/200/200 (Pluribus counts). Pick the largest variant that fits the budget.
- If no 3a arm beats D0, skip 3b: counts are not the binding constraint, and 3c runs at 50/50/20.

**3c. Bucketing method:** `--method potential` (k-means over next-street equity histograms on the flop and turn, river percentile) at the 3b counts, against the percentile map at the same counts.

**Gate 3:** the gate policy below, applied per stage: each 3a arm against D0, the 3b arm against the 3a winner, 3c against 3b. Then one sanity arm at the runner-up C with the final D map, to check that C and D don't interact strongly.

### Phase 4: B (facing-bet size and raise depth)

- Add a facing-bet-size or pot-odds field to the key, and raise the per-street raise cap above 3 (more bits).
- Version-bump as in Phase 2.
- One arm against the Phase 3 winner, under the gate policy below.
- The field's bucketing is left to the plan, informed by the R²=0.32 measurement method from the leak-2 analysis.

#### Phase 4a: C5, last raise size per street (owner-approved 2026-10-23, run ahead of Phase 3)

Why now: the C2/C4 head-to-head split showed the key cannot tell open sizes apart. After a 2.5/3.5/5 BB open the pot is 3.5/4.5/6.0 BB, all one pot bucket with the same raise count, so the big blind answers every open with one strategy (C4 folds 58/56/56%). The published bots key on the full action sequence (Pluribus: 664,845,654 sequences, 1-14 raise sizes; supplementary p. 12-13); this is the cheap partial version.

- **Field:** the size class of the last bet or raise on each of the 4 streets, 3 bits per street in key bits 30-41. Class 0 = no raise on that street, 1-3 = the raise's rank (smallest first) among the vocabulary's sizes of the same unit (the BB-unit opens or the pot-fraction sizes), 4 = all-in. A vocabulary with more than 3 sizes in one unit is refused while the field is on. (Owner: every street, not only the current one, so the preflop open size stays visible postflop.)
- **Opt-in:** `[abstraction] bet_history = "off" | "last_raise"` (default `"off"`). Off keeps every key, hash and checkpoint byte unchanged. On mixes a tag into the abstraction hash and writes a new magic, `SIXBP004` = SIXBP003 plus an int32 bet-history mode after the pot layout; it requires a bucket map, like SIXBP003. A resume or load with a different setting is refused.
- **One source of truth:** the trainer reads the class from the vocab index it applies. The deployed agent replays the live hand into an `EngineGameState` (the search path's replay, mapping each live bet to the nearest menu action) and reads the same classes from it, then packs the key through the same `Abstraction.pack_key`. The nearest-bucket fallback keeps working (it only varies the pot field).
- **Evaluators:** the Slumbot stand-in state must carry the action history the replay needs. The openpoker bridge, which calls `decide` without a hand history, refuses bet-history checkpoints. Search already refuses non-legacy layouts.
- **Arm C5:** C4 (`hu200_c4.toml`: pot edges 3..192, opens 2/3.5/5 BB, map pct 50/50/20) plus `bet_history = "last_raise"`. One change against C4. Infosets are measured by a 2M-iteration local probe before any cloud spend; the cap follows the usual formula.
- **Gates:** the LBR trend rule; head-to-head against C4 (same menu, so no translation confound); Slumbot 13k hands.
- **Out of scope here:** pseudo-harmonic translation (it slots into the replay's bet mapping later) and full action sequences.

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
  - Revised 2026-10-21 from the measured 3,935 it/s benchmark and the Phase 3 cost model: Phase 2 about $6, Phase 3 about $22 (3a about $8, 3b about $7, 3c about $7), about **$28** in total before Phase 4. Re-estimate at the budget checkpoint and again after Gate 2, when the winner's context counts are known.
  - Ask before exceeding **$40 total**, or before any single arm exceeds 2x its benchmark-based estimate.
- **Slumbot:** about 13k hands per gate × 3-4 gates, plus 50k final, all local and free. At most one evaluator process at a time.
- **Wall time:** serial, so roughly 4-6 days of machine time across the phases.

## Left to the builder

- Exact key bit positions and the version mechanism, consistent with the SIXBP002 pattern.
- The B field's bucketing.
- How `cloud_train.sh` carries the map: an upload flag, or baking it into a volume.
- Head-to-head hands and seeds per comparison: enough for SE ≤ 5 BB/100, measured from the first comparison.
- Whether C3's stack-to-pot ratio uses effective stack at the street start or at the decision.
