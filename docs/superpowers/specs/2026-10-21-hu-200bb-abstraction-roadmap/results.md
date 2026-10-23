# HU 200 BB roadmap results

| Date | Step | Arm | Iterations | Infosets | it/s | H2H | Slumbot BB/100 | Cost |
|------|------|-----|-----------:|---------:|-----:|-----|----------------|-----:|
| 2026-10-21 | Local benchmark (M-series laptop) | C0 hu200.toml | 600,000 | 33,647 | 939 cumulative (994/999/841 per 200k window) | - | - | $0 |
| 2026-10-21 | ccx33 benchmark (8 dedicated vCPU) | C0 hu200.toml | 600,000 | 33,637 | 3,935 | - | - | $0.01 (351 s incl. setup) |
| 2026-10-21 | Slumbot freeze check (guest) | hu_blueprint_50000000.bin, hu.toml | 50,000,000 | - | - | - | raw -107.2 ±42.5, baseline -95.1 ±41.1 (3,000 hands); PASS vs -118.3 ±42 | $0 |
| 2026-10-21 | C0 training (ccx33), stopped at plateau | C0 hu200.toml, pct 50/50/20 | 29.3M run; snapshots to 25M | 33,848 | 3,542 | plateau: 15M vs 5M +6.86 (t=1.71, converged); 20M vs 10M -2.03 (t=-1.04, converged) | - | $0.49 (2.35 h) |
| 2026-10-22 | P0 training (ccx33) + gate | P0 hu200.toml, potential 50/50/20 | 25M | 33,850 | 3,515 | gate P0@25M vs C0@25M: +2.68 ±3.0 (t=0.91), keep C0; plateau 25M vs 15M: -7.21 ±3.1 (t=-2.31), converged | - | $0.42 (2.0 h) |

Estimates at 3,935 it/s and about $0.22/h (before larger arms grow the infoset table): C0 to the 60M cap is about 4.2 h, about $1. Six to eight arms at that scale is under $10, well inside the $40 budget. Re-estimate after C0 shows how it/s holds as infosets grow.

## C0 plateau

First converged snapshot: 15M (t=1.71, borderline; 7 of 8 seeds positive). Confirmed at 20M vs 10M (t=-1.04). Run stopped at about 29M; snapshots 5M-25M are local. For arm caps use C0_plateau = 20M (my call: the confirmed point, not the borderline one; costs about $0.3 more per arm than 15M). Gate 2 compares C0 at its 25M snapshot, the most-trained converged checkpoint.

## Map scoring (3-pre), 2026-10-21

`scripts/score_bucket_map.py`, local, free. Spread left = within-bucket spread / one-bucket spread (lower is better). Contexts 73/242/753 (C0 layout).

| Map | Flop | Turn | River | Verdict |
|---|---:|---:|---:|---|
| pct 50/50/20 (D0) | 15.4% | 21.0% | 4.9% | reference |
| pct 100/100/50 (D1) | 15.1% | 20.8% | 2.0% | keep (river step is the best step) |
| pct 200/200/20 (DF) | 15.0% | 20.8% | 4.9% | drop (flop 9%, turn 2% of best) |
| pct 50/50/100 (DR) | 15.4% | 21.0% | 1.0% | drop (river step 21% of best) |
| potential 50/50/20 | 12.1% | 12.6% | 4.9% | same infosets as D0 |

Equity-percentile flop/turn buckets plateau near 15%/21% whatever the count; potential-aware clustering at equal counts cuts that by 21%/40%. Raw rows: `map_scores_3a.json`.

## Phase 1.5 result (P0), 2026-10-22

P0 (potential-aware 50/50/20) did not beat C0 at 25M: +2.68 ±3.0 BB/100, t=0.91 (per seed +16.6, +2.9, -5.7, +5.2, -0.3, +10.9, +0.9, -9.1). Per the spec, C0 stays the control and Phase 2 runs on the percentile map. The map-score gain (flop/turn spread left 15.4/21.0% -> 12.1/12.6%) did not translate into a measurable head-to-head edge at these counts.

Note: P0's own 25M snapshot lost to its 15M snapshot (-7.21, t=-2.31). Head-to-head between two near-converged strategies is not monotone in training (a less-equilibrium strategy can beat a closer one), so the plateau rule reads this as converged, not as training making P0 worse.

## Paper LBR (Lisy & Bowling 2017), 2026-10-22

`scripts/eval_sixmax_lbr.py` (branch feature/lbr-paper). LBR acts on turn and river with the checkpoint's own bet sizes and checks or calls before that. 4,000 duplicate decks, seed 1, 200 BB, about 19 min each when measured (12 s after the evaluator speedup below), local and free. The opponent unseen-infoset rate was 0.00% for all four. Raw output: `lbr/`.

| Checkpoint | LBR BB/100 (95% CI) | mBB/h | Seat 0 (SB) | Seat 1 (BB) |
|---|---:|---:|---:|---:|
| C0 @ 15M | +425.7 ±48.4 | 4,257 | +494.8 | +356.7 |
| C0 @ 25M | +425.0 ±50.2 | 4,250 | +487.9 | +362.1 |
| P0 @ 15M | +398.1 ±53.6 | 3,981 | +487.9 | +308.3 |
| P0 @ 25M | +417.7 ±52.0 | 4,177 | +554.0 | +281.4 |

- C0 at 15M and 25M agree within 1 BB/100: C0 had converged by 15M, which matches the head-to-head plateau checks.
- P0 vs C0 is not significant: the differences are 7-28 BB/100 against about ±70 combined. This matches the head-to-head gate (+2.68, t=0.91). The four runs share decks (same seed), so a paired per-deck comparison would be tighter, but the driver does not save per-deck values yet.
- For scale, the paper reports Slumbot 2016 at about 4,020 mBB/h with LBR acting on turn and river (its "fcpa 3-4" row). By this probe these blueprints are about as exploitable as Slumbot was, even though Slumbot beats them by about 100 BB/100 head-to-head.

Speedup (2026-10-22, before merge): the shared 7-card evaluator was rewritten allocation-free (60 ns vs ~14.5 us per hand, bit-identical scores over all 133,784,560 hands) and RangeTracker::observe memoises by infoset key. 4,000 decks: ~19 min -> 12 s on 8 local cores, with byte-identical output (C0@25M still +425.00 +-50.17). Acting on the flop as well (`--act-streets 1,2,3`) now costs about 2 min per 4,000 decks.

The same evaluator speeds up blueprint training, which calls it at every showdown: hu200 + map_pct_50_50_20, 300k iterations on 8 local threads, 947 -> 4,236 it/s (4.5x). A 20k-iteration single-thread run with seed 7 writes a byte-identical checkpoint on both builds. Cloud (ccx33) rates measured before this change (about 3,500 it/s) are stale; re-measure on the next run before estimating cost.

## Paired LBR plateau check, 2026-10-22

`scripts/arm_plateau.py` (now LBR by default), 25M vs 15M, 40,000 paired decks, seed 1, about 2.7 min each:

| Arm | LBR 25M | LBR 15M | 25M - 15M | se | t | Verdict |
|---|---:|---:|---:|---:|---:|---|
| C0 | +413.6 | +422.3 | -8.68 | 8.01 | -1.08 | converged |
| P0 | +404.3 | +400.8 | +3.41 | 8.57 | +0.40 | converged |

Pairing helps less than hoped. The per-deck sd of the difference (1,602 and 1,714) is close to a single run's (about 1,620), so the two snapshots' results on a deck correlate only about 0.5: after their first differing action the random streams diverge. Pairing is worth about 2x the decks. The default is therefore 100k decks (se about 5 BB/100, about 7 min).


## C1 (pot edges 4/8/16/40/100 BB), 2026-10-22

- Probe (2M iterations, local): 46,507 infosets, 1.39x C0's 33,572 at 25M. Cap = 20M x 46,507 / 33,572 = 27,705,826.
- Cloud: ccx33, CI trainer image with the new evaluator. 27,705,826 iterations at 12,827 it/s average (window 12,100-13,400), 38 min, about $0.13. Snapshots every 5M plus the cap.
- Plateau (paired LBR, 100k decks, seed 1): 20M vs 10M, -1.05 BB/100, se 5.61, t=-0.19, converged. C1_plateau = 20M. LBR(20M) +391.6, LBR(10M) +392.6 BB/100.
- ccx33 is now about 3.7x the pre-speedup 3,500 it/s; the Phase 2-3 cost table should use about 12,800 it/s.

## C2 (pot edges 3/6/12/24/48/96/192 BB), 2026-10-22

- Probe (2M iterations, local): 64,589 infosets, 1.92x C0. Cap = 20M x 64,589 / 33,572 = 38,477,898.
- Cloud: ccx33, 38,477,898 iterations at 10,584 it/s average (the larger table is slower than C1's 12,827), 63 min, about $0.22. 65,016 infosets at the cap.
- Plateau (paired LBR, 100k decks, seed 1): 15M vs 5M, -4.24 BB/100, se 6.25, t=-0.68, converged at the first check. C2_plateau = 15M. LBR(15M) +363.2, LBR(5M) +367.5 BB/100.
- LBR at plateau falls as pot buckets get finer: C0 (4) about +414-422, C1 (6) +391.6, C2 (8) +363.2. These are separate runs, not paired; Gate 2's head-to-head decides strength.

## LBR learning curves (sweep), 2026-10-22

`scripts/lbr_sweep.py`: LBR (turn+river) on every snapshot of every arm, 100k decks, seed 1, so every point is played on the same cards and any two can be compared paired. Rows: `ablations/lbr_sweep.jsonl`; per-deck values: `sixmax/checkpoints/lbr_per_deck/` (gitignored, regenerable). Per-arm facts: `ablations/arms.json`.

| Snapshot | C0 | P0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|
| 5M | 389.2 | 381.1 | 376.7 | 367.5 | 365.6 |
| 10M | 400.7 | 396.8 | 392.6 | 369.2 | 379.4 |
| 15M | 416.8 | 405.2 | 395.0 | 363.2 | 387.8 |
| 20M | 415.5 | 400.5 | 391.6 | 384.4 | 390.3 |
| 25M | 409.4 | 408.7 | 398.1 | 381.7 | 374.2 |
| cap / later | - | - | 392.3 (27.7M) | 380.1 (38.5M) | 391.2 (42.5M) |

(BB/100, each ±about 10 at 95%; paired differences have se about 5-6.)

- Every arm gets more exploitable early in training: 15-20M minus 5M is +18 to +28 (t 3-5) for C0, C1, C3; P0 +16 by 10M; C2 is flat to 15M and rises after (+17 at 20M). CFR converges in the abstract game, which need not lower real-game exploitability (or LBR, a local lower bound, may punish a sharper strategy more); either way more iterations is not monotonically better by LBR.
- Snapshot-to-snapshot LBR moves by 10-18 BB/100 beyond deck noise (C3: 25M vs 15M -13.6, t=-2.69; 35M vs 25M +17.8, t=+3.82). A single-pair plateau test is fragile at this level: the one-sided rule called C3 converged at 15M despite a significant rise (+22, t=+3.84), and a two-sided rule flips between converged and moving from snapshot to snapshot. A trend test over the last 3-4 snapshots would be sounder (proposal, not adopted).
- Cross-arm, paired, each arm at its cap vs C0@25M: C2 -29.3 (se 6.3, t=-4.62), C3 -18.1 (t=-2.91), C1 -17.1 (t=-2.74), P0 -0.7 (t=-0.11). C3 vs C2 at caps: +11.1 (t=+1.83). Finer pot layouts are significantly less exploitable; SPR (C3) matches C1 with 54% more infosets; the potential card map does not move LBR.

## Engine bug: head-to-head results before 2026-10-22 evening are invalid

`game/poker.py` `_reset_street_bets` ran before `betting_round` advanced, so it zeroed the raise count of the street that had just ended (since 2026-06-25, commit 5c0f15d). Every postflop decision in the live engine saw 0 raises for the earlier streets, unlike the trainer's keys. Measured on C1 vs C0: SixmaxAgent missed its own table on 24-43% of postflop decisions (uniform fallback), and some lookups silently aliased onto other infosets. After the fix (commit d8097b7): 0-4%.

- Invalid: every live-engine head-to-head (eval_blueprint_h2h / arm_plateau --method h2h): the C0 plateau h2h checks, the P0 gate (+2.68), and the first Gate 2 run (kept as `ablations/gate2_h2h_INVALID_raise_reset_bug.jsonl`). Also cfr_agent / neural_agent play through the live engine, and the deployed web bot.
- Unaffected: all LBR numbers (trainer key path, 0.00% opponent misses) and Slumbot (eval_slumbot builds its own state from Slumbot's action string).

## Gate 2 (fixed engine), 2026-10-22

Each arm at its cap (P0 at 25M) vs C0@25M, eval_blueprint_h2h, 8 seeds x 20k duplicate hands, 200 BB. Rows: `ablations/gate2_h2h.jsonl`.

| Arm | H2H vs C0 BB/100 | se | t | Gate | LBR vs C0 (paired) | Infosets |
|---|---:|---:|---:|---|---:|---:|
| C2 (bb 3..192, 8 pot buckets) | +23.66 | 2.81 | +8.43 | pass | -29.3 (t -4.62) | 65,016 |
| C3 (SPR, 6) | +15.51 | 2.55 | +6.08 | pass | -18.1 (t -2.91) | 71,916 |
| C1 (bb 4..100, 6) | +14.04 | 3.53 | +3.98 | pass | -17.1 (t -2.74) | 46,795 |
| P0 (potential card map) | +6.76 | 3.53 | +1.92 | fail (just) | -0.7 (t -0.11) | 33,850 |

- Head-to-head and LBR now rank the arms the same way: C2 > C3 ~ C1 > P0 ~ C0.
- Phase 2 winner on the local gate: C2. Next per spec: C2 to Slumbot at 13k hands.
- C3's SPR layout costs 54% more infosets than C1 for the same result.
- P0's corrected gate is borderline (+6.76, t=1.92; LBR no difference). Keep C0's map per the rule, but the card-map question stays open for Phase 3.


## Plateau rule: LBR trend test (spec 1d revised), 2026-10-22

`arm_plateau.py --method lbr` now t-tests the paired LBR slope over the last 4 snapshots and needs the last two windows flat (|t| < 2). On the cached sweep values (100k decks, seed 1):

| Arm | Window | Slope BB/100 per 10M | se | t | Window | Slope | se | t | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| C0 | 5-20M | +19.01 | 3.88 | +4.90 | 10-25M | +4.93 | 3.65 | +1.35 | training (snapshots end at 25M) |
| P0 | 5-20M | +13.31 | 4.17 | +3.19 | 10-25M | +6.15 | 3.86 | +1.59 | training (snapshots end at 25M) |
| C1 | 10-25M | +2.57 | 3.85 | +0.67 | 15-27.7M | +0.01 | 4.27 | +0.00 | converged |
| C2 | 20-35M | -1.72 | 3.58 | -0.48 | 25-38.5M | +1.03 | 3.83 | +0.27 | converged |
| C3 | 25-40M | +12.84 | 3.31 | +3.88 | 30-42.5M | +6.44 | 3.80 | +1.70 | training at cap: flagged possibly undertrained |

- C0 and P0 read "training" only because their last snapshot is 25M (the saved final checkpoints are also 25M), so the early-rise window is still one of the two. One more 5M snapshot each (`--resume`, about $0.03 each) would settle it.
- Gate 2's ranking does not depend on this: C2 is converged by either rule, and C0 at 25M sits on its flat stretch (15-25M within 7 BB/100).

Ruling (2026-10-23): C0 and P0 are treated as converged at 25M without a second trend window. C2 (about 2x their infosets) passed the trend rule by 35M, so about 18M is expected for them; their 10-25M windows are flat (t=+1.35, +1.59) and C0's 15-25M LBR spans 7 BB/100; and no extra training could plausibly close C2's +23.7 (t=8.4) head-to-head lead. Cost if wrong: none for the Phase 2 decision; verify with a 5M `--resume` (about $0.03 each) if either becomes a finalist.

## Slumbot: C2 (Phase 2 winner), 2026-10-23

`scripts/eval_slumbot.py --checkpoint hu200_c2_38477898.bin --config hu200_c2.toml --hands 13000`, guest, sequential (about 3.24 requests per hand, 1-4 per second, no added delay). Log: `slumbot_c2_38m_13000.jsonl`. The evaluator's +/- is one standard error.

- raw **-50.5 +/- 21.2** BB/100 (95%: -92 to -9); baseline -40.0 +/- 21.9. Seats 6,500/6,500; baseline by seat BB -44.5, SB -35.4.
- Regression check vs the previous confirmed score (-118.3 +/- 42): +67.8 better, so PASS (blocks only if worse by > 2 combined SE).
- Phase 2 extra stop ("improved by at least 2 SE over -118"): 3.2 SE on C2's own SE (pass, my reading); 1.44 combined SE. Recorded as passing; the spec's wording is ambiguous.
- Breakdown (baseline BB summed): pots < 10 BB +4,245 (n 11,227); 10-40 -105 (n 1,366); 40-100 -1,471 (n 236); 100+ -7,863 (n 171). Showdown -10,893 (n 1,711); folds +5,699 (n 11,289); 3-bet pots -1,155 (n 4,614). Small pots and fold equity win; about 400 big pots, mostly at showdown, lose more than all of it. Points at deep big-pot decisions (bet-size history, leak B / Phase 4; later-street card buckets, Phase 3).

## Diagnosis: C2's Slumbot big-pot losses are off-tree infoset misses, 2026-10-23

Every logged decision of ours (29,096) was replayed through the deploy agent's key builder (`SixmaxAgent`, as `eval_slumbot` plays it) and looked up in the table.

- C2 missed its table on **41.1% of preflop decisions** (5,840), all in one context: preflop, 1 raise, pot 3.0 BB, the big blind facing Slumbot's 2 BB min-open. Our opens are 2.5/3.5/5 BB, so training never produces a 3.0 BB pot facing an open, and C2's first pot edge (3 BB) files it in a bucket of its own. Postflop misses 5.9/8.3/6.5% (pots of exactly 3.0 or 6.0 BB after Slumbot's small bets).
- A miss plays uniformly over the legal mask. At the preflop miss C2 raised 2,894, **jammed 200 BB 997 times**, called 985, folded 964 (1/6 each of six legal actions).
- Hands with a miss: raw -85.7 +/- 36.1 BB/100 (n 6,225); hands without: -18.1 +/- 23.6 (n 6,775; not a random sample, since it excludes Slumbot's min-open hands). 114 of the 171 pots >= 100 BB had a miss and lost -6,351 of the -7,863 baseline BB.
- Same decisions through other tables: C0 (edges 7/15/40) 0.0% preflop, 0.0/1.8/2.5% postflop; C1 (edges 4/8/16/...) 0.0% preflop but 23.1% on the flop (pot 4.0 BB after a called 2 BB open), 14.5/10.5% turn/river.
- So finer pot edges open holes at off-tree sizes below the smallest trained pots. Self-play head-to-head and LBR use only our own bet sizes and cannot see this; only an outside opponent's sizes do.
- Proposed fix (no retraining): in the deploy bridge, on a miss fall back to the nearest pot bucket present in the table before uniform, and make the final fallback safer than uniform (uniform includes the 200 BB jam). Then re-run C2 on Slumbot. The C2 Slumbot number above is a lower bound on the abstraction's quality.

## C4: C2 with the 2.5 BB open moved to 2 BB, 2026-10-23

`sixmax/configs/hu200_c4.toml` (opens 2/3.5/5 BB; C2's pot edges; its own vocab hash). Slumbot opens 2 BB in 89.8% of its small-blind hands and never 2.5 BB.

- Training: ccx33, 38,477,898 iterations (C2's cap reused, my call; formula cap with C4's infosets would be about 41.6M), 10,892 it/s, 61 min, about $0.21. 69,812 infosets (C2: 65,016).
- LBR (100k decks, seed 1): 5M 351.6, 10M 358.5, 15M 359.5, 20M 365.6, 25M 367.0, 30M 369.4, 35M 371.1, cap 365.9. About 10-19 BB/100 below C2 at every snapshot (cap: 365.9 vs 380.1).
- Trend rule: windows 20-35M slope +3.82/10M (t=+1.10) and 25-38.5M +0.01 (t=0.00): converged.
- Head-to-head (fixed engine, nearest fallback, each checkpoint with its own config): C4 vs C2 **-7.40** (se 3.05, t=-2.43; 7 of 8 seeds negative); C4 vs C0 **+19.13** (se 2.97, t=+6.44).
- Menu-mismatch confound: in C4 vs C2, C2 falls back to the nearest bucket on 9.3% of decisions (C4's 2 BB open lands C2's 3.0 BB hole) and C4 on 0.1%; but C4 files C2's 2.5 BB open (pot 3.5) with its own 3.5/5 BB opens, an alias the counter does not see. Same-menu reference C2 vs C0: 0.3% / 0.2%. So head-to-head between different menus measures cross-menu translation as well as strength.
- Evidence splits: LBR (own-menu) favours C4 by about 14; head-to-head favours C2 by 7.4 (confounded); both beat C0. C4 was built for Slumbot's 2 BB open, so Slumbot is the direct test. By the letter of the gate policy C2 stays the Phase 2 winner.
