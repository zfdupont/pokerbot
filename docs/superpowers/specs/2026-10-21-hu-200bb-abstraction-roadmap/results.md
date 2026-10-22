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

`scripts/eval_sixmax_lbr.py` (branch feature/lbr-paper). LBR acts on turn and river with the checkpoint's own bet sizes and checks or calls before that. 4,000 duplicate decks, seed 1, 200 BB, about 19 min each, local and free. The opponent unseen-infoset rate was 0.00% for all four. Raw output: `lbr/`.

| Checkpoint | LBR BB/100 (95% CI) | mBB/h | Seat 0 (SB) | Seat 1 (BB) |
|---|---:|---:|---:|---:|
| C0 @ 15M | +425.7 ±48.4 | 4,257 | +494.8 | +356.7 |
| C0 @ 25M | +425.0 ±50.2 | 4,250 | +487.9 | +362.1 |
| P0 @ 15M | +398.1 ±53.6 | 3,981 | +487.9 | +308.3 |
| P0 @ 25M | +417.7 ±52.0 | 4,177 | +554.0 | +281.4 |

- C0 at 15M and 25M agree within 1 BB/100: C0 had converged by 15M, which matches the head-to-head plateau checks.
- P0 vs C0 is not significant: the differences are 7-28 BB/100 against about ±70 combined. This matches the head-to-head gate (+2.68, t=0.91). The four runs share decks (same seed), so a paired per-deck comparison would be tighter, but the driver does not save per-deck values yet.
- For scale, the paper reports Slumbot 2016 at about 4,020 mBB/h with LBR acting on turn and river (its "fcpa 3-4" row). By this probe these blueprints are about as exploitable as Slumbot was, even though Slumbot beats them by about 100 BB/100 head-to-head.
