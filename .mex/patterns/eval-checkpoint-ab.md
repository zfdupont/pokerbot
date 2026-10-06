---
name: eval-checkpoint-ab
description: Honestly A/B two candidate checkpoints (or search on/off) — paired seeds, sample size, and the t-test that stops small-sample noise from deciding a deployment.
triggers:
  - "A/B"
  - "is it better"
  - "compare checkpoints"
  - "swap the checkpoint"
  - "search on vs off"
edges:
  - target: patterns/eval-checkpoint.md
    condition: when measuring against OpenSpiel or random rather than against another checkpoint
  - target: context/setup.md
    condition: for the eval commands and the droplet swap procedure
last_updated: 2026-10-06
---

# A/B Two Candidates Honestly

## Context

Settling "is candidate X better than candidate Y" — two blueprint checkpoints, an HU-vs-6-max
checkpoint, or search-on vs search-off. The harnesses (`scripts/eval_hu_sanity.py` for heads-up,
`scripts/eval_sixmax_baseline.py` for 6-max) each play candidate vs a **fixed third party**; to
compare two candidates you run both against that same party **on the same seeds** and difference the
results.

The single most important fact: **this measurement is far noisier than it looks.** Stack-offs at
100–200 BB give per-hand standard deviations of tens of BB, so 4 000 hands has a standard error of
roughly ±10 BB/100 per arm, and seed-to-seed swings of ±50 BB/100 are normal.

## Steps

1. Pick N seeds (≥ 8) and a hand count (**≥ 20 000 hands per seed**). Do not judge on 2 000–4 000.
2. Run **both** candidates on **exactly the same seeds** — pairing cancels the deal luck and is far
   more sensitive than comparing two independent means.
3. Per seed, compute `d = BB100(A) − BB100(B)`. Report the per-seed table, then `mean(d)`, `sd(d)`,
   `se = sd/√N`, and `t = mean/se`.
4. **Only claim a difference at |t| ≥ 2.** Below that, the honest answer is "no measurable
   difference" — keep the incumbent.
5. Always print the raw vs-opponent BB/100 too, as a sanity floor (a candidate that cannot beat the
   weak baseline is broken, not merely unlucky).

## Gotchas

- **Small samples lie in both directions.** `hu_blueprint` 10M vs the 6-max blueprint measured **+14
  BB/100 at 9×4 000 hands (t = 0.95)** and **−8.04 BB/100 at 8×20 000 hands (t = −1.05)**. Same
  checkpoints, same opponent. Only the larger run was worth believing.
- **The absolute number is nearly meaningless.** BB/100 vs the tabular bot swings from −10 to +70
  across seeds for the *same* checkpoint, because the fixed opponent's own exploitability varies by
  seed. The paired difference is the only stable signal.
- **Seed count, not hands alone, controls the between-seed variance.** 20 000 hands sharpens each
  point; more seeds sharpen the mean.
- **Shell precedence trap:** `export A=... && (cmd1) & (cmd2) & wait` puts `A=...` *inside* the first
  background job, so `cmd2` runs with an empty variable and the harness fails with a confusing
  `expected one argument`. `export` the shared vars on their own, before backgrounding.
- **Run it detached** (`nohup ... &` writing to a file), then poll — a foreground multi-seed sweep
  can outlive the terminal and lose its output.
- A null or negative result is a **legitimate outcome**. Do not swap in a candidate because it is
  newer, more expensive to produce, or more clever.

## Verify

- [ ] Both arms used identical seeds and identical hand counts.
- [ ] ≥ 8 seeds, ≥ 20 000 hands per seed.
- [ ] The paired per-seed table is reported, not just two means.
- [ ] `t` is stated; a claim of "better" carries |t| ≥ 2.
- [ ] Raw vs-opponent BB/100 reported for both arms as a sanity floor.

## Debug

- Both arms identical to the last decimal → you forgot to vary the checkpoint path, or reused a
  cached file.
- Enormous single-seed outlier dragging the mean → report the median alongside the mean, and check
  whether that seed's opponent went on an all-in streak (a legitimate heavy tail, not a bug).
- Results flip when you change the seed set → that *is* the answer: no detectable difference.

## Update Scaffold

- [ ] Update `.mex/ROUTER.md` "Current Project State" with the decided outcome (promoted / kept
      incumbent / not yet A/B'd).
- [ ] If a checkpoint swap happened, update `.mex/context/setup.md` (Production section).
