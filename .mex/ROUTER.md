---
name: router
description: Session bootstrap and navigation hub. Read at the start of every session before any task. Contains project state, routing table, and behavioural contract.
edges:
  - target: context/architecture.md
    condition: when working on system design, integrations, or understanding how components connect
  - target: context/stack.md
    condition: when working with specific technologies, libraries, or making tech decisions
  - target: context/conventions.md
    condition: when writing new code, reviewing code, or unsure about project patterns
  - target: context/decisions.md
    condition: when making architectural choices or understanding why something is built a certain way
  - target: context/setup.md
    condition: when setting up the dev environment or running the project for the first time
  - target: context/cfr-training.md
    condition: when working on the tabular MCCFR pipeline (cfr/) or exploitability
  - target: context/neural-cfr.md
    condition: when working on the C++ Deep CFR subsystem (neural_cfr/)
  - target: patterns/INDEX.md
    condition: when starting a task — check the pattern index for a matching pattern file
last_updated: 2026-07-22
---

# Session Bootstrap

If you haven't already read `AGENTS.md`, read it now — it contains the project identity, non-negotiables, and commands.

Then read this file fully before doing anything else in this session.

## Current Project State

**Working:**
- Live N-player Texas Hold'em engine (`game/poker.py`) with pluggable agents and side-pot handling
- Tabular MCCFR pipeline (`cfr/`) trained to 9.04M iterations — 574 mbb/h within-abstraction exploitability, 7,877 infosets
- C++ Deep CFR subsystem (`neural_cfr/`, libtorch + pybind11 + Buck2) with multithreaded traversal and ε-greedy opponent exploration
- Neural CFR correctness fixes (paper-faithful training regime, MLP reset, TOML config, inference parity): validated by from-scratch acceptance retrain (2026-07-14/15, 5M iters, ~20h) — best checkpoint **+36.4 BB/100 vs tabular @ 4.5M iters** (10k-hand eval), first neural win over the 9.04M-iter tabular baseline; `best_checkpoint.pt` selection and global linear-CFR iteration counter both worked end-to-end
- Six-max subsystem Phase 0 (`sixmax/`): pybind11 module skeleton, config-defined `ActionVocab` (BB/pot units, pseudo-harmonic translation, TOML loader, canonical ordering contract); C++ evaluator extracted into `common/` (Buck2 `//common:evaluator`) with opaque `safe_eval::HandRank` API; `scripts/setup_dev.sh` idempotent one-shot dev setup
- Six-max Phase 1a solver core (`sixmax/src/blueprint/` + `sixmax/src/engine/`, merged 2026-07-19 at `99d6203`): abstract `GameState`/`Game` interface, external-sampling MCCFR with linear weighting (gated on Kuhn −1/18 ±0.01), 2–6 player NLHE engine (doubles, SB 0.5/BB 1.0/stack 100) with side-pot settlement property-tested against the Python `PotManager` oracle (300 scenarios), and `EngineGame` — the vocab-masked bridge (naive exact keyer remains the no-abstraction default)
- Six-max Phase 1b blueprint pipeline (merged 2026-07-19 at `3c670d6`): card abstraction (`sixmax/src/abstraction/` — lossless 169 preflop classes, deterministic MC equity-percentile buckets 50/50/20 seeded per (hole,board,salt)); bit-packed abstraction infoset keys (card/street/raise-counts≤3/pot-bucket/live_opps/after — table-size-agnostic: MP 6-handed ≡ UTG 5-handed); multithreaded `BlueprintTrainer` (64-shard table, atomic linear-CFR counter, passes the Kuhn gate 1- and 4-threaded); self-describing binary checkpoints (magic SIXBP001, embeds vocab hash + abstraction edges, loaders refuse mismatches) + `BlueprintStrategy` loader; `scripts/train_sixmax.py` (CLI>TOML>builtin, chunked saves, best-checkpoint selection) and `scripts/eval_sixmax.py` (duplicate-deal seat-rotated A/B, BB/100)
- Six-max Phase 1c deployment (merged 2026-07-20): host-agnostic `SixmaxDeployStrategy` bridge (`agents/sixmax_agent.py`) reconstructs the trainer's infoset key from public state via the shared C++ `pack_abstract_key` (extracted so bridge and trainer cannot drift), re-masks to legal actions, samples, and translates the chosen vocab index to a raise-to (BB) through `ActionVocab`; `SixmaxAgent(PokerAgent)` and `NeuralAgent(PokerAgent)` (`agents/neural_agent.py`) adapt the blueprint and neural strategies to the live `game/poker.py` engine; `scripts/eval_hu_sanity.py` runs duplicate-deal HU-mode evals (blueprint vs frozen tabular/neural, BB/100); `scripts/openpoker_bot.py` auto-detects `.bin` → `SixmaxHandTracker`. `game/poker.py`'s `raises_per_street` is now faithful (uncapped) — each consumer applies its own cap (tabular 2, sixmax 3, neural 2)
- Six-max throughput optimization + first full blueprint run (2026-07-21, commit `9693e6c`): profiled the live trainer with macOS `sample` (valgrind/callgrind are unavailable on arm64) — `Abstraction::bucket()`'s per-postflop-node 100-rollout MC equity dominated self-time (card-eval ~16.7k + alloc churn ~45k; `clone()` was noise at 10). Memoized `bucket()` (64-shard mutex+map, deterministic `(hole,board,street)` key, output-identical, `hash()`/serialization untouched), added `-O3 -mcpu=native -DNDEBUG`, and swapped per-node heap `sigma`/`u` for per-frame `std::array` → **~3× throughput** (2.71× wall / 2.96× CPU; ~30→~90 iters/sec on 8 cores). Ran the first real blueprint to **1M iters / 117,524 infosets** (~2h45m; `sixmax/checkpoints/best_checkpoint.bin`), self-play selection converged. **BUT 1M is smoke-scale for 6-max (~8 visits/infoset) and NOT yet competitive:** the HU sanity eval has the blueprint (2-player) losing **−226.86 BB/100 vs the 9.04M tabular** and **−259.65 vs best neural** (4000 duplicate-deal hands). Root-caused via instrumentation (systematic-debugging, 2026-07-21): NOT a deployment bug and NOT the untrained-key uniform fallback (95% of decisions hit trained keys; 0% masking mismatch; 0 premium hands folded; legal & sane actions/sizes). The loss is **severe undertraining** — at 1M iters / 117k infosets ≈ 8.5 visits/infoset, regret-matched policies are immature (near-uniform over raise sizes: `fold≈0 call≈0`, ~0.20 each across 2.5/3.5/5bb/allin), so the bot open-jams trash (8-high, K-high) and gets stacked; doubly punished by the frozen HU-*specialist* opponents. Fix is training scale + a real 6-max benchmark, not code. Deployment path proven end-to-end. **[CORRECTION 2026-07-22: this "severe undertraining, not code" conclusion was WRONG — the near-uniform open-jam policy was actually the trainer deal-loop bug fixed in `b739363` (see the autopsy + fix entries above). The 2026-07-21 instrumentation checked deployment/masking/fallback but not per-seat regret coverage, so it missed the phase-locked traverser.]**
- Six-max baseline eval harness (`scripts/eval_sixmax_baseline.py`, 2026-07-21): the missing *6-max* (not HU) benchmark — the blueprint via `SixmaxAgent` in one seat of an n-max live table vs n−1 fixed baseline agents (pluggable, default `PotOddsAgent`), seat-rotated over identical decks (each deck's n mirrored hands cancel card luck; only the shuffle touches global RNG). Reports **BB/100 + standard error** treating each deck as one independent sample (`block_stats`, ddof=1), and sweeps a series of `blueprint_<iters>.bin` snapshots into a per-checkpoint curve (optional `--csv`). Pure functions are buck2/checkpoint-free and unit-tested; bridges only through `agents/sixmax_agent.py`. Spec: `docs/superpowers/specs/2026-07-21-sixmax-baseline-eval-design.md`. **Sharpened result (2026-07-22, `sixmax/checkpoints/curve.csv`):** re-ran all 10 snapshots (this-run increments 100k–1M, i.e. 1.1M–2.0M total iters) at **20,000 decks/ckpt (120k hands, ±~50 BB/100)** vs PotOdds, 10-way parallel (`sixmax/checkpoints/run_curve_20k.sh`, ~35 min). The curve is **confidently flat at ~−485 BB/100** — no monotonic trend across iterations; the earlier 3k-deck run's apparent −377/−529 spread was pure variance and regressed to the mean once CIs tightened 4×. This is the strongest evidence for the documented undertraining root cause: even the extremes (100k −454, 700k −535) barely separate, and the blueprint loses ~5 BB/hand to a *passive* baseline. Next rigor lever if a sub-±50 trend is suspected: per-deck **paired** differences across checkpoints (all share the identical seed-1 decks), which cancels shared card-variance — not yet built (harness reports absolute BB/100 only); the 2026-07-22 policy autopsy (below) now motivates building it as the plateau tie-breaker.
- Blueprint plateau autopsy (`scripts/diagnose_blueprint.py` + read-only `sixmax.dump_infosets` binding, 2026-07-22): offline diagnostic over the 10 `blueprint_<iters>.bin` snapshots — `dump_infosets` recovers the per-infoset visit-weight (`strategy_sum` L1) + regret L1 the normalized loaders discard; the script decodes abstract keys in Python (mirrors `abstract_key.h` layout) and reports per-checkpoint visit-weight distribution (Gini/top-1% share), visit-weighted policy entropy (all + top-mass tier), and canonical preflop probe traces (AA/KK/AKs aggression, 72o/83o fold). Trend-based `verdict()`: two gates — top-tier entropy monotone-**down**, mean-probe-mass monotone-**up** (each ≥80% of steps AND |net|>1e-6); `avg_regret_top` is context-only because linear-CFR cumulative regret can't shrink. Pure metrics are stdlib-only + unit-tested; every sixmax/agents import is lazy (module import is Buck2-free). **Finding across 1.1M→2.0M: VERDICT `mixed`** — `entropy_top` falls strictly monotonically 9/9 (0.665→0.642 bits, real but slow concentration) while the premium/trash preflop probes are FROZEN across all 10 ckpts (AA/KK/AKs aggr 0.67 ≈ uniform, 72o/83o fold 0.16). Spec/plan: `docs/superpowers/specs/2026-07-22-blueprint-plateau-autopsy-design.md`, `docs/superpowers/plans/2026-07-22-blueprint-plateau-autopsy.md`. **FOLLOW-UP (2026-07-22, systematic-debugging): the frozen probes were root-caused to a trainer bug, not scale/abstraction — see the trainer deal-loop fix below.**
- Trainer deal-loop fix (2026-07-22, commit `b739363`): both `BlueprintTrainer` (`trainer.cpp`) and reference `MCCFRTrainer` (`mccfr.cpp`) now deal ONE hand per iteration and traverse it once per seat on clones (`root = new_hand(); for p: traverse(root->clone(), p)`), instead of dealing a fresh hand per traverser. The old loop advanced the button a full cycle each iteration (`new_hand` does `button=(button+1)%n`), phase-locking `(first_to_act − traverser)` to a constant so the traverser only ever acted from one button-relative seat — starving all other seats' regret (the frozen-open pathology). Regression `test_utg_open_decisions_receive_regret` (dumps a 1500-iter checkpoint, asserts the UTG-open node has regret>0); both Kuhn −1/18 gates still pass, single-thread determinism preserved. **This invalidates every existing `.bin` blueprint checkpoint under `sixmax/checkpoints/` — retrain from scratch.**
- Tooling: interactive play, range charts, OpenSpiel head-to-head eval, openpoker.ai WebSocket deployment (auto-detects `.pt`/`.pkl`/`.bin`)
- 199-test pytest suite with fast-mode conftests

**Not yet built:**
- Six-max Phases 2–3: depth-limited search with RangeTracker, batched-ReBeL value net (spec in `docs/superpowers/`)
- A **converged / competitive** 6-max blueprint (needs a fresh retrain): the flat ~−485 ±50 BB/100 curve (20k-deck sweep, 2026-07-21/22) was **root-caused on 2026-07-22 to a trainer bug, NOT undertraining or a coarse abstraction** (commit `b739363`). Both MCCFR trainers dealt a fresh hand per traverser, advancing the button a full cycle per iteration and phase-locking the traverser to a single button-relative seat; every other seat's decisions (incl. the UTG open where AA/KK first act) accrued average strategy but ZERO regret → frozen uniform policy → open-jams trash. Fixed (deal one hand/iter, traverse all seats on clones; verified the traverser now covers all 6 positions). **All existing `.bin` blueprint checkpoints under `sixmax/checkpoints/` were trained on the broken objective and are invalid — a from-scratch retrain on the fixed trainer is the next step, then re-run `eval_sixmax_baseline.py` to confirm the curve moves.** Still-open independent levers (unchanged by the fix): the per-deck **paired** eval slope (cancels card variance; not built), stronger reference opponents than PotOdds, and the 169-hand exact preflop abstraction.
- 169-hand exact preflop abstraction (documented as the largest single quality win for tabular)
- Suit-texture-aware board abstraction (flush draws / monotone boards currently bucket like rainbow)
- Full 7-card lookup table for `util/util.py` (fallback lacks kicker discrimination)

**Known issues:**
- Neural inference is chip-scale sensitive: inputs from other engines must be rescaled to the 100BB/1BB training frame or the bot misplays
- Tabular exploitability numbers are only comparable within the current abstraction

## Routing Table

Load the relevant file based on the current task. Always load `context/architecture.md` first if not already in context this session.

| Task type | Load |
|-----------|------|
| Understanding how the system works | `context/architecture.md` |
| Working with a specific technology | `context/stack.md` |
| Writing or reviewing code | `context/conventions.md` |
| Making a design decision | `context/decisions.md` |
| Setting up or running the project | `context/setup.md` |
| Tabular CFR: abstraction, RegretTable, exploitability | `context/cfr-training.md` |
| Neural CFR: C++ code, encodings, training, checkpoints | `context/neural-cfr.md` |
| Six-max subsystem: vocab, engine, blueprint, search | `context/architecture.md` + `docs/superpowers/specs/2026-07-17-sixmax-search-design.md` |
| Any specific task | Check `patterns/INDEX.md` for a matching pattern |

## Behavioural Contract

For every task, follow this loop:

1. **CONTEXT** — Load the relevant context file(s) from the routing table above. Check `patterns/INDEX.md` for a matching pattern. If one exists, follow it. Narrate what you load: "Loading architecture context..."
2. **BUILD** — Do the work. If a pattern exists, follow its Steps. If you are about to deviate from an established pattern, say so before writing any code — state the deviation and why.
3. **VERIFY** — Load `context/conventions.md` and run the Verify Checklist item by item. State each item and whether the output passes. Do not summarise — enumerate explicitly.
4. **DEBUG** — If verification fails or something breaks, check `patterns/INDEX.md` for a debug pattern. Follow it. Fix the issue and re-run VERIFY.
5. **GROW** — After meaningful work, run this binary checklist:
   - **Ground:** What changed in reality? Name the changed behavior, system, command, dependency, or workflow.
   - **Record:** If project state changed, update the "Current Project State" section above. If documented facts changed, update the relevant `context/` file surgically.
   - **Orient:** If this task can recur and no pattern exists, create one in `patterns/` using `patterns/README.md`, then add it to `patterns/INDEX.md`. If a pattern exists but you learned a gotcha, update it.
   - **Write:** Bump `last_updated` in every scaffold file you changed. If the why matters, run `mex log --type decision "<what changed and why>"` or `mex log "<note>"`.
