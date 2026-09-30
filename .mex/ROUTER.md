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
last_updated: 2026-07-25
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
- Six-max throughput optimization (2026-07-21, commit `9693e6c`): profiled the live trainer with macOS `sample` (valgrind/callgrind are unavailable on arm64) — `Abstraction::bucket()`'s per-postflop-node 100-rollout MC equity dominated self-time (card-eval ~16.7k + alloc churn ~45k). Memoized `bucket()` (64-shard mutex+map, deterministic `(hole,board,street)` key, output-identical, `hash()`/serialization untouched), added `-O3 -mcpu=native -DNDEBUG`, and swapped per-node heap `sigma`/`u` for per-frame `std::array` → **~3× throughput** (2.71× wall / 2.96× CPU; ~30→~90 iters/sec on 8 cores). (The first blueprint runs made possible by this were later invalidated by the deal-loop bug; see below.)
- Six-max baseline eval harness (`scripts/eval_sixmax_baseline.py`, 2026-07-21): the *6-max* (not HU) benchmark — the blueprint via `SixmaxAgent` in one seat of an n-max live table vs n−1 fixed baseline agents (pluggable, default `PotOddsAgent`), seat-rotated over identical decks (each deck's n mirrored hands cancel card luck; only the shuffle touches global RNG). Reports **BB/100 + standard error** treating each deck as one independent sample (`block_stats`, ddof=1), and sweeps a series of `blueprint_<iters>.bin` snapshots into a per-checkpoint curve (optional `--csv`). Pure functions are buck2/checkpoint-free and unit-tested; bridges only through `agents/sixmax_agent.py`. Spec: `docs/superpowers/specs/2026-07-21-sixmax-baseline-eval-design.md`. **Its pre-fix result (flat ~−485 BB/100 across all snapshots) came from the now-invalid broken-trainer checkpoints — re-run after the retrain (see the deal-loop-fix validation below). Open rigor lever: per-deck *paired* differences across checkpoints (all share the seed-1 decks) cancel card-variance — not yet built; harness reports absolute BB/100 only.**
- Blueprint plateau autopsy (`scripts/diagnose_blueprint.py` + read-only `sixmax.dump_infosets` binding, 2026-07-22): offline diagnostic over `blueprint_<iters>.bin` snapshots — `dump_infosets` recovers the per-infoset visit-weight (`strategy_sum` L1) + regret L1 the normalized loaders discard; the script decodes abstract keys in Python (mirrors `abstract_key.h` layout) and reports per-checkpoint visit-weight distribution (Gini/top-1% share), visit-weighted policy entropy (all + top-mass tier), and canonical preflop probe traces (AA/KK/AKs aggression, 72o/83o fold). Trend-based `verdict()`: two gates — top-tier entropy monotone-**down**, mean-probe-mass monotone-**up** (each ≥80% of steps AND |net|>1e-6); `avg_regret_top` is context-only because linear-CFR cumulative regret can't shrink. Pure metrics are stdlib-only + unit-tested; every sixmax/agents import is lazy (module import is Buck2-free). Spec/plan: `docs/superpowers/specs/2026-07-22-blueprint-plateau-autopsy-design.md`, `docs/superpowers/plans/2026-07-22-blueprint-plateau-autopsy.md`. **This tool root-caused the frozen preflop probes to the trainer deal-loop bug and, post-fix, confirmed they thawed — see the two entries below.**
- Trainer deal-loop fix (2026-07-22, commit `b739363`): both `BlueprintTrainer` (`trainer.cpp`) and reference `MCCFRTrainer` (`mccfr.cpp`) now deal ONE hand per iteration and traverse it once per seat on clones (`root = new_hand(); for p: traverse(root->clone(), p)`), instead of dealing a fresh hand per traverser. The old loop advanced the button a full cycle each iteration (`new_hand` does `button=(button+1)%n`), phase-locking `(first_to_act − traverser)` to a constant so the traverser only ever acted from one button-relative seat — starving all other seats' regret (the frozen-open pathology). Regression `test_utg_open_decisions_receive_regret` (dumps a 1500-iter checkpoint, asserts the UTG-open node has regret>0); both Kuhn −1/18 gates still pass, single-thread determinism preserved. **This invalidates every existing `.bin` blueprint checkpoint under `sixmax/checkpoints/` — retrain from scratch.**
- Deal-loop fix VALIDATED on the fresh retrain (2026-07-22, `scripts/diagnose_blueprint.py` over the first two from-scratch snapshots `blueprint_00500000.bin`/`blueprint_01000000.bin`): the frozen preflop probes have **thawed**. **72o/83o UTG-unraised fold mass jumped 0.16 (frozen across all 10 pre-fix snapshots) → ~0.82**; the bot no longer open-jams trash. Premium aggression is rising off the old uniform 0.67 floor (AA 0.65→0.69, KK 0.80/0.77, AKs 0.70→0.75). Both `verdict()` gating signals now move the learning way (`entropy_top` 0.770→0.743 falling, mean probe aggression 0.752→0.771 rising) so the label flipped **`mixed` (frozen) → `undertraining`** — the *good* diagnosis: the policy responds to hand strength and just needs iterations, not a redesign. The 0.16→0.82 level shift is unambiguous; the 2-point trend firms up as later snapshots land.
- Six-max deal-loop fix FULLY VALIDATED — retrain complete + eval curve moved (2026-07-23): the from-scratch 5M-iteration retrain finished (`train_5m.log`: 5,000,000 iters, 142,986 infosets; `sixmax/checkpoints/blueprint.bin` is the live blueprint). Re-ran `scripts/eval_sixmax_baseline.py` over all 10 snapshots at 20,000 decks/ckpt (~120k hands, ±~29 BB/100; `sixmax/checkpoints/curve_20k/curve_20k.csv`). **The pre-fix curve that was confidently FLAT at ~−485 BB/100 now climbs monotonically into positive territory:** 500k −51.1 → 1.5M −12.6 → 2.5M −7.5 → 3M +12.7 → 4.5M +24.7 → **5M +32.5 BB/100 vs PotOdds**. The deal-loop fix is confirmed to be the root cause and the blueprint is now learning. (Post-fix evals after 2.5M are still ±~29 non-overlapping-from-zero only at the ends; the trend is what matters.)
- DREAM neural blueprint subsystem (`sixmax/src/dream/`, 2026-07-24/25, branch `feature/neural-cfr`, commits `1fbe357`..`9812fa8`): a DREAM-style (Steinberger et al. 2020) advantage-net + strategy-net blueprint trained by outcome-sampling MCCFR, replacing the tabular `BlueprintTrainer` so strategy generalizes across variable stack sizes (20–250 BB) and player counts (2–6) — which the 64-bit abstract-key table cannot do. Layout: `features.{h,cpp}` (154-dim infoset tensor), `nets.{h,cpp}` (libtorch MLP 154→256→256→256→N), `reservoir.{h,cpp}` (thread-safe weighted M_v/M_π), `trainer.{h,cpp}` (`DreamTrainer`, stochastic external-sampling + IS-weighted advantage targets), `checkpoint.{h,cpp}` (SIXDM001 magic + `DreamStrategy` inference). `src/blueprint/` stays frozen until DREAM passes the Kuhn convergence gate; `bindings.cpp` exposes both `BlueprintStrategy` and `DreamStrategy` implementing the same `get_probs()` so `SixmaxDeployStrategy` is unchanged. `scripts/train_dream.py` + openpoker `dream_*.pt` auto-detect. SDD ledger `.superpowers/sdd/dream-progress.md`: **all 10 tasks complete, "READY TO MERGE" after Critical+Important fixes (`9812fa8`)** — residual minors (multi-thread retrain races, hand_count approximation under variable n_players, hero-seat stack ordering, libtorch ABI workaround). **Not yet merged to main.**
- Mixed-table multi-agent eval (in progress, plan `docs/superpowers/plans/2026-07-23-mixed-table-eval.md`): `scripts/eval_mixed_table.py` seats all four trained agents (sixmax/neural/tabular/potodds×3) at one 6-handed table and reports per-seat BB/100 + a net chip matrix. Uncommitted work-in-progress adds per-hand **paired** BB/100 CIs (sixmax vs each potodds seat) and raises the table stack 200→2000; plan tasks not yet checked off.
- Tooling: interactive play, range charts, OpenSpiel head-to-head eval, openpoker.ai WebSocket deployment (auto-detects `.pt`/`.pkl`/`.bin`)
- Cloud training tooling hardening (2026-09-29): `scripts/cloud_train.sh` now passes `--checkpoint-interval` (default 1M) and `--snapshots` to the trainer — previously `checkpoint_interval=0` meant save-at-end-only, so a watchdog-stopped run lost everything (the 2026-09-29 incident: ~14M iters unsaved) — and rejects a non-positive interval up front; `scripts/cloud_benchmark.sh` rsync calls now use the same `SSH_OPTS` host-key hardening as `cloud_train.sh`. With periodic checkpoints, `--max-hours` watchdog caps are safe again.
- Cloud-run observability fix (2026-09-30): the 2026-09-29 01:26 cloud run (25M-resume, `--max-hours 14`) sat in save-at-end-only mode with **no observable progress** — `print()` was block-buffered into the ssh pipe (so no `Effective config`/`Running`/chunk lines ever reached the log) and the single 25M chunk emitted nothing until the end; `docker logs` was likewise empty. Fixed: `scripts/train_sixmax.py` now logs via `logging`→stderr (line-buffered, appears immediately even when piped) and runs a daemon `_ProgressReporter` that prints `[done/total] rate it/s ETA infosets` every `--report-interval`/`[train.blueprint].report_interval` seconds (default 60, `0`=off) while `train()` runs. It polls the atomic `iterations()` counter (the `train` binding releases the GIL) rather than splitting `train()` calls, deliberately avoiding the per-call `mt19937_64` reseed so the training trajectory is unchanged. `cloud_train.sh`/`cloud_benchmark.sh` also run `python3.12 -u` with `-e PYTHONUNBUFFERED=1`, and `docker/Dockerfile` sets `ENV PYTHONUNBUFFERED=1`. Tests: `format_progress`/`_format_duration`/`_ProgressReporter.report_once` + `report_interval` config precedence in `tests/sixmax/test_train_sixmax.py`. NOTE: the already-running 2026-09-29 job cannot pick this up (old code); it remains save-at-end-only.
- 270-test pytest suite with fast-mode conftests — full `uv run pytest tests/` runs green in ONE process (~185s): 265 passed, 2 skipped, 3 xpassed

**Not yet built:**
- Six-max Phases 2–3: depth-limited search with RangeTracker, batched-ReBeL value net (spec in `docs/superpowers/`); the DREAM neural blueprint (built, above) is the intended replacement for the tabular blueprint once merged.
- A **fully converged** 6-max blueprint: the deal-loop bug (commit `b739363`) is fixed and the from-scratch 5M retrain on `blueprint.bin` is confirmed learning (curve −485 flat → +32.5 BB/100 vs PotOdds; see the validated entry above). **Still open quality levers (independent of the fix):** per-deck **paired** eval slope (cancels card variance; being built in `eval_mixed_table.py`), stronger reference opponents than PotOdds, the 169-hand exact preflop abstraction, and merging/validating the DREAM neural blueprint (`feature/neural-cfr`).
- 169-hand exact preflop abstraction (documented as the largest single quality win for tabular)
- Suit-texture-aware board abstraction (flush draws / monotone boards currently bucket like rainbow)
- Full 7-card lookup table for `util/util.py` (fallback lacks kicker discrimination)

**Known issues:**
- **RESOLVED (2026-07-25) — full-suite `neural_cfr` GIL crash.** Previously `tests/neural_cfr/` failed (1 failed + 4 errors) with `RuntimeError: The autograd engine was called while holding the GIL` **only when `tests/sixmax` ran first**: importing Python `torch` (as `tests/sixmax/test_dream_kuhn.py` does) registers the `PythonEngine`, and neural_cfr's bindings invoked `backward()` while holding the GIL (PythonEngine forbids this). Fixed by adding `py::call_guard<py::gil_scoped_release>()` to `Trainer.run` / `train_strategy` / `checkpoint` in `neural_cfr/src/bindings/bindings.cpp`, mirroring the existing pattern in `sixmax/src/bindings/bindings.cpp`. Regression: `tests/neural_cfr/test_torch_interop.py` (imports torch, then trains + checkpoints). Full suite now green in one process. Root cause was a binding-level defect, not test ordering/variance.
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
