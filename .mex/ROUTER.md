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
last_updated: 2026-07-20
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
- Tooling: interactive play, range charts, OpenSpiel head-to-head eval, openpoker.ai WebSocket deployment (auto-detects `.pt`/`.pkl`/`.bin`)
- 178-test pytest suite with fast-mode conftests

**Not yet built:**
- Six-max Phases 2–3: depth-limited search with RangeTracker, batched-ReBeL value net (spec in `docs/superpowers/`)
- A real 6-max blueprint training run (the pipeline is ready; no full-scale run yet — dev-scale smoke runs only)
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
