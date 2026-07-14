# Global Iteration Counter + Best-Checkpoint Selection — Design

**Date:** 2026-07-13
**Scope:** `neural_cfr/src/cfr/trainer.{h,cpp}`, `neural_cfr/src/bindings/bindings.cpp`, `scripts/train_neural.py`, `neural_cfr/configs/*.toml`, tests, `.mex` docs

## Background

Two gaps surfaced during the 2026-07 acceptance and resumed-training runs:

1. **Per-chunk linear-CFR weights.** `Trainer::run` resets its iteration counter
   (`completed`) on every call, and the Python driver calls `run()` once per
   checkpoint interval — so sample weights `t` restart at 1 every 500k
   iterations, and again on every resume. The intended linear-CFR weighting
   (later iterations dominate the averages) degrades to
   uniform-across-chunks / linear-within-chunk.
2. **Checkpoint quality oscillates.** Since 3.5M iterations, every measured
   checkpoint lands in a −30 ± 40 BB/100 band vs the tabular baseline; the
   strategy net's from-scratch retrain (2,000 batches) makes each checkpoint a
   noisy draw. The best artifact so far (+3.2 at 4M) was found only by manual
   ad-hoc evals. Selection should be automatic.

## Feature 1 — Global iteration counter

### C++ changes (`trainer.h` / `trainer.cpp`)

- New private member: `int64_t total_iters_ = 0;`
- `run()`: linear-CFR weight becomes `global_iter = total_iters_ + index + 1`
  inside the worker; after each thread join, `total_iters_ += batch`. The
  progress bar keeps using the per-call local counter (unchanged UX).
- `checkpoint()`: write an additive `"meta"` sub-archive containing
  `total_iters` as an int64 scalar tensor, alongside the existing
  `adv0`/`adv1`/`strat` sub-archives.
- `load()`: attempt to read `"meta"`/`total_iters`; on any failure (legacy
  checkpoint without meta), set `total_iters_ = 0` and continue silently.
- New public getter `int64_t total_iterations() const;`, bound to Python as
  `Trainer.total_iterations()`.

### Compatibility

- Old checkpoints load fine (counter starts at 0 — current behavior).
- New checkpoints load in old code only if the reader ignores unknown keys —
  not guaranteed; acceptable because the branch has a single consumer and
  `Strategy` reads only `"strat"` either way.
- `BufferEntry.weight` stays `float`: exact to 16.7M iterations; precision
  loss beyond that is harmless for weighting.
- `external_sample`'s `int iteration` parameter: passes `(int)global_iter`;
  safe to 2.1B iterations.

### Tests (`neural_cfr/tests/test_trainer.cpp`)

- Two consecutive `run()` calls accumulate: `total_iterations() == n1 + n2`.
- `checkpoint()` → fresh `Trainer` → `load()` restores `total_iterations()`.
- Legacy-format archive (written by the test with only `adv0`/`adv1`/`strat`
  sub-archives) loads without throwing and yields `total_iterations() == 0`.
- Buffer weight check: after a second `run()`, max weight in a buffer exceeds
  the first run's iteration count (proves weights use the global counter).

## Feature 2 — Best-checkpoint selection by eval

### Driver changes (`scripts/train_neural.py`)

After each `trainer.checkpoint(path)` + config snapshot in the main loop,
when selection is enabled:

1. Run subprocess (same pattern as the existing `--eval-interval` code:
   `[sys.executable, <script>, ...]` with `cwd=repo_root`):
   `scripts/eval_neural_vs_tabular.py --neural-checkpoint <path>
   --hands <selection_hands>`
   (plus `--tabular-checkpoint <selection_tabular_checkpoint>` when non-empty).
2. Parse stdout with module-level `parse_bb100(text) -> float | None`
   (regex on the `Neural win rate` line).
3. Compare with the sidecar via module-level
   `update_best(bb100, ckpt_path, total_iters, hands, best_dir) -> bool`:
   - Sidecar `best_checkpoint.json` (same directory as the checkpoint):
     `{"bb100": float, "total_iters": int, "hands": int, "timestamp": iso8601,
     "source_checkpoint": str}`.
   - On strict improvement (`bb100 > best.bb100`, or no sidecar): copy the
     checkpoint to `best_checkpoint.pt`, rewrite the sidecar, return True.
4. `total_iters` comes from `trainer.total_iterations()` (Feature 1).

**Failure policy:** any eval subprocess failure, timeout, or unparseable
output logs a warning and training continues. Selection must never abort a
run.

### Config / CLI

Three new flat keys in `BUILTIN_DEFAULTS` and the `[training]` TOML section
(reusing existing resolve machinery — no new section handling):

| Key | Builtin default | default.toml | smoke.toml |
|---|---|---|---|
| `selection_enabled` | `false` | `true` | absent (off) |
| `selection_hands` | `10_000` | `10_000` | absent |
| `selection_tabular_checkpoint` | `""` (eval script auto-detects) | `""` | absent |

Matching CLI flags (`--selection-enabled/--no-selection-enabled`,
`--selection-hands`, `--selection-tabular-checkpoint`), standard
CLI > TOML > builtin precedence.

**Metric choice (decided):** vs-tabular head-to-head at 10k hands
(SE ≈ ±10 BB/100). Brown et al. tracked exact-best-response exploitability,
which is intractable for full HUNL vs a neural strategy; vs-tabular is the
stable fixed yardstick already used for acceptance. Strict-improvement
replacement plus a fixed hands budget bounds winner's-curse inflation.

### Tests (`tests/scripts/test_train_neural_config.py` or sibling file)

- `parse_bb100`: extracts `-46.7` / `+3.2` style values from real eval output;
  returns `None` on garbage.
- `update_best`: no sidecar → creates sidecar + copies file; worse score →
  no-op; better score → replaces both; sidecar JSON round-trips.
- Config precedence for the three new keys (reuse existing test pattern).

## Validation / rollout

- `~/bin/buck2 test //neural_cfr/tests:test_trainer` (new counter tests).
- `uv run pytest tests/scripts/ -v` (new parse/sidecar/config tests).
- End-to-end smoke: short run (e.g. 100k iters, `train_interval=10_000`,
  selection enabled, `selection_hands=500` for speed) → confirm
  `best_checkpoint.pt` + `best_checkpoint.json` appear and the log shows the
  selection decision.
- Docs: `.mex/context/neural-cfr.md` — counter persistence in the checkpoint
  format note; selection behavior + sidecar in the training regime/commands
  notes. Config file comments updated.

## Out of scope (follow-up candidates)

- Local Best Response (LBR) exploiter as a convergence metric.
- Serializing M_π / advantage buffers for true resume continuity.
- Variance reduction (AIVAT-style) for evals.
