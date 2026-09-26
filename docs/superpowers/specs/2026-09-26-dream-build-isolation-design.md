# DREAM Build Isolation — Design Spec

Date: 2026-09-26
Status: Approved for planning

## Problem

The DREAM neural blueprint (`sixmax/src/dream/`, ~799 LOC) is compiled into the
same `core` Buck library and the same `sixmax.so` Python extension as the
tabular blueprint. Because DREAM depends on libtorch, this couples libtorch into
the *entire* sixmax subsystem, which breaks the blueprint's cheap CPU path:

- The CPU Docker image (`docker/Dockerfile`) excludes `third_party/libtorch/`
  (correct for a torch-free blueprint) but `core` now needs libtorch, so
  `make docker-push` cannot rebuild cleanly. The GHCR image is stale (pre-DREAM).
- The local blueprint tooling fails to load `sixmax.so` without a libtorch on the
  dynamic-loader path (`diagnose_blueprint.py` needs a `DYLD_LIBRARY_PATH` hack).
- DREAM's GPU/ABI concerns leak into blueprint builds and the 270-test suite.

The blueprint is the maintained, still-improving strategy (diagnostic verdict
`undertraining` at 5M iters; +582 BB/100 vs uniform; +101 BB/100 gain 500k→5M).
DREAM is gated R&D. They must build and evolve independently.

## Goal

Build isolation via a **modular monolith** (one repo, hard build boundary). After
this change `sixmax.so` links **zero** libtorch; DREAM lives in its own
`sixmax_dream.so` extension. No separate repo / submodule — DREAM depends heavily
*on* sixmax internals (`engine`, `vocab`, `abstraction`, `blueprint/engine_game`)
and has no runtime service surface (its only integration point is the checkpoint
file consumed by the deploy agent), so a service/repo split is the wrong axis.

## Non-Goals

- DREAM batched-inference refactor / GPU throughput fix (separate future work;
  DREAM is ~12 iters/sec and GPU does not help until batched — tracked separately).
- Any change to blueprint or DREAM training behavior or outputs.
- Building/publishing a DREAM (GPU) Docker image.

## Current State

- `sixmax/BUCK`:
  - `core` = blueprint sources **+** `src/dream/*.cpp`, deps `//common:evaluator`,
    `//third_party:libtorch`.
  - `sixmax` extension = `src/bindings/bindings.cpp`, deps `:core`,
    `//third_party:pybind11`, `//third_party:libtorch`.
- `src/bindings/bindings.cpp`: single `PYBIND11_MODULE(sixmax, m)`. DREAM lives in
  includes (lines 17-21) and a registration block (~lines 362-524):
  `DreamMLP`, `DreamConfig`, `DreamTrainer`, `DreamStrategy`,
  `save_dream_checkpoint`, plus `encode_state_vec` helper.
- DREAM Python consumers:
  - `scripts/train_dream.py` — builds `//sixmax:sixmax`, constructs
    `sixmax.DreamTrainer(vocab.size(), vocab, abstraction, cfg, device)` where
    `vocab`/`abstraction` are sixmax objects.
  - `tests/sixmax/test_dream_{features,kuhn,checkpoint,nets}.py` — `import sixmax`.
  - `agents/sixmax_agent.py` (deploy) — references a Dream symbol (DreamStrategy /
    DreamDeployStrategy path).
- Extension loading: `agents/sixmax_agent.py::_ensure_sixmax_extension()` runs
  `buck2 build //sixmax:sixmax --show-output`, then loads the `.so` as
  `sys.modules['sixmax']` (the repo-root `sixmax/` package otherwise shadows it).
  `train_dream.py` does its own equivalent build+load. Test loading force-loads
  the `.so` via a conftest/fixture.

## Design

### Buck targets (`sixmax/BUCK`)

| Target | Sources | libtorch |
|--------|---------|----------|
| `core` | blueprint only (remove the 5 `src/dream/*.cpp`) | **removed** |
| `dream` (new `cxx_library`) | `src/dream/*.cpp` | yes; deps `:core`, `//third_party:libtorch` |
| `sixmax` (extension) | `src/bindings/bindings.cpp` | **removed**; deps `:core`, `//third_party:pybind11` |
| `sixmax_dream` (new extension) | `src/bindings/bindings_dream.cpp` | yes; deps `:core`, `:dream`, `//third_party:pybind11`, `//third_party:libtorch`; `soname = "sixmax_dream.so"` |

`core` keeps `exported_deps = ["//common:evaluator"]`. `dream` gets
`exported_preprocessor_flags`/headers as needed for its own headers.

### Bindings split (`sixmax/src/bindings/`)

- `bindings.cpp`: delete DREAM includes (17-21) and the DREAM registration block
  (~362-524, through the `encode_state_vec` def). Keep
  `PYBIND11_MODULE(sixmax, m)` with the blueprint/search API only. No torch types
  remain.
- `bindings_dream.cpp` (new): `PYBIND11_MODULE(sixmax_dream, m)` containing the
  moved DREAM block verbatim (DreamMLP, DreamConfig, DreamTrainer, DreamStrategy,
  save_dream_checkpoint, encode_state_vec). Includes the dream headers plus the
  core headers its signatures need (`vocab/vocab.h`, `blueprint/engine_game.h`,
  `abstraction/abstraction.h`).

### Cross-module type passing (primary risk)

`sixmax_dream.DreamTrainer(...)` receives `ActionVocab` and `Abstraction` objects
whose `py::class_` are registered in the **`sixmax`** module, and
`DreamStrategy.act` takes an `EngineGameState`. pybind11 shares its type registry
across extensions in one process via the global internals capsule, so this works
provided (a) both extensions link the same pybind11, and (b) `sixmax` is imported
before `sixmax_dream`.

Mitigation / sequencing: **validate this first**, before editing consumers, with a
smoke test — build both extensions, `import sixmax; import sixmax_dream`,
construct `sixmax.Abstraction(...)` + a vocab, pass into
`sixmax_dream.DreamTrainer(...)`, run `train(1)`. If it fails, fall back to
passing vocab/abstraction across the boundary by opaque handle (e.g. a
pointer/int) rather than the typed object. Only proceed to the mechanical
consumer edits once the smoke test passes.

### Python consumers

- Add `_ensure_sixmax_dream_extension()` (mirror of the sixmax loader; builds
  `//sixmax:sixmax_dream`, loads as `sys.modules['sixmax_dream']`). No repo-root
  `sixmax_dream/` package exists, so shadowing is not a concern. Factor the shared
  loader logic if clean.
- `scripts/train_dream.py`: import `sixmax` (vocab/abstraction) **and**
  `sixmax_dream` (trainer); `sixmax` first. Build both targets.
- `tests/sixmax/test_dream_*.py`: use `sixmax_dream` for Dream symbols; keep
  `sixmax` where they use blueprint types. Ensure the test conftest/fixture
  force-loads `sixmax_dream` too.
- `agents/sixmax_agent.py` (deploy path): retarget Dream symbol(s) to
  `sixmax_dream`, loading it via the new loader.

### Docker

No Dockerfile change needed: it targets `//sixmax:sixmax`, now torch-free, so the
existing `.dockerignore` exclusion of `third_party/libtorch/` is correct again and
`make docker-push` rebuilds cleanly. (A DREAM/GPU image is out of scope.)

## Verification

- `buck2 build //sixmax:sixmax` succeeds; `ldd`/`otool -L` on the `.so` shows
  **no** libtorch/c10.
- `buck2 build //sixmax:sixmax_dream` succeeds and links libtorch.
- Cross-module smoke test passes (see risk section).
- `uv run pytest tests/` — full suite green (dream tests on `sixmax_dream`).
- `diagnose_blueprint.py` and `eval_sixmax.py` run with **no** `DYLD_LIBRARY_PATH`
  hack.
- `make docker-build` produces a torch-free CPU image (blueprint trains in it).

## Rollout

1. Cross-module smoke test (spike) — gate.
2. Split bindings + BUCK targets; get `sixmax` torch-free and `sixmax_dream`
   building.
3. Retarget consumers (train_dream, tests, deploy agent) + add loader.
4. Verify (suite, ldd, diagnostics, docker build).
5. Rebuild + push CPU image; resume blueprint training on Hetzner (separate task).
