# DREAM Build Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sixmax.so` link zero libtorch by moving the DREAM neural code into its own `sixmax_dream.so` extension, restoring the blueprint's torch-free CPU build/image/test path.

**Architecture:** Modular monolith — one repo, hard build boundary. Split the Buck `core` library into a torch-free blueprint `core` plus a `dream` library, and split the single pybind module into a torch-free `sixmax` extension and a new libtorch-linked `sixmax_dream` extension. DREAM keeps depending on sixmax core internals (engine/vocab/abstraction) at compile time; the runtime integration seam stays the checkpoint file.

**Tech Stack:** C++17, libtorch (C++), pybind11, Buck2 (`~/bin/buck2`), Python ≥3.10, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-26-dream-build-isolation-design.md`

## Global Constraints

- Python ≥3.10; Buck2 invoked as `~/bin/buck2` (user-local, not on PATH).
- Never touch vendored `third_party/` files; `third_party/libtorch` is macOS CPU libtorch locally.
- `sixmax/` vocab is config-defined; never reorder/filter action storage.
- `cfr/`, `neural_cfr/`, `sixmax/` never import each other or `game/poker.py`; shared C++ only via `common/`.
- After this change, `sixmax.so` must contain **no** libtorch/c10 dynamic dependency.
- Commit messages: no Co-Authored-By / Claude-Session / "Generated with" trailers (per user global rules).
- Work stays on branch `dream-build-isolation`.

## Review Focus

- **Extension import order** — importing `sixmax_dream` before `sixmax` can break cross-module pybind type resolution. Loader must import `sixmax` first. (Task 1 test.)
- **Blueprint runtime path re-importing libtorch** — the whole goal regresses if `SixmaxDeployStrategy` / a bare blueprint import pulls in `sixmax_dream`/libtorch. Deploy must load `sixmax_dream` lazily. (Task 4 test.)
- **DREAM deploy construction after retarget** — `DreamDeployStrategy` must still construct from a real DREAM checkpoint via `sixmax_dream.DreamStrategy.load`, loaded lazily. (Task 4 test.)
- **openpoker_bot `.pt` auto-detect** — DREAM checkpoint routing must still resolve after the deploy retarget. (Task 4 test.)
- **Cross-module custom-type passing** — `sixmax.Abstraction`/`ActionVocab` passed into `sixmax_dream.DreamTrainer` must actually work in one process. (Task 1 gate test.)

---

### Task 1: Split Buck targets + pybind module; dual-load conftest; isolation gate

**Files:**
- Modify: `sixmax/BUCK`
- Modify: `sixmax/src/bindings/bindings.cpp` (remove DREAM include block lines 17-21 and DREAM registration block ~lines 362-524)
- Create: `sixmax/src/bindings/bindings_dream.cpp`
- Modify: `tests/sixmax/conftest.py`
- Test (create): `tests/sixmax/test_build_isolation.py`

**Interfaces:**
- Produces: Buck targets `//sixmax:core` (torch-free), `//sixmax:dream`, `//sixmax:sixmax` (torch-free ext, `sixmax.so`), `//sixmax:sixmax_dream` (`sixmax_dream.so`). Python module `sixmax_dream` exposing `DreamMLP`, `DreamConfig`, `DreamTrainer(n_actions, vocab, abstraction, cfg, device_str)`, `DreamStrategy`, `save_dream_checkpoint`, `encode_state_vec`.
- Consumes: `sixmax` module types `ActionVocab`, `Abstraction`, `EngineGameState` (registered in `sixmax`, used across the module boundary).

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_build_isolation.py`:

```python
import os
import platform
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUCK = os.path.expanduser("~/bin/buck2")


def _so_path(target, soname):
    out = subprocess.run([_BUCK, "build", target, "--show-output"],
                         capture_output=True, text=True, cwd=_ROOT)
    assert out.returncode == 0, out.stderr
    for line in out.stdout.splitlines():
        if soname in line:
            return os.path.join(_ROOT, line.split()[-1])
    raise AssertionError(f"{soname} not in buck output: {out.stdout}")


def test_sixmax_so_is_torch_free():
    so = _so_path("//sixmax:sixmax", "sixmax.so")
    tool = ["otool", "-L", so] if platform.system() == "Darwin" else ["ldd", so]
    linkage = subprocess.run(tool, capture_output=True, text=True).stdout.lower()
    assert "libtorch" not in linkage
    assert "libc10" not in linkage


def test_sixmax_dream_builds_and_links_torch():
    so = _so_path("//sixmax:sixmax_dream", "sixmax_dream.so")
    tool = ["otool", "-L", so] if platform.system() == "Darwin" else ["ldd", so]
    linkage = subprocess.run(tool, capture_output=True, text=True).stdout.lower()
    assert "libtorch" in linkage


def test_cross_module_dreamtrainer(default_vocab):
    # sixmax and sixmax_dream are force-loaded by conftest (sixmax first)
    import sixmax
    import sixmax_dream
    abstraction = sixmax.Abstraction(
        flop_buckets=4, turn_buckets=4, river_buckets=4,
        equity_rollouts=5, quantile_samples=100, seed=1)
    cfg = sixmax_dream.DreamConfig()
    cfg.train_interval = 4
    cfg.sgd_steps = 1
    cfg.batch_size = 8
    cfg.reservoir_size = 1000
    tr = sixmax_dream.DreamTrainer(default_vocab.size(), default_vocab,
                                   abstraction, cfg, "cpu")
    tr.train(1)
    assert tr.total_iterations() >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_build_isolation.py -v`
Expected: FAIL — `sixmax.so` currently links libtorch and `//sixmax:sixmax_dream` / module `sixmax_dream` do not exist.

- [ ] **Step 3: Rewrite `sixmax/BUCK`**

Replace the two existing `cxx_library` targets with four. `core` drops the 5 `src/dream/*.cpp` sources and the `//third_party:libtorch` dep; `sixmax` drops `//third_party:libtorch`; add `dream` and `sixmax_dream`:

```python
_python_include = read_config("python", "include_path", "")
_cpu_tune = read_config("build", "cpu_tune", "native")

cxx_library(
    name = "core",
    srcs = [
        "src/vocab/vocab.cpp",
        "src/blueprint/kuhn.cpp",
        "src/blueprint/mccfr.cpp",
        "src/blueprint/trainer.cpp",
        "src/engine/engine.cpp",
        "src/blueprint/engine_game.cpp",
        "src/abstraction/abstraction.cpp",
        "src/abstraction/abstract_key.cpp",
        "src/blueprint/checkpoint.cpp",
    ],
    headers = glob(["src/**/*.h"]),
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"],
    exported_preprocessor_flags = ["-Isixmax/src"],
    deps = ["//common:evaluator"],
    exported_deps = ["//common:evaluator"],
    preferred_linkage = "static",
    visibility = ["PUBLIC"],
)

cxx_library(
    name = "dream",
    srcs = [
        "src/dream/features.cpp",
        "src/dream/nets.cpp",
        "src/dream/reservoir.cpp",
        "src/dream/trainer.cpp",
        "src/dream/checkpoint.cpp",
    ],
    headers = glob(["src/dream/*.h"]),
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"],
    exported_preprocessor_flags = ["-Isixmax/src"],
    deps = [":core", "//third_party:libtorch"],
    exported_deps = [":core", "//third_party:libtorch"],
    preferred_linkage = "static",
    visibility = ["PUBLIC"],
)

cxx_library(
    name = "sixmax",
    srcs = ["src/bindings/bindings.cpp"],
    headers = [],
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"] +
                     (["-I" + _python_include] if _python_include else []),
    deps = [":core", "//third_party:pybind11"],
    preferred_linkage = "shared",
    soname = "sixmax.so",
    linker_flags = ["-undefined", "dynamic_lookup"] if host_info().os.is_macos else [],
    visibility = ["PUBLIC"],
)

cxx_library(
    name = "sixmax_dream",
    srcs = ["src/bindings/bindings_dream.cpp"],
    headers = [],
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"] +
                     (["-I" + _python_include] if _python_include else []),
    deps = [":core", ":dream", "//third_party:pybind11", "//third_party:libtorch"],
    preferred_linkage = "shared",
    soname = "sixmax_dream.so",
    linker_flags = ["-undefined", "dynamic_lookup"] if host_info().os.is_macos else [],
    visibility = ["PUBLIC"],
)
```

- [ ] **Step 4: Move the DREAM block out of `bindings.cpp`**

In `sixmax/src/bindings/bindings.cpp`: delete the DREAM includes (lines 17-21: `dream/features.h`, `dream/nets.h`, `dream/reservoir.h`, `dream/trainer.h`, `dream/checkpoint.h`) and the entire DREAM registration block (from the `// --- Dream neural CFR (Task 7) ---` comment ~line 362 through the `encode_state_vec` def ending ~line 524). Leave `PYBIND11_MODULE(sixmax, m)` with only blueprint/search registrations. Confirm no `torch::` token remains: `grep -n torch sixmax/src/bindings/bindings.cpp` returns nothing.

- [ ] **Step 5: Create `sixmax/src/bindings/bindings_dream.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vocab/vocab.h"
#include "blueprint/engine_game.h"
#include "abstraction/abstraction.h"
#include "dream/features.h"
#include "dream/nets.h"
#include "dream/reservoir.h"
#include "dream/trainer.h"
#include "dream/checkpoint.h"

namespace py = pybind11;

PYBIND11_MODULE(sixmax_dream, m) {
    m.doc() = "Six-max DREAM neural CFR (isolated libtorch extension)";

    // <<< PASTE the DREAM registration block removed from bindings.cpp here,
    //     verbatim: PyDreamMLP struct, DreamMLP/DreamConfig/DreamTrainer/
    //     DreamStrategy py::class_ bindings, save_dream_checkpoint and
    //     encode_state_vec m.def(...) — unchanged except the enclosing module
    //     is now `m` from PYBIND11_MODULE(sixmax_dream, m). >>>
}
```

- [ ] **Step 6: Add dual-load to `tests/sixmax/conftest.py`**

Keep the existing libtorch dylib preload (it must run before `sixmax_dream` is dlopen'd). Replace the single-module discovery/load block with a helper that loads both, `sixmax` first:

```python
def _discover_so_dir(target: str, soname: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", target, "--show-output"],
                            capture_output=True, text=True, cwd=_REPO_ROOT)
    for line in result.stdout.splitlines():
        if soname in line:
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError(f"Could not locate {soname}\nstdout: {result.stdout}\nstderr: {result.stderr}")


def _force_load(module_name: str, target: str) -> None:
    soname = module_name + ".so"
    so_dir = _discover_so_dir(target, soname)
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    so_path = os.path.join(so_dir, soname)
    spec = _ilu.spec_from_file_location(module_name, so_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod  # register before exec so circular refs work
    spec.loader.exec_module(mod)


_force_load("sixmax", "//sixmax:sixmax")            # must be first (cross-module types)
_force_load("sixmax_dream", "//sixmax:sixmax_dream")
```

- [ ] **Step 7: Run the isolation gate tests**

Run: `uv run pytest tests/sixmax/test_build_isolation.py -v`
Expected: PASS (all three). If `test_cross_module_dreamtrainer` fails with a type-conversion error, the pybind cross-module registry is not shared — STOP and report; fallback is passing vocab/abstraction by opaque handle.

- [ ] **Step 8: Commit**

```bash
git add sixmax/BUCK sixmax/src/bindings/bindings.cpp sixmax/src/bindings/bindings_dream.cpp tests/sixmax/conftest.py tests/sixmax/test_build_isolation.py
git commit -m "refactor(sixmax): isolate DREAM into sixmax_dream extension; sixmax.so torch-free"
```

---

### Task 2: Retarget DREAM unit tests to `sixmax_dream`

**Files:**
- Modify: `tests/sixmax/test_dream_features.py`
- Modify: `tests/sixmax/test_dream_nets.py`
- Modify: `tests/sixmax/test_dream_checkpoint.py`
- Modify: `tests/sixmax/test_dream_kuhn.py`

**Interfaces:**
- Consumes: `sixmax_dream` (Dream symbols), `sixmax` (blueprint types like `ActionVocab`, `Abstraction`), both force-loaded by the Task 1 conftest.

- [ ] **Step 1: Point each test's Dream symbols at `sixmax_dream`**

In each of the four files, change usages of `sixmax.DreamMLP` / `sixmax.DreamConfig` / `sixmax.DreamTrainer` / `sixmax.DreamStrategy` / `sixmax.save_dream_checkpoint` / `sixmax.encode_state_vec` to the `sixmax_dream.` namespace. Add `import sixmax_dream` where a file uses a Dream symbol. Keep `import sixmax` where the file uses blueprint types (vocab/abstraction/engine).

Example (in `test_dream_nets.py`):

```python
import sixmax          # blueprint types
import sixmax_dream    # DREAM types

def test_dreammlp_forward_vec_shapes():
    net = sixmax_dream.DreamMLP(10, 64, 2, 6)
    out = net.forward_vec([0.0] * 10)
    assert len(out) == 6
    assert all(math.isfinite(v) for v in out)
```

- [ ] **Step 2: Run the DREAM unit tests**

Run: `uv run pytest tests/sixmax/test_dream_features.py tests/sixmax/test_dream_nets.py tests/sixmax/test_dream_checkpoint.py tests/sixmax/test_dream_kuhn.py -v`
Expected: PASS (all previously-passing assertions, now via `sixmax_dream`).

- [ ] **Step 3: Commit**

```bash
git add tests/sixmax/test_dream_features.py tests/sixmax/test_dream_nets.py tests/sixmax/test_dream_checkpoint.py tests/sixmax/test_dream_kuhn.py
git commit -m "test(sixmax): retarget DREAM unit tests to sixmax_dream"
```

---

### Task 3: Retarget `scripts/train_dream.py` to `sixmax_dream`

**Files:**
- Modify: `scripts/train_dream.py`

**Interfaces:**
- Consumes: `sixmax` (vocab via `vocab_config`, `Abstraction`), `sixmax_dream` (`DreamConfig`, `DreamTrainer`).

- [ ] **Step 1: Build and load `sixmax_dream` alongside `sixmax`**

In `train_dream.py`, after the existing `//sixmax:sixmax` build+load, add a second build+load for `//sixmax:sixmax_dream` registered as `sys.modules["sixmax_dream"]` (mirror the existing discovery loop, matching `sixmax_dream.so`). Keep the libtorch dylib preload (needed for `sixmax_dream`).

- [ ] **Step 2: Route Dream constructors through `sixmax_dream`**

Change `dream_cfg = sixmax.DreamConfig()` → `sixmax_dream.DreamConfig()`, and `trainer = sixmax.DreamTrainer(...)` → `sixmax_dream.DreamTrainer(...)`. Keep `sixmax.Abstraction(...)` and the sixmax-loaded `vocab` as the objects passed in.

- [ ] **Step 3: Smoke-run one iteration**

Run: `uv run python scripts/train_dream.py --config sixmax/configs/default.toml --checkpoint /tmp/dream_smoke.pt --iterations 1 --device cpu`
Expected: exits 0, prints `Saved -> /tmp/dream_smoke.pt`, and `test -f /tmp/dream_smoke.pt` succeeds. (Abstraction build adds ~30-60s; that is expected.)

- [ ] **Step 4: Commit**

```bash
git add scripts/train_dream.py
git commit -m "refactor(scripts): train_dream.py uses sixmax_dream extension"
```

---

### Task 4: Retarget DREAM deploy with lazy `sixmax_dream` load

**Files:**
- Modify: `agents/sixmax_agent.py` (the `DreamDeployStrategy` class, ~lines 150-190)
- Test (create): `tests/sixmax/test_dream_deploy_isolation.py`

**Interfaces:**
- Consumes: `sixmax_dream.DreamStrategy.load(path, device_str, vocab)`.
- Produces: `DreamDeployStrategy` that loads `sixmax_dream` lazily (only when a DREAM checkpoint is deployed), leaving the blueprint deploy path (`SixmaxDeployStrategy`) torch-free.

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_dream_deploy_isolation.py`:

```python
import sys
import importlib


def test_importing_agent_module_does_not_load_dream():
    # Fresh import of the deploy module must not pull in sixmax_dream/libtorch.
    for m in list(sys.modules):
        if m == "sixmax_dream":
            del sys.modules[m]
    importlib.import_module("agents.sixmax_agent")
    assert "sixmax_dream" not in sys.modules, \
        "blueprint deploy path must not import sixmax_dream at module load"


def test_dream_deploy_loads_and_acts(tmp_path, default_vocab):
    # Train a tiny DREAM checkpoint, then deploy it and get a legal action.
    import sixmax
    import sixmax_dream
    abstraction = sixmax.Abstraction(
        flop_buckets=4, turn_buckets=4, river_buckets=4,
        equity_rollouts=5, quantile_samples=100, seed=1)
    cfg = sixmax_dream.DreamConfig()
    cfg.train_interval = 4; cfg.sgd_steps = 1; cfg.batch_size = 8; cfg.reservoir_size = 1000
    tr = sixmax_dream.DreamTrainer(default_vocab.size(), default_vocab, abstraction, cfg, "cpu")
    tr.train(1)
    ckpt = str(tmp_path / "dream.pt")
    tr.save(ckpt, default_vocab.hash())

    from agents.sixmax_agent import DreamDeployStrategy
    strat = DreamDeployStrategy(ckpt, "sixmax/configs/default.toml")
    assert strat is not None                # constructed from a real DREAM checkpoint
    assert "sixmax_dream" in sys.modules    # loaded lazily on construction, not at import
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/sixmax/test_dream_deploy_isolation.py -v`
Expected: FAIL — `DreamDeployStrategy` currently references `sixmax.DreamStrategy`, which no longer exists after Task 1.

- [ ] **Step 3: Add a lazy loader and retarget `DreamDeployStrategy`**

In `agents/sixmax_agent.py`, add a module-level lazy loader (do NOT call it at import time):

```python
def _ensure_sixmax_dream_extension():
    """Lazily force-load sixmax_dream (keeps blueprint deploy path torch-free)."""
    mod = sys.modules.get("sixmax_dream")
    if mod is not None and hasattr(mod, "DreamTrainer"):
        return mod
    _ensure_sixmax_extension()  # sixmax first (cross-module types)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax_dream", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = next((os.path.join(_ROOT, os.path.dirname(l.split()[-1]))
                   for l in result.stdout.splitlines() if "sixmax_dream.so" in l), None)
    if so_dir is None:
        raise RuntimeError("Could not locate sixmax_dream.so in buck2 output")
    so_path = os.path.join(so_dir, "sixmax_dream.so")
    spec = importlib.util.spec_from_file_location("sixmax_dream", so_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sixmax_dream"] = m
    spec.loader.exec_module(m)
    return m
```

In `DreamDeployStrategy.__init__`, replace the `sixmax.DreamStrategy.load(...)` call with:

```python
sixmax_dream = _ensure_sixmax_dream_extension()
strategy = sixmax_dream.DreamStrategy.load(path, "cpu", vocab)
```

- [ ] **Step 4: Run to verify passing**

Run: `uv run pytest tests/sixmax/test_dream_deploy_isolation.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Verify openpoker_bot `.pt` routing still resolves**

Run: `uv run python -c "import ast,sys; ast.parse(open('scripts/openpoker_bot.py').read()); print('parse-ok')"` then grep to confirm it references `DreamDeployStrategy` from `agents.sixmax_agent` (import path unchanged):
Run: `grep -n "DreamDeployStrategy" scripts/openpoker_bot.py`
Expected: import resolves to `agents.sixmax_agent.DreamDeployStrategy` (unchanged); no code change needed since the class name/location is preserved.

- [ ] **Step 6: Commit**

```bash
git add agents/sixmax_agent.py tests/sixmax/test_dream_deploy_isolation.py
git commit -m "refactor(agents): DreamDeployStrategy lazily loads sixmax_dream; blueprint deploy stays torch-free"
```

---

### Task 5: Full verification sweep + torch-free CPU Docker image

**Files:**
- None (verification); if the emulated Docker build surfaces a libtorch reference, fix is confined to `docker/.dockerignore` / `docker/Dockerfile`.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/`
Expected: all tests green (270+ including the new isolation and deploy tests).

- [ ] **Step 2: Prove blueprint tooling runs without the libtorch hack**

Run (NO `DYLD_LIBRARY_PATH` set): `uv run python scripts/diagnose_blueprint.py --checkpoints "sixmax/checkpoints/blueprint_0*.bin" --top-frac 0.05`
Expected: prints the per-checkpoint metric table and a `VERDICT` line, with no `@rpath/libtorch.dylib` load error.

- [ ] **Step 3: Build the torch-free CPU Docker image**

Run: `make docker-build`
Expected: build succeeds through `buck2 build //sixmax:sixmax` and `test -f /opt/sixmax.so`, with `third_party/libtorch/` still excluded by `docker/.dockerignore`. (This is an emulated `linux/amd64` build on arm64 macOS and may take several minutes.)

- [ ] **Step 4: Commit any final touches**

```bash
git add -A
git commit -m "chore(sixmax): verify torch-free blueprint build + CPU image" --allow-empty
```
