# Global Iteration Counter + Best-Checkpoint Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make linear-CFR sample weights use a globally monotonic iteration counter that persists across `run()` calls and checkpoint resume, and automatically track the best checkpoint (by vs-tabular eval) during training.

**Architecture:** The counter is a `Trainer` member serialized into the checkpoint as an additive `"meta"` sub-archive (legacy checkpoints load with counter 0). Best-checkpoint selection lives entirely in the Python driver (`train_neural.py`): after each save it shells out to the existing `eval_neural_vs_tabular.py`, parses the BB/100, and maintains `best_checkpoint.pt` + a JSON sidecar on strict improvement.

**Tech Stack:** C++17 / libtorch serialization / pybind11 / Buck2 (`~/bin/buck2`), Python 3.10+ (`uv run`), GoogleTest, pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-global-iters-best-checkpoint-design.md`

## Global Constraints

- Build/test: `~/bin/buck2 build //neural_cfr:neural_cfr`, `~/bin/buck2 test //neural_cfr/tests:test_trainer` (Buck2 is at `~/bin/buck2`, NOT on PATH). Python: `uv run …` from repo root.
- Checkpoint format compatibility: existing sub-archives `adv0`/`adv1`/`strat` unchanged; `"meta"` is additive; legacy checkpoints (no meta) must load with `total_iters_ = 0` without throwing. `Strategy` reads only `"strat"` — untouched.
- Python 3.10 compatibility for everything `train_neural.py` imports at top level (uv env is 3.10; `tomllib` fallback to `tomli` already in place — keep it).
- Config precedence contract: CLI flag > TOML file > `BUILTIN_DEFAULTS`. New keys (exact names/defaults): `selection_enabled=False`, `selection_hands=10_000`, `selection_tabular_checkpoint=""`; `default.toml` sets `selection_enabled = true`; `smoke.toml` omits all three.
- Selection must never abort training: eval subprocess failure/timeout/unparseable output → warn and continue.
- No training process is currently running; the branch is `feature/neural-cfr` — work directly on it (small sequential feature, no worktrees needed).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Global iteration counter (C++ + binding)

**Files:**
- Modify: `neural_cfr/src/cfr/trainer.h` (member, getter)
- Modify: `neural_cfr/src/cfr/trainer.cpp` (`run()`, `checkpoint()`, `load()`)
- Modify: `neural_cfr/src/bindings/bindings.cpp` (bind getter — the `py::class_<Trainer>` block only)
- Test: `neural_cfr/tests/test_trainer.cpp` (append)

**Interfaces:**
- Consumes: existing `Trainer` API (8-arg constructor `(reservoir_size, batch_size, lr, train_interval, num_threads, epsilon, sgd_steps, reinit_adv)`; `run/checkpoint/load`).
- Produces: `int64_t Trainer::total_iterations() const;` — cumulative traversal-pair count across all `run()` calls and restored by `load()`. Bound to Python as `trainer.total_iterations()` (Task 3 passes it into the sidecar).

- [ ] **Step 1: Write the failing tests**

Append to `neural_cfr/tests/test_trainer.cpp`:

```cpp
// --- Global iteration counter -------------------------------------------

TEST(Trainer, TotalIterationsAccumulatesAcrossRuns) {
    // Cheap continual config: no reinit, 1 SGD step, tiny batches.
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    EXPECT_EQ(t.total_iterations(), 0);
    t.run(50);
    EXPECT_EQ(t.total_iterations(), 50);
    t.run(70);   // second run() call must continue, not reset
    EXPECT_EQ(t.total_iterations(), 120);
}

TEST(Trainer, TotalIterationsRoundTripsThroughCheckpoint) {
    const std::string path = "/tmp/neural_cfr_test_meta_ckpt.pt";
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t.run(50);
    t.checkpoint(path);

    Trainer t2(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t2.load(path);
    EXPECT_EQ(t2.total_iterations(), 50);
    std::remove(path.c_str());
}

TEST(Trainer, LegacyCheckpointWithoutMetaLoadsAsZero) {
    // Craft a legacy-format checkpoint: only the three net sub-archives.
    const std::string path = "/tmp/neural_cfr_test_legacy_ckpt.pt";
    {
        MLP m0, m1, ms;
        torch::serialize::OutputArchive root, a0, a1, s;
        m0.save(a0); m1.save(a1); ms.save(s);
        root.write("adv0", a0);
        root.write("adv1", a1);
        root.write("strat", s);
        root.save_to(path);
    }
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t.run(50);                      // counter nonzero before load
    t.load(path);                   // must not throw
    EXPECT_EQ(t.total_iterations(), 0);  // legacy ⇒ counter resets to 0
    std::remove(path.c_str());
}
```

Note: the spec also suggested asserting buffer weights directly; the buffers
are private and adding an accessor just for the test violates YAGNI. The
weight is computed as `total_iters_ + index + 1` from the same counter the
getter reports (verified in Step 3's code), so the getter tests cover it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: compile FAILURE — `total_iterations` is not a member of `Trainer`.

- [ ] **Step 3: Implement**

In `neural_cfr/src/cfr/trainer.h`, inside `class Trainer`'s public section
(after `void load(const std::string& path);`), add:

```cpp
    // Cumulative traversal-pair count across all run() calls; persisted in
    // checkpoints ("meta" sub-archive) so linear-CFR weights stay globally
    // monotonic across chunks and resumes.
    int64_t total_iterations() const { return total_iters_; }
```

and in the private section (after `bool reinit_adv_;`):

```cpp
    int64_t total_iters_ = 0;
```

In `neural_cfr/src/cfr/trainer.cpp`:

1. In `run()`, the worker lambda currently computes
   `int global_iter = completed + index + 1;` — replace with:

```cpp
                int global_iter = (int)(total_iters_ + index + 1);
```

   (`total_iters_` is stable while workers run — it is only updated after
   `join()`. The `int` cast is safe to 2.1B iterations; `BufferEntry.weight`
   is float, exact to 16.7M — both fine at our scale.)

   Immediately after the existing `completed += batch;` line add:

```cpp
        total_iters_ += batch;
```

2. In `checkpoint()`, after the three existing `root.write(...)` calls and
   before `root.save_to(path);`, add:

```cpp
    torch::serialize::OutputArchive meta;
    meta.write("total_iters", torch::tensor((int64_t)total_iters_));
    root.write("meta", meta);
```

3. In `load()`, after the three existing sub-archive loads and before the
   final `std::cout` line, add:

```cpp
    try {
        torch::serialize::InputArchive meta;
        root.read("meta", meta);
        torch::Tensor t;
        meta.read("total_iters", t);
        total_iters_ = t.item<int64_t>();
    } catch (const std::exception&) {
        total_iters_ = 0;  // legacy checkpoint without meta
    }
```

In `neural_cfr/src/bindings/bindings.cpp`, in the `py::class_<Trainer>`
block, after the `.def("load", ...)` line add:

```cpp
        .def("total_iterations", &Trainer::total_iterations,
             "Cumulative traversal-pair count (persists across checkpoints)");
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: all tests PASS (3 pre-existing + 3 new; the tiny runs take ~1–2 min total).

- [ ] **Step 5: Build the extension and check the binding**

Run: `~/bin/buck2 build //neural_cfr:neural_cfr --show-output`
Expected: `BUILD SUCCEEDED`. Then (substitute the printed .so dir):

```bash
uv run python -c "
import ctypes, sys
for lib in ['libc10.dylib','libtorch_cpu.dylib','libtorch.dylib']:
    ctypes.CDLL('third_party/libtorch/lib/'+lib)
sys.path.insert(0, '<SO_DIR_FROM_BUCK_OUTPUT>')
import neural_cfr
t = neural_cfr.Trainer(reservoir_size=1000, batch_size=64, train_interval=50, num_threads=2, sgd_steps=1, reinit_adv=False)
t.run(50)
print('total_iterations:', t.total_iterations())
assert t.total_iterations() == 50
print('OK')
"
```

Expected: `total_iterations: 50` then `OK`.

- [ ] **Step 6: Commit**

```bash
git add neural_cfr/src/cfr/trainer.h neural_cfr/src/cfr/trainer.cpp neural_cfr/src/bindings/bindings.cpp neural_cfr/tests/test_trainer.cpp
git commit -m "feat: global iteration counter for linear-CFR weights

Weights previously reset to t=1 on every run() call (once per checkpoint
interval) and on every resume, degrading linear CFR to per-chunk weighting.
The counter now accumulates across run() calls and persists in checkpoints
via an additive 'meta' sub-archive; legacy checkpoints load with counter 0.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Selection helpers + config keys (Python, pure functions)

**Files:**
- Modify: `scripts/train_neural.py` (add `parse_bb100`, `update_best`, three `BUILTIN_DEFAULTS` keys, three argparse flags — NOT the loop integration; that's Task 3)
- Test: `tests/scripts/test_train_neural_config.py` (append)

**Interfaces:**
- Consumes: existing module-level pattern in `train_neural.py` (`BUILTIN_DEFAULTS`, `resolve_config`, stdlib-only top-level imports; `json`, `os` already imported — add `shutil` and `datetime` at top level, both stdlib).
- Produces (Task 3 calls these exact signatures):
  - `parse_bb100(text: str) -> float | None` — extracts the win rate from `eval_neural_vs_tabular.py` stdout.
  - `update_best(bb100: float, ckpt_path: str, total_iters: int, hands: int) -> bool` — sidecar + copy on strict improvement; sidecar and `best_checkpoint.pt` live in `os.path.dirname(ckpt_path)`.
  - Config keys `selection_enabled` (bool, False), `selection_hands` (int, 10_000), `selection_tabular_checkpoint` (str, "").

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_train_neural_config.py`:

```python
# --- best-checkpoint selection helpers -------------------------------------

EVAL_OUTPUT = """\
Hands              : 10000
Running 5000 hands (neural=P0, tabular=P1) ...
Running 5000 hands (tabular=P0, neural=P1) ...
=== Neural CFR vs Tabular MCCFR (10000 hands) ===
  Neural win rate     : -46.7 BB/100
  Near 0 BB/100 = strategies agree; positive = neural has edge; negative = tabular wins
"""


def test_parse_bb100_extracts_value(train_neural):
    assert train_neural.parse_bb100(EVAL_OUTPUT) == pytest.approx(-46.7)
    assert train_neural.parse_bb100(
        EVAL_OUTPUT.replace("-46.7", "+3.2")) == pytest.approx(3.2)


def test_parse_bb100_returns_none_on_garbage(train_neural):
    assert train_neural.parse_bb100("no win rate here") is None
    assert train_neural.parse_bb100("") is None


def _fake_ckpt(tmp_path, name="checkpoint.pt", content=b"weights"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_update_best_creates_sidecar_and_copy(train_neural, tmp_path):
    import json
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    assert train_neural.update_best(-46.7, ckpt, total_iters=500_000,
                                    hands=10_000) is True
    best = tmp_path / "best_checkpoint.pt"
    sidecar = tmp_path / "best_checkpoint.json"
    assert best.read_bytes() == b"v1"
    meta = json.loads(sidecar.read_text())
    assert meta["bb100"] == pytest.approx(-46.7)
    assert meta["total_iters"] == 500_000
    assert meta["hands"] == 10_000
    assert meta["source_checkpoint"].endswith("checkpoint.pt")
    assert "timestamp" in meta


def test_update_best_replaces_on_strict_improvement(train_neural, tmp_path):
    import json
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    train_neural.update_best(-46.7, ckpt, total_iters=1, hands=10)
    (tmp_path / "checkpoint.pt").write_bytes(b"v2")
    assert train_neural.update_best(-9.3, ckpt, total_iters=2,
                                    hands=10) is True
    assert (tmp_path / "best_checkpoint.pt").read_bytes() == b"v2"
    meta = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert meta["bb100"] == pytest.approx(-9.3)


def test_update_best_noop_on_worse_or_equal(train_neural, tmp_path):
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    train_neural.update_best(-9.3, ckpt, total_iters=1, hands=10)
    (tmp_path / "checkpoint.pt").write_bytes(b"v2")
    assert train_neural.update_best(-46.7, ckpt, total_iters=2,
                                    hands=10) is False   # worse
    assert train_neural.update_best(-9.3, ckpt, total_iters=2,
                                    hands=10) is False   # equal (strict)
    assert (tmp_path / "best_checkpoint.pt").read_bytes() == b"v1"


def test_selection_config_defaults_and_precedence(train_neural, tmp_path):
    cfg = train_neural.resolve_config(_args(), str(tmp_path))
    assert cfg["selection_enabled"] is False
    assert cfg["selection_hands"] == 10_000
    assert cfg["selection_tabular_checkpoint"] == ""

    toml = tmp_path / "cfg.toml"
    toml.write_text("[training]\nselection_enabled = true\nselection_hands = 500\n")
    cfg = train_neural.resolve_config(
        _args(config=str(toml), selection_hands=777), str(tmp_path))
    assert cfg["selection_enabled"] is True   # from file
    assert cfg["selection_hands"] == 777      # CLI wins
```

Also extend the `_args` helper's `from_keys` list in the same file with the
three new keys:

```python
        "selection_enabled", "selection_hands", "selection_tabular_checkpoint",
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scripts/test_train_neural_config.py -v`
Expected: new tests FAIL with `module 'train_neural' has no attribute 'parse_bb100'`; pre-existing 5 still pass.

- [ ] **Step 3: Implement in `scripts/train_neural.py`**

Add `import re`, `import shutil`, and `from datetime import datetime, timezone`
to the top-level imports (all stdlib, py3.10-safe).

Add the three keys to `BUILTIN_DEFAULTS` (after `"eval_hands": 500,`):

```python
    "selection_enabled":            False,
    "selection_hands":              10_000,
    "selection_tabular_checkpoint": "",
```

Add module-level functions (after `write_config_snapshot`):

```python
_BB100_RE = re.compile(r"Neural win rate\s*:\s*([+-]?\d+(?:\.\d+)?)\s*BB/100")


def parse_bb100(text: str) -> "float | None":
    """Extract the BB/100 win rate from eval_neural_vs_tabular.py output."""
    m = _BB100_RE.search(text)
    return float(m.group(1)) if m else None


def update_best(bb100: float, ckpt_path: str, total_iters: int,
                hands: int) -> bool:
    """Keep best_checkpoint.pt + sidecar next to ckpt_path.

    Replaces on strict improvement only (bounds winner's-curse churn).
    Returns True if the best checkpoint was replaced.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    sidecar = os.path.join(ckpt_dir, "best_checkpoint.json")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            if bb100 <= json.load(f)["bb100"]:
                return False
    shutil.copy2(ckpt_path, os.path.join(ckpt_dir, "best_checkpoint.pt"))
    with open(sidecar, "w") as f:
        json.dump({
            "bb100": bb100,
            "total_iters": total_iters,
            "hands": hands,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": os.path.abspath(ckpt_path),
        }, f, indent=2)
    return True
```

Add the three argparse flags in `main()` (next to the existing `--config` group):

```python
    parser.add_argument("--selection-enabled", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Track best_checkpoint.pt by vs-tabular eval after each save (default: off)")
    parser.add_argument("--selection-hands", type=int, default=None,
                        help="Hands per selection eval (default: 10000)")
    parser.add_argument("--selection-tabular-checkpoint", type=str, default=None,
                        help="Tabular checkpoint for selection evals (default: auto-detect)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_train_neural_config.py -v`
Expected: all tests PASS (5 pre-existing + 7 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_neural.py tests/scripts/test_train_neural_config.py
git commit -m "feat: parse_bb100/update_best helpers + selection config keys

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Driver loop integration, configs, smoke, docs

**Files:**
- Modify: `scripts/train_neural.py` (selection call in the main loop)
- Modify: `neural_cfr/configs/default.toml` (enable selection)
- Modify: `.mex/context/neural-cfr.md` (checkpoint-format + training notes)
- Test: end-to-end smoke run (no new test files)

**Interfaces:**
- Consumes: Task 1's `trainer.total_iterations()`; Task 2's `parse_bb100`, `update_best`, and the three `cfg["selection_*"]` keys.
- Produces: `run_selection(cfg: dict, repo_root: str, trainer) -> None` in `scripts/train_neural.py` — called after every checkpoint save; never raises.

- [ ] **Step 1: Implement `run_selection` and wire the loop**

Add after `update_best` in `scripts/train_neural.py`:

```python
def run_selection(cfg: dict, repo_root: str, trainer) -> None:
    """Eval the just-saved checkpoint vs tabular; update best_checkpoint.pt.

    Never raises: selection is advisory and must not kill training.
    """
    ckpt = cfg["checkpoint"]
    cmd = [sys.executable,
           os.path.join(repo_root, "scripts", "eval_neural_vs_tabular.py"),
           "--neural-checkpoint", ckpt,
           "--hands", str(cfg["selection_hands"])]
    if cfg["selection_tabular_checkpoint"]:
        cmd += ["--tabular-checkpoint", cfg["selection_tabular_checkpoint"]]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=repo_root, timeout=3600)
        bb100 = parse_bb100(result.stdout)
        if result.returncode != 0 or bb100 is None:
            print(f"[selection] eval failed (exit {result.returncode}); skipping. "
                  f"stderr tail: {result.stderr[-300:]}")
            return
        replaced = update_best(bb100, ckpt, trainer.total_iterations(),
                               cfg["selection_hands"])
        print(f"[selection] {bb100:+.1f} BB/100 vs tabular @ "
              f"{trainer.total_iterations():,} iters — "
              f"{'NEW BEST → best_checkpoint.pt' if replaced else 'kept existing best'}")
    except Exception as e:  # noqa: BLE001 — advisory path, never fatal
        print(f"[selection] skipped ({type(e).__name__}: {e})")
```

In `main()`'s training loop, immediately after each
`write_config_snapshot(cfg, cfg["checkpoint"])` call, add:

```python
        if cfg["selection_enabled"]:
            run_selection(cfg, repo_root, trainer)
```

- [ ] **Step 2: Enable in `default.toml`**

In `neural_cfr/configs/default.toml`, append to the `[training]` section:

```toml
selection_enabled          = true      # keep best_checkpoint.pt by vs-tabular eval
selection_hands            = 10_000    # SE ≈ ±10 BB/100 per selection eval
selection_tabular_checkpoint = ""      # "" = auto-detect latest tabular ckpt
```

(`smoke.toml` intentionally unchanged — selection stays off there.)

- [ ] **Step 3: Full test regression**

Run: `uv run pytest tests/ -q` and `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: all pass (Python suite ~102 tests + 6 trainer tests).

- [ ] **Step 4: End-to-end smoke**

Run (~10–15 min: 2 CFR iterations + two 500-hand selection evals):

```bash
uv run python scripts/train_neural.py \
    --iterations 20000 --train-interval 10000 --sgd-steps 100 \
    --checkpoint-interval 10000 \
    --checkpoint /tmp/ncfr_sel_smoke.pt \
    --selection-enabled --selection-hands 500
```

Expected: two `[selection] … BB/100 vs tabular @ 10,000 iters` /
`@ 20,000 iters` lines (the first always `NEW BEST`); afterwards
`/tmp/best_checkpoint.pt` and `/tmp/best_checkpoint.json` exist, and the
sidecar's `total_iters` is `10000` or `20000`. Verify:

```bash
cat /tmp/best_checkpoint.json
```

- [ ] **Step 5: Update docs**

In `.mex/context/neural-cfr.md`:
1. Checkpoint-format note: append — `Checkpoints also carry a "meta"
   sub-archive with the cumulative iteration counter (total_iterations());
   legacy checkpoints load with counter 0. Linear-CFR weights are globally
   monotonic across chunks and resumes.`
2. Training regime/commands notes: append — `With selection_enabled (on in
   default.toml), the driver evals each save vs the tabular baseline
   (selection_hands, default 10k) and keeps the best in best_checkpoint.pt
   + best_checkpoint.json (strict-improvement replacement).`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_neural.py neural_cfr/configs/default.toml .mex/context/neural-cfr.md
git commit -m "feat: best_checkpoint selection by vs-tabular eval after each save

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
