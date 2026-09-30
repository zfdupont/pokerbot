# Heads-Up Blueprint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a dedicated heads-up six-max blueprint checkpoint (`num_players = 2`) for the web service, and select it by heads-up evaluation.

**Architecture:** The sixmax `BlueprintTrainer` already supports 2 players and the deploy bridge (`SixmaxAgent`/`canonical_live_after`) is HU-ready. This plan adds a HU training config, a cloud-training config passthrough, the training run itself, and a heads-up evaluation/selection step. No engine or bridge code changes.

**Tech Stack:** C++ sixmax trainer (Buck2), `scripts/train_sixmax.py`, `scripts/cloud_train.sh` (Hetzner + Docker), `scripts/eval_hu_sanity.py`.

**Spec:** `docs/superpowers/specs/2026-09-30-poker-web-design.md`

## Global Constraints

- Checkpoints are never committed (`sixmax/checkpoints/` is gitignored); the artifact is produced locally/cloud and mounted at deploy time (Plan 4).
- The HU checkpoint must embed `num_players = 2` and the `sixmax/configs/hu.toml` vocab hash; `BlueprintStrategy.load` refuses mismatches.
- The abstraction config must match between training and evaluation within a comparison.
- Never commit secrets (`HCLOUD_TOKEN`, `GHCR_PAT`, `OPENSPIEL`, etc.).

## Corrections to the Spec

- **The "~15× slower HU" risk is stale.** Measured on this 8-core machine: HU = 3,000 iters / 54s (~55 it/s); 6-max = 3,000 iters / 61s (~49 it/s). The old note predates the bounded-bucket cache (2026-09-29). Budget HU ≈ 6-max per iteration.

## Review Focus

1. **Config precedence** — CLI must beat `[train.blueprint]` in `hu.toml`; unknown keys must still be rejected.
2. **Cloud passthrough** — `cloud_train.sh --config` must actually train with the chosen config (not silently fall back to `default.toml`).
3. **Checkpoint/vocab consistency** — the produced `.bin` must load through `BlueprintStrategy.load` with the `blueprint` vocab section.
4. **Selection honesty** — the chosen checkpoint is the best *evaluated* HU snapshot, not merely the last.
5. **No committed artifacts** — no `.bin`/`.pkl`/`.pt` enters git.

---

### Task 1: Heads-up training config

**Files:**
- Create: `sixmax/configs/hu.toml`
- Test: `tests/sixmax/test_train_sixmax.py` (add one test)

**Interfaces:**
- Consumes: `scripts.train_sixmax.resolve_config` (existing).
- Produces: `sixmax/configs/hu.toml` readable by `--config`, yielding `num_players=2`, a HU checkpoint path, and a `snapshot`-friendly `checkpoint_interval`.

- [ ] **Step 1: Write the failing test**

```python
def test_hu_config_sets_two_players():
    mod = _load_script()
    cfg = mod.resolve_config(
        _Args(config=os.path.join(_ROOT, "sixmax", "configs", "hu.toml")), _ROOT)
    assert cfg["num_players"] == 2
    assert cfg["checkpoint"].endswith("hu_blueprint.bin")
    assert cfg["checkpoint_interval"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_train_sixmax.py::test_hu_config_sets_two_players -v`
Expected: FAIL — `FileNotFoundError` (hu.toml missing).

- [ ] **Step 3: Create `sixmax/configs/hu.toml`**

A copy of `default.toml` with `[train.blueprint] num_players = 2`, `checkpoint = "sixmax/checkpoints/hu_blueprint.bin"`, `iterations = 5000000`, `checkpoint_interval = 500000`, `report_interval = 60`. Keep `[actions.blueprint]` and `[abstraction]` identical to `default.toml` so vocab + abstraction hashes stay compatible with the deploy bridge.

- [ ] **Step 4: Run the test, then the sixmax train tests**

Run: `uv run pytest tests/sixmax/test_train_sixmax.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sixmax/configs/hu.toml tests/sixmax/test_train_sixmax.py
git commit -m "feat(sixmax): add heads-up training config"
```

---

### Task 2: `cloud_train.sh --config` passthrough

**Files:**
- Modify: `scripts/cloud_train.sh`
- Test: `tests/scripts/test_cloud_train_flags.py` (create) — static/`bash -n` check

**Interfaces:**
- Consumes: the config path relative to the repo.
- Produces: `cloud_train.sh --config sixmax/configs/hu.toml [--checkpoint-name hu_blueprint.bin]` runs the trainer with `--config /pokerbot/<path>` and writes `--checkpoint /pokerbot/sixmax/checkpoints/<name>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_cloud_train_flags.py
import os, subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "cloud_train.sh")


def test_cloud_train_accepts_config_flag():
    # Syntax is valid and the flag is documented/parsed.
    assert subprocess.run(["bash", "-n", _SCRIPT]).returncode == 0
    text = open(_SCRIPT).read()
    assert "--config)" in text
    assert "--checkpoint-name)" in text
    assert "/pokerbot/$CONFIG" in text or "--config /pokerbot/$CONFIG" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_cloud_train_flags.py -v`
Expected: FAIL on the `--config)` assertion (flag not present).

- [ ] **Step 3: Add the flags**

In the arg-parse loop add `--config` and `--checkpoint-name`; default `CONFIG=""`, `CKPT_NAME="checkpoint.bin"`. Build `CONFIG_FLAG`/`CKPT_NAME` and splice them into the `docker run … train_sixmax.py` command (mirroring `RESUME_FLAG`). Update the `--help` usage line.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_cloud_train_flags.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud_train.sh tests/scripts/test_cloud_train_flags.py
git commit -m "feat(cloud): add --config/--checkpoint-name passthrough to cloud_train.sh"
```

---

### Task 3: Train the heads-up checkpoint (operational)

**Files:**
- Produces: `sixmax/checkpoints/hu_blueprint_<iters>.bin` (+ `hu_blueprint.bin`)

- [ ] **Step 1: Choose the compute path (ask the human partner)**

Local: 8 cores × ~55 it/s → 5M ≈ 25h (background). Cloud (`cloud_train.sh`, Hetzner ccx53+): faster, costs money — **requires explicit approval**.

- [ ] **Step 2: Run training with periodic snapshots**

Local example: `uv run python scripts/train_sixmax.py --config sixmax/configs/hu.toml --snapshots --checkpoint-interval 500000`
Cloud example: `nohup caffeinate scripts/cloud_train.sh --config sixmax/configs/hu.toml --checkpoint-name hu_blueprint.bin --iters 5000000 --checkpoint-interval 500000 &`

- [ ] **Step 3: Verify a snapshot loads**

Run: `uv run python -c "from agents.sixmax_agent import SixmaxDeployStrategy; s=SixmaxDeployStrategy.load('sixmax/checkpoints/hu_blueprint.bin','sixmax/configs/default.toml'); print(s.num_players)"`
Expected: prints `2`.

---

### Task 4: Evaluate and select the checkpoint (operational)

**Files:**
- Produces: a chosen `sixmax/checkpoints/hu_blueprint.bin`

- [ ] **Step 1: Evaluate snapshots heads-up**

Run: `uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/hu_blueprint_<iters>.bin --tabular cfr/checkpoints/checkpoint_09040000.pkl`
Compare against the 6-max blueprint for the same opponent: `--blueprint sixmax/checkpoints/blueprint.bin`.

- [ ] **Step 2: Pick the best snapshot**

Copy the best-evaluated snapshot to `sixmax/checkpoints/hu_blueprint.bin`.

- [ ] **Step 3: Record the result**

Add the achieved BB/100 numbers to `.mex/ROUTER.md`.

---

## Self-Review

**Spec coverage:** HU training config (Task 1), cloud passthrough (Task 2), training run (Task 3), selection by HU eval (Task 4). Deployment of the checkpoint is Plan 4. The frontend is Plan 3.

**Placeholders:** none.

**Type consistency:** `hu.toml` keys are validated against `resolve_config`'s `BUILTIN_DEFAULTS`; `checkpoint_interval`/`report_interval`/`snapshots` already exist.

## Execution Handoff

No subagent tool assumption needed for the code tasks; **Native** execution (executing-plans). Task 3's compute choice is a spend decision — it stops and asks.
