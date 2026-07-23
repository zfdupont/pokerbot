# Cloud Training on Hetzner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship two scripts — `cloud_benchmark.sh` (validates scaffold, measures throughput) and `cloud_train.sh` (full lifecycle: provision → train → sync → destroy) — backed by a Docker image that pre-compiles the `sixmax` C++ extension so remote runs need no build toolchain.

**Architecture:** Multi-stage Dockerfile bakes `sixmax.so` into a lean Ubuntu runtime image (~300MB, no libtorch needed). `train_sixmax.py` gains a `SIXMAX_SO_PATH` env var to skip Buck2 at runtime. Two shell scripts drive the full VPS lifecycle via the `hcloud` CLI; both share a common rsync-over-SSH checkpoint strategy.

**Tech Stack:** Hetzner Cloud (`hcloud` CLI), Docker + GHCR, Buck2 (`2026-06-30` release), Ubuntu 22.04 (x86_64), Python 3.10, Bash, rsync.

## Global Constraints

- Target architecture: `linux/amd64` (Hetzner CCX series = AMD x86_64). All Docker builds must use `--platform linux/amd64`.
- Buck2 version to pin: `2026-06-30` (matches local `~/bin/buck2 --version`).
- Never commit `HCLOUD_TOKEN`, `HETZNER_SSH_KEY_NAME`, or any other secrets. All credentials live in `.env` (already gitignored).
- Do not modify any file under `sixmax/src/`, `common/`, `cfr/`, or `neural_cfr/` except `sixmax/BUCK` (Task 1). No changes to `scripts/train_sixmax.py` internals except the `SIXMAX_SO_PATH` hook (Task 2).
- The spec says no changes to existing training code — only the two targeted modifications in Tasks 1 and 2 are permitted.
- Hourly rates used in scripts are approximate — verify at https://www.hetzner.com/cloud at implementation time.

---

## File Map

| Status | Path | Responsibility |
|--------|------|----------------|
| Modify | `sixmax/BUCK` | Platform-conditional linker flag; configurable cpu_tune |
| Modify | `scripts/train_sixmax.py` | `SIXMAX_SO_PATH` env var bypass for Buck2 build |
| Create | `docker/Dockerfile` | Multi-stage image: builder compiles sixmax.so; runtime has Python + .so |
| Create | `docker/.dockerignore` | Exclude third_party/libtorch, buck-out, checkpoints from build context |
| Create | `Makefile` | `docker-push`, `install-hooks` targets |
| Create | `scripts/cloud_benchmark.sh` | Scaffold validation + throughput measurement + cost table |
| Create | `scripts/cloud_train.sh` | Full VPS lifecycle: provision → train → sync → destroy |

---

## Task 1: Fix sixmax/BUCK for Linux

The current `sixmax/BUCK` has two macOS-isms that break Linux builds:
- `linker_flags = ["-undefined", "dynamic_lookup"]` — Apple ld64 only; GNU ld rejects it.
- `"-mcpu=native"` — when cross-compiling from ARM Mac via Docker emulation, "native" resolves to ARM features. Make it configurable via `.buckconfig.local`.

**Files:**
- Modify: `sixmax/BUCK`
- Test: run local build to confirm it still works

**Interfaces:**
- Produces: `sixmax/BUCK` with `host_info().os.is_macos` guard and `read_config("build", "cpu_tune", "native")` for the CPU flag. Later tasks rely on this for the Docker builder stage to succeed.

- [ ] **Step 1: Read the current BUCK file**

```
cat sixmax/BUCK
```

Confirm the two flags are present as described above.

- [ ] **Step 2: Apply the fix**

Replace the entire file content:

```python
_python_include = read_config("python", "include_path", "")
_cpu_tune = read_config("build", "cpu_tune", "native")

cxx_library(
    name = "sixmax",
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp", "src/blueprint/mccfr.cpp", "src/blueprint/trainer.cpp",
            "src/engine/engine.cpp", "src/blueprint/engine_game.cpp", "src/abstraction/abstraction.cpp", "src/abstraction/abstract_key.cpp", "src/blueprint/checkpoint.cpp"],
    headers = glob(["src/**/*.h"]),
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"] +
                     (["-I" + _python_include] if _python_include else []),
    deps = ["//common:evaluator", "//third_party:pybind11"],
    exported_preprocessor_flags = ["-Isixmax/src"],
    preferred_linkage = "shared",
    soname = "sixmax.so",
    linker_flags = ["-undefined", "dynamic_lookup"] if host_info().os.is_macos else [],
    visibility = ["PUBLIC"],
)
```

- [ ] **Step 3: Verify local build still passes**

```bash
~/bin/buck2 build //sixmax:sixmax
```

Expected: exits 0, no linker errors.

- [ ] **Step 4: Verify the extension loads**

```bash
uv run python scripts/train_sixmax.py --iterations 100 --checkpoint /tmp/smoke_task1.bin
```

Expected: prints "Effective config: ..." and runs 100 iterations without error.

- [ ] **Step 5: Commit**

```bash
git add sixmax/BUCK
git commit -m "fix(sixmax): platform-conditional linker flag + configurable cpu_tune for Docker cross-compile"
```

---

## Task 2: Add SIXMAX_SO_PATH bypass to train_sixmax.py

`train_sixmax.py` always calls Buck2 via `_build_and_get_so_dir()` at startup. The Docker runtime image has no Buck2, so it would fail immediately. A `SIXMAX_SO_PATH` env var lets the container skip the build and load a pre-compiled `.so` directly.

**Files:**
- Modify: `scripts/train_sixmax.py` (the `_force_load_sixmax` function only)
- Create: `tests/test_train_sixmax_env.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `_force_load_sixmax(repo_root)` checks `os.environ.get("SIXMAX_SO_PATH")` before calling `_build_and_get_so_dir`. The Docker image (Task 3) will set `ENV SIXMAX_SO_PATH=/opt/sixmax.so`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_sixmax_env.py`:

```python
import importlib
import sys
from unittest.mock import patch


def test_sixmax_so_path_skips_build(monkeypatch, tmp_path):
    """When SIXMAX_SO_PATH is set, _build_and_get_so_dir must not be called."""
    monkeypatch.setenv("SIXMAX_SO_PATH", str(tmp_path / "fake.so"))

    # Force reload so the module re-reads the env var
    import scripts.train_sixmax as ts
    importlib.reload(ts)

    build_calls = []
    with patch.object(ts, "_build_and_get_so_dir", side_effect=lambda _: build_calls.append(1) or ""):
        try:
            ts._force_load_sixmax("/fake/root")
        except Exception:
            pass  # Expected — fake .so won't load

    assert build_calls == [], "_build_and_get_so_dir must not be called when SIXMAX_SO_PATH is set"


def test_sixmax_so_path_absent_calls_build(monkeypatch):
    """When SIXMAX_SO_PATH is unset, _build_and_get_so_dir must be called."""
    monkeypatch.delenv("SIXMAX_SO_PATH", raising=False)

    import scripts.train_sixmax as ts
    importlib.reload(ts)

    build_calls = []
    with patch.object(ts, "_build_and_get_so_dir", side_effect=lambda _: build_calls.append(1) or (_ for _ in ()).throw(RuntimeError("stop"))):
        try:
            ts._force_load_sixmax("/fake/root")
        except Exception:
            pass

    assert build_calls == [1], "_build_and_get_so_dir must be called when SIXMAX_SO_PATH is absent"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
uv run pytest tests/test_train_sixmax_env.py -v
```

Expected: both tests FAIL (function not yet modified).

- [ ] **Step 3: Apply the minimal fix to _force_load_sixmax**

In `scripts/train_sixmax.py`, find `_force_load_sixmax` (currently ~lines 47–56) and replace it with:

```python
def _force_load_sixmax(repo_root: str):
    """Load the .so and register it as sys.modules['sixmax'] (the repo-root
    sixmax/ directory is a namespace package that would win otherwise)."""
    so_path = os.environ.get("SIXMAX_SO_PATH")
    if not so_path:
        so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/test_train_sixmax_env.py -v
```

Expected: both PASS.

- [ ] **Step 5: Confirm normal training still works (no env var)**

```bash
uv run python scripts/train_sixmax.py --iterations 100 --checkpoint /tmp/smoke_task2.bin
```

Expected: builds via Buck2 as before, runs 100 iterations.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/ -v
```

Expected: all 178+ tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_sixmax.py tests/test_train_sixmax_env.py
git commit -m "feat(train): SIXMAX_SO_PATH env var skips Buck2 build for Docker runtime"
```

---

## Task 3: Docker Image + Makefile

Multi-stage Dockerfile. Builder stage installs build tools, Buck2, and Python 3.10 headers, then compiles `sixmax.so`. Runtime stage copies only the `.so` and installs Python deps — no build toolchain, no `third_party/libtorch/` (~200MB), no `buck-out/`.

The image is built for `linux/amd64` regardless of host architecture (important on Apple Silicon). `Makefile` provides `make docker-push` and `make install-hooks`.

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/.dockerignore`
- Create: `Makefile`

**Interfaces:**
- Consumes: Task 1 (`sixmax/BUCK` with Linux-compatible flags), Task 2 (`SIXMAX_SO_PATH` env var convention).
- Produces:
  - Docker image `ghcr.io/zfdupont/pokerbot-trainer:latest` with `sixmax.so` at `/opt/sixmax.so` and `ENV SIXMAX_SO_PATH=/opt/sixmax.so`.
  - `make docker-push` command usable in Tasks 4 and 5.

- [ ] **Step 1: Create docker/.dockerignore**

```
# Large vendored deps not needed to build sixmax (only needed for neural_cfr)
third_party/libtorch/

# Build artifacts
buck-out/
**/__pycache__/
**/*.pyc

# Checkpoints (large, don't belong in build context)
sixmax/checkpoints/
neural_cfr/checkpoints/
cfr/checkpoints/

# Git
.git/

# Dev / local files
.env
.venv/
*.local
```

- [ ] **Step 2: Create docker/Dockerfile**

```dockerfile
# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        python3.10 \
        python3.10-dev \
        curl \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Pin Buck2 to match local version (~/bin/buck2 --version = 2026-06-30)
RUN curl -fL \
    "https://github.com/facebook/buck2/releases/download/2026-06-30/buck2-x86_64-unknown-linux-gnu.zst" \
    | zstd -d - -o /usr/local/bin/buck2 \
    && chmod +x /usr/local/bin/buck2

WORKDIR /build

# Copy Buck2 config and build infrastructure (cached layers — changes rarely)
COPY .buckconfig /build/
COPY toolchains/ /build/toolchains/
COPY third_party/ /build/third_party/
COPY common/ /build/common/
COPY BUCK /build/

# Set python include path and cpu_tune for cross-compile (x86-64, not native)
RUN python3.10 -c \
    "import sysconfig; p=sysconfig.get_path('include'); \
     open('.buckconfig.local','w').write(f'[python]\n  include_path = {p}\n[build]\n  cpu_tune = x86-64\n')"

# Copy sixmax source (separate layer so C++ changes don't bust the cache above)
COPY sixmax/ /build/sixmax/

# Build the extension
RUN /usr/local/bin/buck2 build //sixmax:sixmax

# Extract the .so to a known path
RUN find /build/buck-out -name "sixmax.so" -exec cp {} /opt/sixmax.so \; \
    && test -f /opt/sixmax.so

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3.10 \
        python3.10-distutils \
        curl \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fLsS https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy compiled extension
COPY --from=builder /opt/sixmax.so /opt/sixmax.so
ENV SIXMAX_SO_PATH=/opt/sixmax.so

# Install Python deps (system-wide — no venv needed in container)
COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system \
        "numpy>=1.26" \
        "tomli>=2.0" \
        "tqdm>=4.66"

WORKDIR /pokerbot
```

- [ ] **Step 3: Create Makefile**

```makefile
IMAGE := ghcr.io/zfdupont/pokerbot-trainer
PLATFORM := linux/amd64

.PHONY: docker-build docker-push install-hooks

docker-build:
	docker buildx build --platform $(PLATFORM) \
		-f docker/Dockerfile \
		-t $(IMAGE):latest \
		.

docker-push: docker-build
	docker push $(IMAGE):latest

install-hooks:
	@cp scripts/hooks/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "pre-push hook installed"
```

- [ ] **Step 4: Create the pre-push hook**

Create directory and file `scripts/hooks/pre-push`:

```bash
#!/usr/bin/env bash
# Warn if sixmax C++ source is dirty but the Docker image hasn't been rebuilt.
set -euo pipefail

DIRTY=$(git diff --name-only HEAD -- sixmax/src/ common/src/ third_party/ sixmax/BUCK common/BUCK 2>/dev/null)
if [ -n "$DIRTY" ]; then
    echo "⚠️  WARNING: C++ source changed since last Docker push:"
    echo "$DIRTY" | sed 's/^/   /'
    echo "   Run 'make docker-push' before starting a cloud training run."
fi
exit 0
```

- [ ] **Step 5: Test the Docker build**

```bash
make docker-build
```

Expected: exits 0. Takes 5–15 min on first build (QEMU emulation on Apple Silicon), ~1 min on subsequent runs with layer cache.

- [ ] **Step 6: Verify sixmax imports correctly in the container**

```bash
docker run --rm ghcr.io/zfdupont/pokerbot-trainer:latest \
    python3.10 -c "import sixmax; print('sixmax loaded:', sixmax)"
```

Expected: prints something like `sixmax loaded: <module 'sixmax' from '/opt/sixmax.so'>`.

- [ ] **Step 7: Commit**

```bash
git add docker/Dockerfile docker/.dockerignore Makefile scripts/hooks/pre-push
git commit -m "feat(docker): multi-stage image with pre-compiled sixmax.so for cloud training"
```

---

## Task 4: cloud_benchmark.sh

Validates the full provision → pull → run → sync → destroy scaffold cheaply (~$0.05, ~5 min). Also supports `--local` mode (free, macOS profiling via `sample`) and `--profile` (callgrind on Linux, `sample` on macOS).

**Files:**
- Create: `scripts/cloud_benchmark.sh`

**Interfaces:**
- Consumes: Task 3 Docker image (`ghcr.io/zfdupont/pokerbot-trainer:latest`). Requires `HCLOUD_TOKEN` and `HETZNER_SSH_KEY_NAME` in env (from `.env`).
- Produces: printed cost table; synced test checkpoint to `sixmax/checkpoints/bench_<timestamp>.bin`; optional `callgrind.out` or `sample.txt` in `sixmax/checkpoints/`.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# cloud_benchmark.sh — provision a Hetzner VPS, run a short timed training job,
# print throughput + cost extrapolation table, sync results, destroy VPS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTANCE_TYPE="ccx53"
ITERS=100000
PROFILE=false
LOCAL=false
IMAGE="ghcr.io/zfdupont/pokerbot-trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCH_CHECKPOINT="sixmax/checkpoints/bench_${TIMESTAMP}.bin"

# Approximate hourly rates in USD — verify at https://www.hetzner.com/cloud
declare -A HOURLY_RATES=(
    ["ccx23"]="0.14"
    ["ccx33"]="0.21"
    ["ccx43"]="0.30"
    ["ccx53"]="0.38"
    ["ccx63"]="0.56"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)     INSTANCE_TYPE="$2"; shift 2 ;;
        --iters)    ITERS="$2";         shift 2 ;;
        --profile)  PROFILE=true;       shift ;;
        --local)    LOCAL=true;         shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

HOURLY_RATE="${HOURLY_RATES[$INSTANCE_TYPE]:-0.38}"

# ── Helpers ───────────────────────────────────────────────────────────────────
print_cost_table() {
    local iters_per_sec="$1"
    local iters_per_hr
    iters_per_hr=$(echo "$iters_per_sec * 3600" | bc)

    echo ""
    echo "  Instance:   $INSTANCE_TYPE"
    echo "  Throughput: ${iters_per_sec} iter/sec  (${iters_per_hr} iter/hr)"
    echo "  Cost/hr:    \$${HOURLY_RATE}"
    echo ""
    echo "  Extrapolated:"
    for target in 1000000 10000000 50000000; do
        hrs=$(echo "scale=1; $target / $iters_per_hr" | bc)
        cost=$(echo "scale=2; $hrs * $HOURLY_RATE" | bc)
        label=$(printf "%8d" $target | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')
        echo "    ${label} iters → ${hrs} hrs → \$${cost}"
    done
    echo ""
}

# ── Local mode ────────────────────────────────────────────────────────────────
run_local() {
    echo "[cloud_benchmark] Running locally (no VPS)"
    cd "$REPO_ROOT"

    if $PROFILE; then
        echo "[cloud_benchmark] Profiling with sample (macOS)"
        uv run python scripts/train_sixmax.py \
            --iterations "$ITERS" \
            --checkpoint "$BENCH_CHECKPOINT" &
        TRAIN_PID=$!
        sleep 2
        sample "$TRAIN_PID" 30 -f /tmp/pokerbot_sample_${TIMESTAMP}.txt 2>/dev/null || true
        wait "$TRAIN_PID"
        echo "[cloud_benchmark] Profile saved to /tmp/pokerbot_sample_${TIMESTAMP}.txt"
    else
        START=$(date +%s)
        uv run python scripts/train_sixmax.py \
            --iterations "$ITERS" \
            --checkpoint "$BENCH_CHECKPOINT"
        END=$(date +%s)
        ELAPSED=$((END - START))
        ITERS_PER_SEC=$((ITERS / ELAPSED))
        echo ""
        echo "Benchmark complete (local)"
        print_cost_table "$ITERS_PER_SEC"
    fi
}

# ── Remote mode ───────────────────────────────────────────────────────────────
run_remote() {
    : "${HCLOUD_TOKEN:?HCLOUD_TOKEN not set — add to .env}"
    : "${HETZNER_SSH_KEY_NAME:?HETZNER_SSH_KEY_NAME not set — add to .env}"

    SERVER_NAME="pokerbot-bench-${TIMESTAMP}"
    SERVER_IP=""

    cleanup() {
        echo ""
        echo "[cloud_benchmark] Syncing results..."
        if [ -n "$SERVER_IP" ]; then
            rsync -avz --ignore-errors \
                "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
                "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
            if $PROFILE; then
                rsync -avz --ignore-errors \
                    "root@${SERVER_IP}:/tmp/pokerbot/callgrind.out" \
                    "${REPO_ROOT}/sixmax/checkpoints/callgrind_${TIMESTAMP}.out" 2>/dev/null || true
            fi
            echo "[cloud_benchmark] Destroying $SERVER_NAME..."
            hcloud server delete "$SERVER_NAME" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT ERR INT

    echo "[cloud_benchmark] Creating $INSTANCE_TYPE server..."
    hcloud server create \
        --name "$SERVER_NAME" \
        --type "$INSTANCE_TYPE" \
        --image ubuntu-22.04 \
        --ssh-key "$HETZNER_SSH_KEY_NAME" \
        --location nbg1

    SERVER_IP=$(hcloud server describe "$SERVER_NAME" -o json \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['public_net']['ipv4']['ip'])")
    echo "[cloud_benchmark] Server IP: $SERVER_IP"

    echo "[cloud_benchmark] Waiting for SSH..."
    until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        "root@$SERVER_IP" echo ok 2>/dev/null; do sleep 5; done

    echo "[cloud_benchmark] Syncing repo source..."
    rsync -avz --exclude='third_party/' --exclude='buck-out/' \
        --exclude='sixmax/checkpoints/' --exclude='neural_cfr/checkpoints/' \
        --exclude='cfr/checkpoints/' --exclude='.git/' \
        --exclude='__pycache__/' --exclude='*.pyc' \
        "${REPO_ROOT}/" "root@${SERVER_IP}:/tmp/pokerbot/"

    echo "[cloud_benchmark] Pulling Docker image..."
    ssh "root@$SERVER_IP" "docker pull $IMAGE"

    DOCKER_CMD="docker run --rm \
        -v /tmp/pokerbot/sixmax/configs:/pokerbot/sixmax/configs:ro \
        -v /tmp/pokerbot/sixmax/checkpoints:/pokerbot/sixmax/checkpoints \
        -v /tmp/pokerbot/scripts:/pokerbot/scripts:ro \
        -v /tmp/pokerbot/sixmax:/pokerbot/sixmax:ro \
        -v /tmp/pokerbot/agents:/pokerbot/agents:ro \
        -e SIXMAX_SO_PATH=/opt/sixmax.so \
        --name pokerbot-bench \
        $IMAGE \
        python3.10 /pokerbot/scripts/train_sixmax.py \
            --iterations $ITERS \
            --checkpoint /pokerbot/sixmax/checkpoints/bench_${TIMESTAMP}.bin"

    if $PROFILE; then
        echo "[cloud_benchmark] Running with callgrind profiling (this will be slow)..."
        ssh "root@$SERVER_IP" "apt-get install -y valgrind -qq && \
            valgrind --tool=callgrind --callgrind-out-file=/tmp/pokerbot/callgrind.out \
            $DOCKER_CMD"
    else
        echo "[cloud_benchmark] Running timed benchmark..."
        START_REMOTE=$(date +%s)
        ssh "root@$SERVER_IP" "$DOCKER_CMD"
        END_REMOTE=$(date +%s)
        ELAPSED=$((END_REMOTE - START_REMOTE))
        ITERS_PER_SEC=$((ITERS / ELAPSED))

        echo ""
        echo "Benchmark complete"
        print_cost_table "$ITERS_PER_SEC"
    fi
}

# ── Entry point ───────────────────────────────────────────────────────────────
# Load .env if present
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

if $LOCAL; then
    run_local
else
    run_remote
fi
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/cloud_benchmark.sh
```

- [ ] **Step 3: Test local mode (free, ~1 min)**

```bash
scripts/cloud_benchmark.sh --local --iters 10000
```

Expected: runs 10k iterations locally, prints a cost table with "Throughput:", "Cost/hr:", and "Extrapolated:" sections. A checkpoint file appears at `sixmax/checkpoints/bench_<timestamp>.bin`.

- [ ] **Step 4: Verify cost table math**

Manually check one row: if throughput = 90 iter/sec → 324,000 iter/hr → 1M iters = 3.09 hrs × $0.38 = $1.17. Confirm the printed numbers match.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud_benchmark.sh
git commit -m "feat(cloud): cloud_benchmark.sh — scaffold validation + throughput measurement"
```

---

## Task 5: cloud_train.sh

Full VPS lifecycle: provision → rsync source → pull image → run training → mid-run sync every 30 min → watchdog kills run at `--max-hours` → final sync → destroy.

**Files:**
- Create: `scripts/cloud_train.sh`

**Interfaces:**
- Consumes: Task 3 Docker image. Requires `HCLOUD_TOKEN` and `HETZNER_SSH_KEY_NAME` in env. Optionally consumes a local checkpoint path via `--resume`.
- Produces: checkpoint files synced to `sixmax/checkpoints/` on the local machine.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# cloud_train.sh — provision Hetzner VPS, run full sixmax blueprint training,
# sync checkpoints every 30 min + on exit, then destroy the VPS.
#
# Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] \
#            [--resume path/to/ckpt.bin] [--max-hours 24]
#
# Run long jobs with: nohup caffeinate scripts/cloud_train.sh ... &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTANCE_TYPE="ccx53"
ITERS=10000000
RESUME_CHECKPOINT=""
MAX_HOURS=24
IMAGE="ghcr.io/zfdupont/pokerbot-trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SERVER_NAME="pokerbot-train-${TIMESTAMP}"
SERVER_IP=""
SYNC_PID=""
WATCHDOG_PID=""
START_TIME=$(date +%s)

declare -A HOURLY_RATES=(
    ["ccx23"]="0.14"
    ["ccx33"]="0.21"
    ["ccx43"]="0.30"
    ["ccx53"]="0.38"
    ["ccx63"]="0.56"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)       INSTANCE_TYPE="$2"; shift 2 ;;
        --iters)      ITERS="$2";         shift 2 ;;
        --resume)     RESUME_CHECKPOINT="$2"; shift 2 ;;
        --max-hours)  MAX_HOURS="$2";     shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

HOURLY_RATE="${HOURLY_RATES[$INSTANCE_TYPE]:-0.38}"

# ── Cleanup (runs on any exit path) ──────────────────────────────────────────
cleanup() {
    # Kill background loops
    [ -n "$SYNC_PID" ]     && kill "$SYNC_PID"     2>/dev/null || true
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null || true

    if [ -n "$SERVER_IP" ]; then
        echo ""
        ELAPSED=$(( $(date +%s) - START_TIME ))
        COST=$(echo "scale=2; $ELAPSED / 3600 * $HOURLY_RATE" | bc)
        echo "[cloud_train] Final sync from $SERVER_IP..."
        rsync -avz --ignore-errors \
            "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
            "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
        echo "[cloud_train] Total elapsed: ${ELAPSED}s — estimated cost: \$${COST}"
        echo "[cloud_train] Destroying $SERVER_NAME..."
        hcloud server delete "$SERVER_NAME" 2>/dev/null || true
    fi
}
trap cleanup EXIT ERR INT

# ── Validation ────────────────────────────────────────────────────────────────
: "${HCLOUD_TOKEN:?HCLOUD_TOKEN not set — add to .env}"
: "${HETZNER_SSH_KEY_NAME:?HETZNER_SSH_KEY_NAME not set — add to .env}"

if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "Error: resume checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
fi

# ── Provision ─────────────────────────────────────────────────────────────────
echo "[cloud_train] Creating $INSTANCE_TYPE server ($SERVER_NAME)..."
hcloud server create \
    --name "$SERVER_NAME" \
    --type "$INSTANCE_TYPE" \
    --image ubuntu-22.04 \
    --ssh-key "$HETZNER_SSH_KEY_NAME" \
    --location nbg1

SERVER_IP=$(hcloud server describe "$SERVER_NAME" -o json \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['public_net']['ipv4']['ip'])")
echo "[cloud_train] Server IP: $SERVER_IP"

echo "[cloud_train] Waiting for SSH..."
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    "root@$SERVER_IP" echo ok 2>/dev/null; do sleep 5; done

# ── Sync source ───────────────────────────────────────────────────────────────
echo "[cloud_train] Syncing repo source..."
rsync -avz \
    --exclude='third_party/' --exclude='buck-out/' \
    --exclude='sixmax/checkpoints/' --exclude='neural_cfr/checkpoints/' \
    --exclude='cfr/checkpoints/' --exclude='.git/' \
    --exclude='__pycache__/' --exclude='*.pyc' \
    "${REPO_ROOT}/" "root@${SERVER_IP}:/tmp/pokerbot/"

# Sync resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "[cloud_train] Uploading resume checkpoint..."
    ssh "root@$SERVER_IP" "mkdir -p /tmp/pokerbot/sixmax/checkpoints"
    rsync -avz "$RESUME_CHECKPOINT" \
        "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/resume.bin"
fi

# ── Pull image ────────────────────────────────────────────────────────────────
echo "[cloud_train] Pulling Docker image..."
ssh "root@$SERVER_IP" "docker pull $IMAGE"

# ── Mid-run sync loop (background) ───────────────────────────────────────────
sync_count=0
mid_sync_loop() {
    while true; do
        sleep 1800  # 30 min
        sync_count=$((sync_count + 1))
        ELAPSED=$(( $(date +%s) - START_TIME ))
        COST=$(echo "scale=2; $ELAPSED / 3600 * $HOURLY_RATE" | bc)
        REMAINING=$(echo "scale=2; $MAX_HOURS - $ELAPSED / 3600" | bc)
        echo "[sync ${sync_count}] elapsed ${ELAPSED}s  est. cost: \$${COST}  remaining budget hrs: ${REMAINING}"
        rsync -avz --ignore-errors \
            "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
            "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
    done
}
mid_sync_loop &
SYNC_PID=$!

# ── Watchdog (background) ─────────────────────────────────────────────────────
(
    sleep $((MAX_HOURS * 3600))
    echo "[cloud_train] Max hours ($MAX_HOURS) reached — stopping container..."
    ssh "root@$SERVER_IP" "docker stop pokerbot-trainer 2>/dev/null || true"
) &
WATCHDOG_PID=$!

# ── Training ──────────────────────────────────────────────────────────────────
RESUME_FLAG=""
[ -n "$RESUME_CHECKPOINT" ] && RESUME_FLAG="--resume /pokerbot/sixmax/checkpoints/resume.bin"

echo "[cloud_train] Starting training ($ITERS iterations)..."
ssh "root@$SERVER_IP" "docker run \
    --name pokerbot-trainer \
    -v /tmp/pokerbot/sixmax/configs:/pokerbot/sixmax/configs:ro \
    -v /tmp/pokerbot/sixmax/checkpoints:/pokerbot/sixmax/checkpoints \
    -v /tmp/pokerbot/scripts:/pokerbot/scripts:ro \
    -v /tmp/pokerbot/sixmax:/pokerbot/sixmax:ro \
    -v /tmp/pokerbot/agents:/pokerbot/agents:ro \
    -e SIXMAX_SO_PATH=/opt/sixmax.so \
    $IMAGE \
    python3.10 /pokerbot/scripts/train_sixmax.py \
        --iterations $ITERS \
        --checkpoint /pokerbot/sixmax/checkpoints/checkpoint.bin \
        $RESUME_FLAG"

# Watchdog no longer needed (training completed normally)
kill "$WATCHDOG_PID" 2>/dev/null || true
WATCHDOG_PID=""

echo "[cloud_train] Training complete."
# cleanup() handles final sync + destroy via EXIT trap
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/cloud_train.sh
```

- [ ] **Step 3: Validate the script parses flags correctly (free, no VPS)**

Add a temporary `--dry-run` check by running with a fake token but checking validation:

```bash
HCLOUD_TOKEN=fake HETZNER_SSH_KEY_NAME=fake \
    scripts/cloud_train.sh --type ccx53 --iters 5000000 --max-hours 12 2>&1 | head -5
```

Expected: prints "[cloud_train] Creating ccx53 server..." then fails at the actual `hcloud` call (not at flag parsing). Confirms flags parsed correctly.

- [ ] **Step 4: Run the real end-to-end validation via cloud_benchmark.sh**

Before trusting `cloud_train.sh` with a long run, validate the full remote scaffold is working:

```bash
# Requires real HCLOUD_TOKEN and HETZNER_SSH_KEY_NAME in .env
source .env
scripts/cloud_benchmark.sh --type ccx23 --iters 50000
```

Expected: provisions a CCX23 (cheapest, ~$0.14/hr), runs 50k iterations (~30 sec), prints cost table, syncs a checkpoint to `sixmax/checkpoints/`, destroys VPS. Total cost < $0.05.

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud_train.sh
git commit -m "feat(cloud): cloud_train.sh — full Hetzner VPS lifecycle with mid-run sync + watchdog"
```

---

## Post-Implementation Checklist

After all tasks are committed:

- [ ] Run `make docker-push` to publish the image to GHCR.
- [ ] Run `make install-hooks` to install the pre-push warning hook.
- [ ] Add to `.env` (never commit):
  ```
  HCLOUD_TOKEN=<your-hetzner-api-token>
  HETZNER_SSH_KEY_NAME=<key-name-registered-in-hetzner-console>
  ```
- [ ] Set a monthly spending limit of $25–30 in the Hetzner web console as a backstop.
- [ ] Run the first real benchmark: `scripts/cloud_benchmark.sh` (no flags) to get actual throughput numbers and validate the complete remote pipeline.
