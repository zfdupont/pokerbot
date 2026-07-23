# Cloud Training Design — Six-Max Blueprint on Hetzner

**Date:** 2026-07-22  
**Status:** Approved  
**Scope:** Six-max blueprint (`sixmax/`) only. Tabular CFR and Neural CFR are out of scope.

## Problem

The six-max blueprint is confirmed flat at −485 BB/100 vs a passive baseline at 2M iterations (~8.5 visits/infoset). Convergence requires 10–100× more iterations. Local throughput (~90 iter/sec on 8 cores) makes multi-day runs expensive in wall-clock time. Cloud compute provides 4–16× more CPU cores for $5–20/run.

## Decision: Option A — Hetzner VPS + Docker

**Why not alternatives:**
- **Google Colab Pro:** only 2 vCPUs for CPU runtimes — slower than local Mac; sessions reclaimed on inactivity; Buck2 recompiles every session.
- **AWS spot:** cheaper per core-hour but more IAM/S3 setup; spot interruptions require extra handling (viable future option once pipeline is proven).
- **Modal:** $1.80/CPU-hr — most expensive per core; free tier burns fast; 8-core cap on CPU containers.
- **Horizontal scaling:** not viable. The regret table is shared in-process memory. Multiple machines produce diverged checkpoints with no merge path. Vertical scaling (more cores, one machine) is the correct axis.

**Target instance:** Hetzner CCX53 (32 AMD vCPU, ~$0.38/hr). At 4× local core count, expected throughput ~360–410k iters/hr. Within the $5–20/run budget:

| Budget | Duration | Est. iterations |
|--------|----------|-----------------|
| $5     | 13 hrs   | ~5M             |
| $10    | 26 hrs   | ~10M            |
| $20    | 52 hrs   | ~20M            |

## Architecture

```
[your Mac]                    [GHCR]                    [Hetzner CCX53]
    │                            │                             │
    │── hcloud server create ────────────────────────────────▶│
    │── docker pull ─────────────────────────────────────────▶│
    │                            │                             │ train_sixmax.py
    │◀── rsync (every 30 min) ───────────────────────────────│
    │◀── rsync (final) ──────────────────────────────────────│
    │── hcloud server delete ────────────────────────────────▶│
```

Three components, each with one responsibility:

1. **Docker image** (`ghcr.io/zfdupont/pokerbot-trainer`) — compiled `sixmax.so` + Python deps, rebuilt only when C++ source changes.
2. **`scripts/cloud_benchmark.sh`** — validates the full scaffold cheaply (~$0.05, ~5 min) and measures real throughput on target hardware.
3. **`scripts/cloud_train.sh`** — provisions, runs a full training job, syncs checkpoints, destroys VPS.

## Docker Image

Multi-stage build. Builder stage compiles `sixmax.so`; runtime stage copies only the `.so` and installs Python deps. `third_party/`, Buck2 binary, and libtorch headers stay in the builder stage.

**Baked into image:**
- `sixmax.so` (compiled C++ extension, installed to site-packages)
- Python deps from `pyproject.toml` (numpy, tqdm, etc.)
- uv + Python 3.10

**Mounted at runtime:**
- `sixmax/configs/` — TOML config files
- `sixmax/checkpoints/` — checkpoint output directory
- `scripts/`, `sixmax/*.py`, `agents/` — Python source

Config and script changes never require an image rebuild. Rebuild is triggered only by changes to `sixmax/src/`, `BUCK`, or `third_party/`.

**libtorch:** CPU-only build vendored in `docker/libtorch-cpu/` — hermetic, no download at build time.

**Build and push:** `make docker-push` (build + tag + push to GHCR). A git pre-push hook warns if C++ source is dirty but the image hasn't been rebuilt.

Estimated image sizes: builder ~2GB, runtime ~400MB.

## cloud_benchmark.sh

```
Usage: scripts/cloud_benchmark.sh [--type ccx53] [--iters 100000] [--profile] [--local]
```

Runs the full scaffold lifecycle (provision → pull → run → sync → destroy) with a short timed job. Validates the pipeline end-to-end before committing to a long run.

**Flags:**
- `--type`: Hetzner instance type (default: `ccx53`)
- `--iters`: iteration count (default: 100k, ~1 min on CCX53)
- `--profile`: wraps training with a profiler
  - Linux (cloud): `valgrind --tool=callgrind --callgrind-out-file=/checkpoints/callgrind.out`
  - macOS (`--local`): `sample <pid> 30 -f /tmp/pokerbot_sample.txt`
- `--local`: skips Hetzner provisioning entirely; runs locally (macOS profiling at zero cost)

**Output:**
```
Benchmark complete
  Instance:   ccx53 (32 vCPU)
  Throughput: 114 iter/sec  (410k iter/hr)
  Runtime:    52s for 100k iters
  Cost/hr:    $0.38

  Extrapolated:
     1M iters →  2.4 hrs →  $0.91
    10M iters →  24 hrs  →  $9.12
    50M iters →  122 hrs →  $46.40  ← exceeds $20 cap
```

Rsyncs the test checkpoint back alongside any profiler output (proves sync works).

## cloud_train.sh

```
Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] [--resume checkpoint.pt] [--max-hours 24]
```

**Lifecycle (linear, blocking):**
1. `hcloud server create` → capture server IP and ID
2. Poll SSH until ready (~30s)
3. rsync Python source files to `/tmp/pokerbot` on VPS
4. `docker pull ghcr.io/zfdupont/pokerbot-trainer:latest`
5. `docker run` (attached, streaming logs to terminal)
6. Final rsync of `sixmax/checkpoints/` back to local
7. `hcloud server delete`

**Safety:** `trap 'rsync && hcloud server delete $SERVER_ID' EXIT ERR INT` fires on any exit path — normal completion, uncaught error, or Ctrl+C. Without this, a dropped connection leaves a billing VPS running with no record of its ID.

**Long runs:** launch with `nohup caffeinate scripts/cloud_train.sh ... &` to prevent macOS sleep from interrupting the blocking SSH connection.

**Resume:** `--resume path/to/checkpoint.pt` rsyncs the checkpoint to the VPS before training and appends `--resume /checkpoints/checkpoint.pt` to the training command. Per existing trainer behavior, buffers restart from scratch; only network weights are restored.

## Checkpoint Sync

Two sync moments:

**Mid-run (every 30 min):** a background rsync loop runs alongside the training container. At ~410k iters/hr with `checkpoint_interval=500k`, a new checkpoint lands every ~73 min; the 30-min poll picks it up within one cycle. A VPS crash loses at most ~30 min of training.

**Final:** runs after container exits, before `hcloud server delete`. Syncs: `checkpoint_<iter>.bin`, `best_checkpoint.bin`, `best_checkpoint.json`, `curve.csv`.

Transport: `rsync -avz` over the same SSH key used to provision the VPS. No S3, no rclone — one fewer credential to manage.

## Cost Controls

**`--max-hours N`** (default: 24 for `cloud_train.sh`, 1 for `cloud_benchmark.sh`): background watchdog kills the container and triggers sync → destroy if the deadline hits before training completes. Watchdog is cancelled on clean completion.

**Running cost display** at each mid-run sync:
```
[sync 3/? @ 1.5M iters]  elapsed: 3.6 hrs  est. cost: $1.37  remaining budget: $18.63
```
Calculated from `elapsed_seconds * hourly_rate` — rate known at script start from `--type`, no Hetzner API call needed.

**Hetzner console spending limit:** set to $25–30 as a backstop against the `trap` failing (e.g., Mac loses power before teardown). Configured once in the Hetzner web console, not in code.

## New Files

```
docker/
  Dockerfile                  # multi-stage build
  .dockerignore               # excludes third_party/, .git, checkpoints from build context
  libtorch-cpu/               # vendored CPU-only libtorch tarball
Makefile                      # docker-push target + pre-push hook install
scripts/
  cloud_benchmark.sh          # scaffold validation + throughput measurement
  cloud_train.sh              # full training run lifecycle
```

No changes to `sixmax/`, `scripts/train_sixmax.py`, or any existing training code.

## Prerequisites

- Hetzner account with API token in `HCLOUD_TOKEN` env var (add to `.env`, already gitignored)
- `hcloud` CLI installed (`brew install hcloud`)
- GHCR write access (`docker login ghcr.io`)
- SSH key registered with Hetzner (one-time console setup)
