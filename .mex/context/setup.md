---
name: setup
description: Dev environment setup and commands. Load when setting up the project for the first time or when environment issues arise.
triggers:
  - "setup"
  - "install"
  - "environment"
  - "getting started"
  - "how do I run"
  - "local development"
edges:
  - target: context/stack.md
    condition: when specific technology versions or library details are needed
  - target: context/neural-cfr.md
    condition: when building or troubleshooting the C++ extension
  - target: context/cfr-training.md
    condition: when running or resuming tabular training
  - target: patterns/debug-bot-misplay.md
    condition: when a trained bot behaves nonsensically after setup
  - target: patterns/eval-checkpoint-ab.md
    condition: when deciding whether one checkpoint (or search on/off) beats another before deploying it
  - target: patterns/eval-checkpoint.md
    condition: when measuring a checkpoint against OpenSpiel or random baselines
  - target: patterns/deploy-openpoker.md
    condition: when deploying a bot to openpoker.ai over WebSocket
last_updated: 2026-10-21
---

# Setup

## Prerequisites

- Python ≥3.10 with **uv** installed (all commands run through `uv run`).
- **Buck2** at `~/bin/buck2` (user-local binary, not on PATH) — needed for both C++ extensions: `//neural_cfr:neural_cfr` and `//sixmax:sixmax`.
- macOS (primary dev platform); libtorch and other C++ deps are vendored in `third_party/`.
- Optional: OpenSpiel installed in the environment for `scripts/eval_openspiel*.py`.

## First-time Setup

1. `./scripts/setup_dev.sh` — idempotent one-shot setup for clones, worktrees, and cloud machines: pins/verifies Python 3.10 (extension ABI), `uv sync`, creates missing `third_party/` symlinks (worktrees don't inherit the untracked vendored deps), writes `.buckconfig.local` with the python include path, builds `//neural_cfr:neural_cfr` + `//sixmax:sixmax`, and runs the full suite. Ends with `SETUP OK`.
2. `mkdir -p neural_cfr/checkpoints` — checkpoint dirs are gitignored and not auto-created.

Manual equivalents, if you need one piece: `uv sync`; `uv run pytest tests/`; `~/bin/buck2 build //neural_cfr:neural_cfr //sixmax:sixmax`.

## Environment Variables

- `OPENPOKER_API_KEY` (required only for `scripts/openpoker_bot.py`) — bearer token for `wss://openpoker.ai/ws`. Never commit it.
- `POKERBOT_CHECKPOINT` (web service) — path to a sixmax `.bin`; required by `web.main`.
- `POKERBOT_STACK` / `POKERBOT_SMALL_BLIND` (web service, defaults `200` / `1` = 100 BB), `POKERBOT_SESSION_TTL` (default `3600`), `POKERBOT_CORS_ORIGINS` (default `https://zfdupont.com`).
- `SIXMAX_SO_PATH` — prebuilt `sixmax.so` path; when set, scripts and the web service skip the Buck2 build (used in containers).
- `SIXMAX_BUCKET_MAP` — full path to a `SIXBM001` map used to resolve **every** `SIXBP002` load in the process (wins over the checkpoint-relative and tables-dir search). Because it is process-wide, a head-to-head between checkpoints on *different* maps must leave it unset (pass the map per load instead: `sixmax.load_abstraction(path, bucket_map=PATH)`, `train_sixmax.py --resume ... --bucket-map PATH`). A wrong map is refused with `map hash mismatch`, never used silently.
- `SIXMAX_TABLES_DIR` — directory searched for a checkpoint's map filename hint (default `sixmax/tables`, relative to cwd); only after `$SIXMAX_BUCKET_MAP` and the explicit/checkpoint-adjacent candidates.
- Artifact formats (one line each): `SIXEQ001` = exact equity store (magic + hand-index version + sizes + hash + dense `float32` tables); `SIXBM001` = bucket map (magic + hand-index version + per-street sizes/bucket counts + method/params + equity-store hash + content hash + dense `uint8` buckets); `SIXBP002` = map-backed blueprint checkpoint (kind=1, map content hash and filename hint after the common v1 header); `SIXBP003` = SIXBP002 **plus the persisted `PotLayout`** (mode `bb`/`spr`, bucket count, edges) — written whenever the pot layout is non-legacy, so the pot index lives at key bits 26-29 and bits 18-19 are zero; loading or resuming under a different layout refuses with `pot layout mismatch`; `SIXBP001` = legacy blueprint checkpoint (abstraction config + quantile edges). The legacy layout (`bb`, edges `[7, 15, 40]`) is what SIXBP001/002 imply and keeps the old bits and hash byte-identical.
- Otherwise training and evaluation are configured via CLI flags.

## Common Commands

- `uv run pytest tests/` — full test suite; single test: `uv run pytest tests/path/test_file.py::ClassName::test_name -v`.
- `uv run python scripts/train_cfr.py --iterations 1000000` — tabular training; `--resume cfr/checkpoints/checkpoint_XXXXXXXX.pkl` to continue.
- `uv run python scripts/train_neural.py --iterations 5_000_000 --checkpoint neural_cfr/checkpoints/checkpoint.pt` — neural training (add `--resume` to continue).
- `uv run python scripts/build_equity_tables.py --out sixmax/tables/equity_v1.bin [--threads 0]` — build the exact equity feature store (`SIXEQ001`, ~10 min on 8 cores, 528 MB). Format: magic + hand-index version + per-street sizes + content hash, then dense `float32` equity per suit-canonical (hole, board) (river enumerated exactly; turn/flop are exact next-street means). Build artifact, never deployed.
- `uv run python scripts/build_bucket_map.py --equity sixmax/tables/equity_v1.bin --method {percentile,potential} --counts 50,50,20 --out sixmax/tables/map.bin [--seed 0] [--threads 0]` — build a bucket map (`SIXBM001`, ~140 MB; **~30 s** percentile, **~5 min** potential k-means). Opening a map memory-maps it read-only and hashes the whole file once, so each process touches ~140 MB (pages are shared between processes on one host).
- `uv run python scripts/train_sixmax.py` — six-max blueprint training (`[abstraction]` + `[train.blueprint]` in `sixmax/configs/default.toml`; CLI overrides; `--bucket-map PATH` overrides `[abstraction].bucket_map` and is also used as the map candidate on `--resume`; `--resume PATH` restores table + iteration counter; `--checkpoint-interval N` saves every N iters (`0` = save-at-end-only; add `--snapshots` to keep each save as `<base>_<total-iters>.bin` — named by the trainer's TOTAL iterations, so a `--resume` run continues the numbering instead of overwriting earlier snapshots); `--report-interval S` logs a live `[done/total] rate it/s ETA infosets` heartbeat every S seconds (`0` = off, default 60); `--selection-enabled` maintains `best_checkpoint.bin` via A/B vs previous best). All progress goes to stderr via `logging` (byte-identical message text, so log greps/watcher regexes are unaffected). Each checkpoint save emits a machine-readable `[rate] iters=<N> it/s=<cumulative> window_it/s=<since previous save>` line (one decimal place, no thousands separators) — `window_it/s` is the iterations since the previous save over the wall time since the previous save.
- `./scripts/cloud_train.sh --config sixmax/configs/hu200.toml --bucket-map sixmax/tables/map_pct_50_50_20.bin --checkpoint-name hu200_c0.bin` — same, but ship **one** bucket map with the run: the map is uploaded as `/tmp/pokerbot/tables/<basename>` and mounted read-only at `/pokerbot/tables`, the container gets `SIXMAX_BUCKET_MAP=/pokerbot/tables/<basename>` and the trainer gets `--bucket-map /pokerbot/tables/<basename>`. The source sync always excludes `sixmax/tables/` (130–530 MB of gitignored artifacts), so a `--config` whose TOML sets `[abstraction].bucket_map` **requires** `--bucket-map`; without it the script exits 2 with `config needs a bucket map; pass --bucket-map`.
- `./scripts/cloud_train.sh --dry-run [all other flags]` — print the rsync exclude flags, the bucket-map upload target and the exact `docker run` command, then exit 0. Stops before sourcing `.env` and before any `hcloud`/`ssh`/`rsync`/`docker` call, so it needs no credentials and no map file on disk: use it to check an arm's command line before spending money.
- `./scripts/cloud_train.sh --config sixmax/configs/hu.toml --checkpoint-name hu_blueprint.bin --iters 50000000 --type ccx33 --max-hours 48` — provision a Hetzner VPS, train, rsync snapshots back every 30 min + on exit, then self-destroy. **The Hetzner account is capped at 8 dedicated cores**: ccx33 (8 cores) works, ccx43/ccx53 are rejected with `resource_limit_exceeded` — do not plan around ccx53. Measured **~840–895 it/s on ccx33** ⇒ **50M iters ≈ 16.7 h ≈ $3.50** (ccx33 @ $0.21/h). Needs `HCLOUD_TOKEN`, `HETZNER_SSH_KEY_NAME`, `GHCR_PAT` in `.env`; run detached (`nohup caffeinate -i ... &`) so a closed terminal doesn't abort the run. **Watch out for the plan-level rate figures**: the "~55 it/s" in older plans is *total* wall time divided by iters and is dominated by the one-time ~48 s abstraction build — the actual trainer rate is ~415 it/s locally, ~850 it/s on ccx33.
- `uv run python scripts/eval_sixmax.py --a CKPT [--b CKPT|uniform] --hands 500` — duplicate-deal seat-rotated A/B eval; prints `Blueprint A win rate: ±X.XX BB/100`.
- `uv run python scripts/eval_blueprint_h2h.py --a CKPT --b CKPT --hands 20000 --seeds 1,2,3,4,5,6,7,8 --bb 2 --stack 400` — head-to-head of two checkpoints directly (all RNGs pinned, duplicate seats, so self-match is exactly 0 and each seed's score is a paired diff). **For 200 BB arms always pass `--bb 2 --stack 400`** (stack is in chips, so 400 chips at bb 2 = 200 BB); `scripts/arm_plateau.py` defaults to exactly that. Omitting them silently evaluates the arm at 100 BB.
- `uv run python scripts/diagnose_blueprint.py [--checkpoints "sixmax/checkpoints/blueprint_0*.bin"] [--top-frac 0.05] [--csv PATH]` — offline blueprint plateau autopsy over a checkpoint series; prints per-checkpoint visit-weight/entropy/probe metrics + a trend-based `VERDICT` (undertraining | structural | mixed). Reads `sixmax.dump_infosets`; run from repo root.
- `uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/hu_blueprint.bin --tabular cfr/checkpoints/checkpoint_09040000.pkl --neural neural_cfr/checkpoints/checkpoint.pt` — HU-mode sanity eval: the six-max blueprint (2-player) vs the frozen tabular/neural bots in the live engine (BB/100). `--tabular`/`--neural` optional.
- `uv run python scripts/arm_plateau.py --snapshots 'sixmax/checkpoints/hu200_c0_*.bin' [--lag 10000000] [--hands 20000] [--seeds 1,2,3,4,5,6,7,8] [--bb 2] [--stack 400]` — plateau check for an abstraction arm: plays its newest snapshot against the newest one at least `--lag` iterations older (duplicate-deal head-to-head via `eval_blueprint_h2h.run_h2h`) and prints `VERDICT: converged|training` (converged unless the newer beats the older at mean>0 and |t|>=2). Names must end `_<iters>.bin`; unsuffixed files and closer snapshots are ignored, and a lag with no qualifying older snapshot is an error. `--infosets CKPT` prints only the queryable infoset count; `--cap C0_PLATEAU C0_INFOSETS ARM_INFOSETS` prints only the arm's iteration cap (`round(C0_plateau * arm_infosets / C0_infosets)`, clamped to 20M-150M) — both machine-readable.
- `uv run python scripts/openpoker_bot.py --checkpoint sixmax/checkpoints/blueprint.bin` — deploy the six-max blueprint (`.bin` auto-detected → `SixmaxHandTracker`).
- `uv run python scripts/eval_openspiel.py --hands 2000` — head-to-head eval (BB/100); `eval_openspiel_neural.py` for `.pt` checkpoints.
- `uv run python scripts/play.py [--stack 1000] [--big-blind 10]` — interactive heads-up CLI vs the bot.
- `uv run python -m web.main --port 8100` — serve the heads-up **web API** (FastAPI). Requires `POKERBOT_CHECKPOINT` (a sixmax `.bin`); loads `sixmax.so` from `SIXMAX_SO_PATH` or builds it via Buck2. Single uvicorn worker (in-memory TTL sessions). Endpoints: `GET /health`, `POST /api/session`, `GET /api/session/{token}`, `POST /api/session/{token}/{action|next-hand|rebuy}`, `DELETE /api/session/{token}`. Browser UI lives in the separate `blogfolio` site.
- `uv run python scripts/range_chart.py [--scenario sb-open|bb-vs-raise]` — 13×13 ANSI preflop range chart.
- `uv run python scripts/run_simulation.py [--hands 100] [--agent simple|position|cfr]` — agent-vs-agent simulation; `python main.py` runs one demo hand at a 4-player table.
- `./scripts/run_openpoker.sh {start|stop|status}` — managed openpoker deployment: tracks the python PID (pidfile `.openpoker.pid`), `caffeinate -w` alongside, SIGTERM sends `leave_table` before exit (banks the table stack). Env overrides: `CHECKPOINT`, `BUY_IN`. Needs `OPENPOKER_API_KEY` in `.env`.
- `uv run python scripts/openpoker_bot.py --buy-in 2000` — raw deployment (prefer the run script).
- `~/bin/buck2 build //neural_cfr:neural_cfr` — rebuild the C++ extension after any `neural_cfr/src/` or `common/src/` change; `//sixmax:sixmax` for the six-max module.

## Common Issues

**OpenSpiel prints noise on import:** the `pokerkit_wrapper` print is suppressed in `scripts/eval_openspiel.py` (commit `83d9653`) — route new OpenSpiel imports through the same suppression.

**Neural bot plays absurdly (folds strong hands):** almost always a chip-scaling or encoding problem, not a training problem — inputs must be rescaled to the `starting_stack=100, big_blind=1` training frame, and the evaluator's `12 - rank` kicker inversion must be intact. See `patterns/debug-bot-misplay.md`.

**Slow tests:** you're bypassing the conftests. `tests/conftest.py` sets `MONTE_CARLO_SAMPLES=10`; run tests via pytest from the repo root so `pythonpath = ["."]` and the patches apply.

**`buck2: command not found`:** it's at `~/bin/buck2`, not on PATH.

**`pybind11/pybind11.h not found` when building `//sixmax` or `//neural_cfr`:** `third_party/pybind11/include` is a machine-local symlink that rots when the Python env it points at is rebuilt. Run `./scripts/setup_dev.sh` — since Phase 1b it detects dead third_party symlinks and repoints `pybind11/include` at the venv's pybind11 package automatically.

**`hcloud: no active context or token`:** `cloud_train.sh` sources `.env` itself (`set -a; source .env; set +a`); running `hcloud` by hand needs the same prefix, or the token is simply absent from the environment.

**Checkpoint A/B results that flip between runs:** the eval is far noisier than it looks — 4 000-hand runs swing by tens of BB/100. Always use the paired multi-seed harness in `patterns/eval-checkpoint-ab.md` before concluding anything.

## Production (droplet)

- `poker.zfdupont.com` → a docker-compose stack on the droplet under `/var/www/poker-service` (image `ghcr.io/zfdupont/pokerbot-web:latest`, bound to `127.0.0.1:8100`, the checkpoint directory bind-mounted read-only at `/checkpoints`, `POKERBOT_CHECKPOINT=/checkpoints/blueprint.bin`, single uvicorn worker). nginx vhost at `/etc/nginx/sites-enabled/poker` (Cloudflare terminates TLS at the edge).
- **Swapping the served checkpoint:** back up, overwrite `checkpoints/blueprint.bin`, then `docker compose restart pokerbot-web` — no image rebuild. The dedicated HU artifact is already staged there as `blueprint-hu.bin` (inactive, deliberately not wired up).
- **Rebuild the image:** push to `main` triggers `.github/workflows/build-web-image.yml` → `ghcr.io/zfdupont/pokerbot-web:latest` (it cannot be built locally — linux/amd64 buck2 SIGILLs under QEMU on arm64).
