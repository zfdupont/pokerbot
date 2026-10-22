#!/usr/bin/env bash
# cloud_train.sh — provision Hetzner VPS, run full sixmax blueprint training,
# sync checkpoints every 30 min + on exit, then destroy the VPS.
#
# Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] \
#            [--checkpoint-interval 1000000] [--no-snapshots] \
#            [--config sixmax/configs/hu.toml] [--checkpoint-name out.bin] \
#            [--bucket-map path/to/map.bin] [--resume path/to/ckpt.bin] \
#            [--max-hours 24] [--dry-run]
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
CKPT_INTERVAL=1000000   # save every N iters (never 0 = save-at-end-only)
SNAPSHOTS=true          # keep a numbered <checkpoint>_<iters>.bin per save
CONFIG=""               # e.g. sixmax/configs/hu.toml (empty = trainer default)
CKPT_NAME="checkpoint.bin"  # output file name under sixmax/checkpoints/
BUCKET_MAP=""           # local SIXBM001 path; uploaded as tables/<basename>
DRY_RUN=false           # print the commands, provision nothing
IMAGE="ghcr.io/zfdupont/pokerbot-trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SERVER_NAME="pokerbot-train-${TIMESTAMP//_/-}"  # hostnames disallow underscores
SERVER_IP=""
SYNC_PID=""
WATCHDOG_PID=""
START_TIME=$(date +%s)
# Ephemeral VPS: never trust/persist host keys (IPs get recycled across runs).
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

# rsync exclude flags for the source sync (--dry-run prints these verbatim).
# sixmax/tables/ holds gitignored build artifacts (equity store + bucket maps,
# 130-530 MB each); the one map an arm needs is uploaded via --bucket-map.
RSYNC_EXCLUDES=(
    --exclude='third_party/'
    --exclude='buck-out/'
    --exclude='sixmax/tables/'
    --exclude='sixmax/checkpoints/'
    --exclude='neural_cfr/checkpoints/'
    --exclude='cfr/checkpoints/'
    --exclude='.git/'
    --exclude='.worktrees/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
)

# Does the config's [abstraction] block set a non-empty bucket_map?
# Prints "yes" or "no". Uses tomllib on Python 3.11+; otherwise a minimal
# fallback that only understands `bucket_map = "..."` under [abstraction].
config_sets_bucket_map() {
    python3 - "$1" <<'PY' 2>/dev/null
import sys

text = open(sys.argv[1], encoding="utf-8").read()
value = ""
try:
    import tomllib
except ImportError:  # Python < 3.11: barely-parse the one line we care about.
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            section = line.strip("[]").strip()
            continue
        if section == "abstraction" and "=" in line:
            key, _, raw_value = line.partition("=")
            if key.strip() == "bucket_map":
                value = raw_value.strip().strip("\"'")
                break
else:
    value = tomllib.loads(text).get("abstraction", {}).get("bucket_map", "")
print("yes" if value else "no")
PY
}

# Print each exclude flag with its pattern quoted: --exclude='sixmax/tables/'.
print_rsync_excludes() {
    local _flag
    for _flag in "${RSYNC_EXCLUDES[@]}"; do
        printf "%s='%s'\n" "${_flag%%=*}" "${_flag#*=}"
    done
}

# Approximate hourly rates in USD — verify at https://www.hetzner.com/cloud
# (plain case avoids bash 4 associative-array requirement on macOS)
case_hourly_rate() {
    case "$1" in
        ccx23) echo "0.14" ;;
        ccx33) echo "0.21" ;;
        ccx43) echo "0.30" ;;
        ccx53) echo "0.38" ;;
        ccx63) echo "0.56" ;;
        *)     echo "0.38" ;;
    esac
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)                INSTANCE_TYPE="$2";  shift 2 ;;
        --iters)               ITERS="$2";          shift 2 ;;
        --resume)              RESUME_CHECKPOINT="$2"; shift 2 ;;
        --max-hours)           MAX_HOURS="$2";      shift 2 ;;
        --checkpoint-interval) CKPT_INTERVAL="$2";  shift 2 ;;
        --snapshots)           SNAPSHOTS=true;      shift ;;
        --no-snapshots)        SNAPSHOTS=false;     shift ;;
        --config)              CONFIG="$2";         shift 2 ;;
        --checkpoint-name)     CKPT_NAME="$2";      shift 2 ;;
        --bucket-map)          BUCKET_MAP="$2";     shift 2 ;;
        --dry-run)             DRY_RUN=true;        shift ;;
        --help)
            echo "Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] [--checkpoint-interval 1000000] [--no-snapshots] [--config path/to/cfg.toml] [--checkpoint-name out.bin] [--bucket-map path/to/map.bin] [--resume path/to/ckpt.bin] [--max-hours 24] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] [--checkpoint-interval 1000000] [--no-snapshots] [--config path/to/cfg.toml] [--checkpoint-name out.bin] [--bucket-map path/to/map.bin] [--resume path/to/ckpt.bin] [--max-hours 24] [--dry-run]"
            exit 1
            ;;
    esac
done

# Fail loud: 0 = save-at-end-only, i.e. the whole run is lost if the container
# is stopped before its single chunk finishes (the 2026-09-29 incident).
if ! [[ "$CKPT_INTERVAL" =~ ^[0-9]+$ ]] || [ "$CKPT_INTERVAL" -le 0 ]; then
    echo "Error: --checkpoint-interval must be a positive integer (got '${CKPT_INTERVAL}')"
    exit 1
fi

# ── Pre-flight (never needs credentials; --dry-run stops here) ────────────────
# A bucket map is shipped under tables/<basename>: the source rsync excludes
# sixmax/tables/, so a config that names a bucket_map cannot resolve on the box
# unless the operator names the map explicitly (it is uploaded, then mounted
# read-only for the container).
MAP_BASENAME=""
if [ -n "$BUCKET_MAP" ]; then
    MAP_BASENAME=$(basename "$BUCKET_MAP")
fi

CONFIG_PATH=""
if [ -n "$CONFIG" ]; then
    case "$CONFIG" in
        /*) CONFIG_PATH="$CONFIG" ;;
        *)  CONFIG_PATH="${REPO_ROOT}/$CONFIG" ;;
    esac
fi

if [ -z "$BUCKET_MAP" ] && [ -n "$CONFIG_PATH" ]; then
    # `|| true`: a missing/invalid config must reach the explicit error below
    # instead of aborting inside the command substitution under `set -e`.
    CONFIG_HAS_MAP=$(config_sets_bucket_map "$CONFIG_PATH" || true)
    if [ "$CONFIG_HAS_MAP" = "yes" ]; then
        echo "Error: config needs a bucket map; pass --bucket-map PATH (the config names one, but sixmax/tables/ is never shipped to the box)"
        exit 2
    elif [ "$CONFIG_HAS_MAP" != "no" ]; then
        # Never guess: an unreadable config could hide a bucket_map requirement.
        echo "Error: could not read [abstraction].bucket_map from $CONFIG_PATH (needs python3)"
        exit 1
    fi
fi

if [ "$DRY_RUN" != true ] && [ -n "$BUCKET_MAP" ] && [ ! -f "$BUCKET_MAP" ]; then
    echo "Error: bucket map not found: $BUCKET_MAP"
    exit 1
fi

CKPT_PATH="/pokerbot/sixmax/checkpoints/$CKPT_NAME"

# The exact docker run invocation, built here so --dry-run prints what runs.
# The payload is handed to ssh as one string, so each flag+value is one element.
# Broad ro mount first; the narrow rw checkpoints mount overrides it.
DOCKER_ARGS=(
    --name pokerbot-trainer
    -v /tmp/pokerbot/sixmax:/pokerbot/sixmax:ro
    -v /tmp/pokerbot/sixmax/checkpoints:/pokerbot/sixmax/checkpoints
    -v /tmp/pokerbot/scripts:/pokerbot/scripts:ro
    -v /tmp/pokerbot/agents:/pokerbot/agents:ro
    -e SIXMAX_SO_PATH=/opt/sixmax.so
    -e PYTHONUNBUFFERED=1
)
if [ -n "$MAP_BASENAME" ]; then
    DOCKER_ARGS+=(-v /tmp/pokerbot/tables:/pokerbot/tables:ro \
                  -e "SIXMAX_BUCKET_MAP=/pokerbot/tables/$MAP_BASENAME")
fi
DOCKER_ARGS+=("$IMAGE" python3.12 -u /pokerbot/scripts/train_sixmax.py \
              "--iterations $ITERS" "--checkpoint $CKPT_PATH")
if [ "$SNAPSHOTS" = true ]; then DOCKER_ARGS+=(--snapshots); fi
if [ -n "$CONFIG" ]; then DOCKER_ARGS+=("--config /pokerbot/$CONFIG"); fi
if [ -n "$MAP_BASENAME" ]; then
    DOCKER_ARGS+=("--bucket-map /pokerbot/tables/$MAP_BASENAME")
fi
DOCKER_ARGS+=("--checkpoint-interval $CKPT_INTERVAL")
if [ -n "$RESUME_CHECKPOINT" ]; then
    DOCKER_ARGS+=("--resume /pokerbot/sixmax/checkpoints/resume.bin")
fi

if [ "$DRY_RUN" = true ]; then
    echo "[cloud_train] DRY RUN — nothing is provisioned, uploaded or executed."
    echo "rsync excludes:"
    print_rsync_excludes
    if [ -n "$MAP_BASENAME" ]; then
        echo "bucket map upload target: root@<server>:/tmp/pokerbot/tables/$MAP_BASENAME"
    fi
    echo "docker run command (sent over ssh):"
    printf 'docker run'
    printf ' %s' "${DOCKER_ARGS[@]}"
    printf '\n'
    exit 0
fi

HOURLY_RATE=$(case_hourly_rate "$INSTANCE_TYPE")

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
        rsync -avz --ignore-errors -e "ssh $SSH_OPTS" \
            "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
            "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
        echo "[cloud_train] Total elapsed: ${ELAPSED}s — estimated cost: \$${COST}"
        echo "[cloud_train] Destroying $SERVER_NAME..."
        hcloud server delete "$SERVER_NAME" 2>/dev/null || true
    fi
}
trap cleanup EXIT ERR INT

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

# ── Validation ────────────────────────────────────────────────────────────────
: "${HCLOUD_TOKEN:?HCLOUD_TOKEN not set — add to .env}"
: "${HETZNER_SSH_KEY_NAME:?HETZNER_SSH_KEY_NAME not set — add to .env}"
: "${GHCR_PAT:?GHCR_PAT not set — add to .env (GitHub PAT with read:packages scope)}"

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
_ssh_attempts=0
until ssh $SSH_OPTS "root@$SERVER_IP" echo ok 2>/dev/null; do
    sleep 5
    (( _ssh_attempts++ ))
    [[ $_ssh_attempts -ge 24 ]] && { echo "[cloud_train] SSH timeout after 2 min — destroying server"; exit 1; }
done

echo "[cloud_train] Installing Docker..."
ssh $SSH_OPTS "root@$SERVER_IP" "curl -fsSL https://get.docker.com | sh"

echo "[cloud_train] Logging in to GHCR..."
ssh $SSH_OPTS "root@$SERVER_IP" "echo '$GHCR_PAT' | docker login ghcr.io -u zfdupont --password-stdin"

# ── Sync source ───────────────────────────────────────────────────────────────
echo "[cloud_train] Syncing repo source..."
rsync -avz -e "ssh $SSH_OPTS" "${RSYNC_EXCLUDES[@]}" \
    "${REPO_ROOT}/" "root@${SERVER_IP}:/tmp/pokerbot/"

# Create checkpoints dir on VPS (excluded from rsync above)
ssh $SSH_OPTS "root@$SERVER_IP" "mkdir -p /tmp/pokerbot/sixmax/checkpoints"

# Ship exactly one bucket map, by basename (excluded from the source rsync)
if [ -n "$MAP_BASENAME" ]; then
    echo "[cloud_train] Uploading bucket map $MAP_BASENAME..."
    ssh $SSH_OPTS "root@$SERVER_IP" "mkdir -p /tmp/pokerbot/tables"
    rsync -avz -e "ssh $SSH_OPTS" "$BUCKET_MAP" \
        "root@${SERVER_IP}:/tmp/pokerbot/tables/${MAP_BASENAME}"
fi

# Sync resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "[cloud_train] Uploading resume checkpoint..."
    rsync -avz -e "ssh $SSH_OPTS" "$RESUME_CHECKPOINT" \
        "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/resume.bin"
fi

# ── Pull image ────────────────────────────────────────────────────────────────
echo "[cloud_train] Pulling Docker image..."
ssh $SSH_OPTS "root@$SERVER_IP" "docker pull $IMAGE"

# ── Mid-run sync loop (background) ───────────────────────────────────────────
mid_sync_loop() {
    local _count=0
    while true; do
        sleep 1800  # 30 min
        _count=$(( _count + 1 ))
        local _elapsed
        _elapsed=$(( $(date +%s) - START_TIME ))
        local _cost
        _cost=$(echo "scale=2; $_elapsed / 3600 * $HOURLY_RATE" | bc)
        echo "[sync ${_count} @ ~${_elapsed}s elapsed] est. cost: \$${_cost}"
        rsync -avz --ignore-errors -e "ssh $SSH_OPTS" \
            "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
            "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
    done
}
mid_sync_loop &
SYNC_PID=$!

# ── Watchdog (background) ─────────────────────────────────────────────────────
(
    sleep $(( MAX_HOURS * 3600 ))
    echo "[cloud_train] Max hours ($MAX_HOURS) reached — stopping container..."
    ssh $SSH_OPTS "root@$SERVER_IP" \
        "docker stop pokerbot-trainer 2>/dev/null || true"
) &
WATCHDOG_PID=$!

# ── Training ──────────────────────────────────────────────────────────────────
echo "[cloud_train] Starting training ($ITERS iterations)..."
ssh $SSH_OPTS "root@$SERVER_IP" "docker run ${DOCKER_ARGS[*]}"

# Watchdog no longer needed (training completed normally)
kill "$WATCHDOG_PID" 2>/dev/null || true
WATCHDOG_PID=""

echo "[cloud_train] Training complete."
# cleanup() handles final sync + destroy via EXIT trap
