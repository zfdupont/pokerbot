#!/usr/bin/env bash
# cloud_train.sh — provision Hetzner VPS, run full sixmax blueprint training,
# sync checkpoints every 30 min + on exit, then destroy the VPS.
#
# Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] \
#            [--checkpoint-interval 1000000] [--no-snapshots] \
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
CKPT_INTERVAL=1000000   # save every N iters (never 0 = save-at-end-only)
SNAPSHOTS=true          # keep a numbered <checkpoint>_<iters>.bin per save
IMAGE="ghcr.io/zfdupont/pokerbot-trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SERVER_NAME="pokerbot-train-${TIMESTAMP//_/-}"  # hostnames disallow underscores
SERVER_IP=""
SYNC_PID=""
WATCHDOG_PID=""
START_TIME=$(date +%s)
# Ephemeral VPS: never trust/persist host keys (IPs get recycled across runs).
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

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
        --help)
            echo "Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] [--checkpoint-interval 1000000] [--no-snapshots] [--resume path/to/ckpt.bin] [--max-hours 24]"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: scripts/cloud_train.sh [--type ccx53] [--iters 10000000] [--checkpoint-interval 1000000] [--no-snapshots] [--resume path/to/ckpt.bin] [--max-hours 24]"
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
rsync -avz -e "ssh $SSH_OPTS" \
    --exclude='third_party/' --exclude='buck-out/' \
    --exclude='sixmax/checkpoints/' --exclude='neural_cfr/checkpoints/' \
    --exclude='cfr/checkpoints/' --exclude='.git/' \
    --exclude='__pycache__/' --exclude='*.pyc' \
    "${REPO_ROOT}/" "root@${SERVER_IP}:/tmp/pokerbot/"

# Create checkpoints dir on VPS (excluded from rsync above)
ssh $SSH_OPTS "root@$SERVER_IP" "mkdir -p /tmp/pokerbot/sixmax/checkpoints"

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
RESUME_FLAG=""
[ -n "$RESUME_CHECKPOINT" ] && RESUME_FLAG="--resume /pokerbot/sixmax/checkpoints/resume.bin"
CKPT_FLAG="--checkpoint-interval $CKPT_INTERVAL"
SNAP_FLAG=""
[ "$SNAPSHOTS" = true ] && SNAP_FLAG="--snapshots"

echo "[cloud_train] Starting training ($ITERS iterations)..."
# Broad ro mount FIRST, then narrow rw checkpoints mount overrides it (Task 4 pattern)
ssh $SSH_OPTS "root@$SERVER_IP" "docker run \
    --name pokerbot-trainer \
    -v /tmp/pokerbot/sixmax:/pokerbot/sixmax:ro \
    -v /tmp/pokerbot/sixmax/checkpoints:/pokerbot/sixmax/checkpoints \
    -v /tmp/pokerbot/scripts:/pokerbot/scripts:ro \
    -v /tmp/pokerbot/agents:/pokerbot/agents:ro \
    -e SIXMAX_SO_PATH=/opt/sixmax.so \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE \
    python3.12 -u /pokerbot/scripts/train_sixmax.py \
        --iterations $ITERS \
        --checkpoint /pokerbot/sixmax/checkpoints/checkpoint.bin \
        $CKPT_FLAG $SNAP_FLAG \
        $RESUME_FLAG"

# Watchdog no longer needed (training completed normally)
kill "$WATCHDOG_PID" 2>/dev/null || true
WATCHDOG_PID=""

echo "[cloud_train] Training complete."
# cleanup() handles final sync + destroy via EXIT trap
