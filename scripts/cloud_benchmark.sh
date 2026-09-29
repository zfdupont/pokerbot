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
BENCH_CHECKPOINT="${REPO_ROOT}/sixmax/checkpoints/bench_${TIMESTAMP}.bin"

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

# Approximate hourly rates in USD — verify at https://www.hetzner.com/cloud
# (plain case avoids bash 4 associative-array requirement on macOS)
case "$INSTANCE_TYPE" in
    ccx23) HOURLY_RATE="0.14" ;;
    ccx33) HOURLY_RATE="0.21" ;;
    ccx43) HOURLY_RATE="0.30" ;;
    ccx53) HOURLY_RATE="0.38" ;;
    ccx63) HOURLY_RATE="0.56" ;;
    *)     HOURLY_RATE="0.38" ;;
esac

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
        label=$(printf "%'d" $target 2>/dev/null || printf "%d" $target)
        echo "    ${label} iters → ${hrs} hrs → \$${cost}"
    done
    echo ""
}

# ── Local mode ────────────────────────────────────────────────────────────────
run_local() {
    echo "[cloud_benchmark] Running locally (no VPS)"
    cd "$REPO_ROOT"

    START=$(date +%s)
    if $PROFILE; then
        echo "[cloud_benchmark] Profiling with sample (macOS)"
        uv run python scripts/train_sixmax.py \
            --iterations "$ITERS" \
            --checkpoint "$BENCH_CHECKPOINT" &
        TRAIN_PID=$!
        sleep 2
        sample "$TRAIN_PID" 30 -f "${REPO_ROOT}/sixmax/checkpoints/sample_${TIMESTAMP}.txt" 2>/dev/null || true
        wait "$TRAIN_PID"
        echo "[cloud_benchmark] Profile saved to ${REPO_ROOT}/sixmax/checkpoints/sample_${TIMESTAMP}.txt"
    else
        uv run python scripts/train_sixmax.py \
            --iterations "$ITERS" \
            --checkpoint "$BENCH_CHECKPOINT"
    fi
    END=$(date +%s)
    ELAPSED=$((END - START))
    [[ $ELAPSED -eq 0 ]] && ELAPSED=1
    ITERS_PER_SEC=$((ITERS / ELAPSED))
    echo ""
    echo "Benchmark complete (local)"
    print_cost_table "$ITERS_PER_SEC"
}

# ── Remote mode ───────────────────────────────────────────────────────────────
run_remote() {
    : "${HCLOUD_TOKEN:?HCLOUD_TOKEN not set — add to .env}"
    : "${HETZNER_SSH_KEY_NAME:?HETZNER_SSH_KEY_NAME not set — add to .env}"
    : "${GHCR_PAT:?GHCR_PAT not set — add to .env (GitHub PAT with read:packages scope)}"

    SERVER_NAME="pokerbot-bench-${TIMESTAMP//_/-}"  # hostnames disallow underscores
    # Ephemeral VPS: never trust/persist host keys (IPs get recycled across runs).
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
    SERVER_IP=""

    cleanup() {
        echo ""
        echo "[cloud_benchmark] Syncing results..."
        if [ -n "$SERVER_IP" ]; then
            rsync -avz --ignore-errors -e "ssh $SSH_OPTS" \
                "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/" \
                "${REPO_ROOT}/sixmax/checkpoints/" 2>/dev/null || true
            if $PROFILE; then
                rsync -avz --ignore-errors -e "ssh $SSH_OPTS" \
                    "root@${SERVER_IP}:/tmp/pokerbot/sixmax/checkpoints/callgrind_${TIMESTAMP}.out" \
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
    _ssh_attempts=0
    until ssh $SSH_OPTS "root@$SERVER_IP" echo ok 2>/dev/null; do
        sleep 5
        (( _ssh_attempts++ ))
        [[ $_ssh_attempts -ge 24 ]] && { echo "[cloud_benchmark] SSH timeout after 2 min — destroying server"; exit 1; }
    done

    echo "[cloud_benchmark] Installing Docker..."
    ssh $SSH_OPTS "root@$SERVER_IP" "curl -fsSL https://get.docker.com | sh"

    echo "[cloud_benchmark] Logging in to GHCR..."
    ssh $SSH_OPTS "root@$SERVER_IP" "echo '$GHCR_PAT' | docker login ghcr.io -u zfdupont --password-stdin"

    echo "[cloud_benchmark] Syncing repo source..."
    rsync -avz -e "ssh $SSH_OPTS" --exclude='third_party/' --exclude='buck-out/' \
        --exclude='sixmax/checkpoints/' --exclude='neural_cfr/checkpoints/' \
        --exclude='cfr/checkpoints/' --exclude='.git/' \
        --exclude='__pycache__/' --exclude='*.pyc' \
        "${REPO_ROOT}/" "root@${SERVER_IP}:/tmp/pokerbot/"

    echo "[cloud_benchmark] Pulling Docker image..."
    ssh $SSH_OPTS "root@$SERVER_IP" "docker pull $IMAGE"

    # -v broad ro mount FIRST, then write mount for checkpoints overrides it
    DOCKER_BASE="docker run --rm \
        -v /tmp/pokerbot/sixmax:/pokerbot/sixmax:ro \
        -v /tmp/pokerbot/sixmax/checkpoints:/pokerbot/sixmax/checkpoints \
        -v /tmp/pokerbot/scripts:/pokerbot/scripts:ro \
        -v /tmp/pokerbot/agents:/pokerbot/agents:ro \
        -e SIXMAX_SO_PATH=/opt/sixmax.so \
        --name pokerbot-bench \
        $IMAGE"

    TRAIN_CMD="python3.12 /pokerbot/scripts/train_sixmax.py \
        --iterations $ITERS \
        --checkpoint /pokerbot/sixmax/checkpoints/bench_${TIMESTAMP}.bin"

    START_REMOTE=$(date +%s)
    if $PROFILE; then
        echo "[cloud_benchmark] Installing valgrind on VM..."
        ssh $SSH_OPTS "root@$SERVER_IP" "apt-get install -y valgrind -qq"
        echo "[cloud_benchmark] Running with callgrind profiling inside container (this will be slow)..."
        ssh $SSH_OPTS "root@$SERVER_IP" "$DOCKER_BASE \
            valgrind --tool=callgrind \
            --callgrind-out-file=/pokerbot/sixmax/checkpoints/callgrind_${TIMESTAMP}.out \
            $TRAIN_CMD"
    else
        echo "[cloud_benchmark] Running timed benchmark..."
        ssh $SSH_OPTS "root@$SERVER_IP" "$DOCKER_BASE $TRAIN_CMD"
    fi
    END_REMOTE=$(date +%s)
    ELAPSED=$((END_REMOTE - START_REMOTE))
    [[ $ELAPSED -eq 0 ]] && ELAPSED=1
    ITERS_PER_SEC=$((ITERS / ELAPSED))

    echo ""
    echo "Benchmark complete"
    print_cost_table "$ITERS_PER_SEC"
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
