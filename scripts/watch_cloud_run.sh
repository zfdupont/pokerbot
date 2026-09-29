#!/usr/bin/env bash
# watch_cloud_run.sh — armed watcher for a running cloud_train.sh job.
#
# Pings (macOS notification + audible cue + terminal bell + status log) the moment
# the run EITHER reaches training (the container is up, i.e. the fragile
# SSH -> Docker -> GHCR-pull sequence cleared) OR hits an error / stops.
#
# Usage:
#   scripts/watch_cloud_run.sh [--log PATH] [--container NAME] [--interval SECS]
#
# Defaults:
#   --log       newest /tmp/pokerbot_cloud_train_*.log
#   --container pokerbot-trainer
#   --interval  15
#
# Runs until it fires one event. Launch detached, e.g.:
#   nohup scripts/watch_cloud_run.sh --log /tmp/pokerbot_cloud_train_XXXX.log \
#       >/tmp/pokerbot_watch.out 2>&1 &
set -uo pipefail

CONTAINER="pokerbot-trainer"
INTERVAL=15
LOG=""
# Never trust/persist recycled host keys (same policy as cloud_train.sh).
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o BatchMode=yes"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log)       LOG="$2";       shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --interval)  INTERVAL="$2";  shift 2 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ -z "$LOG" ]; then
    LOG=$(ls -t /tmp/pokerbot_cloud_train_*.log 2>/dev/null | head -1)
fi
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "[watch] No log found (pass --log PATH)"; exit 1
fi

STATUS="/tmp/pokerbot_watch_status.log"
STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
LOG_LINES=0
TRAINING_PINGED=0

# ── Notify ────────────────────────────────────────────────────────────────────
notify() {
    local title="$1"; shift
    local msg="$*"
    local safe="${msg//[\"\\]/}"
    safe="${safe//$'\n'/ }"
    osascript -e "display notification \"$safe\" with title \"$title\"" >/dev/null 2>&1 || true
    afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &
    # Ring the terminal that is tailing this log (best effort).
    local t
    for t in $(ps -o tty= -p $(pgrep -f "[t]ail -f ${LOG}" 2>/dev/null) 2>/dev/null); do
        [ "$t" != "??" ] && printf '\a' > "/dev/$t" 2>/dev/null || true
    done
    echo "[watch $STARTED_AT] $title: $msg" >> "$STATUS"
    echo "[watch $(date '+%H:%M:%S')] $title: $msg"
}

announce_training() {   # idempotent: a relaunch must not re-ping the same milestone
    if grep -q 'TRAINING reached' "$STATUS" 2>/dev/null; then
        echo "[watch] training already announced — guarding for errors"
    else
        notify "pokerbot TRAINING reached" "$1"
    fi
}

# ── Log scanning ──────────────────────────────────────────────────────────────
# Curated error signatures (avoid the generic word "error" — docker pull noise).
ERROR_RE='Host key verification failed|Permission denied|no matching manifest|manifest unknown|Error response from daemon|[Cc]annot connect to the Docker daemon|[Cc]onnection refused|[Cc]onnection timed out|[Tt]raceback \(most recent call last\)|RuntimeError|terminate called|std::|Segmentation fault|Killed|Out of memory|No space left on device|docker: Error|command not found|python3\.[0-9]+: not found|Aborted|FAILED'
# Signatures that the training script itself got past startup into the loop.
TRAIN_RE='\[cloud_train\] Starting training|Effective config:|Building abstraction|Running [0-9,]+ iterations|Resuming from|snapshot ->|^\[[0-9,]+/[0-9,]+\]'

read_new() {   # print lines appended since last call; sets LAST_CHUNK
    local total
    total=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    LAST_CHUNK=""
    if [ "$total" -gt "$LOG_LINES" ]; then
        LAST_CHUNK=$(tail -n +"$((LOG_LINES + 1))" "$LOG" 2>/dev/null)
        LOG_LINES=$total
    fi
}

server_ip() {
    grep -m1 'Server IP:' "$LOG" 2>/dev/null | awk '{print $NF}'
}

run_alive() {
    pgrep -f "[c]loud_train.sh" >/dev/null 2>&1
}

# "true" = running, "false" = exited, "MISSING" = no such container, "" = ssh failed.
probe_container() {
    local ip="$1" out
    [ -z "$ip" ] && { echo ""; return; }
    out=$(ssh $SSH_OPTS "root@$ip" \
        "docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null || echo MISSING" 2>/dev/null)
    echo "$out"
}

echo "[watch] Armed at $STARTED_AT on $LOG (container=$CONTAINER, interval=${INTERVAL}s)"
echo "[watch] Status file: $STATUS"

# ── Phase 1: wait for training (or error) ─────────────────────────────────────
while :; do
    read_new
    if printf '%s' "$LAST_CHUNK" | grep -Eq "$ERROR_RE"; then
        notify "pokerbot run ERROR (pre-training)" "$(printf '%s' "$LAST_CHUNK" | grep -E "$ERROR_RE" | tail -1)"
        exit 1
    fi

    out=$(probe_container "$(server_ip)")
    if [ "$out" = "true" ]; then
        announce_training "container $CONTAINER is up on $(server_ip)"
        TRAINING_PINGED=1
        break
    fi
    if printf '%s' "$LAST_CHUNK" | grep -Eq "$TRAIN_RE"; then
        announce_training "$(printf '%s' "$LAST_CHUNK" | grep -E "$TRAIN_RE" | tail -1)"
        TRAINING_PINGED=1
        break
    fi

    if ! run_alive; then
        notify "pokerbot run DIED before training" "cloud_train.sh exited; last log: $(tail -1 "$LOG" 2>/dev/null)"
        exit 1
    fi

    sleep "$INTERVAL"
done

# ── Phase 2: guard for errors / unexpected end ────────────────────────────────
ticks=0
while :; do
    read_new
    if printf '%s' "$LAST_CHUNK" | grep -Eq "$ERROR_RE"; then
        notify "pokerbot run ERROR" "$(printf '%s' "$LAST_CHUNK" | grep -E "$ERROR_RE" | tail -1)"
        exit 1
    fi

    if ! run_alive; then
        if printf '%s' "$LAST_CHUNK" | grep -q 'Total elapsed:'; then
            notify "pokerbot run ENDED" "cloud_train.sh finished (see $LOG)"
        else
            notify "pokerbot run ENDED unexpectedly" "cloud_train.sh gone; last: $(tail -1 "$LOG" 2>/dev/null)"
        fi
        exit 0
    fi

    # Every ~2 min, confirm the container is still up (skip transient ssh failures).
    if [ $((ticks % 8)) -eq 0 ] && [ "$ticks" -gt 0 ]; then
        out=$(probe_container "$(server_ip)")
        if [ "$out" = "false" ] || [ "$out" = "MISSING" ]; then
            notify "pokerbot CONTAINER stopped" "docker inspect says '$out' (training likely crashed)"
            exit 1
        fi
    fi

    ticks=$((ticks + 1))
    sleep "$INTERVAL"
done
