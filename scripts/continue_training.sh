#!/bin/zsh
# Continue neural CFR training from the latest checkpoint, fully detached.
#
# Usage:
#   ./scripts/continue_training.sh [iterations]     (default: 5_000_000)
#
# - Resumes nets from neural_cfr/checkpoints/checkpoint.pt and keeps saving there.
# - Detached via nohup + caffeinate: survives terminal close and idle sleep.
# - A watcher archives every checkpoint save as archive_<MMDD_HHMM>.pt so
#   intermediate checkpoints are never lost to overwrites.
# - Known limitation (by design): reservoir buffers are not serialized, so the
#   resumed run's strategy net trains only on post-resume M_pi. Buffers refill
#   within the first ~2 CFR iterations; loaded advantage nets drive traversal
#   until the first training event reinitializes them.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=neural_cfr/checkpoints/checkpoint.pt
ITERS=${1:-5000000}
LOG=neural_cfr/checkpoints/train_$(date +%Y%m%d_%H%M).log

[[ -f $CKPT ]] || { echo "error: no checkpoint at $CKPT" >&2; exit 1; }
if pgrep -f "train_neural.py" > /dev/null; then
    echo "error: a train_neural.py process is already running" >&2
    exit 1
fi

nohup caffeinate -i .venv/bin/python3 scripts/train_neural.py \
    --config neural_cfr/configs/default.toml \
    --resume "$CKPT" \
    --iterations "$ITERS" \
    --checkpoint "$CKPT" > "$LOG" 2>&1 &
PID=$!
disown

# Archive each new save; seed with current mtime so the pre-resume checkpoint
# isn't re-archived.
nohup zsh -c "
  last=\$(stat -f %m $CKPT)
  while kill -0 $PID 2>/dev/null; do
    m=\$(stat -f %m $CKPT 2>/dev/null || true)
    if [[ -n \$m && \$m != \$last ]]; then
      cp $CKPT neural_cfr/checkpoints/archive_\$(date -r \$m +%m%d_%H%M).pt
      last=\$m
    fi
    sleep 60
  done
  m=\$(stat -f %m $CKPT 2>/dev/null || true)
  [[ -n \$m && \$m != \$last ]] && cp $CKPT neural_cfr/checkpoints/archive_\$(date -r \$m +%m%d_%H%M).pt
" > /dev/null 2>&1 &
disown

echo "training PID $PID  ($ITERS iterations, resumed from $CKPT)"
echo "log:      tail -f $LOG | tr '\r' '\n'"
echo "progress: ps -o etime=,time=,%cpu= -p $PID"
