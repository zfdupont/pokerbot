#!/bin/zsh
# Manage the openpoker bot: tracks the *python* PID (not a wrapper).
set -euo pipefail
cd "$(dirname "$0")/.."
PIDFILE=.openpoker.pid
LOG=neural_cfr/checkpoints/openpoker_$(date +%Y%m%d).log

case "${1:-}" in
  start)
    [ -f .env ] && { set -a; source .env; set +a; }
    [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null && \
      { echo "already running ($(cat $PIDFILE))"; exit 1; }
    nohup uv run python scripts/openpoker_bot.py \
        --checkpoint "${CHECKPOINT:-neural_cfr/checkpoints/best_checkpoint.pt}" \
        --buy-in "${BUY_IN:-2000}" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    caffeinate -w "$(cat $PIDFILE)" &
    echo "started $(cat $PIDFILE), log: $LOG" ;;
  stop)
    kill -TERM "$(cat $PIDFILE)" && rm -f "$PIDFILE" && echo "stopped" ;;
  status)
    kill -0 "$(cat $PIDFILE)" 2>/dev/null && echo "running $(cat $PIDFILE)" || echo "not running" ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
