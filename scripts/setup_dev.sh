#!/bin/zsh
# Idempotent dev setup: run from a fresh clone, worktree, or cloud machine.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "SETUP FAIL: $1" >&2; exit 1; }

# 1. Python: extension ABI is pinned to 3.10 (sixmax.so / neural_cfr.so).
uv python pin 3.10 >/dev/null 2>&1 || true
uv sync
PYV=$(uv run python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[ "$PYV" = "3.10" ] || fail "venv resolved python $PYV, need 3.10 (extension ABI)"

# 2. third_party symlinks (worktrees don't inherit them).
MAIN_REPO=$(git rev-parse --path-format=absolute --git-common-dir)/..
for d in libtorch pybind11 indicators; do
  [ -e "third_party/$d" ] || ln -s "$MAIN_REPO/third_party/$d" "third_party/$d"
done

# 3. buck2 + python include path for pybind targets.
[ -x ~/bin/buck2 ] || fail "~/bin/buck2 not found (see .mex/context/setup.md)"
if [ ! -f .buckconfig.local ]; then
  INC=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))")
  printf '[python]\n  include_path = %s\n' "$INC" > .buckconfig.local
fi

# 4. Prove it.
~/bin/buck2 build //neural_cfr:neural_cfr //sixmax:sixmax
uv run pytest tests/ -q
echo "SETUP OK"
