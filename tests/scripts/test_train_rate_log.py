"""train_sixmax.py logs a machine-readable [rate] line at every save.

Format (one line, decimals to one place):
    [rate] iters=<N> it/s=<cumulative> window_it/s=<since the previous save>

The window is iterations since the previous save divided by wall time since the
previous save, so a stalled/uneven arm is visible from the log alone.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from tests.sixmax.conftest import _write_idx_mod_map  # same builder as the fixture

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")
_RATE_RE = re.compile(
    r"^\[rate\] iters=(\d+) it/s=(\d+\.\d+) window_it/s=(\d+\.\d+)$", re.M)

# Same full-size (real indexer sizes) synthetic map the sixmax fixtures use,
# written once per module so the gitignored sixmax/tables/ artifacts are never
# touched.
@pytest.fixture(scope="module")
def rate_bucket_map(tmp_path_factory):
    path = tmp_path_factory.mktemp("rate_maps") / "rate_map.bin"
    return _write_idx_mod_map(path, (7, 5, 3))


def test_rate_line_per_checkpoint_save(tmp_path, rate_bucket_map):
    toml = tmp_path / "rate.toml"
    toml.write_text(
        "[abstraction]\n"
        f"bucket_map = {json.dumps(rate_bucket_map)}\n"
        "[actions.blueprint]\n"
        'preflop_opens = [{size = 2.5, unit = "bb"}]\n'
        'bet_sizes = [{size = 0.33, unit = "pot"}]\n'
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\n"
        "iterations = 150\n"
        "checkpoint_interval = 50\n"
        "num_threads = 1\n"
        "seed = 3\n"
        "report_interval = 0\n")
    ckpt = tmp_path / "rate.bin"
    env = {**os.environ, "SIXMAX_BUCKET_MAP": rate_bucket_map}
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, env=env, timeout=600)
    assert result.returncode == 0, result.stderr

    rates = _RATE_RE.findall(result.stderr)
    assert len(rates) == 3, result.stderr          # 50, 100, 150 — final included
    assert [int(iters) for iters, _, _ in rates] == [50, 100, 150]
    assert result.stderr.count("saved ") == 3      # one save per [rate] line
    for _, cumulative, window in rates:
        assert float(cumulative) > 0.0, result.stderr
        assert float(window) > 0.0, result.stderr
