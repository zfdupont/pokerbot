"""Snapshots are named by the trainer's TOTAL iterations, not by this run's.

`--resume` restarts the chunk counter at 0, so naming a snapshot by it reuses
(and silently overwrites) earlier names and breaks the `_<iters>.bin` contract
scripts/arm_plateau.py relies on. The `[rate] iters=` field carries the same
total so a resumed log stays monotone.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from tests.sixmax.conftest import _write_idx_mod_map  # same builder as the fixture

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")
_RATE_RE = re.compile(r"^\[rate\] iters=(\d+) it/s=", re.M)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture(scope="module")
def snapshot_bucket_map(tmp_path_factory):
    path = tmp_path_factory.mktemp("snap_maps") / "snap_map.bin"
    return _write_idx_mod_map(path, (7, 5, 3))


def _run(args, env):
    result = subprocess.run([sys.executable, _SCRIPT] + args,
                            capture_output=True, text=True, cwd=_ROOT,
                            env=env, timeout=900)
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_resume_snapshots_use_total_iterations(tmp_path, snapshot_bucket_map):
    toml = tmp_path / "snap.toml"
    toml.write_text(
        "[abstraction]\n"
        f"bucket_map = {json.dumps(snapshot_bucket_map)}\n"
        "[actions.blueprint]\n"
        'preflop_opens = [{size = 2.5, unit = "bb"}]\n'
        'bet_sizes = [{size = 0.33, unit = "pot"}]\n'
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\n"
        "iterations = 100\n"
        "checkpoint_interval = 50\n"
        "num_threads = 1\n"
        "seed = 3\n"
        "report_interval = 0\n"
        "snapshots = true\n")
    ckpt = tmp_path / "snap.bin"
    env = {**os.environ, "SIXMAX_BUCKET_MAP": snapshot_bucket_map}
    base = ["--config", str(toml), "--checkpoint", str(ckpt)]

    first = _run(base, env)
    assert [int(n) for n in _RATE_RE.findall(first)] == [50, 100]
    snap50 = tmp_path / "snap_00000050.bin"
    snap100 = tmp_path / "snap_00000100.bin"
    assert snap50.exists() and snap100.exists()
    sha50, sha100 = _sha(snap50), _sha(snap100)

    second = _run(base + ["--resume", str(ckpt), "--iterations", "50"], env)

    snap150 = tmp_path / "snap_00000150.bin"
    assert sorted(p.name for p in tmp_path.glob("snap_[0-9]*.bin")) == [
        "snap_00000050.bin", "snap_00000100.bin", "snap_00000150.bin"]
    # No resumed save overwrote the earlier snapshots.
    assert _sha(snap50) == sha50
    assert _sha(snap100) == sha100
    assert snap150.exists()
    # The rate line reports the trainer's total, so resumed logs stay monotone.
    assert [int(n) for n in _RATE_RE.findall(second)] == [150]
