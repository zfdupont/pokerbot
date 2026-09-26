import os
import platform
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUCK = os.path.expanduser("~/bin/buck2")


def _so_path(target, soname):
    out = subprocess.run([_BUCK, "build", target, "--show-output"],
                         capture_output=True, text=True, cwd=_ROOT)
    assert out.returncode == 0, out.stderr
    for line in out.stdout.splitlines():
        if soname in line:
            return os.path.join(_ROOT, line.split()[-1])
    raise AssertionError(f"{soname} not in buck output: {out.stdout}")


def test_sixmax_so_is_torch_free():
    so = _so_path("//sixmax:sixmax", "sixmax.so")
    tool = ["otool", "-L", so] if platform.system() == "Darwin" else ["ldd", so]
    linkage = subprocess.run(tool, capture_output=True, text=True).stdout.lower()
    assert "libtorch" not in linkage
    assert "libc10" not in linkage


def test_sixmax_dream_builds_and_links_torch():
    so = _so_path("//sixmax:sixmax_dream", "sixmax_dream.so")
    tool = ["otool", "-L", so] if platform.system() == "Darwin" else ["ldd", so]
    linkage = subprocess.run(tool, capture_output=True, text=True).stdout.lower()
    assert "libtorch" in linkage


def test_cross_module_dreamtrainer(default_vocab):
    # sixmax and sixmax_dream are force-loaded by conftest (sixmax first)
    import sixmax
    import sixmax_dream
    abstraction = sixmax.Abstraction(
        flop_buckets=4, turn_buckets=4, river_buckets=4,
        equity_rollouts=5, quantile_samples=100, seed=1)
    cfg = sixmax_dream.DreamConfig()
    cfg.train_interval = 4
    cfg.sgd_steps = 1
    cfg.batch_size = 8
    cfg.reservoir_size = 1000
    tr = sixmax_dream.DreamTrainer(default_vocab.size(), default_vocab,
                                   abstraction, cfg, "cpu")
    tr.train(1)
    assert tr.total_iterations() >= 0
