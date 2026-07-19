"""Config resolution + end-to-end smoke for scripts/train_sixmax.py."""
import importlib.util
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_sixmax", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """argparse.Namespace stand-in; unset flags are None."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __getattr__(self, _):
        return None


def test_resolve_config_precedence(tmp_path):
    mod = _load_script()
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        "[abstraction]\nflop_buckets = 8\n"
        "[train.blueprint]\niterations = 500\nnum_players = 3\n")
    cfg = mod.resolve_config(_Args(config=str(toml), iterations=250), _ROOT)
    assert cfg["iterations"] == 250          # CLI beats TOML
    assert cfg["num_players"] == 3           # TOML beats builtin
    assert cfg["flop_buckets"] == 8          # [abstraction] section merged
    assert cfg["turn_buckets"] == 50         # builtin default survives


def test_resolve_config_rejects_unknown_keys(tmp_path):
    mod = _load_script()
    toml = tmp_path / "bad.toml"
    toml.write_text("[train.blueprint]\nnot_a_key = 1\n")
    try:
        mod.resolve_config(_Args(config=str(toml)), _ROOT)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_end_to_end_smoke(tmp_path):
    """Tiny full run: builds abstraction, trains, saves a loadable artifact."""
    toml = tmp_path / "smoke.toml"
    toml.write_text(
        "[abstraction]\n"
        "flop_buckets = 6\nturn_buckets = 6\nriver_buckets = 4\n"
        "equity_rollouts = 20\nquantile_samples = 150\nseed = 5\n"
        "[actions.blueprint]\n"
        "preflop_opens = [{size = 2.5, unit = \"bb\"}]\n"
        "bet_sizes = [{size = 0.33, unit = \"pot\"}]\n"
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 40\nnum_threads = 1\nseed = 3\n")
    ckpt = tmp_path / "bp.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    assert ckpt.exists()
    assert (tmp_path / "bp.bin.config.toml").exists()   # effective-config snapshot
    import sixmax  # conftest already force-loaded the extension
    abs_ = sixmax.load_abstraction(str(ckpt))
    assert abs_.num_buckets(1) == 6
