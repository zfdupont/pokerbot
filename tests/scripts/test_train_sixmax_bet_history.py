"""bet_history plumbing in scripts/train_sixmax.py: config, fresh path, resume.

The setting must reach a fresh abstraction BEFORE the trainer builds game
states, and a resume whose config disagrees with the checkpoint must be
refused rather than training one key space over the other.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import sixmax

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")
_C4 = os.path.join(_ROOT, "sixmax", "configs", "hu200_c4.toml")
_C5 = os.path.join(_ROOT, "sixmax", "configs", "hu200_c5.toml")
_CFG = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
C4_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]


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


def _load_vocab(toml_path):
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(toml_path, "blueprint")


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, _SCRIPT, *argv],
        capture_output=True, text=True, cwd=_ROOT, timeout=900)


def test_config_default_is_off():
    mod = _load_script()
    cfg = mod.resolve_config(_Args(config=_C4), _ROOT)
    assert cfg["bet_history"] == "off"
    # No --config at all resolves default.toml, still off.
    assert mod.resolve_config(_Args(), _ROOT)["bet_history"] == "off"


def test_c5_config_reads_last_raise():
    mod = _load_script()
    cfg = mod.resolve_config(_Args(config=_C5), _ROOT)
    assert cfg["bet_history"] == "last_raise"
    assert cfg["checkpoint"].endswith("hu200_c5.bin")
    assert cfg["num_players"] == 2 and cfg["starting_stack"] == 200.0
    # One change against C4: the vocabulary (and its hash) is identical.
    assert _load_vocab(_C5).hash() == _load_vocab(_C4).hash()


def test_resume_bet_history_mismatch_refused(tmp_path,
                                             synthetic_bucket_map,
                                             default_vocab):
    directory = tmp_path / "ck"
    directory.mkdir()
    shutil.copyfile(synthetic_bucket_map, directory / "idxmod.bin")
    abs_ = sixmax.Abstraction.from_bucket_map(str(directory / "idxmod.bin"))
    abs_.set_pot_layout("bb", C4_EDGES)  # exactly C4's layout
    abs_.set_bet_history("last_raise")
    trainer = sixmax.BlueprintTrainer(_CFG, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(20)
    ckpt = str(directory / "c5.bin")
    trainer.save(ckpt, default_vocab, _CFG, abs_)
    with open(ckpt, "rb") as f:
        assert f.read(8) == b"SIXBP004"

    # C4 has the same pot layout and vocabulary, but bet_history = "off".
    # --bucket-map pins the checkpoint's own map: C4's TOML names the real
    # sixmax/tables map, which exists in some checkouts and would be refused
    # on content hash before the bet-history check is reached.
    result = _run_cli("--config", _C4, "--resume", ckpt,
                      "--bucket-map", str(directory / "idxmod.bin"),
                      "--checkpoint", str(tmp_path / "out.bin"))
    assert result.returncode != 0
    assert "bet history mismatch" in result.stderr


def test_cli_trains_a_bet_history_checkpoint(tmp_path,
                                             synthetic_bucket_map):
    """End-to-end: the fresh map path turns the field on before training."""
    directory = tmp_path / "ck"
    directory.mkdir()
    shutil.copyfile(synthetic_bucket_map, directory / "idxmod.bin")
    ckpt = directory / "c5_fresh.bin"
    map_path = directory / "idxmod.bin"
    # C4's vocab blocks unchanged (same vocab hash), abstraction/train tweaked
    # down to a 20-iteration local run that carries the field.
    text = open(_C4).read()
    for old, new in [
            ('bucket_map = "sixmax/tables/map_pct_50_50_20.bin"',
             f'bucket_map = "{map_path}"'),
            ('pot_edges = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]',
             'pot_edges = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]\n'
             'bet_history = "last_raise"'),
            ('iterations = 60000000', 'iterations = 20'),
            ('num_threads = 0', 'num_threads = 1'),
            ('checkpoint_interval = 5000000', 'checkpoint_interval = 0'),
            ('snapshots = true', 'snapshots = false'),
            ('checkpoint = "sixmax/checkpoints/hu200_c4.bin"',
             f'checkpoint = "{ckpt}"')]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    toml = tmp_path / "c5.toml"
    toml.write_text(text)
    result = _run_cli("--config", str(toml))
    assert result.returncode == 0, result.stderr[-2000:]
    with open(ckpt, "rb") as f:
        assert f.read(8) == b"SIXBP004"
    assert sixmax.load_abstraction(str(ckpt)).bet_history() == "last_raise"
