"""Config resolution for scripts/train_neural.py (CLI > TOML > builtin)."""
import argparse
import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train_neural.py")


@pytest.fixture()
def train_neural():
    spec = importlib.util.spec_from_file_location("train_neural", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level imports are stdlib-only
    return mod


def _args(**overrides):
    """Namespace with every config key unset (None) unless overridden."""
    from_keys = [
        "iterations", "checkpoint_interval", "checkpoint", "reservoir_size",
        "batch_size", "lr", "train_interval", "sgd_steps", "reinit_adv",
        "num_threads", "epsilon", "eval_interval", "eval_hands", "config",
    ]
    ns = argparse.Namespace(**{k: None for k in from_keys})
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_builtin_defaults_when_no_config(train_neural, tmp_path):
    cfg = train_neural.resolve_config(_args(), str(tmp_path))  # no configs/ dir
    assert cfg["train_interval"] == 10_000
    assert cfg["sgd_steps"] == 2_000
    assert cfg["reinit_adv"] is True
    assert cfg["lr"] == pytest.approx(1e-3)


def test_config_file_overrides_builtin(train_neural, tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text("[trainer]\ntrain_interval = 10\nsgd_steps = 1\nreinit_adv = false\n")
    cfg = train_neural.resolve_config(_args(config=str(toml)), str(tmp_path))
    assert cfg["train_interval"] == 10
    assert cfg["sgd_steps"] == 1
    assert cfg["reinit_adv"] is False
    assert cfg["batch_size"] == 4096  # untouched builtin


def test_cli_overrides_config_file(train_neural, tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text("[trainer]\ntrain_interval = 10\n")
    cfg = train_neural.resolve_config(
        _args(config=str(toml), train_interval=77), str(tmp_path))
    assert cfg["train_interval"] == 77


def test_default_config_file_autoloaded(train_neural, tmp_path):
    configs = tmp_path / "neural_cfr" / "configs"
    configs.mkdir(parents=True)
    (configs / "default.toml").write_text("[training]\niterations = 123\n")
    cfg = train_neural.resolve_config(_args(), str(tmp_path))
    assert cfg["iterations"] == 123


def test_snapshot_roundtrips(train_neural, tmp_path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # py3.10 backport, declared in pyproject
    cfg = {"iterations": 5, "lr": 1e-3, "reinit_adv": True,
           "checkpoint": "a/b.pt", "eval_interval": None}
    ckpt = tmp_path / "ckpt.pt"
    train_neural.write_config_snapshot(cfg, str(ckpt))
    with open(str(ckpt) + ".config.toml", "rb") as f:
        snap = tomllib.load(f)["resolved"]
    assert snap["iterations"] == 5
    assert snap["reinit_adv"] is True
    assert snap["checkpoint"] == "a/b.pt"
    assert "eval_interval" not in snap  # None keys omitted
