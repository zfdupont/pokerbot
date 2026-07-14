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
        "selection_enabled", "selection_hands", "selection_tabular_checkpoint",
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


# --- best-checkpoint selection helpers -------------------------------------

EVAL_OUTPUT = """\
Hands              : 10000
Running 5000 hands (neural=P0, tabular=P1) ...
Running 5000 hands (tabular=P0, neural=P1) ...
=== Neural CFR vs Tabular MCCFR (10000 hands) ===
  Neural win rate     : -46.7 BB/100
  Near 0 BB/100 = strategies agree; positive = neural has edge; negative = tabular wins
"""


def test_parse_bb100_extracts_value(train_neural):
    assert train_neural.parse_bb100(EVAL_OUTPUT) == pytest.approx(-46.7)
    assert train_neural.parse_bb100(
        EVAL_OUTPUT.replace("-46.7", "+3.2")) == pytest.approx(3.2)


def test_parse_bb100_returns_none_on_garbage(train_neural):
    assert train_neural.parse_bb100("no win rate here") is None
    assert train_neural.parse_bb100("") is None


def test_parse_bb100_integer_value(train_neural):
    assert train_neural.parse_bb100(
        EVAL_OUTPUT.replace("-46.7", "3")) == pytest.approx(3.0)


def _fake_ckpt(tmp_path, name="checkpoint.pt", content=b"weights"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_update_best_creates_sidecar_and_copy(train_neural, tmp_path):
    import json
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    assert train_neural.update_best(-46.7, ckpt, total_iters=500_000,
                                    hands=10_000) is True
    best = tmp_path / "best_checkpoint.pt"
    sidecar = tmp_path / "best_checkpoint.json"
    assert best.read_bytes() == b"v1"
    meta = json.loads(sidecar.read_text())
    assert meta["bb100"] == pytest.approx(-46.7)
    assert meta["total_iters"] == 500_000
    assert meta["hands"] == 10_000
    assert meta["source_checkpoint"].endswith("checkpoint.pt")
    assert "timestamp" in meta


def test_update_best_replaces_on_strict_improvement(train_neural, tmp_path):
    import json
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    train_neural.update_best(-46.7, ckpt, total_iters=1, hands=10)
    (tmp_path / "checkpoint.pt").write_bytes(b"v2")
    assert train_neural.update_best(-9.3, ckpt, total_iters=2,
                                    hands=10) is True
    assert (tmp_path / "best_checkpoint.pt").read_bytes() == b"v2"
    meta = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert meta["bb100"] == pytest.approx(-9.3)


def test_update_best_noop_on_worse_or_equal(train_neural, tmp_path):
    ckpt = _fake_ckpt(tmp_path, content=b"v1")
    train_neural.update_best(-9.3, ckpt, total_iters=1, hands=10)
    (tmp_path / "checkpoint.pt").write_bytes(b"v2")
    assert train_neural.update_best(-46.7, ckpt, total_iters=2,
                                    hands=10) is False   # worse
    assert train_neural.update_best(-9.3, ckpt, total_iters=2,
                                    hands=10) is False   # equal (strict)
    assert (tmp_path / "best_checkpoint.pt").read_bytes() == b"v1"


def test_selection_config_defaults_and_precedence(train_neural, tmp_path):
    cfg = train_neural.resolve_config(_args(), str(tmp_path))
    assert cfg["selection_enabled"] is False
    assert cfg["selection_hands"] == 10_000
    assert cfg["selection_tabular_checkpoint"] == ""

    toml = tmp_path / "cfg.toml"
    toml.write_text("[training]\nselection_enabled = true\nselection_hands = 500\n")
    cfg = train_neural.resolve_config(
        _args(config=str(toml), selection_hands=777), str(tmp_path))
    assert cfg["selection_enabled"] is True   # from file
    assert cfg["selection_hands"] == 777      # CLI wins
