"""Narrowing knobs (`narrow_preflop`, `prune_keep_mass`) resolve with the same
builtin < TOML < env precedence as the rest of the [search] block, and are kept
separate from the C++ SearchConfig."""
import os

from agents.sixmax_agent import _load_narrow_config

_TOML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "sixmax", "configs", "hu.toml")


def test_narrow_defaults_from_toml():
    cfg = _load_narrow_config(_TOML)
    assert cfg["narrow_preflop"] is True
    assert cfg["prune_keep_mass"] == 0.99


def test_narrow_env_override(monkeypatch):
    monkeypatch.setenv("SIXMAX_SEARCH_PRUNE_KEEP_MASS", "0.90")
    monkeypatch.setenv("SIXMAX_SEARCH_NARROW_PREFLOP", "false")
    cfg = _load_narrow_config(_TOML)
    assert cfg["prune_keep_mass"] == 0.90
    assert cfg["narrow_preflop"] is False
