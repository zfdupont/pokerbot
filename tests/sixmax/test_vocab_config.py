import importlib.util
import os

import pytest

sixmax = pytest.importorskip("sixmax")

# Import vocab_config via importlib to bypass the sys.modules["sixmax"] registration
# (which binds sixmax to the .so, not a package). Mirror the conftest pattern.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_vocab_config_path = os.path.join(_repo_root, "sixmax", "vocab_config.py")
_spec = importlib.util.spec_from_file_location("vocab_config", _vocab_config_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_vocab = _mod.load_vocab


def test_load_spec_default(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text("""
[actions.blueprint]
preflop_opens = [ { size = 2.5, unit = "bb" },
                  { size = 3.5, unit = "bb" },
                  { size = 5.0, unit = "bb" } ]
bet_sizes     = [ { size = 0.33, unit = "pot" },
                  { size = 0.75, unit = "pot" },
                  { size = 1.5,  unit = "pot" } ]
include_allin = true
""")
    v = load_vocab(str(cfg), "blueprint")
    assert v.size() == 10  # fold, check, call, 3 opens, 3 bets, allin
    assert v.at(0).type == sixmax.ActionType.Fold
    assert v.at(3).size == 2.5 and v.at(3).unit == sixmax.SizeUnit.BB
    assert v.at(6).size == 0.33 and v.at(6).unit == sixmax.SizeUnit.Pot
    assert v.at(9).type == sixmax.ActionType.AllIn


def test_repo_default_config_loads():
    v = load_vocab("sixmax/configs/default.toml", "blueprint")
    assert v.size() == 10


def test_missing_section_raises(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text("[actions.blueprint]\ninclude_allin = true\n")
    with pytest.raises(KeyError):
        load_vocab(str(cfg), "search")
