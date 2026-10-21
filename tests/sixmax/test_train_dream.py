"""scripts/train_dream.py refuses (warns about) bucket-map configs."""
import importlib.util
import os

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "train_dream.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_dream", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_warns_when_bucket_map_configured(capsys):
    mod = _load_script()
    mod.warn_if_bucket_map_ignored(
        {"bucket_map": "sixmax/tables/map_pct_50_50_20.bin"})
    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "bucket_map" in err
    assert "map_pct_50_50_20.bin" in err


def test_silent_without_bucket_map(capsys):
    mod = _load_script()
    mod.warn_if_bucket_map_ignored({})
    mod.warn_if_bucket_map_ignored({"bucket_map": ""})
    assert capsys.readouterr().err == ""
