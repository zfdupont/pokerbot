import importlib
import sys
from unittest.mock import patch


def test_sixmax_so_path_skips_build(monkeypatch, tmp_path):
    """When SIXMAX_SO_PATH is set, _build_and_get_so_dir must not be called."""
    monkeypatch.setenv("SIXMAX_SO_PATH", str(tmp_path / "fake.so"))

    # Force reload so the module re-reads the env var
    import scripts.train_sixmax as ts
    importlib.reload(ts)

    build_calls = []
    with patch.object(ts, "_build_and_get_so_dir", side_effect=lambda _: build_calls.append(1) or ""):
        try:
            ts._force_load_sixmax("/fake/root")
        except Exception:
            pass  # Expected — fake .so won't load

    assert build_calls == [], "_build_and_get_so_dir must not be called when SIXMAX_SO_PATH is set"


def test_sixmax_so_path_absent_calls_build(monkeypatch):
    """When SIXMAX_SO_PATH is unset, _build_and_get_so_dir must be called."""
    monkeypatch.delenv("SIXMAX_SO_PATH", raising=False)

    import scripts.train_sixmax as ts
    importlib.reload(ts)

    build_calls = []
    with patch.object(ts, "_build_and_get_so_dir", side_effect=lambda _: build_calls.append(1) or (_ for _ in ()).throw(RuntimeError("stop"))):
        try:
            ts._force_load_sixmax("/fake/root")
        except Exception:
            pass

    assert build_calls == [1], "_build_and_get_so_dir must be called when SIXMAX_SO_PATH is absent"
