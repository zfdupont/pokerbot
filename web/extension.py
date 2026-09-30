"""Load the compiled sixmax extension without invoking Buck2 at runtime.

Mirrors `scripts/train_sixmax.py`: honor `SIXMAX_SO_PATH` (set in the container)
and fall back to a Buck2 build only for local dev. The repo-root `sixmax/`
directory is a namespace package that would shadow the `.so`, so the extension
is force-loaded and registered in `sys.modules["sixmax"]` directly.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _so_path() -> str:
    explicit = os.environ.get("SIXMAX_SO_PATH")
    if explicit:
        return explicit
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            return os.path.join(_ROOT, line.split()[-1])
    raise RuntimeError("could not locate sixmax.so in buck2 output")


def load_sixmax() -> None:
    if hasattr(sys.modules.get("sixmax"), "SizeUnit"):
        return
    so = _so_path()
    spec = importlib.util.spec_from_file_location("sixmax", so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod  # register before exec so cross-refs resolve
    spec.loader.exec_module(mod)
