"""
conftest.py for tests/sixmax/

Handles the Buck2-generated .so directory discovery before any test can
import sixmax. The output path contains a content-hash component that
changes on rebuild, so we discover it dynamically via
`buck2 build --show-output`.
"""
import importlib.util as _ilu
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/sixmax/ → root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Discover Buck2 output path and insert into sys.path
# ---------------------------------------------------------------------------
def _discover_so_dir() -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            # Line format: "root//sixmax:sixmax  buck-out/v2/art/.../sixmax.so"
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError(
        f"Could not locate sixmax.so in buck2 output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


_so_dir = _discover_so_dir()
if _so_dir not in sys.path:
    sys.path.insert(0, _so_dir)

# The project root contains a `sixmax/` directory which Python 3
# treats as a namespace package, taking precedence over sixmax.so
# even when so_dir is at sys.path[0]. Force-load the .so and register
# it as sys.modules['sixmax'] so every subsequent `import sixmax`
# gets the C++ extension.
_so_path = os.path.join(_so_dir, "sixmax.so")
_spec = _ilu.spec_from_file_location("sixmax", _so_path)
_mod = _ilu.module_from_spec(_spec)
sys.modules["sixmax"] = _mod  # register BEFORE exec so circular refs work
_spec.loader.exec_module(_mod)
