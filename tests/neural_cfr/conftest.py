"""
conftest.py for tests/neural_cfr/

Handles two macOS-specific requirements before any test can import neural_cfr:
1. Pre-load libtorch dylibs (DYLD_LIBRARY_PATH is ignored under SIP).
2. Insert the Buck2-generated .so directory into sys.path (the output path
   contains a content-hash component that changes on rebuild, so we discover
   it dynamically via `buck2 build --show-output`).
"""
import ctypes
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/neural_cfr/ → root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1. Pre-load libtorch shared libraries
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(_REPO_ROOT, "third_party", "libtorch", "lib")
for _lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
    ctypes.CDLL(os.path.join(_LIB_DIR, _lib))

# ---------------------------------------------------------------------------
# 2. Discover Buck2 output path and insert into sys.path
# ---------------------------------------------------------------------------
def _discover_so_dir() -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//neural_cfr:neural_cfr", "--show-output"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            # Line format: "root//neural_cfr:neural_cfr  buck-out/v2/art/.../neural_cfr.so"
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError(
        f"Could not locate neural_cfr.so in buck2 output.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


_so_dir = _discover_so_dir()
if _so_dir not in sys.path:
    sys.path.insert(0, _so_dir)

# The project root contains a `neural_cfr/` directory which Python 3
# treats as a namespace package, taking precedence over neural_cfr.so
# even when so_dir is at sys.path[0]. Force-load the .so and register
# it as sys.modules['neural_cfr'] so every subsequent `import neural_cfr`
# gets the C++ extension.
import importlib.util as _ilu
import types as _types

_so_path = os.path.join(_so_dir, "neural_cfr.so")
_spec = _ilu.spec_from_file_location("neural_cfr", _so_path)
_mod = _ilu.module_from_spec(_spec)
sys.modules["neural_cfr"] = _mod  # register BEFORE exec so circular refs work
_spec.loader.exec_module(_mod)
