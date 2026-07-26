"""
conftest.py for tests/sixmax/

Handles the Buck2-generated .so directory discovery before any test can
import sixmax. The output path contains a content-hash component that
changes on rebuild, so we discover it dynamically via
`buck2 build --show-output`.
"""
import ctypes
import importlib.util as _ilu
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/sixmax/ → root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Pre-load libtorch shared libraries (macOS SIP ignores DYLD_LIBRARY_PATH)
# Must happen before Buck2-built sixmax.so is dlopen'd below.
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(_REPO_ROOT, "third_party", "libtorch", "lib")
for _lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
    _lib_path = os.path.join(_LIB_DIR, _lib)
    if os.path.exists(_lib_path):
        ctypes.CDLL(_lib_path)

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

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
import pytest
import importlib.util as _ilu2


def _load_vocab_mod():
    path = os.path.join(_REPO_ROOT, "sixmax", "vocab_config.py")
    spec = _ilu2.spec_from_file_location("vocab_config", path)
    mod = _ilu2.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TOML = os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml")


@pytest.fixture(scope="session")
def default_vocab():
    """Session-scoped ActionVocab loaded from the default blueprint config."""
    vc = _load_vocab_mod()
    return vc.load_vocab(_TOML, "blueprint")
