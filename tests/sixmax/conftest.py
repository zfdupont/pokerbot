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
# Discover Buck2 output path and force-load extension modules
# ---------------------------------------------------------------------------
def _discover_so_dir(target: str, soname: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", target, "--show-output"],
                            capture_output=True, text=True, cwd=_REPO_ROOT)
    for line in result.stdout.splitlines():
        if soname in line:
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError(f"Could not locate {soname}\nstdout: {result.stdout}\nstderr: {result.stderr}")


def _force_load(module_name: str, target: str) -> None:
    soname = module_name + ".so"
    so_dir = _discover_so_dir(target, soname)
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    so_path = os.path.join(so_dir, soname)
    spec = _ilu.spec_from_file_location(module_name, so_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[module_name] = mod  # register before exec so circular refs work
    spec.loader.exec_module(mod)


_force_load("sixmax", "//sixmax:sixmax")            # must be first (cross-module types)
_force_load("sixmax_dream", "//sixmax:sixmax_dream")

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


def _write_idx_mod_map(path, counts):
    """Write a SIXBM001 map whose bucket is `index % k` on each street."""
    import numpy as np

    import sixmax  # force-loaded above

    arrays = []
    for street, k in zip((1, 2, 3), counts):
        size = sixmax.hand_index_size(street)
        values = np.empty(size, dtype=np.uint8)
        chunk = 1 << 22
        for start in range(0, size, chunk):
            stop = min(size, start + chunk)
            values[start:stop] = (
                np.arange(start, stop, dtype=np.uint64) % k).astype(np.uint8)
        arrays.append(values)
    sixmax._write_bucket_map_for_test(str(path), *arrays, "percentile")
    return str(path)


@pytest.fixture(scope="session")
def synthetic_bucket_map(tmp_path_factory):
    """Synthetic `index % k` bucket map at the real indexer sizes (7/5/3).

    Shared across test modules so the full-size map is written only once;
    the real (gitignored, multi-hundred-MB) tables are never touched here.
    """
    path = tmp_path_factory.mktemp("maps") / "idxmod.bin"
    return _write_idx_mod_map(path, (7, 5, 3))
