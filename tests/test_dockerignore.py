"""The build context is the repo root, so the ignore file must live there."""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patterns():
    with open(os.path.join(_ROOT, ".dockerignore")) as f:
        return [line.strip() for line in f if line.strip()
                and not line.lstrip().startswith("#")]


def test_root_dockerignore_covers_tables_and_checkpoints():
    assert os.path.exists(os.path.join(_ROOT, ".dockerignore"))
    patterns = _patterns()
    assert "sixmax/tables/" in patterns
    assert "sixmax/checkpoints/" in patterns
    # Patterns moved from docker/.dockerignore must survive the merge.
    for pattern in ("third_party/libtorch/", "buck-out/", ".venv/"):
        assert pattern in patterns
