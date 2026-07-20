# Force-load the sixmax C++ extension the same way tests/sixmax/conftest.py does.
from tests.sixmax.conftest import _mod  # noqa: F401  (side effect: registers sys.modules['sixmax'])
