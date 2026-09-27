# Force-load the sixmax + sixmax_dream C++ extensions the same way tests/sixmax/conftest.py does.
import tests.sixmax.conftest  # noqa: F401  (side effect: force-loads sixmax and sixmax_dream into sys.modules)
