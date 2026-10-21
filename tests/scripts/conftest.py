# Force-load the sixmax + sixmax_dream C++ extensions the same way tests/sixmax/conftest.py does.
import tests.sixmax.conftest  # noqa: F401  (side effect: force-loads sixmax and sixmax_dream into sys.modules)

# Re-export the shared sixmax fixtures this directory's tests need (pytest only
# sees fixtures from conftest files on the test's own directory path).
from tests.sixmax.conftest import default_vocab, synthetic_bucket_map  # noqa: F401,E402
