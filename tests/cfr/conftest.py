import pytest


@pytest.fixture(autouse=True)
def patch_trainer_exploitability(monkeypatch):
    """Patch compute_exploitability in trainer to skip expensive BR computation in tests.
    Must patch at usage site (cfr.trainer), not definition site (cfr.mccfr), because
    `from x import y` binds a local name that won't see module-level patches."""
    try:
        import cfr.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "compute_exploitability", lambda *a, **kw: 0.0)
    except ImportError:
        pass  # trainer not yet imported in tasks that don't need it
