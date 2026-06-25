import pytest
import cfr.abstraction as abstraction_module


@pytest.fixture(autouse=True)
def fast_monte_carlo(monkeypatch):
    """Patch MC sample count to 10 for fast test runs (production uses 500)."""
    monkeypatch.setattr(abstraction_module, "MONTE_CARLO_SAMPLES", 10)
    # Patch compute_exploitability wherever it was imported — must patch at the
    # usage site (cfr.trainer), not the definition site (cfr.mccfr), because
    # `from x import y` binds a local name that won't see module-level patches.
    try:
        import cfr.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "compute_exploitability", lambda *a, **kw: 0.0)
    except ImportError:
        pass  # trainer not yet imported in tasks that don't need it
