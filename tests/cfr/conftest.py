import pytest
import cfr.abstraction as abstraction_module


@pytest.fixture(autouse=True)
def fast_monte_carlo(monkeypatch):
    """Patch MC sample count to 10 for fast test runs (production uses 500)."""
    monkeypatch.setattr(abstraction_module, "MONTE_CARLO_SAMPLES", 10)
