import pytest
import cfr.abstraction as abstraction_module


@pytest.fixture(autouse=True)
def fast_monte_carlo(monkeypatch):
    """Patch MC sample count to 10 for fast test runs (production uses 100)."""
    monkeypatch.setattr(abstraction_module, "MONTE_CARLO_SAMPLES", 10)
    abstraction_module._hand_to_bucket_cached.cache_clear()
    abstraction_module._board_to_bucket_cached.cache_clear()
    yield
    abstraction_module._hand_to_bucket_cached.cache_clear()
    abstraction_module._board_to_bucket_cached.cache_clear()
