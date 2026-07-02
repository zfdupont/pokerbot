"""
Eval compatibility tests — verify Strategy exposes the interface expected by
eval_openspiel_neural.py and that error handling is correct.

sys.path and dylib pre-loading are handled by conftest.py in this directory.
"""
import pytest
import neural_cfr


def test_strategy_interface_matches_contract():
    """Verify Strategy exposes the interface expected by eval_openspiel_neural.py."""
    strat_cls = neural_cfr.Strategy
    # get_action_probs must exist
    assert hasattr(strat_cls, "get_action_probs")


def test_missing_checkpoint_raises():
    with pytest.raises(RuntimeError, match="not found"):
        neural_cfr.Strategy("/tmp/nonexistent_neural_cfr_checkpoint.pt")
