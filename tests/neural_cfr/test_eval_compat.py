"""
Eval compatibility tests — verify Strategy exposes the interface expected by
eval_openspiel_neural.py and that error handling is correct.

sys.path and dylib pre-loading are handled by conftest.py in this directory.
"""
import tempfile
import os
import pytest
import neural_cfr


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    """Train briefly and save a checkpoint for strategy tests."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(500)
    path = str(tmp_path_factory.mktemp("ckpt") / "eval_compat_test.pt")
    trainer.checkpoint(path)
    return path


def test_strategy_interface_compatible(checkpoint_path):
    """Verify Strategy exposes the interface expected by eval_openspiel_neural.py."""
    strat_cls = neural_cfr.Strategy
    # get_action_probs must exist
    assert hasattr(strat_cls, "get_action_probs")
    # Construct a Strategy from the checkpoint and call get_action_probs
    strat = neural_cfr.Strategy(checkpoint_path)
    probs = strat.get_action_probs(
        hole_cards=[0, 1],
        board_cards=[],
        street=0,
        pot=1.5,
        stack=99.0,
        to_call=0.5,
        raises_per_street=[0, 0, 0, 0],
        position=0,
    )
    assert isinstance(probs, dict)
    assert len(probs) > 0
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_missing_checkpoint_raises():
    with pytest.raises(RuntimeError, match="not found"):
        neural_cfr.Strategy("/tmp/nonexistent_neural_cfr_checkpoint.pt")
