"""
Integration tests for neural_cfr.Strategy.

sys.path and dylib pre-loading are handled by conftest.py in this directory.
"""
import pytest
import neural_cfr


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    """Train briefly and save a checkpoint for strategy tests."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(20)
    path = str(tmp_path_factory.mktemp("ckpt") / "test.pt")
    trainer.checkpoint(path)
    return path


def test_strategy_loads(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    assert strat is not None


def test_get_action_probs_valid_distribution(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    probs = strat.get_action_probs(
        hole_cards=[0, 1],        # 2c, 2d
        board_cards=[],
        street=0,
        pot=1.5,
        stack=99.0,
        to_call=0.5,              # SB faces BB raise of 0.5
        raises_per_street=[0, 0, 0, 0],
        position=0,
    )
    assert isinstance(probs, dict)
    assert len(probs) > 0
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"
    for v in probs.values():
        assert 0.0 <= v <= 1.0


def test_get_action_probs_postflop(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    # Flop: Ah Kh Qh board
    board = [48, 44, 40]  # Ah=12*4+3=51? Let's use indices consistent with the engine
    probs = strat.get_action_probs(
        hole_cards=[2, 6],    # 3c, 4c
        board_cards=board,
        street=1,
        pot=4.0,
        stack=96.0,
        to_call=0.0,          # first to act postflop, no bet facing
        raises_per_street=[1, 0, 0, 0],
        position=1,
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-5
