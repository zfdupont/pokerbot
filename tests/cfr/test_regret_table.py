import os
import tempfile
import numpy as np
import pytest
from cfr.regret_table import RegretTable
from cfr.info_set import InfoSet


def make_infoset(**kwargs):
    defaults = dict(
        player=0, hand_bucket=2, street=0, board_bucket=0,
        betting_history=(), stack_bucket=3,
    )
    defaults.update(kwargs)
    return InfoSet(**defaults)


ACTIONS = ["fold", "check", "b0.5", "b1.0", "allin"]


class TestRegretMatching:
    def test_uniform_when_no_regrets(self):
        t = RegretTable()
        i = make_infoset()
        strat = t.get_strategy(i, ACTIONS)
        expected = 1.0 / len(ACTIONS)
        assert np.allclose(strat, expected)

    def test_positive_regrets_proportional(self):
        t = RegretTable()
        i = make_infoset()
        # ALL_ACTIONS order: fold=0, check=1, call=2, b0.5=3, b1.0=4, allin=5
        t.regrets[i] = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        strat = t.get_strategy(i, ACTIONS)
        assert strat[1] == pytest.approx(1.0)  # check gets all probability
        assert strat[0] == pytest.approx(0.0)

    def test_negative_regrets_clipped_to_zero(self):
        t = RegretTable()
        i = make_infoset()
        # fold=-5, check=2, call=0, b0.5=-1, b1.0=0, allin=0
        t.regrets[i] = np.array([-5.0, 2.0, 0.0, -1.0, 0.0, 0.0])
        strat = t.get_strategy(i, ACTIONS)
        assert strat[0] == pytest.approx(0.0)  # fold clipped
        assert strat[1] > 0                     # check has prob

    def test_strategy_sums_to_one(self):
        t = RegretTable()
        i = make_infoset()
        # fold=1, check=2, call=0, b0.5=0.5, b1.0=0, allin=-1
        t.regrets[i] = np.array([1.0, 2.0, 0.0, 0.5, 0.0, -1.0])
        strat = t.get_strategy(i, ACTIONS)
        assert np.sum(strat) == pytest.approx(1.0)


class TestUpdateAndAverage:
    def test_update_regrets(self):
        t = RegretTable()
        i = make_infoset()
        action_values = {"fold": -2.0, "check": 1.0, "b0.5": 0.5, "b1.0": 0.0, "allin": -1.0}
        t.update_regrets(i, action_values, node_value=0.0, actions=ACTIONS)
        assert t.regrets[i][1] == pytest.approx(1.0)   # check
        assert t.regrets[i][0] == pytest.approx(-2.0)  # fold

    def test_accumulate_strategy(self):
        t = RegretTable()
        i = make_infoset()
        strat = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        t.accumulate_strategy(i, strat, ACTIONS)
        t.accumulate_strategy(i, strat, ACTIONS)
        avg = t.get_average_strategy(i, ACTIONS)
        assert np.allclose(avg, strat)

    def test_average_strategy_uniform_when_empty(self):
        t = RegretTable()
        i = make_infoset()
        avg = t.get_average_strategy(i, ACTIONS)
        assert np.allclose(avg, 1.0 / len(ACTIONS))


class TestSaveLoad:
    def test_roundtrip(self):
        t = RegretTable()
        i = make_infoset()
        t.regrets[i] = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        t.strategy[i] = np.array([0.3, 0.4, 0.1, 0.1, 0.1])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            t.save(path)
            t2 = RegretTable()
            t2.load(path)
            assert np.allclose(t2.regrets[i], t.regrets[i])
            assert np.allclose(t2.strategy[i], t.strategy[i])
        finally:
            os.unlink(path)
