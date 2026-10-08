"""The Phase-2 correctness gate: the vector CFR+ solver must reproduce the
closed-form Kuhn equilibrium value (-1/18 to player 0) before it is allowed
near hold'em. Mirrors the existing -1/18 gate on the tabular trainer."""
import math

import sixmax


def _solver(max_iters=20000):
    cfg = sixmax.VectorCfrConfig()
    cfg.max_iters = max_iters
    cfg.budget_ms = 60000.0
    return sixmax.VectorCfr(cfg)


def test_kuhn_vector_root_value_matches_closed_form():
    solver = _solver()
    res = solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    assert abs(res.root_value - (-1.0 / 18.0)) < 0.01, res.root_value
    assert not res.deadline_hit


def test_kuhn_vector_root_strategy_bets_the_best_card():
    """P0's first decision: bet the best card (hand 2) with probability 1.
    A solver that returns near-uniform is failing."""
    solver = _solver()
    solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    assert solver.rank_root_action(0, 2) == 1     # 1 == "bet" in the fixture


def test_kuhn_vector_average_strategy_sums_to_one():
    solver = _solver(max_iters=200)
    solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    for hand in range(3):
        probs = list(solver.average_strategy(solver.root_id(), hand))
        assert len(probs) == 2
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
