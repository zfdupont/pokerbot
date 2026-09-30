import os

from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_from_strategy_shares_strategy_and_has_independent_rng(blueprint_hu_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_hu_ckpt, _TOML)
    a1 = SixmaxAgent.from_strategy(strat, _TOML)
    a2 = SixmaxAgent.from_strategy(strat, _TOML)
    assert a1._deploy is a2._deploy is strat
    assert a1._rng is not a2._rng
