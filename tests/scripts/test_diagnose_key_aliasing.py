"""aliasing_report: how much of the price faced the betting-state key explains."""
from scripts.diagnose_key_aliasing import aliasing_report, required_equity


def test_required_equity():
    assert required_equity(1.0, 3.0) == 0.25


def test_key_that_determines_price_explains_everything():
    r = aliasing_report([(1, 0.2), (1, 0.2), (2, 0.4), (2, 0.4)])
    assert r["r2_key"] == 1.0 and r["share_in_wide_keys"] == 0.0


def test_one_key_hiding_two_prices_explains_nothing():
    r = aliasing_report([(1, 0.2), (1, 0.4), (1, 0.2), (1, 0.4)])
    assert r["r2_key"] == 0.0 and r["share_in_wide_keys"] == 1.0
