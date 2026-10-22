"""score_bucket_map.py: per-street reference selection, gain-per-infoset
efficiency, and the keep/drop aggregation."""
import argparse

import pytest

from scripts import score_bucket_map as sbm

_CTX = [73, 242, 753]


def _row(m, street, buckets, error, baseline=1.0):
    return {"map": m, "street": street, "buckets": buckets, "error": error,
            "baseline": baseline}


def _rows():
    # D0 50/50/20, DF 200/200/20, DR 50/50/100. Baselines differ per street,
    # as river deviation and flop/turn EMD do.
    base = (0.2, 0.2, 0.3)
    rows = []
    for m, counts, errs in [
        ("d0", (50, 50, 20), (0.10, 0.10, 0.10)),
        ("df", (200, 200, 20), (0.05, 0.095, 0.10)),
        ("dr", (50, 50, 100), (0.10, 0.10, 0.02)),
    ]:
        rows += [_row(m, s + 1, counts[s], errs[s], base[s])
                 for s in range(3)]
    return rows


def test_keeps_rule():
    assert sbm.keeps(1.0, 2.0, 0.5)
    assert not sbm.keeps(0.99, 2.0, 0.5)
    assert not sbm.keeps(0.0, 0.0, 0.5)   # removes nothing


def test_steps_efficiency_is_baseline_share_per_infoset():
    by = {(s["map"], s["street"]): s for s in sbm.steps(_rows(), _CTX)}
    flop = by[("df", "flop")]
    assert flop["extra_infosets"] == 73 * 150
    assert flop["efficiency"] == pytest.approx((0.05 / 0.2) / (73 * 150))
    river = by[("dr", "river")]
    assert river["extra_infosets"] == 753 * 80
    assert river["efficiency"] == pytest.approx((0.08 / 0.3) / (753 * 80))
    assert river["gain"] == pytest.approx(0.8)
    assert set(by) == {("df", "flop"), ("df", "turn"), ("dr", "river")}


def test_big_river_gain_can_lose_on_cost():
    # The river removes 80% of its coarser spread but costs about 10x more
    # per bucket: about 11% of the flop step's efficiency, so it is dropped.
    calls = sbm.verdicts(_rows(), _CTX, 0.5)
    assert calls["dr"]["keep"] is False
    assert calls["dr"]["streets"][0]["relative"] == pytest.approx(
        ((0.08 / 0.3) / (753 * 80)) / ((0.05 / 0.2) / (73 * 150)))
    assert calls["df"]["keep"] is True
    assert calls["d0"] == {"streets": [], "keep": True}


def test_fraction_zero_keeps_every_improving_step():
    calls = sbm.verdicts(_rows(), _CTX, 0.0)
    assert calls["dr"]["keep"] is True
    assert all(s["keep"] for s in calls["df"]["streets"])


def test_reference_is_next_smaller_count():
    rows = [_row("a", 3, 20, 0.10), _row("b", 3, 50, 0.05),
            _row("c", 3, 100, 0.049)]
    calls = sbm.verdicts(rows, _CTX, 0.5)
    assert calls["c"]["streets"][0]["vs"] == "b"
    assert calls["c"]["keep"] is False
    assert calls["b"]["keep"] is True


def test_contexts_parser():
    assert sbm._contexts("73,242,753") == [73, 242, 753]
    for bad in ("73,242", "0,1,2", "a,b,c"):
        with pytest.raises(argparse.ArgumentTypeError):
            sbm._contexts(bad)
