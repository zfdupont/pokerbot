"""The harness's arithmetic and its gate must be unit-testable without a
checkpoint: paired differences, the t statistic, and the >0 AND |t|>=2 rule."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_search_ab", os.path.join(_ROOT, "scripts", "eval_search_ab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_stats_reports_t_statistic():
    mod = _load()
    stats = mod.paired_stats([10.0, 12.0, 14.0, 16.0])
    assert abs(stats["mean"] - 13.0) < 1e-9
    assert stats["t"] > 5.0


def test_verdict_requires_positive_and_significant():
    mod = _load()
    assert mod.verdict({"mean": 12.0, "t": 3.1}) == "promote"
    assert mod.verdict({"mean": 12.0, "t": 1.2}) == "keep-blueprint"
    assert mod.verdict({"mean": -3.0, "t": -4.0}) == "keep-blueprint"
