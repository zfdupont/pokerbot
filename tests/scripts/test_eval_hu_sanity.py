"""Smoke test: the duplicate-deal HU eval returns a finite BB/100 over a few
hands with the blueprint on both sides (expected ~0 by symmetry, but we only
assert finiteness and duplicate-seat cancellation runs)."""
import math
import os

from scripts.eval_hu_sanity import run_duplicate_match
from agents.sixmax_agent import SixmaxAgent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_run_duplicate_match_finite(blueprint_hu_ckpt):
    def mk(_):
        return SixmaxAgent(blueprint_hu_ckpt, config_toml=_TOML)

    bb100 = run_duplicate_match(mk, mk, hands=20, seed=1, bb=2, stack=200)
    assert math.isfinite(bb100)
