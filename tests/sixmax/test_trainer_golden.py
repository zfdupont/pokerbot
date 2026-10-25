"""Golden digests for the sixmax blueprint trainer.

Every change under sixmax/src/{blueprint,engine} must leave a single-threaded
training run bit-identical. These digests train a fixed scenario, dump the
saved checkpoint's infosets and hash the values in key order, so they pin the
floating-point results while ignoring `unordered_map` iteration order.

Baseline recorded on unmodified main, macOS arm64 (libc++), Task 1 of the
2026-10-24 trainer-speedup plan. Regenerate with:
    uv run python scripts/trainer_perf_check.py digest
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.trainer_perf_check import digest

GOLDEN = {
    "hu_bet_history": "b8777ee6baf5218bea5cfa85abfb880fd5578976cb0553fee2b332ec6ded3912",
    "sixmax_legacy": "e88267531873fa13eb781476b17fa21c347bf532eab249a3cdf688255bc23b6f",
    "hu_resume": "cd08c781e750f402a6860ce31591f2daec51d4256e16911625f8c102eeafaa0a",
}


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="std::uniform_real_distribution differs across C++ standard libraries")
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_trainer_digest_unchanged(name, tmp_path):
    assert digest(name, str(tmp_path)) == GOLDEN[name]
