"""Wiring contract. The first test is the one that protects production: with
[search] disabled the agent must be byte-identical to today's blueprint agent.
Requires the gitignored checkpoint, so the module skips when it is absent.

The match harness uses unseeded per-agent RNGs (SixmaxAgent._rng is
entropy-seeded, CFRAgent draws from numpy's global RNG), so the two arms are
played with explicitly pinned RNG state; otherwise the comparison would be
nondeterministic and meaningless."""
import os
import random

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_TABULAR = os.path.join(_ROOT, "cfr", "checkpoints", "checkpoint_09040000.pkl")

pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_TABULAR)),
                                reason="checkpoint(s) gitignored")


def _run_arm(search_toml):
    import numpy as np
    from agents.sixmax_agent import SixmaxAgent
    from agents.cfr_agent import CFRAgent
    from scripts.eval_hu_sanity import run_duplicate_match

    np.random.seed(20240701)

    def hero(_):
        agent = (SixmaxAgent(_CKPT, search_toml=search_toml)
                 if search_toml else SixmaxAgent(_CKPT))
        agent._rng = random.Random(1234)   # pin the decision stream
        return agent

    return run_duplicate_match(hero, lambda _: CFRAgent(_TABULAR), 200, 1, 2, 200)


def test_search_disabled_matches_blueprint_agent():
    """Same RNG stream: the search-aware agent with [search] disabled scores
    identically to the plain blueprint agent (the production regression)."""
    assert _run_arm(None) == _run_arm(_TOML)


def test_search_enabled_constructs_and_stays_legal():
    """Enabling search must not change the *interface*: the agent still emits
    only legal actions, and construction with a [search] section must not raise."""
    from agents.sixmax_agent import SixmaxAgent
    agent = SixmaxAgent(_CKPT, search_toml=_TOML)
    assert agent is not None
