"""The eval harness must pin each agent's private action RNG, not just the deck,
or paired A/B comparisons mix in uncontrolled action-sampling noise. A hand
whose decisions depend on an agent's entropy-seeded `_rng` must still replay
identically at the same seed once the harness pins it."""
import random

from models.enums import Action
from scripts.eval_hu_sanity import run_duplicate_match


class _RandomAgent:
    """Mirrors SixmaxAgent: a private, entropy-seeded _rng drives a mixed
    (call/fold) action — so without a harness pin, two runs diverge."""

    def __init__(self):
        self._rng = random.Random()

    def get_action(self, player, game_state):
        return (Action.CALL, None) if self._rng.random() < 0.5 else (Action.FOLD, None)


def _factory(_seat):
    return _RandomAgent()


def test_duplicate_match_reproducible_with_private_agent_rng():
    """Over many hands, two runs at the same seed must score identically once the
    harness pins each agent's private _rng. Aggregating ~25 hands makes a
    pre-pin (entropy) divergence effectively certain, so the RED is not flaky."""
    r1 = run_duplicate_match(_factory, _factory, 25, 7, 2, 200)
    r2 = run_duplicate_match(_factory, _factory, 25, 7, 2, 200)
    assert r1 == r2
