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


def test_search_enabled_passes_pruned_prior():
    """With search + narrowing on, the agent calls SearchStrategy.probs with a
    non-empty, normalised, pruned opp_prior (fewer than the unblocked-combo
    count). A fake search captures the kwarg and forces the blueprint fallback."""
    from game.poker import PokerGame
    from models.player import Player
    from models.enums import Action
    from agents.sixmax_agent import SixmaxAgent
    from tests.test_action_history import ScriptedAgent

    sx = SixmaxAgent(_CKPT, search_toml=_TOML)

    class FakeSearch:
        enabled = True

        def __init__(self):
            self.captured = {}

        def probs(self, state, hero, prefix, opp_prior=None):
            self.captured["opp_prior"] = list(opp_prior) if opp_prior else []
            return []                       # force blueprint fallback

    fake = FakeSearch()
    sx._search = fake
    done = {}

    class HeroAgent(ScriptedAgent):
        def get_action(self, player, game_state):
            if game_state.betting_round == 1 and "x" not in done:
                sx.get_action(player, game_state)   # drive the search branch
                done["x"] = True
            return super().get_action(player, game_state)

    # p0 button/SB opens to 6 (3bb); p1 BB calls -> flop; BB acts first postflop.
    p0 = Player("sb", 100, ScriptedAgent([(Action.RAISE, 6), (Action.FOLD, None)]))
    p1 = Player("bb", 100, HeroAgent([(Action.CALL, None), (Action.CHECK, None)]))
    PokerGame([p0, p1], small_blind=1).play_hand()

    assert "x" in done
    prior = fake.captured["opp_prior"]
    nonzero = [p for p in prior if p > 0]
    assert nonzero                               # a prior was passed
    assert abs(sum(prior) - 1.0) < 1e-6          # normalised after pruning
    assert len(nonzero) < 1081                   # pruned below the unblocked count


def test_search_falls_back_to_blueprint_on_reconstruction_error():
    """If reconstruction raises, the decision equals the pure-blueprint decision
    (same pinned RNG) and never propagates an exception to the engine."""
    import random
    from game.poker import PokerGame
    from models.player import Player
    from models.enums import Action
    from agents.sixmax_agent import SixmaxAgent
    from tests.test_action_history import ScriptedAgent

    class EnabledSearch:
        enabled = True

        def probs(self, *a, **k):
            return []

    out = {}

    class HeroAgent(ScriptedAgent):
        def get_action(self, player, game_state):
            if game_state.betting_round == 1 and "fb" not in out:
                fb = SixmaxAgent(_CKPT, search_toml=_TOML)
                fb._search = EnabledSearch()
                fb._rng = random.Random(7)
                fb._replay_to_street_root = lambda *a, **k: (_ for _ in ()).throw(
                    RuntimeError("forced replay failure"))
                plain = SixmaxAgent(_CKPT)
                plain._rng = random.Random(7)
                out["fb"] = fb.get_action(player, game_state)
                out["plain"] = plain.get_action(player, game_state)
            return super().get_action(player, game_state)

    p0 = Player("sb", 100, ScriptedAgent([(Action.CALL, None), (Action.FOLD, None)]))
    p1 = Player("bb", 100, HeroAgent([(Action.CHECK, None), (Action.FOLD, None)]))
    PokerGame([p0, p1], small_blind=1).play_hand()

    assert "fb" in out
    assert out["fb"] == out["plain"]             # fallback == pure blueprint
