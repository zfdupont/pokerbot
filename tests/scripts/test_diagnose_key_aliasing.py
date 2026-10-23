"""The script's key must equal the key the agent actually queries.

`diagnose_key_aliasing` used to rebuild keys with the legacy
`sixmax.pack_abstract_key`, which is wrong for a SIXBP003 checkpoint (pot index
lives at key bits 26-29). It now keys through the checkpoint's own abstraction.
"""
import os
import random
import shutil

import pytest

import sixmax
from scripts.diagnose_key_aliasing import aliasing_report, decision_key, \
    required_equity

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
C1_EDGES = [4.0, 8.0, 16.0, 40.0, 100.0]


def test_required_equity():
    assert required_equity(1.0, 3.0) == 0.25


def test_key_that_determines_price_explains_everything():
    r = aliasing_report([(1, 0.2), (1, 0.2), (2, 0.4), (2, 0.4)])
    assert r["r2_key"] == 1.0 and r["share_in_wide_keys"] == 0.0


def test_one_key_hiding_two_prices_explains_nothing():
    r = aliasing_report([(1, 0.2), (1, 0.4), (1, 0.2), (1, 0.4)])
    assert r["r2_key"] == 0.0 and r["share_in_wide_keys"] == 1.0


class _RecordingStrategy:
    """Delegate to a BlueprintStrategy while recording the keys it is asked."""

    def __init__(self, inner):
        self.inner = inner
        self.keys = []

    def abstraction(self):
        return self.inner.abstraction()

    def probs(self, key):
        self.keys.append(key)
        return self.inner.probs(key)

    def num_players(self):
        return self.inner.num_players()

    def num_infosets(self):
        return self.inner.num_infosets()


@pytest.fixture(scope="module")
def c1_checkpoint(tmp_path_factory, synthetic_bucket_map, default_vocab):
    """A tiny non-legacy (C1 pot edges) checkpoint beside its bucket map."""
    directory = tmp_path_factory.mktemp("key_alias_c1")
    map_copy = directory / "idxmod.bin"
    shutil.copyfile(synthetic_bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    abs_.set_pot_layout("bb", C1_EDGES)
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    trainer = sixmax.BlueprintTrainer(cfg, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(60)
    path = str(directory / "c1.bin")
    trainer.save(path, default_vocab, cfg, abs_)
    return path


def test_script_key_matches_agent_key(c1_checkpoint):
    from agents.sixmax_agent import SixmaxDeployStrategy

    # fallback="uniform": one lookup per decision. The default retries
    # neighbouring pot buckets on a miss, which this tiny checkpoint hits.
    deploy = SixmaxDeployStrategy.load(c1_checkpoint, _TOML,
                                       fallback="uniform")
    recorder = _RecordingStrategy(deploy._strategy)
    deploy._strategy = recorder

    # Values are in the big_blind=1 training frame, as decide() expects.
    kwargs = dict(
        hole=[0, 5], board=[9, 20, 31], street=1,
        raises_per_street=[1, 1, 0, 0], pot_bb=7.5,
        current_bet_bb=2.0, to_call_bb=1.0,
        stack_bb=98.0, eff_stack_bb=95.0,
        live=1, after=0, legal=[True, False, False, False, False, False],
        rng=random.Random(1))

    script_key = decision_key(deploy, **kwargs)
    deploy.decide(**kwargs)

    assert len(recorder.keys) == 1
    assert script_key == recorder.keys[0]
    # Non-legacy layout really is in play: bits 18-19 stay zero and the pot
    # index lives at bits 26-29.
    assert (script_key >> 18) & 0b11 == 0
    assert (script_key >> 26) & 0b1111 > 0
