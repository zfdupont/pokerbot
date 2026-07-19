"""Checkpoint round-trip, vocab-hash refusal, resume, and BlueprintStrategy."""
import importlib.util
import os

import pytest
import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _vocab_config():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VC = _vocab_config()
TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
VOCAB = VC.load_vocab(TOML, "blueprint")
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
CFG = sixmax.EngineConfig(num_players=2)


def _trained(iters=120, seed=13):
    t = sixmax.BlueprintTrainer(CFG, VOCAB, ABS, num_threads=1, seed=seed)
    t.train(iters)
    return t


def test_round_trip_preserves_everything(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    assert not os.path.exists(path + ".tmp")     # atomic write cleaned up

    abs2 = sixmax.load_abstraction(path)
    assert abs2.hash() == ABS.hash()             # edges stored, not rebuilt

    t2 = sixmax.resume_blueprint(path, CFG, VOCAB, abs2, num_threads=1, seed=13)
    assert t2.iterations() == t.iterations()
    assert sorted(t2.keys()) == sorted(t.keys())
    for k in t.keys():
        assert t2.average_strategy(k) == t.average_strategy(k)


def test_resume_continues_the_global_counter(tmp_path):
    t = _trained(iters=100)
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    t2 = sixmax.resume_blueprint(path, CFG, VOCAB, ABS, num_threads=1, seed=13)
    t2.train(50)
    assert t2.iterations() == 150


def test_wrong_vocab_is_refused(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    other = sixmax.ActionVocab([
        sixmax.AbstractAction(sixmax.ActionType.Fold, 0.0, sixmax.SizeUnit.BB),
        sixmax.AbstractAction(sixmax.ActionType.Check, 0.0, sixmax.SizeUnit.BB),
        sixmax.AbstractAction(sixmax.ActionType.Call, 0.0, sixmax.SizeUnit.BB),
    ])
    with pytest.raises(RuntimeError, match="vocab"):
        sixmax.BlueprintStrategy.load(path, other)
    with pytest.raises(RuntimeError, match="vocab"):
        sixmax.resume_blueprint(path, CFG, other, ABS, num_threads=1, seed=1)


def test_strategy_matches_trainer_and_keys_states(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    strat = sixmax.BlueprintStrategy.load(path, VOCAB)
    assert strat.iterations() == t.iterations()
    assert strat.num_players() == 2
    seen = 0
    for k in t.keys():
        expect = t.average_strategy(k)
        if expect:
            assert strat.probs(k) == expect
            seen += 1
    assert seen > 0
    # probs_for keys a fresh state through the strategy's OWN abstraction
    s = sixmax.EngineGameState(CFG, 0, list(range(52)), VOCAB, [])
    assert strat.probs_for(s) == strat.probs(s.abstract_key(strat.abstraction()))
