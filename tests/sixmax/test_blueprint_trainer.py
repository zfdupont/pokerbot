"""BlueprintTrainer: the multithreaded trainer must pass the same Kuhn
-1/18 gate as the Phase 1a single-threaded trainer, single-threaded runs
must be deterministic, and it must drive the abstracted 6-max EngineGame."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
KUHN_VALUE = -1.0 / 18.0


def test_kuhn_gate_single_thread():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=3)
    t.train(200_000)
    assert abs(sixmax.blueprint_kuhn_value(t) - KUHN_VALUE) < 0.01


def test_kuhn_gate_four_threads():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=4, seed=5)
    t.train(200_000)
    assert abs(sixmax.blueprint_kuhn_value(t) - KUHN_VALUE) < 0.01


def test_single_thread_determinism():
    a = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=11)
    b = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=11)
    a.train(2000)
    b.train(2000)
    assert sorted(a.keys()) == sorted(b.keys())
    for k in a.keys():
        assert a.average_strategy(k) == b.average_strategy(k)


def test_iterations_accumulate_across_train_calls():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=1)
    t.train(100)
    t.train(100)
    assert t.iterations() == 200


def test_drives_abstracted_sixmax_engine():
    cfg = sixmax.EngineConfig(num_players=6)
    t = sixmax.BlueprintTrainer(cfg, VOCAB, ABS, num_threads=2, seed=9)
    t.train(150)
    assert t.num_infosets() > 0
    for k in t.keys():
        probs = t.average_strategy(k)
        if probs:
            assert abs(sum(probs) - 1.0) < 1e-9


def _utg_open_regret_and_mass(records, num_players):
    """Regret L1 and visit-mass over the UTG opening decision: preflop, no
    raises yet, everyone still live and first-to-act (after == n-1). This is
    the highest-stakes opening node (AA/KK opens live here) and the one the
    button-rotation resonance starves."""
    from scripts.diagnose_blueprint import decode_key
    reg = mass = 0.0
    for key, _probs, m, r in records:
        d = decode_key(key)
        if (d["street"] == 0 and sum(d["raises"]) == 0
                and d["after"] == num_players - 1):
            reg += r
            mass += m
    return reg, mass


def test_utg_open_decisions_receive_regret(tmp_path):
    """Regression for the button-rotation resonance in the deal loop.

    The trainer must traverse each dealt hand once per seat as the traverser.
    The buggy loop dealt a *fresh* hand per traverser, so the button advanced
    a full cycle each iteration and the traverser was phase-locked to a single
    button-relative seat. Every other seat's decisions — including the UTG
    open, where AA/KK first act — accumulated average strategy but ZERO regret
    (verified: the UTG-open node carries huge visit-mass yet exactly-zero
    regret), freezing them at a uniform policy and producing the blueprint's
    flat BB/100 curve. A correct deal loop puts the traverser in every seat,
    so the UTG open must accrue regret."""
    cfg = sixmax.EngineConfig(num_players=6)
    t = sixmax.BlueprintTrainer(cfg, VOCAB, ABS, num_threads=1, seed=7)
    t.train(1500)
    ckpt = str(tmp_path / "bp.bin")
    t.save(ckpt, VOCAB, cfg, ABS)
    _iters, records = sixmax.dump_infosets(ckpt)
    reg, mass = _utg_open_regret_and_mass(records, num_players=6)
    assert mass > 0, "the UTG-open decision was never visited at all"
    assert reg > 0, (
        "the UTG open accumulated ZERO regret across 1500 iterations — the "
        "traverser never occupies the opening seat (button-rotation resonance "
        "in the deal loop)")
