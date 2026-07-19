"""Duplicate-deal A/B harness: masking-safe sampling, determinism, and a
trained-blueprint-beats-uniform smoke. Selection helpers unit-tested."""
import importlib.util
import json
import os
import random

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load(os.path.join(_ROOT, "scripts", "eval_sixmax.py"), "eval_sixmax")
VC = _load(os.path.join(_ROOT, "sixmax", "vocab_config.py"), "vocab_config")
VOCAB = VC.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                      "blueprint")
ABS = sixmax.Abstraction(flop_buckets=6, turn_buckets=6, river_buckets=4,
                         equity_rollouts=20, quantile_samples=150, seed=5)
CFG = sixmax.EngineConfig(num_players=2)


def test_sample_action_remasks_and_falls_back():
    rng = random.Random(1)
    # stored probs put all mass on an illegal action -> uniform over legal
    a = EV.sample_action([1.0, 0.0, 0.0], [0, 1, 1], rng)
    assert a in (1, 2)
    # unseen infoset (empty probs) -> uniform over legal
    a = EV.sample_action([], [1, 0, 1], rng)
    assert a in (0, 2)
    # legal mass is respected
    a = EV.sample_action([0.0, 1.0, 0.0], [1, 1, 0], rng)
    assert a == 1


def test_uniform_selfplay_is_deterministic_and_sane():
    u = EV.UniformStrategy()
    r1 = EV.run_match(u, u, VOCAB, CFG, hands=60, seed=17)
    r2 = EV.run_match(u, u, VOCAB, CFG, hands=60, seed=17)
    assert r1 == r2                       # fully seeded -> reproducible
    assert abs(r1) < 1000                 # sanity bound, not a strength claim


def test_trained_blueprint_beats_uniform(tmp_path):
    t = sixmax.BlueprintTrainer(CFG, VOCAB, ABS, num_threads=1, seed=21)
    t.train(6000)
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    strat = sixmax.BlueprintStrategy.load(path, VOCAB)
    bb100 = EV.run_match(strat, EV.UniformStrategy(), VOCAB, CFG,
                         hands=200, seed=33)
    # Trained HU blueprint vs uniform-random must be clearly positive. The
    # run is deterministic; if this fails, double the training iterations
    # once — if it still fails, STOP and escalate (do not weaken the bound).
    assert bb100 > 0


def test_selection_promote_and_replace(tmp_path):
    TR = _load(os.path.join(_ROOT, "scripts", "train_sixmax.py"), "train_sixmax")
    ckpt = tmp_path / "bp.bin"
    ckpt.write_bytes(b"v1")
    best = tmp_path / "best_checkpoint.bin"
    # first save: unconditional promote
    assert TR.update_best(None, str(ckpt), 100) is True
    assert best.read_bytes() == b"v1"
    # improvement: replace
    ckpt.write_bytes(b"v2")
    assert TR.update_best(5.0, str(ckpt), 200) is True
    assert best.read_bytes() == b"v2"
    side = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert side["iterations"] == 200
    # non-positive: keep
    ckpt.write_bytes(b"v3")
    assert TR.update_best(-1.0, str(ckpt), 300) is False
    assert best.read_bytes() == b"v2"


def test_parse_bb100():
    TR = _load(os.path.join(_ROOT, "scripts", "train_sixmax.py"), "train_sixmax")
    line = "Blueprint A win rate: +12.34 BB/100 (2400 hands)\n"
    assert TR.parse_bb100(line) == 12.34
    assert TR.parse_bb100("Blueprint A win rate: -3 BB/100 (10 hands)") == -3.0
    assert TR.parse_bb100("no match") is None
