"""LBR exploitability estimator: engine primitives, policy, decide, run."""
import importlib.util
import math
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


def _card(rank, suit):
    return (rank - 2) * 4 + suit


VOCAB = _load_vocab()
CFG = sixmax.EngineConfig(num_players=2)
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)


def _strategy(tmp_path, iters=120, seed=13):
    t = sixmax.BlueprintTrainer(CFG, VOCAB, ABS, num_threads=1, seed=seed)
    t.train(iters)
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    return sixmax.BlueprintStrategy.load(path, VOCAB)


def test_blueprint_probs_masks_and_normalizes(tmp_path):
    strat = _strategy(tmp_path)
    s = sixmax.EngineGameState(CFG, 0, list(range(52)), VOCAB, [])
    probs = sixmax.blueprint_probs(strat, s)
    mask = s.legal_mask()
    assert len(probs) == VOCAB.size()
    assert all(p >= 0.0 for p in probs)
    assert all(probs[i] == 0.0 for i in range(len(probs)) if not mask[i])
    assert abs(sum(probs) - 1.0) < 1e-9


def test_sample_blueprint_respects_the_mask(tmp_path):
    strat = _strategy(tmp_path)
    s = sixmax.EngineGameState(CFG, 0, list(range(52)), VOCAB, [])
    mask = s.legal_mask()
    for seed in range(20):
        a = sixmax.sample_blueprint(strat, s, False, seed)
        assert mask[a] == 1


def test_keys_use_the_strategys_own_abstraction(tmp_path):
    """When the strategy has a stored policy for the state's key, the masked
    distribution must be exactly that policy (i.e. keyed through the strategy's
    own abstraction); unseen keys legitimately fall back to uniform."""
    strat = _strategy(tmp_path, iters=400)
    premiums = [((14, 0), (14, 1)),    # AA
                ((13, 0), (13, 1)),    # KK
                ((12, 0), (12, 1)),    # QQ
                ((14, 0), (13, 1))]    # AKs
    seen = 0
    for (r0, s0), (r1, s1) in premiums:
        hole = [_card(r0, s0), _card(r1, s1)]
        deck = hole + [c for c in range(52) if c not in set(hole)]
        s = sixmax.EngineGameState(CFG, 0, deck, VOCAB, [])
        stored = strat.probs(s.abstract_key(strat.abstraction()))
        if stored:
            assert sixmax.blueprint_probs(strat, s) == stored
            seen += 1
    assert seen > 0


_AA = [_card(14, 0), _card(14, 1)]
_KK = [_card(13, 0), _card(13, 1)]
_DRY = [_card(2, 0), _card(7, 1), _card(11, 3), _card(3, 2), _card(9, 0)]
_OPEN25 = 3  # default blueprint vocab index


def _hu_deck(p0, p1):
    d = list(p0) + list(p1) + _DRY
    return d + [c for c in range(52) if c not in d]


def _evaluator(strat, **kw):
    return sixmax.LbrEvaluator(strat, VOCAB, CFG, sixmax.LbrConfig(**kw))


def test_decide_never_folds_the_nuts(tmp_path):
    strat = _strategy(tmp_path)
    # `initial` is the hand at the FIRST decision; history = SB's 2.5 BB open.
    initial = sixmax.HandState(CFG, 0, _hu_deck(_KK, _AA))
    assert initial.current_player() == 0
    ev = _evaluator(strat, samples=32, seed=1)
    a = ev.decide(initial.clone(), hero=1, actors=[0], actions=[_OPEN25], seed=1)
    assert a != 0  # 0 == fold in the default vocab


def test_decide_returns_minus_one_when_terminal(tmp_path):
    strat = _strategy(tmp_path)
    s = sixmax.HandState(CFG, 0, _hu_deck(_AA, _KK))
    s.apply_fold()
    ev = _evaluator(strat, samples=4, seed=1)
    assert ev.decide(s.clone(), hero=1, actors=[], actions=[], seed=1) == -1


def test_rollouts_terminate_for_short_stacks(tmp_path):
    strat = _strategy(tmp_path)
    ev = _evaluator(strat, samples=8, seed=1, max_steps=4096)
    for stack in (100.0, 20.0, 5.0):
        s = sixmax.HandState(CFG, 0, _hu_deck(_AA, _KK), [stack, stack])
        ev.decide(s.clone(), hero=0, actors=[], actions=[], seed=3)
    assert ev.sampled_rollouts_exhausted() == 0


def test_sample_weights_normalize(tmp_path):
    strat = _strategy(tmp_path)
    ev = _evaluator(strat, samples=16, seed=1)
    _, weights = ev.debug_sample_deals(
        sixmax.HandState(CFG, 0, _hu_deck(_KK, _AA)), 1, [0], [_OPEN25], seed=1)
    assert len(weights) == 16
    assert all(w >= 0.0 and math.isfinite(w) for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-9


def test_conditioning_changes_the_weights(tmp_path):
    strat = _strategy(tmp_path)
    initial = sixmax.HandState(CFG, 0, _hu_deck(_KK, _AA))
    ev = _evaluator(strat, samples=32, seed=1)
    _, cond = ev.debug_sample_deals(initial.clone(), 1, [0], [_OPEN25], seed=1)
    _, uncond = ev.debug_sample_deals(initial.clone(), 1, [], [], seed=1)
    assert not all(abs(a - b) < 1e-12 for a, b in zip(cond, uncond))


def test_zero_weight_history_falls_back_to_uniform(tmp_path):
    strat = _strategy(tmp_path)
    initial = sixmax.HandState(CFG, 0, _hu_deck(_KK, _AA))
    ev = _evaluator(strat, samples=8, seed=1)
    # Action index -1 matches no bucket: every weight is 0, so the sampler must
    # fall back to uniform weights rather than divide by zero.
    _, weights = ev.debug_sample_deals(initial.clone(), 1, [0], [-1], seed=1)
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(abs(w - 1.0 / 8) < 1e-12 for w in weights)
