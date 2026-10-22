"""LBR (Lisy & Bowling 2017): win probability, LocalBR decide, range removal."""
import importlib.util
import os
import random

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


def test_strategy_reports_its_training_stack(tmp_path):
    # Evaluators read the stack from the checkpoint instead of assuming 100.
    assert _strategy(tmp_path, iters=20).starting_stack() == 100.0
    cfg200 = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    t = sixmax.BlueprintTrainer(cfg200, VOCAB, ABS, num_threads=1, seed=3)
    t.train(20)
    path = str(tmp_path / "bp200.bin")
    t.save(path, VOCAB, cfg200, ABS)
    assert sixmax.BlueprintStrategy.load(path, VOCAB).starting_stack() == 200.0


# --- LocalBR decision core -------------------------------------------------

_A_C = _card(14, 0)   # A♣
_A_H = _card(14, 1)   # A♥
_A_S = _card(14, 3)   # A♠
_K_C = _card(13, 0)   # K♣
_K_D = _card(13, 1)   # K♦
_K_H = _card(13, 2)   # K♥
_K_S = _card(13, 3)   # K♠
_Q_C = _card(12, 0)   # Q♣
_Q_D = _card(12, 1)   # Q♦
_7_C = _card(7, 0)    # 7♣
_7_D = _card(7, 1)    # 7♦
_7_H = _card(7, 2)    # 7♥
_2_H = _card(2, 2)    # 2♥
_4_C = _card(4, 0)    # 4♣
_9_S = _card(9, 3)    # 9♠

_AA = [_A_S, _A_H]
_KK = [_K_D, _K_H]
# No king on the board: AA (pair of aces) beats KK (pair of kings) and loses
# to 7♣7♥ (set of sevens on the 7♦ board card).
_RIVER_BOARD = [_Q_C, _7_D, _2_H, _9_S, _4_C]  # Q♣ 7♦ 2♥ 9♠ 4♣


def _deck(hole0, hole1, board):
    cards = list(hole0) + list(hole1) + list(board)
    assert len(set(cards)) == len(cards), "duplicate card in fixture"
    return cards + [c for c in range(52) if c not in set(cards)]


def _index_range(weights):
    """(c0, c1, weight) triples -> 1,326-length range list."""
    r = [0.0] * 1326
    for c0, c1, w in weights:
        r[sixmax.RangeTracker.combo_index(c0, c1)] = w
    return r


def _evaluator(strat, **kw):
    return sixmax.LbrEvaluator(strat, VOCAB, CFG, sixmax.LbrConfig(**kw))


def _to_river(hero_hole, opp_hole, board=_RIVER_BOARD):
    """Deal `hero_hole` to seat 0 and `opp_hole` to seat 1 and check/call to the
    river; seat 1 acts first postflop."""
    s = sixmax.HandState(CFG, 0, _deck(hero_hole, opp_hole, board))
    for _ in range(3):  # preflop (limp + BB check), flop, turn
        s.apply_check_call()
        s.apply_check_call()
    assert s.street() == sixmax.Street.River
    assert s.current_player() == 1
    return s


def _uniform_range(state, hero):
    tracker = sixmax.RangeTracker(CFG, VOCAB)
    tracker.reset(state.board(), state.hole_cards(hero))
    return tracker.range(0)


def _assert_legal(action, to_call):
    assert 0 <= action < VOCAB.size()
    t = VOCAB.at(action).type
    if to_call > 0.0:
        assert t in (sixmax.ActionType.Fold, sixmax.ActionType.Call,
                     sixmax.ActionType.Bet, sixmax.ActionType.AllIn)
    else:
        assert t in (sixmax.ActionType.Check, sixmax.ActionType.Bet,
                     sixmax.ActionType.AllIn)


def _showdown(hero, opp, board5):
    a = sixmax.rank7(list(hero) + list(board5))
    b = sixmax.rank7(list(opp) + list(board5))
    if a.beats(b):
        return 1.0
    return 0.5 if a.ties(b) else 0.0


def _brute_turn(hero, board4, combos):
    """Exact win probability over all 44 rivers, ties = 1/2."""
    used = set(hero) | set(board4)
    num = den = 0.0
    for c0, c1, w in combos:
        if c0 in used or c1 in used:
            continue
        rivers = [c for c in range(52)
                  if c not in used and c != c0 and c != c1]
        total = 0.0
        for r in rivers:
            total += _showdown(hero, (c0, c1), list(board4) + [r])
        num += w * total / len(rivers)
        den += w
    return num / den


def test_win_probability_river_exact(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    assert abs(ev.win_probability(_AA, _RIVER_BOARD,
                                  _index_range([(_K_D, _K_H, 1.0)]), 1) - 1.0) < 1e-12
    assert abs(ev.win_probability(_AA, _RIVER_BOARD,
                                  _index_range([(_7_C, _7_H, 1.0)]), 1)) < 1e-12
    mixed = _index_range([(_K_D, _K_H, 1.0), (_7_C, _7_H, 1.0)])
    assert abs(ev.win_probability(_AA, _RIVER_BOARD, mixed, 1) - 0.5) < 1e-12


def test_win_probability_ties_count_half(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    board = [_A_C, _K_C, _card(12, 0), _card(11, 0), _card(10, 0)]  # royal
    r = _index_range([(_card(2, 0), _card(3, 1), 1.0)])
    assert abs(ev.win_probability(_AA, board, r, 1) - 0.5) < 1e-12


def test_win_probability_turn_matches_brute_force(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    board4 = [_K_C, _7_D, _2_H, _9_S]
    pool = [c for c in range(52) if c not in set(_AA) | set(board4)]
    rng = random.Random(2026)
    for _ in range(3):
        picked = rng.sample(pool, 8)
        combos = [(picked[2 * i], picked[2 * i + 1],
                   round(rng.uniform(0.05, 1.0), 3)) for i in range(4)]
        want = _brute_turn(_AA, board4, combos)
        got = ev.win_probability(_AA, board4, _index_range(combos), 1)
        assert abs(got - want) < 1e-9, (got, want, combos)


def test_win_probability_ignores_blocked_combos(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    valid = [(_7_C, _7_H, 1.0)]
    # Blocked by the board (Q♣) and by hero (A♠): both must contribute nothing.
    added = [(_Q_C, _Q_D, 5.0), (_A_S, _K_S, 5.0)]
    want = ev.win_probability(_AA, _RIVER_BOARD, _index_range(valid), 1)
    got = ev.win_probability(_AA, _RIVER_BOARD,
                             _index_range(valid + added), 1)
    assert abs(got - want) < 1e-12


def _brute_exact(hero, board, combos):
    """Exact win probability averaged over every completion of `board`."""
    used = set(hero) | set(board)
    pool = [c for c in range(52) if c not in used]
    need = 5 - len(board)
    num = den = 0.0
    for c0, c1, w in combos:
        if c0 in used or c1 in used:
            continue
        runs = [c for c in pool if c != c0 and c != c1]
        total = 0.0
        count = 0
        if need == 2:
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    full = list(board) + [runs[i], runs[j]]
                    total += _showdown(hero, (c0, c1), full)
                    count += 1
        else:
            for r in runs:
                total += _showdown(hero, (c0, c1), list(board) + [r])
                count += 1
        num += w * total / count
        den += w
    return num / den


def test_win_probability_flop_matches_brute_force(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    board3 = [_K_C, _7_D, _2_H]
    combos = [(_A_S, _K_H, 0.75), (_Q_D, _Q_C, 0.25)]
    want = _brute_exact(_AA, board3, combos)
    got = ev.win_probability(_AA, board3, _index_range(combos), 1)
    assert abs(got - want) < 1e-9


def test_decide_returns_minus_one_when_terminal(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    s = sixmax.HandState(CFG, 0, _deck(_AA, _KK, []))
    s.apply_fold()
    assert s.is_terminal()
    assert ev.decide(s, 0, [0.0] * 1326, 1) == -1


def test_decide_returns_minus_one_when_not_hero_to_act(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    s = sixmax.HandState(CFG, 0, _deck(_AA, _KK, []))
    assert s.current_player() == 0
    assert ev.decide(s, 1, _index_range([(_A_S, _A_H, 1.0)]), 1) == -1


def test_decide_never_folds_the_nuts(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    board = [_A_C, _7_D, _2_H, _9_S, _4_C]  # trips aces for hero
    s = _to_river(_AA, _KK, board)
    s.apply_raise_to(2.0)
    assert s.current_player() == 0 and s.to_call() == 2.0
    a = ev.decide(s, 0, _index_range([(_K_D, _K_H, 1.0)]), 1)
    assert VOCAB.at(a).type != sixmax.ActionType.Fold


def test_decide_folds_zero_equity_facing_bet(tmp_path):
    # Uniform opponent so the fold probability is small and fixed: no bluff is
    # profitable, so a hand losing to the whole range must fold.
    ev = _evaluator(_strategy(tmp_path, iters=20), uniform_opponents=True)
    board = [_A_C, _K_D, _2_H, _9_S, _4_C]
    s = _to_river([_7_C, _7_H], _AA, board)
    s.apply_raise_to(2.0)
    assert s.current_player() == 0 and s.to_call() == 2.0
    a = ev.decide(s, 0, _index_range([(_A_S, _A_H, 1.0)]), 1)
    assert VOCAB.at(a).type == sixmax.ActionType.Fold


def test_decide_checks_not_folds_when_unbet(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20), uniform_opponents=True)
    board = [_A_C, _K_D, _2_H, _9_S, _4_C]
    s = _to_river([_7_C, _7_H], _AA, board)
    s.apply_check_call()  # seat 1 checks; hero acts with nothing to call
    assert s.current_player() == 0 and s.to_call() == 0.0
    a = ev.decide(s, 0, _index_range([(_A_S, _A_H, 1.0)]), 1)
    assert VOCAB.at(a).type == sixmax.ActionType.Check


def test_decide_calls_on_non_acting_street(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20), act_streets=[3])
    s = sixmax.HandState(CFG, 0, _deck(_AA, _KK, []))
    assert s.street() == sixmax.Street.Preflop and s.current_player() == 0
    assert s.to_call() == 0.5
    a = ev.decide(s, 0, _index_range([(_K_D, _K_H, 1.0)]), 1)
    assert VOCAB.at(a).type in (sixmax.ActionType.Call, sixmax.ActionType.Check)


def test_fold_call_only_never_bets(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20), fold_call_only=True)
    banned = (sixmax.ActionType.Bet, sixmax.ActionType.AllIn)
    for seed in range(50):
        deck = list(range(52))
        random.Random(seed).shuffle(deck)
        s = sixmax.HandState(CFG, 0, deck)
        for _ in range(3):
            s.apply_check_call()
            s.apply_check_call()
        assert s.street() == sixmax.Street.River
        if seed % 2 == 0:
            s.apply_raise_to(2.0)  # hero faces a bet in half the spots
        hero = s.current_player()
        a = ev.decide(s, hero, _uniform_range(s, hero), seed)
        assert VOCAB.at(a).type not in banned


def test_zero_mass_range_still_returns_a_legal_action(tmp_path):
    ev = _evaluator(_strategy(tmp_path, iters=20))
    s = _to_river(_AA, _KK)
    s.apply_raise_to(2.0)
    _assert_legal(ev.decide(s, 0, [0.0] * 1326, 1), s.to_call())


def test_decide_short_stacks_still_returns_a_legal_action(tmp_path):
    # Short stacks make the all-in targets and the opponent's call cap bite;
    # the decision must stay legal (and must not fold when nothing is owed).
    ev = _evaluator(_strategy(tmp_path, iters=20), uniform_opponents=True)
    for stacks in ([5.0, 5.0], [100.0, 3.0]):
        s = sixmax.HandState(CFG, 0, _deck(_AA, _KK, []), stacks)
        for _ in range(3):
            s.apply_check_call()
            s.apply_check_call()
        assert s.street() == sixmax.Street.River
        s.apply_raise_to(2.0)
        hero = s.current_player()
        _assert_legal(ev.decide(s, hero, _uniform_range(s, hero), 1),
                      s.to_call())


def test_remove_blocked_renormalises(tmp_path):
    tracker = sixmax.RangeTracker(CFG, VOCAB)
    tracker.reset([], [_A_S, _A_H])
    assert abs(sum(tracker.range(0)) - 1.0) < 1e-12
    tracker.remove_blocked([_K_D])
    r = tracker.range(0)
    assert abs(sum(r) - 1.0) < 1e-12
    for i, p in enumerate(r):
        c0, c1 = sixmax.RangeTracker.combo_cards(i)
        if c0 == _K_D or c1 == _K_D:
            assert p == 0.0


def test_remove_blocked_keeps_the_prior_when_everything_is_blocked(tmp_path):
    tracker = sixmax.RangeTracker(CFG, VOCAB)
    tracker.reset([], [_A_S, _A_H])
    before = tracker.range(0)
    tracker.remove_blocked(list(range(52)))  # zero mass left: keep the prior
    assert tracker.range(0) == before
