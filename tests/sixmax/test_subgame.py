"""Street-subgame tree invariants: legality mirrors EngineGameState exactly,
the node cap degrades to leaves rather than throwing, and construction is
deterministic."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _flop_state():
    """Call + check preflop -> flop, BB to act. Same construction as
    tests/sixmax/test_engine_game.py::test_postflop_bet_spot_mask."""
    vocab = _load_vocab("search")
    cfg = sixmax.EngineConfig(num_players=2)
    eng = sixmax.EngineGameState(cfg, 0, list(range(52)), vocab, [])
    eng.apply(CALL)
    eng.apply(CHECK)
    assert eng.street() == 1
    tracker = sixmax.RangeTracker(cfg, vocab)
    tracker.reset(list(eng.board()), tuple(eng.hole_cards(0)))
    return eng, vocab, tracker


def _river_state():
    """Check down preflop -> flop -> turn -> river, hero still in. Mirrors
    tests/sixmax/test_leaf_eval.py::_street_state(3)."""
    vocab = _load_vocab("search")
    cfg = sixmax.EngineConfig(num_players=2)
    eng = sixmax.EngineGameState(cfg, 0, list(range(52)), vocab, [])
    while eng.street() < 3:
        eng.apply(CALL)
        eng.apply(CHECK)
    assert eng.street() == 3
    tracker = sixmax.RangeTracker(cfg, vocab)
    tracker.reset(list(eng.board()), tuple(eng.hole_cards(0)))
    return eng, vocab, tracker


def test_subgame_legality_matches_engine_exactly():
    eng, vocab, tracker = _flop_state()
    hole = tuple(eng.hole_cards(0))
    sg = sixmax.Subgame(eng, vocab, tracker, 0, hole, sixmax.SubgameConfig())
    root = sg.root()
    assert list(root.legal_mask()) == list(eng.legal_mask())
    assert root.actor() == eng.current_player()


def test_subgame_respects_node_cap():
    eng, vocab, tracker = _flop_state()
    hole = tuple(eng.hole_cards(0))
    cfg = sixmax.SubgameConfig()
    cfg.max_nodes = 2
    sg = sixmax.Subgame(eng, vocab, tracker, 0, hole, cfg)
    assert sg.node_count() <= 2


def test_subgame_hand_spaces_are_asymmetric():
    """#6: the hero seat holds exactly one combo; the opponent ranges over the
    posterior's surviving combos, and the hero's own combo is excluded from it."""
    eng, vocab, tracker = _flop_state()
    hole = tuple(eng.hole_cards(0))
    sg = sixmax.Subgame(eng, vocab, tracker, 0, hole, sixmax.SubgameConfig())
    assert sg.num_hands(0) == 1          # hero seat: point mass
    assert sg.num_hands(1) > 1           # opponent seat: a real range
    assert sg.prior_reach(0) == [1.0]


def test_subgame_construction_is_deterministic():
    eng, vocab, tracker = _flop_state()
    hole = tuple(eng.hole_cards(0))
    a = sixmax.Subgame(eng, vocab, tracker, 0, hole, sixmax.SubgameConfig())
    b = sixmax.Subgame(eng, vocab, tracker, 0, hole, sixmax.SubgameConfig())
    assert a.node_ids() == b.node_ids()
    assert list(a.root().legal_mask()) == list(b.root().legal_mask())


def test_subgame_completeness_reflects_leaf_evaluation():
    """Deadline->fallback safety: a subgame whose leaves could not all be valued
    reports complete() == False, so SearchStrategy can fall back to the blueprint
    instead of solving a tree with zero-valued (unevaluated) leaves.

    - Flop subgame, exact-only (no blueprint): flop/turn leaves need a blueprint
      continuation that is absent, so they go unevaluated -> incomplete.
    - River subgame: every leaf is board-complete and valued exactly by
      safe_eval without a blueprint -> complete."""
    eng, vocab, tracker = _flop_state()
    hole = tuple(eng.hole_cards(0))
    flop_sg = sixmax.Subgame(eng, vocab, tracker, 0, hole, sixmax.SubgameConfig())
    assert flop_sg.complete() is False

    reng, rvocab, rtracker = _river_state()
    rhole = tuple(reng.hole_cards(0))
    river_sg = sixmax.Subgame(reng, rvocab, rtracker, 0, rhole, sixmax.SubgameConfig())
    assert river_sg.complete() is True
