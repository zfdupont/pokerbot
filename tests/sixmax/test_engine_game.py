"""Engine <-> ActionVocab bridge. Uses the default blueprint vocab
(10 entries): 0=fold 1=check 2=call 3=open2.5 4=open3.5 5=open5.0
6=bet0.33pot 7=bet0.75pot 8=bet1.5pot 9=allin."""
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
FOLD, CHECK, CALL = 0, 1, 2
OPEN25, OPEN35, OPEN50 = 3, 4, 5
B33, B75, B150 = 6, 7, 8
ALLIN = 9


def _hu_state(stacks=()):
    cfg = sixmax.EngineConfig(num_players=2)
    deck = list(range(52))  # deterministic deal; cards don't matter for masks
    return sixmax.EngineGameState(cfg, 0, deck, VOCAB, list(stacks))


def test_preflop_open_spot_mask():
    s = _hu_state()
    # SB facing the blind: fold/call legal, check not; BB opens legal,
    # pot-fraction entries masked (unopened preflop); jam legal.
    assert s.legal_mask() == [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]


def test_postflop_bet_spot_mask():
    s = _hu_state()
    s.apply(CALL)
    s.apply(CHECK)
    # Flop, pot 2.0, BB to act, no bet yet. 0.33 pot = 0.66 BB is below the
    # 1 BB min bet -> masked. BB-unit opens are preflop-only -> masked.
    assert s.legal_mask() == [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]


def test_preflop_3bet_spot_uses_pot_grid():
    s = _hu_state()
    s.apply(OPEN25)
    # BB facing a 2.5 open: pot-fraction 3-bets now legal, BB opens masked.
    mask = s.legal_mask()
    assert mask[FOLD] and mask[CALL] and not mask[CHECK]
    assert mask[OPEN25] == mask[OPEN35] == mask[OPEN50] == 0
    assert mask[B33] and mask[B75] and mask[B150] and mask[ALLIN]
    # ctx: pot=3.5-1.5=2.0, current_bet=2.5, to_call=1.5 =>
    # 0.33-pot 3-bet raises to 2.5 + 0.33*(2.0+3.0) = 4.15 >= min_raise 4.0.
    ctx = s.bet_context()
    assert abs(ctx.pot - 2.0) < 1e-9
    assert abs(ctx.to_call - 1.5) < 1e-9
    assert abs(VOCAB.target_bb(B33, ctx) - 4.15) < 1e-9


def test_short_stack_dedupes_bet_against_jam():
    # 4 BB stacks: every open's target caps at/over the 4 BB jam -> only
    # the AllIn entry may represent the stack-off.
    s = _hu_state(stacks=(4.0, 4.0))
    mask = s.legal_mask()
    assert mask[OPEN25] == 1          # 2.5 < 4.0 all-in target: distinct raise
    assert mask[OPEN35] == 1          # 3.5 < 4.0
    assert mask[OPEN50] == 0          # would cap to 4.0 == jam -> deduped
    assert mask[ALLIN] == 1


def test_apply_translates_and_terminal_utility_matches_engine():
    s = _hu_state()
    s.apply(OPEN25)
    s.apply(CALL)
    s.apply(CHECK)      # flop: BB checks
    s.apply(B75)        # button bets 0.75 * 5.0 = 3.75
    s.apply(FOLD)
    assert s.is_terminal()
    assert abs(s.utility(0) - 2.5) < 1e-9
    assert abs(s.utility(1) + 2.5) < 1e-9


def test_random_playouts_are_zero_sum_and_legal():
    import random
    rng = random.Random(5)
    for players in (2, 3, 6):
        g = sixmax.EngineGame(sixmax.EngineConfig(num_players=players), VOCAB)
        for trial in range(60):
            s = g.new_hand(seed=rng.randrange(2**60))
            steps = 0
            while not s.is_terminal():
                mask = s.legal_mask()
                assert len(mask) == VOCAB.size()
                legal = [i for i, ok in enumerate(mask) if ok]
                assert legal, "no legal action"
                s.apply(rng.choice(legal))
                steps += 1
                assert steps < 200, "hand failed to terminate"
            total = sum(s.utility(i) for i in range(players))
            assert abs(total) < 1e-6


def test_mccfr_smoke_on_hu_engine():
    g = sixmax.EngineGame(sixmax.EngineConfig(num_players=2), VOCAB)
    t = sixmax.MCCFRTrainer(g, seed=11)
    t.train(300)
    assert t.num_infosets() > 200
    g2 = sixmax.EngineGame(sixmax.EngineConfig(num_players=2), VOCAB)
    u2 = sixmax.MCCFRTrainer(g2, seed=11)
    u2.train(300)
    # Fresh game + same seed reproduces exactly (determinism end to end).
    assert u2.num_infosets() == t.num_infosets()
