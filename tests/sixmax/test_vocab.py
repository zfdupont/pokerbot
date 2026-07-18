import pytest

sixmax = pytest.importorskip("sixmax")
A = sixmax.AbstractAction
T = sixmax.ActionType
U = sixmax.SizeUnit


def spec_vocab():
    return sixmax.ActionVocab([
        A(T.Fold, 0.0, U.BB), A(T.Check, 0.0, U.BB), A(T.Call, 0.0, U.BB),
        A(T.Bet, 2.5, U.BB), A(T.Bet, 3.5, U.BB), A(T.Bet, 5.0, U.BB),
        A(T.Bet, 0.33, U.Pot), A(T.Bet, 0.75, U.Pot), A(T.Bet, 1.5, U.Pot),
        A(T.AllIn, 0.0, U.BB),
    ])


def test_order_is_canonical_and_sized():
    v = spec_vocab()
    assert v.size() == 10
    assert v.at(3).size == 2.5 and v.at(3).unit == U.BB


def test_target_bb_units():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=7.5, current_bet=2.0, to_call=1.0, stack=100.0)
    assert v.target_bb(3, ctx) == 2.5                    # BB unit: raise-to 2.5BB
    # Pot unit: raise-to = current_bet + size * (pot + 2*to_call)   [spec formula]
    assert v.target_bb(7, ctx) == pytest.approx(2.0 + 0.75 * (7.5 + 2.0))
    assert v.target_bb(9, ctx) == 100.0                  # all-in = stack


def test_target_bb_capped_by_stack():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=200.0, current_bet=0.0, to_call=0.0, stack=10.0)
    assert v.target_bb(8, ctx) == 10.0                   # 1.5x pot capped to stack


def test_nearest_is_pseudo_harmonic():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=10.0, current_bet=0.0, to_call=0.0, stack=100.0)
    # Observed bet exactly on a grid point maps there regardless of u —
    # including BB-unit grid points (5.0 == the 5BB open).
    assert v.nearest(0.75 * 10.0, ctx, 0.0) == 7
    assert v.nearest(0.75 * 10.0, ctx, 0.999) == 7
    assert v.nearest(5.0, ctx, 0.0) == 5
    assert v.nearest(5.0, ctx, 0.999) == 5
    # Between 0.75x pot (7.5) and 1.5x pot (15.0), no BB point intervenes:
    # low u -> smaller size, high u -> larger.
    assert v.nearest(10.0, ctx, 0.0) == 7
    assert v.nearest(10.0, ctx, 0.999) == 8


def test_hash_changes_with_vocab():
    v1, v2 = spec_vocab(), spec_vocab()
    assert v1.hash() == v2.hash()
    v3 = sixmax.ActionVocab([A(T.Fold, 0.0, U.BB), A(T.AllIn, 0.0, U.BB)])
    assert v3.hash() != v1.hash()
