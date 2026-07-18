import pytest

sixmax = pytest.importorskip("sixmax")

def c(rank, suit):  # rank 2-14, suit 0-3 (c,d,h,s)
    return (rank - 2) * 4 + suit

def test_aces_beat_deuces():
    board = [c(7, 0), c(8, 1), c(2, 2), c(9, 3), c(13, 0)]
    aces = sixmax.rank7([c(14, 0), c(14, 1)] + board)
    deuces = sixmax.rank7([c(3, 0), c(3, 1)] + board)
    assert aces.beats(deuces)
    assert not deuces.beats(aces)

def test_kicker_order_not_inverted():
    # AK high beats AQ high on the same board — the classic 12-rank bug detector.
    board = [c(2, 0), c(7, 1), c(9, 2), c(4, 3), c(11, 0)]
    ak = sixmax.rank7([c(14, 1), c(13, 2)] + board)
    aq = sixmax.rank7([c(14, 2), c(12, 3)] + board)
    assert ak.beats(aq)

def test_ties():
    board = [c(10, 0), c(10, 1), c(4, 2), c(4, 3), c(9, 0)]
    a = sixmax.rank7([c(2, 0), c(3, 1)] + board)   # board plays
    b = sixmax.rank7([c(2, 2), c(3, 3)] + board)
    assert a.ties(b) and not a.beats(b)
