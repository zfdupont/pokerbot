"""NLHE engine: BB frame (SB=0.5, BB=1.0, stacks in BB). Card code =
(rank-2)*4 + suit. Deck layout: player i holds deck[2i], deck[2i+1];
board = deck[2n .. 2n+4]."""
import sixmax


def _card(rank, suit):
    return (rank - 2) * 4 + suit


def _deck_hu(p0, p1, board):
    d = list(p0) + list(p1) + list(board)
    d += [c for c in range(52) if c not in d]
    return d


AA = [_card(14, 0), _card(14, 1)]
KK = [_card(13, 0), _card(13, 1)]
QQ = [_card(12, 0), _card(12, 1)]
DRY_BOARD = [_card(2, 0), _card(7, 1), _card(11, 3), _card(3, 2), _card(9, 0)]


def _hu(deck=None, stacks=()):
    cfg = sixmax.EngineConfig(num_players=2)
    deck = deck or _deck_hu(AA, KK, DRY_BOARD)
    return sixmax.HandState(cfg, 0, deck, list(stacks))


def test_preflop_setup_hu():
    s = _hu()
    assert s.current_player() == 0          # button/SB acts first HU preflop
    assert s.street() == sixmax.Street.Preflop
    assert abs(s.pot() - 1.5) < 1e-9
    assert abs(s.to_call() - 0.5) < 1e-9
    assert abs(s.min_raise_to() - 2.0) < 1e-9


def test_fold_preflop_awards_blinds():
    s = _hu()
    s.apply_fold()
    assert s.is_terminal()
    assert s.payoffs() == [-0.5, 0.5]
    assert s.board() == []                  # no cards revealed on a fold


def test_full_hand_raise_call_bet_call_showdown():
    s = _hu()
    s.apply_raise_to(2.5)
    assert abs(s.to_call() - 1.5) < 1e-9
    s.apply_check_call()                    # BB calls, pot 5.0
    assert s.street() == sixmax.Street.Flop
    assert s.current_player() == 1          # BB first postflop HU
    assert len(s.board()) == 3
    s.apply_check_call()                    # BB checks
    s.apply_raise_to(3.75)                  # button bets 0.75 pot
    s.apply_check_call()                    # BB calls, pot 12.5
    s.apply_check_call(); s.apply_check_call()   # turn checks through
    s.apply_check_call(); s.apply_check_call()   # river checks through
    assert s.is_terminal()
    assert s.payoffs() == [6.25, -6.25]     # aces win


def test_chopped_pot_on_played_board():
    board = [_card(10, 3), _card(11, 3), _card(12, 1), _card(13, 0), _card(14, 2)]
    lo0, lo1 = [_card(2, 0), _card(3, 1)], [_card(4, 0), _card(6, 1)]
    s = _hu(deck=_deck_hu(lo0, lo1, board))
    s.apply_check_call()                    # SB limps
    s.apply_check_call()                    # BB checks
    for _ in range(3):
        s.apply_check_call(); s.apply_check_call()
    assert s.is_terminal()
    assert s.payoffs() == [0.0, 0.0]        # broadway on board, chop


def test_min_raise_tracking():
    s = _hu()
    s.apply_raise_to(3.0)                   # raise size 2.0
    assert abs(s.min_raise_to() - 5.0) < 1e-9
    s.apply_raise_to(9.0)                   # 3-bet, raise size 6.0
    assert abs(s.min_raise_to() - 15.0) < 1e-9


def test_three_way_allin_side_pots():
    cfg = sixmax.EngineConfig(num_players=3)
    deck = QQ + KK + AA + [_card(2, 0), _card(7, 1), _card(9, 3),
                           _card(3, 2), _card(11, 1)]
    deck += [c for c in range(52) if c not in deck]
    s = sixmax.HandState(cfg, 0, deck, [100.0, 40.0, 10.0])
    # button=0 => SB=1, BB=2, UTG=0 opens.
    assert s.current_player() == 0
    s.apply_raise_to(100.0)                 # UTG jams QQ
    s.apply_check_call()                    # SB calls all-in for 40 (KK)
    s.apply_check_call()                    # BB calls all-in for 10 (AA)
    assert s.is_terminal()
    assert s.street() == sixmax.Street.River
    assert len(s.board()) == 5
    # Main pot 30 -> AA; side pot 60 -> KK; 60 uncalled back to QQ.
    assert s.payoffs() == [-40.0, 20.0, 20.0]
    assert abs(sum(s.payoffs())) < 1e-9


def test_bet_when_opponents_allin_is_skipped():
    cfg = sixmax.EngineConfig(num_players=2)
    s = sixmax.HandState(cfg, 0, _deck_hu(AA, KK, DRY_BOARD), [100.0, 20.0])
    s.apply_raise_to(30.0)
    s.apply_check_call()                    # BB all-in for 20
    assert s.is_terminal()                  # board runs out, no more betting
    assert s.payoffs() == [20.0, -20.0]


def test_settle_pots_ties_split_exactly():
    assert sixmax.settle_pots([100.0, 100.0], [False, False], [0, 0]) == [100.0, 100.0]
    # Main-pot tie plus a side pot: p0 all-in 50, p1/p2 at 200, p0 ties p1.
    got = sixmax.settle_pots([50.0, 200.0, 200.0], [False, False, False], [0, 0, 1])
    assert got == [75.0, 375.0, 0.0]
    # Odd amounts split exactly in the BB double frame.
    assert sixmax.settle_pots([3.0, 3.0, 3.0], [False, False, False],
                              [0, 0, 1]) == [4.5, 4.5, 0.0]


def test_settle_pots_folded_max_contributor_gets_nothing():
    got = sixmax.settle_pots([60.0, 60.0, 60.0], [False, True, False], [2, 0, 1])
    assert got == [0.0, 0.0, 180.0]


def test_deal_is_seed_deterministic():
    cfg = sixmax.EngineConfig(num_players=6)
    a = sixmax.HandState.deal(cfg, 2, 99)
    b = sixmax.HandState.deal(cfg, 2, 99)
    assert [a.hole_cards(i) for i in range(6)] == [b.hole_cards(i) for i in range(6)]
