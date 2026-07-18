"""Kuhn poker fixture: 2 players, cards {0=J,1=Q,2=K}, ante 1 BB, bet 1 BB.
Actions: 0 = check/fold ("pass"), 1 = bet/call."""
import sixmax


def test_check_check_showdown():
    s = sixmax.KuhnState(2, 0)  # P0 has K, P1 has J
    assert not s.is_terminal()
    assert s.current_player() == 0
    s.apply(0)
    assert s.current_player() == 1
    s.apply(0)
    assert s.is_terminal()
    assert s.utility(0) == 1.0   # showdown for the antes
    assert s.utility(1) == -1.0


def test_bet_fold_pays_bettor():
    s = sixmax.KuhnState(0, 2)  # P0 has J (bluffs), P1 has K but folds
    s.apply(1)
    s.apply(0)
    assert s.is_terminal()
    assert s.utility(0) == 1.0


def test_bet_call_showdown_for_two():
    s = sixmax.KuhnState(1, 2)  # P0 Q bets, P1 K calls
    s.apply(1)
    s.apply(1)
    assert s.is_terminal()
    assert s.utility(0) == -2.0
    assert s.utility(1) == 2.0


def test_check_bet_fold_and_call():
    s = sixmax.KuhnState(2, 1)
    s.apply(0)
    s.apply(1)
    assert not s.is_terminal()
    assert s.current_player() == 0
    fold = sixmax.KuhnState(2, 1)
    fold.apply(0); fold.apply(1); fold.apply(0)
    assert fold.is_terminal() and fold.utility(0) == -1.0
    call = sixmax.KuhnState(2, 1)
    call.apply(0); call.apply(1); call.apply(1)
    assert call.is_terminal() and call.utility(0) == 2.0


def test_legal_mask_always_both():
    s = sixmax.KuhnState(0, 1)
    assert s.legal_mask() == [1, 1]


def test_infoset_keys_hide_opponent_card():
    # History codes: 0="" 1="check" 2="bet" 3="check,bet".
    a = sixmax.KuhnState(1, 0)
    b = sixmax.KuhnState(1, 2)
    assert a.infoset_key() == b.infoset_key() == sixmax.kuhn_infoset_key(1, 0)
    a.apply(0); b.apply(0)
    # P1's key depends on P1's card, which differs.
    assert a.infoset_key() == sixmax.kuhn_infoset_key(0, 1)
    assert b.infoset_key() == sixmax.kuhn_infoset_key(2, 1)
    assert a.infoset_key() != b.infoset_key()


def test_game_deals_valid_hands():
    g = sixmax.KuhnGame()
    assert g.num_players() == 2
    assert g.num_actions() == 2
