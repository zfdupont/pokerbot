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


def test_bb_option_three_players():
    """BB retains the option to raise even when button + SB have limped in."""
    cfg = sixmax.EngineConfig(num_players=3)
    # button=0 => SB=1, BB=2, UTG=0 opens preflop
    deck = AA + KK + QQ + DRY_BOARD
    deck += [c for c in range(52) if c not in deck]
    s = sixmax.HandState(cfg, 0, deck)
    assert s.current_player() == 0        # UTG acts first 3-handed
    s.apply_check_call()                  # UTG calls 1.0
    s.apply_check_call()                  # SB calls 0.5 more
    assert not s.is_terminal()            # hand not over — BB still has option
    assert s.current_player() == 2        # BB is next
    s.apply_check_call()                  # BB checks (takes the option)
    assert s.street() == sixmax.Street.Flop
    assert len(s.board()) == 3            # flop was dealt


def test_deal_is_seed_deterministic():
    cfg = sixmax.EngineConfig(num_players=6)
    a = sixmax.HandState.deal(cfg, 2, 99)
    b = sixmax.HandState.deal(cfg, 2, 99)
    assert [a.hole_cards(i) for i in range(6)] == [b.hole_cards(i) for i in range(6)]


def test_clone_is_independent():
    s = _hu()
    c = s.clone()
    c.apply_fold()
    assert c.is_terminal()
    assert not s.is_terminal()          # parent untouched
    assert s.hole_cards(0) == c.hole_cards(0)


def test_reroll_hidden_keeps_hero_and_revealed_board():
    cfg = sixmax.EngineConfig(num_players=6)
    s = sixmax.HandState(cfg, 0, list(range(52)))
    hero_hole = s.hole_cards(3)
    pool = [c for c in range(52) if c not in set(hero_hole)]
    s.reroll_hidden(3, pool, 7)
    assert s.hole_cards(3) == hero_hole             # hero fixed
    assert s.board() == []                          # still preflop
    opp = [s.hole_cards(i) for i in range(6) if i != 3]
    flat = [c for h in opp for c in h]
    assert len(set(flat)) == 10                     # distinct cards
    assert not (set(flat) & set(hero_hole))         # no overlap with hero


def test_reroll_hidden_preserves_revealed_board():
    s = _hu()
    s.apply_check_call(); s.apply_check_call()      # to flop
    revealed = s.board()
    hero_hole = s.hole_cards(0)
    pool = [c for c in range(52) if c not in set(hero_hole) | set(revealed)]
    s.reroll_hidden(0, pool, 3)
    assert s.board() == revealed
    assert s.hole_cards(0) == hero_hole


def test_settle_pots_into_matches_reference_on_random_sidepots():
    # 20k random tables, 2-6 players: unequal all-ins, random folds (never all),
    # random dense ranks with ties.
    # The reference is the current C++ loop, transcribed in Python, so float
    # results are compared exactly (==). It pins the allocation-free rewrite to
    # the pre-change algorithm, addition order included.
    import random

    rng = random.Random(20261024)

    def reference(totals, folded, ranks):
        n = len(totals)
        payout = [0.0] * n
        levels = sorted(t for t in totals if t > 1e-9)
        # std::unique: keep the first of every run of near-equal levels.
        deduped = []
        for level in levels:
            if not deduped or not abs(deduped[-1] - level) < 1e-9:
                deduped.append(level)
        prev = 0.0
        for level in deduped:
            amount = 0.0
            for i in range(n):
                amount += max(0.0, min(totals[i], level) - prev)
            best = None
            for i in range(n):
                if not folded[i] and totals[i] >= level - 1e-9:
                    best = ranks[i] if best is None else min(best, ranks[i])
            if best is not None:
                winners = [i for i in range(n)
                           if not folded[i] and totals[i] >= level - 1e-9
                           and ranks[i] == best]
            else:
                # Defensive: every contributor at this level folded.
                winners = [i for i in range(n) if not folded[i]]
            for w in winners:
                payout[w] += amount / len(winners)
            prev = level
        return payout

    for _ in range(20000):
        n = rng.randint(2, 6)
        # Shared contribution levels so exact duplicates and near-duplicates
        # (within kChipEps) both occur.
        levels = [rng.choice([0.0, 0.0, 5.0, 10.0, 25.0, 100.0])
                  for _ in range(rng.randint(1, 3))]
        totals = []
        for _ in range(n):
            t = rng.choice(levels)
            if t > 0.0 and rng.random() < 0.3:
                t += rng.choice([5e-10, -5e-10])   # inside kChipEps
            totals.append(t)
        # Random folds, but never everyone.
        folded = [rng.random() < 0.35 for _ in range(n)]
        if all(folded):
            folded[rng.randrange(n)] = False
        # Dense ranks 0..k-1 with ties; folded seats often get the best rank so
        # a rewrite that forgets to skip them fails loudly.
        k = rng.randint(1, n)
        ranks = [rng.randrange(k) for _ in range(n)]
        best_vals = sorted(set(ranks))
        ranks = [best_vals.index(r) for r in ranks]
        if rng.random() < 0.5:
            folded_idx = [i for i in range(n) if folded[i]]
            if folded_idx:
                ranks[rng.choice(folded_idx)] = 0

        assert sixmax.settle_pots(totals, folded, ranks) == \
            reference(totals, folded, ranks), (totals, folded, ranks)
