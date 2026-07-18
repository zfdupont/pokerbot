"""External-sampling MCCFR must reproduce Kuhn poker's closed-form
equilibrium: game value -1/18 for P0; P0 bets J with alpha in [0,1/3],
K with 3*alpha, never Q; P1 folds J to a bet, always calls K."""
import sixmax

J, Q, K = 0, 1, 2
ROOT, AFTER_CHECK, AFTER_BET = 0, 1, 2


def _trained(seed=7, iters=200_000):
    g = sixmax.KuhnGame()
    t = sixmax.MCCFRTrainer(g, seed)
    t.train(iters)
    return t


def _bet_prob(t, card, hist):
    sigma = t.average_strategy(sixmax.kuhn_infoset_key(card, hist))
    assert len(sigma) == 2 and abs(sum(sigma) - 1.0) < 1e-9
    return sigma[1]


def test_game_value_converges():
    t = _trained()
    assert abs(sixmax.kuhn_exact_value(t) - (-1.0 / 18.0)) < 0.01


def test_equilibrium_strategy_structure():
    t = _trained()
    alpha = _bet_prob(t, J, ROOT)
    assert 0.0 <= alpha <= 1.0 / 3.0 + 0.05
    assert abs(_bet_prob(t, K, ROOT) - 3.0 * alpha) < 0.10
    assert _bet_prob(t, Q, ROOT) < 0.05          # never bet Q first
    assert _bet_prob(t, J, AFTER_BET) < 0.05     # P1 folds J to a bet
    assert _bet_prob(t, K, AFTER_BET) > 0.95     # P1 always calls K
    assert abs(_bet_prob(t, Q, AFTER_BET) - 1.0 / 3.0) < 0.07  # call Q 1/3


def test_deterministic_given_seed():
    a, b = _trained(seed=3, iters=5_000), _trained(seed=3, iters=5_000)
    assert a.num_infosets() == b.num_infosets() == 12
    for card in (J, Q, K):
        for hist in (ROOT, AFTER_CHECK, AFTER_BET, 3):
            key = sixmax.kuhn_infoset_key(card, hist)
            assert a.average_strategy(key) == b.average_strategy(key)


def test_unseen_key_returns_empty():
    g = sixmax.KuhnGame()
    t = sixmax.MCCFRTrainer(g, 1)
    assert t.average_strategy(12345) == []
