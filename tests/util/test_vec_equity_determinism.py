"""equity_vec must be a pure function of (hole, board, n_samples). It feeds
CFRAgent's lru_cached buckets, so an entropy-seeded RNG made the villain's
bucket for a borderline hand differ run to run, breaking paired A/B evals
(np.random.seed does not reach a numpy Generator)."""
import numpy as np
import pytest

from models.card import Card
from models.enums import Suit
from util import vec_equity

pytestmark = pytest.mark.skipif(vec_equity._RANK7_NP is None,
                                reason="rank table not built")

HOLE = [Card(14, Suit.SPADES), Card(13, Suit.SPADES)]
BOARD = [Card(10, Suit.HEARTS), Card(7, Suit.CLUBS), Card(2, Suit.SPADES)]


def test_equity_vec_same_inputs_same_result():
    assert (vec_equity.equity_vec(HOLE, BOARD, 200)
            == vec_equity.equity_vec(HOLE, BOARD, 200))


def test_equity_vec_ignores_card_order():
    assert (vec_equity.equity_vec(HOLE, BOARD, 200)
            == vec_equity.equity_vec(HOLE[::-1], BOARD[::-1], 200))


def test_equity_vec_unaffected_by_global_numpy_state():
    np.random.seed(0)
    a = vec_equity.equity_vec(HOLE, [], 200)
    np.random.seed(12345)
    assert a == vec_equity.equity_vec(HOLE, [], 200)
