"""SixmaxDeployStrategy's play-time fallback for infosets missing from the
table: nearest pot bucket first, then a safe action that never bets; and the
original uniform draw when asked for. A fake strategy with a known table pins
every branch; the trainer never uses this class."""
import random

import pytest

import agents.sixmax_agent as sa

sixmax = pytest.importorskip("sixmax")

_VOCAB = sa._load_vocab(sa._DEFAULT_TOML, "blueprint")
_T = sixmax.ActionType
_EDGES = [3.0, 6.0, 12.0, 24.0]


def _index(kind):
    return next(i for i in range(_VOCAB.size()) if _VOCAB.at(i).type == kind)


FOLD, CHECK, CALL, ALLIN = (_index(_T.Fold), _index(_T.Check), _index(_T.Call),
                            _index(_T.AllIn))
BET = _index(_T.Bet)


class _Abstraction:
    def __init__(self, mode="bb"):
        self._layout = (mode, _EDGES)

    def pot_layout(self):
        return self._layout

    def bucket(self, hole, board):
        return 0

    def bet_history(self):
        return "off"

    def pack_key(self, card, street, raises, pot_bb, eff_stack_bb, live, after,
                 last_raise=None):
        return (street, tuple(raises),
                sa._pot_index(self._layout, pot_bb, eff_stack_bb))


class _Strategy:
    """Table keyed by (street, raises, pot bucket)."""

    def __init__(self, table, mode="bb"):
        self._abs = _Abstraction(mode)
        self._table = table

    def abstraction(self):
        return self._abs

    def probs(self, key):
        return list(self._table.get(key, []))


def _one_hot(i):
    p = [0.0] * _VOCAB.size()
    p[i] = 1.0
    return p


def _legal(*idx):
    return [1 if i in idx else 0 for i in range(_VOCAB.size())]


def _decide(deploy, pot_bb, legal, to_call_bb=1.0, rng=None):
    return deploy.decide(
        hole=[0, 1], board=[2, 3, 4], street=1, raises_per_street=[1, 0, 0, 0],
        pot_bb=pot_bb, current_bet_bb=to_call_bb, to_call_bb=to_call_bb,
        stack_bb=190.0, live=1, after=0, legal=legal,
        rng=rng or random.Random(0), eff_stack_bb=190.0)


_FACING = _legal(FOLD, CALL, BET, ALLIN)


def _deploy(table, fallback="nearest", mode="bb"):
    return sa.SixmaxDeployStrategy(_Strategy(table, mode), _VOCAB,
                                   fallback=fallback)


def _key(bucket):
    return (1, (1, 0, 0, 0), bucket)


def test_hit_uses_the_table():
    d = _deploy({_key(0): _one_hot(CALL)})
    assert _decide(d, 3.0, _FACING)[0] == CALL
    assert d.lookups["hit"] == 1


def test_miss_uses_the_nearest_present_bucket():
    # 3.0 BB files in bucket 0, which training never reached.
    d = _deploy({_key(1): _one_hot(CALL), _key(2): _one_hot(ALLIN)})
    assert _decide(d, 3.0, _FACING)[0] == CALL
    assert d.lookups == {"hit": 0, "nearest": 1, "safe": 0, "uniform": 0,
                         "desync": 0}


def test_equal_distance_prefers_the_lower_bucket():
    d = _deploy({_key(1): _one_hot(FOLD), _key(3): _one_hot(CALL)})
    assert _decide(d, 9.0, _FACING)[0] == FOLD    # 9 BB is bucket 2


def test_neighbour_with_no_legal_weight_is_skipped():
    d = _deploy({_key(1): _one_hot(CHECK), _key(2): _one_hot(CALL)})
    assert _decide(d, 3.0, _FACING)[0] == CALL    # bucket 1 only checks


def test_beyond_radius_plays_safe():
    d = _deploy({_key(3): _one_hot(ALLIN)})       # 3 buckets from bucket 0
    assert _decide(d, 3.0, _FACING, to_call_bb=1.0)[0] == CALL
    assert d.lookups["safe"] == 1


@pytest.mark.parametrize("legal, to_call, expected", [
    (_legal(CHECK, BET, ALLIN), 0.0, CHECK),       # free: check
    (_FACING, 1.5, CALL),                          # 1.5 <= 0.5 * 3.0
    (_FACING, 2.0, FOLD),                          # 2.0 > 0.5 * 3.0
    (_legal(CALL, ALLIN), 50.0, CALL),             # no fold: call, never jam
])
def test_safe_action_never_bets(legal, to_call, expected):
    d = _deploy({})
    for seed in range(20):
        idx, raise_to = _decide(d, 3.0, legal, to_call_bb=to_call,
                                rng=random.Random(seed))
        assert idx == expected
        assert raise_to == 0.0


def test_uniform_fallback_keeps_the_original_behaviour():
    d = _deploy({}, fallback="uniform")
    picks = {_decide(d, 3.0, _FACING, rng=random.Random(s))[0]
             for s in range(200)}
    assert picks == {FOLD, CALL, BET, ALLIN}
    assert d.lookups["uniform"] == 200


def test_unknown_fallback_refused():
    with pytest.raises(ValueError, match="fallback"):
        _deploy({}, fallback="random")


@pytest.mark.parametrize("mode", ["bb", "spr"])
def test_pot_for_index_round_trips(mode):
    layout = (mode, _EDGES)
    for j in range(len(_EDGES) + 1):
        pot = sa._pot_for_index(layout, j, 190.0)
        assert sa._pot_index(layout, pot, 190.0) == j


def test_spr_without_stack_has_no_representative():
    assert sa._pot_for_index(("spr", _EDGES), 1, 0.0) is None
