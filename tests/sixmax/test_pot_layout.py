"""Config-defined pot layouts in the abstraction infoset key.

Legacy (BB mode, edges 7/15/40) must stay byte-identical: pot at bits 18-19.
Any other layout moves the pot index to bits 26-29, leaves 18-19 zero, and
folds the layout into Abstraction::hash().
"""
import pytest

import sixmax

LEGACY_EDGES = [7.0, 15.0, 40.0]
C1_EDGES = [4.0, 8.0, 16.0, 40.0, 100.0]
C2_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]
SPR_EDGES = [0.5, 1.0, 2.0, 4.0, 8.0]

OPEN25, CALL = 3, 2


def _map_abs(synthetic_bucket_map):
    return sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)


def _pot_idx(key):
    return (key >> 26) & 15


def _mirror_apply(state, hand, action, vocab):
    """Apply the same action to an EngineGameState and a plain HandState."""
    a = vocab.at(action)
    if a.type in (sixmax.ActionType.Bet, sixmax.ActionType.AllIn):
        target = vocab.target_bb(action, state.bet_context())
        state.apply(action)
        hand.apply_raise_to(target)
    elif a.type == sixmax.ActionType.Fold:
        state.apply(action)
        hand.apply_fold()
    else:
        state.apply(action)
        hand.apply_check_call()


def _eff_stack(hand, acting, num_players):
    others = [hand.stack(i) for i in range(num_players)
              if i != acting and not hand.folded(i)]
    return min(hand.stack(acting), max(others, default=0.0))


def _assert_reproduces_engine_key(synthetic_bucket_map, default_vocab,
                                  mode, edges):
    abs_ = _map_abs(synthetic_bucket_map)
    abs_.set_pot_layout(mode, edges)
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    deck = list(range(52))
    state = sixmax.EngineGameState(cfg, 0, deck, default_vocab, [],
                                   abstraction=abs_)
    hand = sixmax.HandState(cfg, 0, deck)
    for action in (OPEN25, CALL):
        _mirror_apply(state, hand, action, default_vocab)
    assert state.street() == 1  # flop decision

    acting = state.current_player()
    card = abs_.bucket(state.hole_cards(acting), state.board())
    eff = _eff_stack(hand, acting, cfg.num_players)
    packed = abs_.pack_key(card, 1, [1, 0, 0, 0], hand.pot(), eff, 1, 1)
    assert packed == state.abstract_key(abs_)


def test_legacy_pack_key_matches_old(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    for pot in (1.0, 7.0, 7.01, 15.0, 15.01, 40.0, 40.01, 500.0):
        for card in (0, 37, 168):
            for street in range(4):
                raises = [1, 2, 3, 0]
                assert abs_.pack_key(card, street, raises, pot, 100.0, 2, 1) \
                    == sixmax.pack_abstract_key(card, street, raises, pot, 2, 1)


def test_legacy_hash_unchanged(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    before = abs_.hash()
    abs_.set_pot_layout("bb", LEGACY_EDGES)
    assert abs_.pot_layout() == ("bb", LEGACY_EDGES)
    assert abs_.hash() == before


def test_c1_bins_and_bits(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    abs_.set_pot_layout("bb", C1_EDGES)
    for pot, idx in ((4.0, 0), (4.01, 1), (100.0, 4), (100.01, 5)):
        key = abs_.pack_key(37, 1, [0, 0, 0, 0], pot, 100.0, 1, 1)
        assert (key >> 18) & 3 == 0
        assert _pot_idx(key) == idx


def test_spr_bins(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    abs_.set_pot_layout("spr", SPR_EDGES)
    key = abs_.pack_key(37, 1, [0, 0, 0, 0], 10.0, 15.0, 1, 1)
    assert _pot_idx(key) == 2  # 15 / 10 = 1.5, edges 0.5 / 1 / 2


def test_spr_allin_opponent(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    abs_.set_pot_layout("spr", SPR_EDGES)
    key = abs_.pack_key(37, 1, [0, 0, 0, 0], 10.0, 0.0, 1, 1)
    assert _pot_idx(key) == 0  # no division by zero, all-in opponent


def test_validation(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    bad_edge_sets = (
        [],
        [5.0, 5.0],
        [-1.0, 2.0],
        [float(i) for i in range(1, 17)],  # 16 edges > the 15 cap
    )
    for edges in bad_edge_sets:
        with pytest.raises(ValueError, match="pot_edges"):
            abs_.set_pot_layout("bb", edges)
    with pytest.raises(ValueError):
        abs_.set_pot_layout("x", LEGACY_EDGES)


def test_layout_hashes_distinct(synthetic_bucket_map):
    abs_ = _map_abs(synthetic_bucket_map)
    hashes = {abs_.hash()}  # legacy
    abs_.set_pot_layout("bb", C1_EDGES)
    hashes.add(abs_.hash())
    abs_.set_pot_layout("bb", C2_EDGES)
    hashes.add(abs_.hash())
    abs_.set_pot_layout("spr", SPR_EDGES)
    hashes.add(abs_.hash())
    assert len(hashes) == 4


def test_pack_key_reproduces_engine_key_c1(synthetic_bucket_map,
                                           default_vocab):
    _assert_reproduces_engine_key(synthetic_bucket_map, default_vocab,
                                  "bb", C1_EDGES)


def test_pack_key_reproduces_engine_key_spr(synthetic_bucket_map,
                                            default_vocab):
    _assert_reproduces_engine_key(synthetic_bucket_map, default_vocab,
                                  "spr", SPR_EDGES)
