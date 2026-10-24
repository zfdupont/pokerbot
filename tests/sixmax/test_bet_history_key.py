"""Opt-in last-raise class per street in the sixmax infoset key.

Key bits 30-41 hold, for each street 0..3, the size class of the last bet or
raise on that street: 0 = none, 1-3 = rank among the vocabulary's sizes of the
same unit, 4 = all-in. Off (the default) leaves every key and hash unchanged.
"""
import importlib.util
import os

import pytest
import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab_mod():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_VOCAB = _load_vocab_mod()


@pytest.fixture
def c4_vocab():
    """hu200_c4.toml's blueprint vocab: 3/4/5 = open 2/3.5/5 BB,
    6/7/8 = bet 0.33/0.75/1.5 pot, 9 = all-in."""
    return _VOCAB.load_vocab(
        os.path.join(_ROOT, "sixmax", "configs", "hu200_c4.toml"), "blueprint")


@pytest.fixture
def abs_c4_layout(synthetic_bucket_map):
    """Map-backed abstraction with C4's pot layout (edges 3..192 BB)."""
    abs_ = sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)
    abs_.set_pot_layout("bb", [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0])
    return abs_


def _hu_state(vocab, stack_bb=200.0, abstraction=None):
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=stack_bb)
    return sixmax.EngineGameState(cfg, 0, list(range(9)), vocab, [],
                                  abstraction=abstraction)


def _vocab_with_pot_sizes(tmp_path, sizes):
    opens = "".join(f'{{ size = {s}, unit = "bb" }},\n' for s in (2.0, 3.5, 5.0))
    bets = "".join(f'{{ size = {s}, unit = "pot" }},\n' for s in sizes)
    path = tmp_path / "vocab.toml"
    path.write_text(
        "[actions.blueprint]\n"
        f"preflop_opens = [{opens}]\n"
        f"bet_sizes = [{bets}]\n"
        "include_allin = true\n")
    return _VOCAB.load_vocab(str(path), "blueprint")


def test_raise_class_ranks_within_unit(c4_vocab):
    assert [c4_vocab.raise_class(i) for i in range(10)] == \
        [0, 0, 0, 1, 2, 3, 1, 2, 3, 4]


def test_raise_class_refuses_four_sizes_in_a_unit(tmp_path):
    vocab = _vocab_with_pot_sizes(tmp_path, [0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="raise_class"):
        vocab.raise_class(6)


def test_raise_class_bounds_checked(c4_vocab):
    # raise_class is exposed to Python; an out-of-range index must raise
    # rather than read past the class table.
    with pytest.raises(IndexError):
        c4_vocab.raise_class(c4_vocab.size())
    with pytest.raises(IndexError):
        c4_vocab.raise_class(-1)


def test_off_mode_ignores_last_raise(abs_c4_layout):
    base = abs_c4_layout.pack_key(7, 2, [1, 1, 0, 0], 20.0, 180.0, 1, 0)
    assert abs_c4_layout.pack_key(7, 2, [1, 1, 0, 0], 20.0, 180.0, 1, 0,
                                  last_raise=[1, 2, 3, 4]) == base
    assert (base >> 30) & 0xFFF == 0


def test_last_raise_bits(abs_c4_layout):
    off = abs_c4_layout.pack_key(7, 3, [1, 1, 1, 1], 60.0, 140.0, 1, 0)
    abs_c4_layout.set_bet_history("last_raise")
    on = abs_c4_layout.pack_key(7, 3, [1, 1, 1, 1], 60.0, 140.0, 1, 0,
                                last_raise=[1, 2, 3, 4])
    assert (on >> 30) & 0xFFF == 1 | (2 << 3) | (3 << 6) | (4 << 9)
    assert on & ((1 << 30) - 1) == off


def test_hash_tracks_bet_history(abs_c4_layout):
    h_off = abs_c4_layout.hash()
    abs_c4_layout.set_bet_history("last_raise")
    assert abs_c4_layout.hash() != h_off
    abs_c4_layout.set_bet_history("off")
    assert abs_c4_layout.hash() == h_off


def test_unknown_bet_history_refused(abs_c4_layout):
    with pytest.raises(ValueError, match="bet_history"):
        abs_c4_layout.set_bet_history("full")


def test_engine_tracks_last_raise_per_street(c4_vocab, abs_c4_layout):
    # Open 3.5 BB (idx 4) -> class 2, call; flop: check, bet 0.33 (6) -> 1,
    # raise 1.5 (8) -> 3, call. The turn keeps the flop's last raise.
    abs_c4_layout.set_bet_history("last_raise")
    st = _hu_state(c4_vocab, abstraction=abs_c4_layout)
    for a in (4, 2, 1, 6, 8, 2):
        st.apply(a)
    assert st.last_raise_classes() == [2, 3, 0, 0]


def test_capped_bet_is_all_in_class(c4_vocab, abs_c4_layout):
    # 20 BB stacks: a 1.5-pot raise over a 5 BB open targets 26 BB > 20 BB, so
    # the engine masks the capped Bet (index 8); the legal AllIn (9) is the
    # only all-in. Class 4 follows from the resulting state either way.
    abs_c4_layout.set_bet_history("last_raise")
    st = _hu_state(c4_vocab, stack_bb=20.0, abstraction=abs_c4_layout)
    st.apply(5)
    assert st.legal_mask()[8] == 0   # a capped Bet cannot be chosen
    st.apply(9)
    assert st.last_raise_classes()[0] == 4


def test_off_mode_four_size_vocab_plays_a_full_hand(tmp_path, abs_c4_layout):
    # Off mode must not ask for a raise class: with 4 pot sizes the class table
    # is unavailable, yet the hand must play through without throwing.
    vocab = _vocab_with_pot_sizes(tmp_path, [0.25, 0.5, 0.75, 1.0])
    st = _hu_state(vocab, abstraction=abs_c4_layout)   # bet history off
    # 3 = open 2 BB, 2 = call; flop 6 = bet 0.25, 7 = raise 0.5, call;
    # turn 8 = bet 0.75, 9 = raise 1.0, call; river check-check.
    for a in (3, 2, 6, 7, 2, 8, 9, 2, 1, 1):
        st.apply(a)
    assert st.is_terminal()
    assert st.last_raise_classes() == [0, 0, 0, 0]


def test_on_mode_four_size_vocab_refused_at_construction(
        tmp_path, abs_c4_layout):
    vocab = _vocab_with_pot_sizes(tmp_path, [0.25, 0.5, 0.75, 1.0])
    abs_c4_layout.set_bet_history("last_raise")
    with pytest.raises(ValueError, match="raise_class"):
        _hu_state(vocab, abstraction=abs_c4_layout)


def test_engine_abstract_key_includes_classes_only_when_on(c4_vocab, abs_c4_layout):
    actions = (4, 2, 1, 7)
    # Off: keys are taken against an off abstraction, which zeroes the field
    # (tracking itself is unconditional, so the evaluator's unattached states
    # key the same as the trainer's).
    off_state = _hu_state(c4_vocab)
    for a in actions:
        off_state.apply(a)
    off_key = off_state.abstract_key(abs_c4_layout)
    # On: same deck and actions -> same low bits, class bits included.
    abs_c4_layout.set_bet_history("last_raise")
    on_state = _hu_state(c4_vocab, abstraction=abs_c4_layout)
    for a in actions:
        on_state.apply(a)
    on_key = on_state.abstract_key(abs_c4_layout)
    assert (off_key >> 30) & 0xFFF == 0
    assert (on_key >> 30) & 0xFFF == 2 | (2 << 3)


def test_unattached_state_keys_include_classes(c4_vocab, abs_c4_layout):
    """An EngineGameState built without an abstraction (eval_sixmax.py's
    _play_hand, the trainer's selection eval) must key exactly like the
    trainer's attached state on the same line -- otherwise a C5 checkpoint
    gets on-mode keys with zero class bits, misses, and silently plays
    uniform."""
    actions = (4, 2, 1, 7)          # open 3.5 (class 2), call, check, bet 0.75
    abs_c4_layout.set_bet_history("last_raise")
    attached = _hu_state(c4_vocab, abstraction=abs_c4_layout)
    plain = _hu_state(c4_vocab)     # no abstraction: how the A/B eval builds
    for a in actions:
        attached.apply(a)
        plain.apply(a)
    assert plain.last_raise_classes() == attached.last_raise_classes()
    assert plain.last_raise_classes() == [2, 2, 0, 0]
    on_key = attached.abstract_key(abs_c4_layout)
    assert (on_key >> 30) & 0xFFF != 0
    assert plain.abstract_key(abs_c4_layout) == on_key


def test_abstract_key_refuses_unsupported_vocab_when_on(tmp_path,
                                                        abs_c4_layout):
    """A 4-size-unit vocab cannot name a class, so an on-mode key would
    silently carry zeros. Keying (not just construction) must refuse."""
    vocab = _vocab_with_pot_sizes(tmp_path, [0.25, 0.5, 0.75, 1.0])
    plain = _hu_state(vocab)        # unattached: construction cannot refuse
    abs_c4_layout.set_bet_history("last_raise")
    with pytest.raises(ValueError, match="raise_class"):
        plain.abstract_key(abs_c4_layout)
    abs_c4_layout.set_bet_history("off")
    assert plain.abstract_key(abs_c4_layout) is not None   # off mode works
