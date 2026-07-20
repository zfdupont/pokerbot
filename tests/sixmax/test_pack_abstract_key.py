"""The Python-exposed packer must reproduce EngineGameState::abstract_key
exactly, and the pinned bit layout must be stable — this is the contract the
deployment bridge relies on to hit trained infosets."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(
        os.path.join(_ROOT, "sixmax", "configs", "default.toml"), "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)


def card(rank, suit):
    return (rank - 2) * 4 + suit


AKo = [card(14, 3), card(13, 1)]  # preflop class 155


def test_pot_bucket_thresholds():
    assert sixmax.pot_bucket(1.5) == 0
    assert sixmax.pot_bucket(7.0) == 0
    assert sixmax.pot_bucket(7.01) == 1
    assert sixmax.pot_bucket(15.0) == 1
    assert sixmax.pot_bucket(40.0) == 2
    assert sixmax.pot_bucket(40.01) == 3


def test_pack_matches_bit_layout():
    # card=155, street=0, no raises, pot 1.5 BB (bucket 0), live=4, after=4
    expected = 155 | (0 << 8) | (0 << 18) | (4 << 20) | (4 << 23)
    assert sixmax.pack_abstract_key(155, 0, [0, 0, 0, 0], 1.5, 4, 4) == expected


def test_pack_reproduces_engine_key():
    # Rebuild the 6-handed MP spot from test_abstract_key and check the packer
    # reproduces the engine's own key from independently supplied fields.
    cfg = sixmax.EngineConfig(num_players=6)
    deck = list(range(17))
    spares = iter(range(40, 52))
    taken = set(AKo)
    deck = [c if c not in taken else next(spares) for c in deck]
    deck[2 * 4], deck[2 * 4 + 1] = AKo[0], AKo[1]  # hero seat 4
    s = sixmax.EngineGameState(cfg, 0, deck, VOCAB, [], abstraction=ABS)
    s.apply(0)  # UTG (seat 3) folds -> hero (seat 4) to act
    # Fields for the packer: preflop AKo -> card 155, street 0, no raises,
    # pot 1.5 BB, live=4 opponents, after=4 (SB, BB, and two behind).
    packed = sixmax.pack_abstract_key(155, 0, [0, 0, 0, 0], 1.5, 4, 4)
    assert packed == s.infoset_key()
