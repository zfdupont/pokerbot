"""Bit-packed abstraction infoset keys: field packing, and the table-size
invariance that motivated the (live_opps, after) encoding — MP 6-handed and
UTG 5-handed are literally the same infoset."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
FOLD, CHECK, CALL, OPEN25, B33 = 0, 1, 2, 3, 6

def card(rank, suit):
    return (rank - 2) * 4 + suit

# key field extractors (bit layout pinned in the plan's Global Constraints)
def f_card(k):   return k & 0xFF
def f_street(k): return (k >> 8) & 3
def f_raises(k, st): return (k >> (10 + 2 * st)) & 3
def f_potb(k):   return (k >> 18) & 3
def f_live(k):   return (k >> 20) & 7
def f_after(k):  return (k >> 23) & 7

AKo = [card(14, 3), card(13, 1)]  # class 11*13+12 = 155


def _state(n, hero_seat, deck_override):
    cfg = sixmax.EngineConfig(num_players=n)
    deck = list(range(2 * n + 5))
    # place hero's hole cards at deck[2*seat], deck[2*seat+1]; keep the rest
    # distinct by remapping any collision onto high spare codes
    spares = iter(range(40, 52))
    taken = set(deck_override)
    deck = [c if c not in taken else next(spares) for c in deck]
    deck[2 * hero_seat] = deck_override[0]
    deck[2 * hero_seat + 1] = deck_override[1]
    return sixmax.EngineGameState(cfg, 0, deck, VOCAB, [], abstraction=ABS)


def test_mp_6handed_equals_utg_5handed():
    # 6-handed, button 0: preflop order UTG=3, HJ=4, CO=5, BTN=0, SB=1, BB=2.
    s6 = _state(6, hero_seat=4, deck_override=AKo)
    s6.apply(FOLD)                      # UTG (seat 3) folds -> HJ (seat 4) acts
    # 5-handed, button 0: UTG=3 acts first.
    s5 = _state(5, hero_seat=3, deck_override=AKo)
    k6, k5 = s6.infoset_key(), s5.infoset_key()
    assert k6 == k5                     # same spot, different table size
    assert f_card(k6) == 155 and f_street(k6) == 0
    assert f_live(k6) == 4 and f_after(k6) == 4
    assert f_potb(k6) == 0              # pot 1.5 BB


def test_key_fields_track_raises_pot_and_position():
    s = _state(2, hero_seat=0, deck_override=AKo)
    k = s.infoset_key()                 # HU preflop: SB/BTN (seat 0) first
    assert f_live(k) == 1 and f_after(k) == 1 and f_raises(k, 0) == 0
    s.apply(OPEN25)                     # raise count street 0 -> 1
    k = s.infoset_key()                 # BB now acting
    assert f_raises(k, 0) == 1 and f_after(k) == 0
    s.apply(CALL)
    s.apply(CHECK)                      # flop dealt; HU postflop BB first
    k = s.infoset_key()
    assert f_street(k) == 1
    assert f_raises(k, 0) == 1 and f_raises(k, 1) == 0
    assert 0 <= f_card(k) < 10          # flop equity bucket


def test_raise_count_caps_at_three():
    s = _state(2, hero_seat=0, deck_override=AKo)
    s.apply(OPEN25)
    for _ in range(4):                  # raise war beyond the cap
        mask = s.legal_mask()
        raise_idx = next(i for i in range(3, 9) if mask[i])
        s.apply(raise_idx)
        if s.is_terminal():
            break
    assert f_raises(s.infoset_key(), 0) == 3 if not s.is_terminal() else True


def test_naive_keyer_still_default_without_abstraction():
    cfg = sixmax.EngineConfig(num_players=2)
    s = sixmax.EngineGameState(cfg, 0, list(range(52)), VOCAB, [])
    t = sixmax.EngineGameState(cfg, 0, list(range(52)), VOCAB, [],
                               abstraction=ABS)
    assert s.infoset_key() != t.infoset_key()  # naive vs packed differ
    # abstract_key with an explicit abstraction works on a plain state too
    assert s.abstract_key(ABS) == t.infoset_key()
