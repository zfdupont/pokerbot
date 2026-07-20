"""Deployment bridge for the six-max blueprint.

SixmaxDeployStrategy is host-agnostic: it takes public state in big-blind units
and returns a chosen vocab index plus a raise-to (also in BB). Hosts (the live
engine via SixmaxAgent, the openpoker connector via HandTracker) compute the
legal mask and the (live, after) counts, and translate BB back to table chips.

Key reconstruction routes through sixmax.pack_abstract_key so it is byte-
identical to the trainer's EngineGameState::abstract_key. Bet legality and
translation route through the C++ ActionVocab (target_bb), the single vocab
translation layer.
"""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vocab(config_toml, section):
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(config_toml, section)


def canonical_live_after(n, button, hero, folded, all_in, street):
    """(live, after) per the pinned canonical action order.

    Mirrors EngineGameState::abstract_key exactly. Positions are table indices
    0..n-1; folded/all_in are length-n bool lists indexed the same way."""
    start = (button if n == 2 else (button + 3) % n) if street == 0 \
        else (button + 1) % n

    def order(seat):
        return (seat - start + n) % n

    live = after = 0
    for s in range(n):
        if s == hero or folded[s]:
            continue
        live += 1
        if not all_in[s] and order(s) > order(hero):
            after += 1
    return live, after


class SixmaxDeployStrategy:
    def __init__(self, strategy, vocab):
        self._strategy = strategy
        self._vocab = vocab

    @classmethod
    def load(cls, path, config_toml, section="blueprint"):
        vocab = _load_vocab(config_toml, section)
        strategy = sixmax.BlueprintStrategy.load(path, vocab)
        return cls(strategy, vocab)

    @property
    def num_players(self):
        return self._strategy.num_players()

    def _card_bucket(self, hole, board, street):
        if street == 0:
            return sixmax.preflop_class(hole)
        return self._strategy.abstraction().bucket(hole, board)

    def decide(self, *, hole, board, street, raises_per_street, pot_bb,
               current_bet_bb, to_call_bb, stack_bb, live, after, legal, rng):
        card = self._card_bucket(hole, board, street)
        key = sixmax.pack_abstract_key(card, street, list(raises_per_street),
                                       pot_bb, live, after)
        probs = self._strategy.probs(key)  # [] if unseen

        legal_idx = [i for i, m in enumerate(legal) if m]
        weights = [probs[i] if i < len(probs) else 0.0 for i in legal_idx]
        total = sum(weights)
        if total <= 0.0:
            idx = rng.choice(legal_idx)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = legal_idx[-1]
            for i, w in zip(legal_idx, weights):
                acc += w
                if r <= acc:
                    idx = i
                    break

        action = self._vocab.at(idx)
        if action.type in (sixmax.ActionType.Bet, sixmax.ActionType.AllIn):
            ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                    to_call=to_call_bb, stack=stack_bb)
            raise_to = self._vocab.target_bb(idx, ctx)
        else:
            raise_to = 0.0
        return idx, raise_to
