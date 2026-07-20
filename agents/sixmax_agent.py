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
import random
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_sixmax_extension():
    """Force-load the sixmax C++ extension as sys.modules['sixmax'].

    The repo-root `sixmax/` directory is a namespace package that shadows the
    Buck2-built `sixmax.so`, so a bare `import sixmax` from a plain script gets
    the Python package (no C++ symbols). Under pytest the conftests already
    force-load the .so, so this is a no-op there (guarded on a C++-only
    attribute); for scripts (eval_hu_sanity, openpoker_bot) it builds and loads
    the extension the same way train_sixmax.py does."""
    mod = sys.modules.get("sixmax")
    if mod is not None and hasattr(mod, "SizeUnit"):
        return  # real extension already loaded (e.g. by a test conftest)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = None
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            so_dir = os.path.join(_ROOT, os.path.dirname(line.split()[-1]))
            break
    if so_dir is None:
        raise RuntimeError("Could not locate sixmax.so in buck2 output")
    so_path = os.path.join(so_dir, "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = m  # register before exec so circular refs resolve
    spec.loader.exec_module(m)


_ensure_sixmax_extension()
import sixmax  # noqa: E402  (must follow _ensure_sixmax_extension)

from agents.base_agent import PokerAgent  # noqa: E402
from models.enums import Action, Suit  # noqa: E402


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


_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_DEFAULT_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_RAISE_CAP = 3  # the sixmax abstraction's per-street raise cap (2 bits)


def _card_to_int(card):
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


class SixmaxAgent(PokerAgent):
    """Live-engine adapter over SixmaxDeployStrategy. Computes the legal mask,
    (live, after), and BB rescale from GameState; translates the blueprint's
    raise-to (BB) into an engine (Action, chips).

    game_state.raises_per_street is now faithful (Step 1 uncapped the engine),
    so we clamp it to the abstraction's cap (3) here at consumption — the same
    pattern neural_cfr uses (it clamps to 2). No observer plumbing needed."""

    def __init__(self, checkpoint_path, config_toml=_DEFAULT_TOML):
        self._deploy = SixmaxDeployStrategy.load(checkpoint_path, config_toml)
        self._rng = random.Random()

    def get_action(self, player, game_state):
        bb = game_state.big_blind
        players = game_state.players
        n = len(players)
        hero = players.index(player)
        street = game_state.betting_round

        hole = [_card_to_int(c) for c in player.hole_cards]
        board = [_card_to_int(c) for c in game_state.community_cards]

        folded = [not p.is_active for p in players]
        all_in = [p.is_all_in for p in players]
        live, after = canonical_live_after(n, game_state.button_pos, hero,
                                           folded, all_in, street)

        to_call_chips = game_state.current_bet - player.current_bet
        pot_bb = game_state.pot / bb
        current_bet_bb = game_state.current_bet / bb
        to_call_bb = to_call_chips / bb
        stack_bb = (player.current_bet + player.stack) / bb  # all-in target

        legal = self._legal_mask(game_state, player, to_call_chips,
                                 current_bet_bb, pot_bb, to_call_bb, stack_bb)

        raises = [min(r, _RAISE_CAP) for r in game_state.raises_per_street]
        idx, raise_to_bb = self._deploy.decide(
            hole=hole, board=board, street=street,
            raises_per_street=raises, pot_bb=pot_bb,
            current_bet_bb=current_bet_bb, to_call_bb=to_call_bb,
            stack_bb=stack_bb, live=live, after=after, legal=legal,
            rng=self._rng)
        return self._translate(idx, to_call_chips, raise_to_bb, bb, player)

    def _legal_mask(self, game_state, player, to_call_chips, current_bet_bb,
                    pot_bb, to_call_bb, stack_bb):
        """Mirror EngineGameState::legal_mask on live state. Bet entries are
        legal when their vocab target lands in [min_raise, all_in) BB."""
        vocab = self._deploy._vocab
        facing = to_call_chips > 1e-9
        can_raise = player.stack > to_call_chips + 1e-9
        street = game_state.betting_round
        unopened_preflop = street == 0 and current_bet_bb <= 1.0 + 1e-9
        min_raise_to_bb = (game_state.current_bet + max(
            game_state.big_blind, to_call_chips)) / game_state.big_blind
        ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                to_call=to_call_bb, stack=stack_bb)
        mask = [0] * vocab.size()
        for i in range(vocab.size()):
            a = vocab.at(i)
            if a.type == sixmax.ActionType.Fold:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Check:
                mask[i] = 0 if facing else 1
            elif a.type == sixmax.ActionType.Call:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Bet:
                bb_unit = a.unit == sixmax.SizeUnit.BB
                if not can_raise or bb_unit != unopened_preflop:
                    continue
                t = vocab.target_bb(i, ctx)
                mask[i] = 1 if (t >= min_raise_to_bb - 1e-9
                                and t < stack_bb - 1e-9) else 0
            elif a.type == sixmax.ActionType.AllIn:
                mask[i] = 1 if can_raise else 0
        return mask

    def _translate(self, idx, to_call_chips, raise_to_bb, bb, player):
        a = self._deploy._vocab.at(idx)
        if a.type == sixmax.ActionType.Fold:
            return Action.FOLD, None
        if a.type == sixmax.ActionType.Check:
            return Action.CHECK, None
        if a.type == sixmax.ActionType.Call:
            return Action.CALL, None
        # Bet / AllIn: game/poker.py:139 treats Action.BET amount as raise-TO
        # (the total street commitment; additional = amount - player.current_bet).
        # This matches CFRAgent._translate, whose all-in returns
        # stack + current_bet and whose bets cap at stack + current_bet.
        max_to = int(round(player.current_bet + player.stack))  # all-in raise-to
        if a.type == sixmax.ActionType.AllIn:
            return Action.BET, max_to
        raise_to = int(round(raise_to_bb * bb))
        return Action.BET, max(0, min(raise_to, max_to))
