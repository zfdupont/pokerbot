"""Live-engine adapter over neural_cfr.Strategy — a frozen HU eval opponent.

Mirrors scripts/openpoker_bot.py's neural loader and decode. Chips are rescaled
to the 100BB/1BB training frame by dividing by game_state.big_blind."""
import ctypes
import os
import subprocess
import sys

import numpy as np

from agents.base_agent import PokerAgent
from models.enums import Action, Suit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_ABSTRACT = ["fold", "check", "call", "b0.5", "b1.0", "allin"]


def _card_to_int(card):
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


def load_neural_strategy(checkpoint):
    lib_dir = os.path.join(_ROOT, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        p = os.path.join(lib_dir, lib)
        if os.path.exists(p):
            ctypes.CDLL(p)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//neural_cfr:neural_cfr",
                             "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = None
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            so_dir = os.path.join(_ROOT, os.path.dirname(line.split()[-1]))
            break
    if so_dir and so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    import neural_cfr
    return neural_cfr.Strategy(checkpoint)


class NeuralAgent(PokerAgent):
    def __init__(self, checkpoint_path):
        self._strategy = load_neural_strategy(checkpoint_path)

    def get_action(self, player, game_state):
        bb = game_state.big_blind
        n = len(game_state.players)
        hero = game_state.players.index(player)
        # HU position: 0=SB/button, 1=BB. Button is SB heads-up.
        position = 0 if hero == game_state.button_pos else 1
        street = game_state.betting_round

        to_call_chips = game_state.current_bet - player.current_bet
        probs = self._strategy.get_action_probs(
            [_card_to_int(c) for c in player.hole_cards],
            [_card_to_int(c) for c in game_state.community_cards],
            street,
            game_state.pot / bb,
            player.stack / bb,
            to_call_chips / bb,
            list(game_state.raises_per_street),
            position,
            my_street_bet=player.current_bet / bb,
            opp_street_bet=(player.current_bet + to_call_chips) / bb,
        )

        facing = to_call_chips > 1e-9
        legal = []
        if facing:
            legal.append("fold")
        else:
            legal.append("check")
        if facing:
            legal.append("call")
        if player.stack > to_call_chips + 1e-9:
            legal.extend(["b0.5", "b1.0", "allin"])
        raw = np.array([probs.get(a, 0.0) for a in legal], dtype=float)
        if raw.sum() <= 0:
            raw = np.ones(len(legal))
        choice = legal[int(np.argmax(raw))]
        return self._translate(choice, game_state, player, to_call_chips)

    def _translate(self, choice, game_state, player, to_call_chips):
        if choice == "fold":
            return Action.FOLD, None
        if choice == "check":
            return Action.CHECK, None
        if choice == "call":
            return Action.CALL, None
        # game/poker.py:139 treats Action.BET amount as raise-TO (total street
        # commitment). All-in raise-to = current_bet + stack; a pot-fraction bet
        # raise-to = current_bet + to_call + size * (pot after the call).
        max_to = int(round(player.current_bet + player.stack))  # all-in raise-to
        if choice == "allin":
            return Action.BET, max_to
        size = float(choice[1:])  # 0.5 or 1.0 pot
        eff_pot = game_state.pot + to_call_chips
        raise_to = int(round(player.current_bet + to_call_chips + size * eff_pot))
        return Action.BET, max(0, min(raise_to, max_to))
