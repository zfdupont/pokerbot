#!/usr/bin/env python3
"""
eval_openspiel_neural.py — eval_openspiel.py adapted for neural_cfr.Strategy.

Evaluates the neural CFR bot against baselines using OpenSpiel's
universal_poker environment (FCPA, 100BB HU NL).

Usage:
    uv run python scripts/eval_openspiel_neural.py \
        --checkpoint neural_cfr/checkpoints/checkpoint.pt \
        [--hands 2000] [--baseline random|cfr] [--cfr-iters 500]
"""
import argparse
import ctypes
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Pre-load libtorch dylibs and discover the Buck2 .so directory.
# DYLD_LIBRARY_PATH is stripped by macOS SIP, so we must do this in-process.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_REPO_ROOT, "third_party", "libtorch", "lib")
for _lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
    ctypes.CDLL(os.path.join(_LIB_DIR, _lib))


def _discover_so_dir() -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//neural_cfr:neural_cfr", "--show-output"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(_REPO_ROOT, os.path.dirname(rel_so))
    raise RuntimeError("Could not locate neural_cfr.so in buck2 output")


_so_dir = _discover_so_dir()
if _so_dir not in sys.path:
    sys.path.insert(0, _so_dir)

import neural_cfr  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers from eval_openspiel (unchanged game/parsing logic)
# ---------------------------------------------------------------------------
sys.path.insert(0, _REPO_ROOT)

import re
import numpy as np
import pyspiel
from open_spiel.python import policy as ospiel_policy

from models.enums import Suit

# Game configuration — identical to eval_openspiel.py
GAME_STR = (
    "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
    "blind=5 10,stack=1000 1000,firstPlayer=2 1 1 1,"
    "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
    "bettingAbstraction=fcpa)"
)

ACTION_FOLD  = 0
ACTION_CALL  = 1
ACTION_BET   = 2
ACTION_ALLIN = 3

BIG_BLIND = 10

SUIT_MAP = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# neural_cfr suit index: 0=clubs 1=diamonds 2=hearts 3=spades
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}

_INFO_RE = re.compile(
    r"\[Round (\d+)\].*?\[Pot: (\d+)\].*?\[Money: (\d+) (\d+)\]"
    r".*?\[Private: ([^\]]*)\].*?\[Public: ([^\]]*)\].*?\[Sequences: ([^\]]*)\]"
)


def _card_to_int(card) -> int:
    """Convert a models.card.Card to neural_cfr integer encoding (0-51)."""
    rank_idx = card.rank - 2          # rank 2→0, A→12
    suit_idx = _SUIT_TO_IDX[card.suit]
    return rank_idx * 4 + suit_idx


def _parse_cards(s: str):
    """'Td2d' → list of Card-like objects with .rank and .suit."""
    from models.card import Card
    s = s.strip()
    if not s:
        return []
    out = []
    for i in range(0, len(s), 2):
        r = RANK_MAP[s[i]]
        suit = SUIT_MAP[s[i + 1]]
        out.append(Card(r, suit))
    return out


def _count_raises(sequences: str) -> list[int]:
    parts = sequences.split("|")
    counts = [0, 0, 0, 0]
    for i, part in enumerate(parts[:4]):
        counts[i] = min(part.count("r"), 2)
    return counts


def _is_check_action(legal: list[int], street: int, sequences: str) -> bool:
    if ACTION_FOLD not in legal:
        return street > 0
    parts = sequences.split("|")
    cur = parts[street] if street < len(parts) else ""
    has_open_raise = bool(re.search(r"r\d+$", cur))
    return not has_open_raise


def _parse_info_state(info_str: str) -> dict | None:
    m = _INFO_RE.search(info_str)
    if not m:
        return None
    street, pot, m0, m1, priv, pub, seqs = m.groups()
    hole = _parse_cards(priv)
    community = _parse_cards(pub)
    raises = _count_raises(seqs)
    return {
        "street":            int(street),
        "pot":               int(pot),
        "stacks":            (int(m0), int(m1)),
        "hole_cards":        hole,
        "community_cards":   community,
        "raises_per_street": raises,
        "sequences":         seqs,
    }


# ---------------------------------------------------------------------------
# Neural policy
# ---------------------------------------------------------------------------
class NeuralCFRPolicy(ospiel_policy.Policy):
    """Wraps neural_cfr.Strategy as an OpenSpiel Policy."""

    def __init__(self, game, strategy: "neural_cfr.Strategy", player_id: int):
        super().__init__(game, [player_id])
        self.strategy  = strategy
        self.player_id = player_id

    def action_probabilities(self, state, player_id=None):
        pid    = player_id if player_id is not None else self.player_id
        legal  = state.legal_actions()
        info_str = state.information_state_string(pid)
        parsed = _parse_info_state(info_str)

        if parsed is None or not parsed["hole_cards"]:
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        street    = parsed["street"]
        hole      = parsed["hole_cards"]
        community = parsed["community_cards"]
        stacks    = parsed["stacks"]
        raises    = parsed["raises_per_street"]
        sequences = parsed["sequences"]
        pot       = parsed["pot"]

        # Convert Card objects to neural_cfr integer encoding
        hole_ints  = [_card_to_int(c) for c in hole]
        board_ints = [_card_to_int(c) for c in community]

        # Determine to_call (in chips, not BB)
        is_check = _is_check_action(legal, street, sequences)
        stack_my = stacks[pid]
        stack_op = stacks[1 - pid]
        to_call  = 0.0 if is_check else float(stack_op - stack_my) if stack_my < stack_op else 0.0

        probs = self.strategy.get_action_probs(
            hole_cards        = hole_ints,
            board_cards       = board_ints,
            street            = street,
            pot               = float(pot),
            stack             = float(stack_my),
            to_call           = to_call,
            raises_per_street = raises,
            position          = pid,
        )

        # Map abstract action names → OpenSpiel FCPA action IDs
        result: dict[int, float] = {}
        if ACTION_FOLD  in legal:
            result[ACTION_FOLD]  = probs.get("fold", 0.0)
        if ACTION_CALL  in legal:
            key = "check" if is_check else "call"
            result[ACTION_CALL]  = probs.get(key, 0.0)
        if ACTION_BET   in legal:
            result[ACTION_BET]   = probs.get("b0.5", 0.0) + probs.get("b1.0", 0.0)
        if ACTION_ALLIN in legal:
            result[ACTION_ALLIN] = probs.get("allin", 0.0)

        total = sum(result.values())
        if total > 0:
            result = {a: v / total for a, v in result.items()}
        else:
            p = 1.0 / len(legal)
            result = {a: p for a in legal}
        return result


# ---------------------------------------------------------------------------
# Simulation & main
# ---------------------------------------------------------------------------
def _sample_action(probs: dict[int, float]) -> int:
    actions = list(probs.keys())
    weights = [probs[a] for a in actions]
    return np.random.choice(actions, p=np.array(weights) / sum(weights))


def simulate_hands(game, policy0, policy1, num_hands: int) -> tuple[float, float]:
    total = [0.0, 0.0]
    for _ in range(num_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, wts = zip(*outcomes)
                a = np.random.choice(acts, p=np.array(wts, dtype=float) / sum(wts))
                state.apply_action(int(a))
            else:
                cp     = state.current_player()
                policy = policy0 if cp == 0 else policy1
                probs  = policy.action_probabilities(state, cp)
                state.apply_action(_sample_action(probs))
        returns = state.returns()
        total[0] += returns[0]
        total[1] += returns[1]
    return tuple(total)


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural CFR bot via OpenSpiel")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to neural_cfr checkpoint .pt file")
    parser.add_argument("--hands",      type=int, default=2000)
    parser.add_argument("--baseline",   choices=["random", "cfr"], default="random")
    parser.add_argument("--cfr-iters",  type=int, default=300)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    strategy = neural_cfr.Strategy(args.checkpoint)

    game = pyspiel.load_game(GAME_STR)
    print(f"Game: {GAME_STR[:60]}...")

    policy0 = NeuralCFRPolicy(game, strategy, player_id=0)
    policy1 = NeuralCFRPolicy(game, strategy, player_id=1)

    if args.baseline == "random":
        print("Baseline: UniformRandom")
        opponent = ospiel_policy.UniformRandomPolicy(game)
    else:
        from open_spiel.python.algorithms import cfr as ospiel_cfr
        print(f"Baseline: OpenSpiel CFR ({args.cfr_iters} iters)")
        cfr_solver = ospiel_cfr.CFRSolver(game)
        for _ in range(args.cfr_iters):
            cfr_solver.evaluate_and_update_policy()
        opponent = cfr_solver.average_policy()

    p0_ret, p1_ret = simulate_hands(game, policy0, opponent, args.hands // 2)
    p1_ret2, p0_ret2 = simulate_hands(game, opponent, policy1, args.hands // 2)
    total0 = p0_ret + p0_ret2
    total1 = p1_ret + p1_ret2

    bb100 = (total0 - total1) / 2 / (args.hands / 100) / BIG_BLIND
    print(f"\nNeural CFR win rate: {bb100:+.2f} BB/100 over {args.hands} hands")


if __name__ == "__main__":
    main()
