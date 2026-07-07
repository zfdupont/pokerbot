#!/usr/bin/env python3
"""
eval_neural_vs_tabular.py — head-to-head: neural CFR vs tabular MCCFR.

Runs both bots on the same OpenSpiel universal_poker game (100BB HU NL, FCPA)
and reports the neural bot's BB/100 win rate against the tabular bot.

Usage:
    uv run python scripts/eval_neural_vs_tabular.py \
        [--neural-checkpoint neural_cfr/checkpoints/checkpoint.pt] \
        [--tabular-checkpoint cfr/checkpoints/checkpoint_09040000.pkl] \
        [--hands 2000]
"""
import argparse
import contextlib
import ctypes
import io
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Pre-load libtorch dylibs (macOS SIP strips DYLD_LIBRARY_PATH).
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
            return os.path.join(_REPO_ROOT, os.path.dirname(line.split()[-1]))
    raise RuntimeError("Could not locate neural_cfr.so in buck2 output")


_so_dir = _discover_so_dir()
if _so_dir not in sys.path:
    sys.path.insert(0, _so_dir)

import neural_cfr  # noqa: E402

# ---------------------------------------------------------------------------
# Shared imports
# ---------------------------------------------------------------------------
sys.path.insert(0, _REPO_ROOT)

import re
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import pyspiel
    from open_spiel.python import policy as ospiel_policy

from cfr.abstraction import hand_to_bucket, board_to_bucket
from cfr.info_set import InfoSet, stack_bucket
from cfr.regret_table import RegretTable
from models.card import Card
from models.enums import Suit

# ---------------------------------------------------------------------------
# Game config (identical to both eval scripts)
# ---------------------------------------------------------------------------
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
OPENSPIEL_STARTING_STACK = 1000.0
TRAINING_STARTING_STACK  = 100.0
_CHIP_SCALE = OPENSPIEL_STARTING_STACK / TRAINING_STARTING_STACK  # 10.0

SUIT_MAP = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}

_INFO_RE = re.compile(
    r"\[Round (\d+)\].*?\[Pot: (\d+)\].*?\[Money: (\d+) (\d+)\]"
    r".*?\[Private: ([^\]]*)\].*?\[Public: ([^\]]*)\].*?\[Sequences: ([^\]]*)\]"
)


def _parse_cards_tabular(s: str) -> list[Card]:
    s = s.strip()
    if not s:
        return []
    return [Card(RANK_MAP[s[i]], SUIT_MAP[s[i + 1]]) for i in range(0, len(s), 2)]


def _card_to_int(card) -> int:
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


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
    return not bool(re.search(r"r\d+$", cur))


def _parse_info_state(info_str: str) -> dict | None:
    m = _INFO_RE.search(info_str)
    if not m:
        return None
    street, pot, m0, m1, priv, pub, seqs = m.groups()
    hole = _parse_cards_tabular(priv)
    community = _parse_cards_tabular(pub)
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
# Policies
# ---------------------------------------------------------------------------
class NeuralCFRPolicy(ospiel_policy.Policy):
    def __init__(self, game, strategy, player_id: int):
        super().__init__(game, [player_id])
        self.strategy  = strategy
        self.player_id = player_id

    def action_probabilities(self, state, player_id=None):
        pid   = player_id if player_id is not None else self.player_id
        legal = state.legal_actions()
        parsed = _parse_info_state(state.information_state_string(pid))

        if parsed is None or not parsed["hole_cards"]:
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        is_check = _is_check_action(legal, parsed["street"], parsed["sequences"])
        stack_my = parsed["stacks"][pid]

        probs = self.strategy.get_action_probs(
            hole_cards        = [_card_to_int(c) for c in parsed["hole_cards"]],
            board_cards       = [_card_to_int(c) for c in parsed["community_cards"]],
            street            = parsed["street"],
            pot               = float(parsed["pot"])  / _CHIP_SCALE,
            stack             = float(stack_my)       / _CHIP_SCALE,
            to_call           = 0.0 if is_check else float(stack_my - parsed["stacks"][1 - pid]) / _CHIP_SCALE,
            raises_per_street = parsed["raises_per_street"],
            position          = pid,
        )

        result: dict[int, float] = {}
        if ACTION_FOLD  in legal: result[ACTION_FOLD]  = probs.get("fold", 0.0)
        if ACTION_CALL  in legal: result[ACTION_CALL]  = probs.get("check" if is_check else "call", 0.0)
        if ACTION_BET   in legal: result[ACTION_BET]   = probs.get("b0.5", 0.0) + probs.get("b1.0", 0.0)
        if ACTION_ALLIN in legal: result[ACTION_ALLIN] = probs.get("allin", 0.0)

        total = sum(result.values())
        if total > 0:
            return {a: v / total for a, v in result.items()}
        p = 1.0 / len(legal)
        return {a: p for a in legal}


class TabularCFRPolicy(ospiel_policy.Policy):
    def __init__(self, game, table: RegretTable, player_id: int):
        super().__init__(game, [player_id])
        self.table     = table
        self.player_id = player_id

    def action_probabilities(self, state, player_id=None):
        legal  = state.legal_actions()
        parsed = _parse_info_state(state.information_state_string(self.player_id))

        if parsed is None or not parsed["hole_cards"]:
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        street    = parsed["street"]
        is_check  = _is_check_action(legal, street, parsed["sequences"])
        my_stack_bb = parsed["stacks"][self.player_id] / BIG_BLIND

        infoset = InfoSet(
            player=0,
            hand_bucket=hand_to_bucket(parsed["hole_cards"], parsed["community_cards"], street),
            street=street,
            board_bucket=board_to_bucket(parsed["community_cards"], street),
            betting_history=tuple(parsed["raises_per_street"]),
            stack_bucket=stack_bucket(my_stack_bb),
        )

        abstract_legal: list[str] = []
        if ACTION_FOLD  in legal: abstract_legal.append("fold")
        if ACTION_CALL  in legal: abstract_legal.append("check" if is_check else "call")
        if ACTION_BET   in legal: abstract_legal.extend(["b0.5", "b1.0"])
        if ACTION_ALLIN in legal: abstract_legal.append("allin")

        probs = self.table.get_average_strategy(infoset, abstract_legal)
        amap  = dict(zip(abstract_legal, probs))

        result: dict[int, float] = {}
        if ACTION_FOLD  in legal: result[ACTION_FOLD]  = amap.get("fold", 0.0)
        if ACTION_CALL  in legal: result[ACTION_CALL]  = amap.get("check" if is_check else "call", 0.0)
        if ACTION_BET   in legal: result[ACTION_BET]   = amap.get("b0.5", 0.0) + amap.get("b1.0", 0.0)
        if ACTION_ALLIN in legal: result[ACTION_ALLIN] = amap.get("allin", 0.0)

        total = sum(result.values())
        if total > 0:
            return {a: v / total for a, v in result.items()}
        p = 1.0 / len(legal)
        return {a: p for a in legal}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _sample_action(probs: dict[int, float]) -> int:
    actions = list(probs.keys())
    weights = np.array([probs[a] for a in actions], dtype=float)
    return np.random.choice(actions, p=weights / weights.sum())


def simulate_hands(game, policy0, policy1, num_hands: int) -> tuple[float, float]:
    total = [0.0, 0.0]
    for _ in range(num_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                acts, wts = zip(*state.chance_outcomes())
                wts = np.array(wts, dtype=float)
                state.apply_action(int(np.random.choice(acts, p=wts / wts.sum())))
            else:
                cp = state.current_player()
                policy = policy0 if cp == 0 else policy1
                state.apply_action(_sample_action(policy.action_probabilities(state, cp)))
        r = state.returns()
        total[0] += r[0]
        total[1] += r[1]
    return tuple(total)


def _latest_pkl(directory="cfr/checkpoints") -> str | None:
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return None
    return os.path.join(directory, sorted(files)[-1]) if files else None


def main():
    parser = argparse.ArgumentParser(description="Neural CFR vs tabular MCCFR head-to-head")
    parser.add_argument("--neural-checkpoint",   default="neural_cfr/checkpoints/checkpoint.pt")
    parser.add_argument("--tabular-checkpoint",  default=None)
    parser.add_argument("--hands",               type=int, default=2000)
    args = parser.parse_args()

    tabular_ckpt = args.tabular_checkpoint or _latest_pkl()
    if not tabular_ckpt or not os.path.exists(tabular_ckpt):
        print("Error: no tabular CFR checkpoint found.")
        sys.exit(1)

    print(f"Neural  checkpoint : {args.neural_checkpoint}")
    print(f"Tabular checkpoint : {tabular_ckpt}")
    print(f"Hands              : {args.hands}")
    print()

    strategy = neural_cfr.Strategy(args.neural_checkpoint)
    table = RegretTable()
    table.load(tabular_ckpt)

    game = pyspiel.load_game(GAME_STR)

    neural_p0  = NeuralCFRPolicy(game, strategy, player_id=0)
    neural_p1  = NeuralCFRPolicy(game, strategy, player_id=1)
    tabular_p0 = TabularCFRPolicy(game, table,    player_id=0)
    tabular_p1 = TabularCFRPolicy(game, table,    player_id=1)

    half = args.hands // 2
    print(f"Running {half} hands (neural=P0, tabular=P1) ...")
    n0, _ = simulate_hands(game, neural_p0, tabular_p1, half)
    print(f"Running {args.hands - half} hands (tabular=P0, neural=P1) ...")
    _, n1  = simulate_hands(game, tabular_p0, neural_p1, args.hands - half)

    total_neural = n0 + n1
    bb100 = total_neural / BIG_BLIND / args.hands * 100

    print()
    print(f"=== Neural CFR vs Tabular MCCFR ({args.hands} hands) ===")
    print(f"  Neural total return : {total_neural:+.0f} chips")
    print(f"  Neural win rate     : {bb100:+.1f} BB/100")
    print()
    print("  Reference: tabular MCCFR exploitability = 574 mbb/h @ 9.04M iterations")
    print("  Near 0 BB/100 = strategies agree; positive = neural has edge; negative = tabular wins")


if __name__ == "__main__":
    main()
