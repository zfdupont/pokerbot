#!/usr/bin/env python3
"""
Evaluate our CFR bot against baselines using OpenSpiel's universal_poker environment.

Two modes:
  --baseline random    : vs uniform-random opponent (sanity check / upper bound)
  --baseline cfr       : vs OpenSpiel's own CFR agent trained on the same game

Usage:
    uv run python scripts/eval_openspiel.py --hands 2000
    uv run python scripts/eval_openspiel.py --hands 500 --baseline cfr --cfr-iters 500
"""
import argparse
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pyspiel
from open_spiel.python import policy as ospiel_policy
from open_spiel.python.algorithms import cfr, exploitability

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr.abstraction import hand_to_bucket, board_to_bucket
from cfr.info_set import InfoSet, stack_bucket
from cfr.regret_table import RegretTable
from models.card import Card
from models.enums import Suit


# ── Game configuration ─────────────────────────────────────────────────────────
# 100BB stacks, blinds 5/10, heads-up NL, FCPA action abstraction.
# FCPA = Fold / Call(check) / Pot-size bet / All-in
GAME_STR = (
    "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
    "blind=5 10,stack=1000 1000,firstPlayer=2 1 1 1,"
    "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
    "bettingAbstraction=fcpa)"
)

# OpenSpiel action IDs for FCPA universal_poker
ACTION_FOLD  = 0
ACTION_CALL  = 1
ACTION_BET   = 2   # pot-size raise
ACTION_ALLIN = 3

BIG_BLIND  = 10
SUIT_MAP   = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
RANK_MAP   = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

_INFO_RE = re.compile(
    r"\[Round (\d+)\].*?\[Pot: (\d+)\].*?\[Money: (\d+) (\d+)\]"
    r".*?\[Private: ([^\]]*)\].*?\[Public: ([^\]]*)\].*?\[Sequences: ([^\]]*)\]"
)


def _parse_cards(s: str) -> list[Card]:
    """'Td2d' → [Card(10, DIAMONDS), Card(2, DIAMONDS)]"""
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
    """Count raises per street from the Sequences field."""
    parts = sequences.split("|")
    counts = [0, 0, 0, 0]
    for i, part in enumerate(parts[:4]):
        counts[i] = min(part.count("r"), 2)
    return counts


def _is_check_action(legal: list[int], street: int, sequences: str) -> bool:
    """Return True if OpenSpiel action 1 (call/check) means 'check' in our vocab.

    Rules:
    - No fold + postflop  → always a free check
    - No fold + preflop   → SB must match BB, so it's a 'call'
    - Fold available      → a raise is outstanding iff current-street sequence ends
                            with an amount digit, otherwise BB can check after a limp
    """
    if ACTION_FOLD not in legal:
        return street > 0   # postflop free-check vs preflop SB call

    # Fold is available — distinguish "call facing a raise" vs "BB check after limp"
    parts = sequences.split("|")
    cur = parts[street] if street < len(parts) else ""
    has_open_raise = bool(re.search(r"r\d+$", cur))
    return not has_open_raise


def _parse_info_state(info_str: str) -> dict | None:
    m = _INFO_RE.search(info_str)
    if not m:
        return None
    street, pot, m0, m1, priv, pub, seqs = m.groups()
    street    = int(street)
    hole      = _parse_cards(priv)
    community = _parse_cards(pub)
    stacks    = (int(m0), int(m1))
    raises    = _count_raises(seqs)
    return {
        "street":            street,
        "pot":               int(pot),
        "stacks":            (int(m0), int(m1)),
        "hole_cards":        hole,
        "community_cards":   community,
        "raises_per_street": raises,
        "sequences":         seqs,
    }


class CFRBotPolicy(ospiel_policy.Policy):
    """
    Wraps our RegretTable as an OpenSpiel Policy.

    For each player decision node, parses the information state string,
    builds our InfoSet, and returns action probabilities mapped to OpenSpiel
    FCPA action IDs.
    """

    def __init__(self, game, table: RegretTable, player_id: int):
        super().__init__(game, [player_id])
        self.table     = table
        self.player_id = player_id

    def action_probabilities(self, state, player_id=None):
        legal = state.legal_actions()
        info_str = state.information_state_string(self.player_id)
        parsed   = _parse_info_state(info_str)

        if parsed is None or not parsed["hole_cards"]:
            # Fallback to uniform if we can't parse
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        street     = parsed["street"]
        hole       = parsed["hole_cards"]
        community  = parsed["community_cards"]
        stacks     = parsed["stacks"]
        raises     = parsed["raises_per_street"]
        sequences  = parsed["sequences"]

        my_stack_bb  = stacks[self.player_id] / BIG_BLIND
        hbucket      = hand_to_bucket(hole, community, street)
        bbucket      = board_to_bucket(community, street)

        infoset = InfoSet(
            player=0,
            hand_bucket=hbucket,
            street=street,
            board_bucket=bbucket,
            betting_history=tuple(raises),
            stack_bucket=stack_bucket(my_stack_bb),
        )

        # Abstract actions relevant to legal OpenSpiel actions.
        # OpenSpiel action 1 is check/call — map to correct abstract vocab.
        is_check = _is_check_action(legal, street, sequences)
        abstract_legal: list[str] = []
        if ACTION_FOLD  in legal: abstract_legal.append("fold")
        if ACTION_CALL  in legal:
            abstract_legal.append("check" if is_check else "call")
        if ACTION_BET   in legal: abstract_legal.extend(["b0.5", "b1.0"])
        if ACTION_ALLIN in legal: abstract_legal.append("allin")

        probs = self.table.get_average_strategy(infoset, abstract_legal)
        amap  = dict(zip(abstract_legal, probs))

        result: dict[int, float] = {}
        if ACTION_FOLD  in legal:
            result[ACTION_FOLD]  = amap.get("fold", 0.0)
        if ACTION_CALL  in legal:
            result[ACTION_CALL]  = amap.get("check" if is_check else "call", 0.0)
        if ACTION_BET   in legal:
            # Merge both bet sizes into pot-size bet
            result[ACTION_BET]   = amap.get("b0.5", 0.0) + amap.get("b1.0", 0.0)
        if ACTION_ALLIN in legal:
            result[ACTION_ALLIN] = amap.get("allin", 0.0)

        # Renormalize (merging introduces floating-point drift)
        total = sum(result.values())
        if total > 0:
            result = {a: p / total for a, p in result.items()}
        else:
            p = 1.0 / len(legal)
            result = {a: p for a in legal}

        return result


def _sample_action(probs: dict[int, float]) -> int:
    actions = list(probs.keys())
    weights = [probs[a] for a in actions]
    return np.random.choice(actions, p=np.array(weights) / sum(weights))


def simulate_hands(
    game,
    policy0: ospiel_policy.Policy,
    policy1: ospiel_policy.Policy,
    num_hands: int,
) -> tuple[float, float]:
    """
    Run num_hands heads-up games, alternating who is player 0.
    Returns (P0_total_return, P1_total_return) in chips.
    """
    total = [0.0, 0.0]
    for h in range(num_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, wts = zip(*outcomes)
                a = np.random.choice(acts, p=np.array(wts, dtype=float) / sum(wts))
                state.apply_action(int(a))
            else:
                cp = state.current_player()
                policy = policy0 if cp == 0 else policy1
                probs = policy.action_probabilities(state, cp)
                a = _sample_action(probs)
                state.apply_action(a)

        returns = state.returns()
        total[0] += returns[0]
        total[1] += returns[1]

    return tuple(total)


def _latest_checkpoint(directory="cfr/checkpoints"):
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return None
    return os.path.join(directory, sorted(files)[-1]) if files else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFR bot via OpenSpiel")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hands",      type=int, default=2000)
    parser.add_argument("--baseline",   choices=["random", "cfr"], default="random")
    parser.add_argument("--cfr-iters",  type=int, default=300,
                        help="OpenSpiel CFR iterations when --baseline cfr")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint()
    if not checkpoint or not os.path.exists(checkpoint):
        print("Error: no checkpoint found.")
        sys.exit(1)

    print(f"Loading {checkpoint} ...")
    table = RegretTable()
    table.load(checkpoint)

    game = pyspiel.load_game(GAME_STR)
    print(f"Game: {GAME_STR[:60]}...")
    print(f"Actions per node: 4 (fold/call/pot/allin)")
    print()

    bot_policy    = CFRBotPolicy(game, table, player_id=0)
    bot_policy_p1 = CFRBotPolicy(game, table, player_id=1)

    if args.baseline == "random":
        print("Baseline: UniformRandom")
        opponent = ospiel_policy.UniformRandomPolicy(game)

    else:
        print(f"Training OpenSpiel CFR for {args.cfr_iters} iterations ...")
        cfr_solver = cfr.CFRSolver(game)
        for i in range(args.cfr_iters):
            cfr_solver.evaluate_and_update_policy()
            if (i + 1) % 100 == 0:
                print(f"  iter {i+1}")
        opponent = cfr_solver.average_policy()
        print("OpenSpiel CFR training done.")

    print(f"Simulating {args.hands} hands ...")

    # Alternate seats every hand to cancel positional bias.
    # First half: our bot = P0, opponent = P1
    half = args.hands // 2
    r0_as0, _ = simulate_hands(game, bot_policy,    opponent, half)
    # Second half: our bot = P1, opponent = P0
    _, r1_as1  = simulate_hands(game, opponent, bot_policy_p1, args.hands - half)

    total_bot_return = r0_as0 + r1_as1
    hands_played     = args.hands
    bb_per_hand      = total_bot_return / BIG_BLIND / hands_played
    bb100            = bb_per_hand * 100

    print()
    print(f"=== Results ({args.baseline} baseline, {hands_played} hands) ===")
    print(f"  Bot total return : {total_bot_return:+.0f} chips")
    print(f"  Win rate         : {bb100:+.1f} BB/100")
    print()
    if args.baseline == "random":
        print("(vs random: a strong bot should win 30-60 BB/100;")
        print(" near 0 = bug, near 100+ = likely correct)")
    else:
        print("(vs OpenSpiel CFR: near 0 = strategies are close;")
        print(" large positive = our bot has an edge; negative = OpenSpiel wins)")


if __name__ == "__main__":
    main()
