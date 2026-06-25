"""
Play N hands between poker agents and print a summary.

Usage:
    uv run python scripts/run_simulation.py [--hands 100] [--stack 1000] [--agent simple|position|cfr]
"""
import sys
import os
import argparse
import glob
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.player import Player
from agents.simple_agent import SimpleAgent
from agents.position_agent import PositionBasedAgent
from agents.hand_strength_agent import HandStrengthAgent
from game.poker import PokerGame
from game.pot_manager import PotManager
from util.observer import GameObserver


class StatsObserver(GameObserver):
    def __init__(self, players: List[Player]):
        self.hands_won = defaultdict(int)
        self.chips_won = defaultdict(int)
        self._start_stacks = {p.name: p.stack for p in players}

    def on_hand_complete(self, winners: List[Tuple[Player, int]], pot: int) -> None:
        split = pot // len(winners)
        for player, _ in winners:
            self.hands_won[player.name] += 1
            self.chips_won[player.name] += split

    def report(self, players: List[Player], num_hands: int, big_blind: int) -> None:
        print(f"\n{'='*55}")
        print(f"  Simulation: {num_hands} hands  (BB=${big_blind})")
        print(f"{'='*55}")
        print(f"{'Player':<20} {'Type':<14} {'Net':>8} {'BB/100':>8} {'Stack':>7}")
        print(f"{'-'*55}")

        for p in players:
            agent_type = type(p.agent).__name__.replace("Agent", "")
            net = p.stack - self._start_stacks[p.name]
            bb_per_100 = (net / big_blind) / num_hands * 100
            net_str = f"+{net}" if net >= 0 else str(net)
            bb_str = f"{bb_per_100:+.1f}"
            print(f"{p.name:<20} {agent_type:<14} {net_str:>8} {bb_str:>8} {p.stack:>7}")

        print(f"{'='*55}")

        # Per agent-type summary
        type_net = defaultdict(int)
        type_count = defaultdict(int)
        for p in players:
            t = type(p.agent).__name__.replace("Agent", "")
            type_net[t] += p.stack - self._start_stacks[p.name]
            type_count[t] += 1

        print(f"\n  By agent type:")
        for t in sorted(type_net):
            net = type_net[t]
            bb_per_100 = (net / big_blind) / num_hands * 100
            net_str = f"+{net}" if net >= 0 else str(net)
            count = type_count[t]
            print(f"    {t:<16} BB/100: {bb_per_100:+.1f}   net: {net_str}  ({count} players)")
        print()


def load_cfr_agent(checkpoint_dir: str = "cfr/checkpoints"):
    from agents.cfr_agent import CFRAgent
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}. Run train_cfr.py first.")
    return CFRAgent(strategy_path=files[-1])


def main():
    parser = argparse.ArgumentParser(description="Poker agent simulation")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--agent", choices=["simple", "position", "cfr"], default="simple")
    args = parser.parse_args()

    starting_stack = args.stack

    if args.agent == "cfr":
        try:
            cfr = load_cfr_agent()
            players = [
                Player("CFR 1", starting_stack, cfr),
                Player("Simple 1", starting_stack, SimpleAgent()),
                Player("CFR 2", starting_stack, cfr),
                Player("HandStr 1", starting_stack, HandStrengthAgent()),
            ]
        except FileNotFoundError as e:
            print(e)
            return
    else:
        players = [
            Player("Simple 1", starting_stack, SimpleAgent()),
            Player("Position 1", starting_stack, PositionBasedAgent()),
            Player("HandStr 1", starting_stack, HandStrengthAgent()),
            Player("HandStr 2", starting_stack, HandStrengthAgent()),
        ]

    stats = StatsObserver(players)
    game = PokerGame(players, small_blind=5, observers=[stats])

    hands_played = 0
    print(f"Running {args.hands} hands... ", end="", flush=True)
    for i in range(args.hands):
        active_count = sum(1 for p in players if p.stack > 0)
        if active_count < 2:
            print(f"\nOnly {active_count} solvent player(s) left after hand {i}. Stopping early.")
            break
        game.play_hand()
        hands_played += 1
        if hands_played % 10 == 0:
            print(f"{hands_played}.", end="", flush=True)

    print(" done.")
    stats.report(players, hands_played, big_blind=10)


if __name__ == "__main__":
    main()
