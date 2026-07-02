#!/usr/bin/env python3
"""
Interactive heads-up poker CLI against a trained CFR bot.

Usage:
    uv run python scripts/play.py --checkpoint cfr/checkpoints/checkpoint_05040000.pkl
    uv run python scripts/play.py --checkpoint cfr/checkpoints/checkpoint_05040000.pkl --stack 1000 --big-blind 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import PokerAgent
from agents.cfr_agent import CFRAgent
from game.poker import PokerGame
from models.enums import Action
from models.player import Player
from util.observer import GameObserver


_RANK_NAMES = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}


def _cards(cards):
    if not cards:
        return "—"
    return "  ".join(
        f"{_RANK_NAMES.get(c.rank, str(c.rank))}{c.suit.value}" for c in cards
    )


def _action_label(action, amount):
    if action == Action.FOLD:
        return "folds"
    if action == Action.CHECK:
        return "checks"
    if action == Action.CALL:
        return f"calls {amount}"
    if action in (Action.BET, Action.RAISE):
        return f"raises to {amount}"
    return str(action)


class HumanAgent(PokerAgent):
    def get_action(self, player, state):
        to_call = state.current_bet - player.current_bet
        pot = state.pot
        stack = player.stack

        bot = next(p for p in state.players if p is not player)

        print()
        print(f"  Pot: {pot}   Your stack: {stack}   Bot stack: {bot.stack}")
        if state.community_cards:
            print(f"  Board:  {_cards(state.community_cards)}")
        print(f"  Hand:   {_cards(player.hole_cards)}")
        print()

        options = {}
        if to_call > 0:
            options["f"] = (Action.FOLD, None)
            call_amt = min(to_call, stack)
            options["c"] = (Action.CALL, int(call_amt))
        else:
            options["c"] = (Action.CHECK, None)

        remaining_after_call = stack - min(to_call, stack)
        if remaining_after_call > 0:
            # minimum legal raise: match the existing raise size, at least 1 BB
            min_total = int(state.current_bet + max(state.current_bet, state.big_blind))
            options["b"] = ("bet", min_total)

        options["a"] = (Action.BET, int(stack + player.current_bet))

        parts = []
        if "f" in options:
            parts.append("[f] fold")
        if "c" in options:
            label = "call " + str(options["c"][1]) if to_call > 0 else "check"
            parts.append(f"[c] {label}")
        if "b" in options:
            parts.append(f"[b] bet <amount>  (min {options['b'][1]})")
        parts.append(f"[a] allin {stack}")
        parts.append("[q] quit")
        print("  " + "   ".join(parts))

        while True:
            try:
                raw = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(0)

            if raw == "q":
                print("\n  Thanks for playing!")
                sys.exit(0)

            if raw in ("f", "fold"):
                if "f" in options:
                    return Action.FOLD, None
                print("  Nothing to fold against — you can check for free.")
                continue

            if raw in ("c", "check", "call"):
                act, amt = options["c"]
                return act, amt

            if raw in ("a", "allin"):
                return options["a"]

            if raw.startswith("b"):
                if "b" not in options:
                    print("  Can't bet — you'd need to go all-in. Use [a].")
                    continue
                parts_raw = raw.split()
                if len(parts_raw) < 2:
                    print(f"  Usage: b <amount>   (min {options['b'][1]})")
                    continue
                try:
                    amount = int(parts_raw[1])
                except ValueError:
                    print("  Not a valid number.")
                    continue
                min_total = options["b"][1]
                if amount < min_total:
                    print(f"  Minimum bet is {min_total}.")
                    continue
                # amount is treated as total chips committed (player.current_bet + more)
                total = min(amount, int(stack + player.current_bet))
                return Action.BET, total

            print("  Unknown action. Options: f, c, b <amount>, a, q")


class DisplayObserver(GameObserver):
    def __init__(self, human_name, bot_name):
        self.human_name = human_name
        self.bot_name = bot_name
        self.hand_num = 0

    def on_hand_start(self, players, button_pos):
        self.hand_num += 1
        human = next(p for p in players if p.name == self.human_name)
        bot = next(p for p in players if p.name == self.bot_name)

        print()
        print("━" * 52)
        print(f"  Hand #{self.hand_num}  │  {human.name}: {human.stack} chips"
              f"  │  {bot.name}: {bot.stack} chips")
        print("━" * 52)
        print(f"  Positions: {human.name} = {human.position.value}   "
              f"{bot.name} = {bot.position.value}")
        print(f"  Your cards:  {_cards(human.hole_cards)}")

    def on_street_start(self, street_name, community_cards):
        if street_name == "Pre-flop":
            return
        print()
        print(f"  ── {street_name}:  {_cards(community_cards)}")

    def on_player_action(self, player, action, amount):
        if player.name != self.human_name:
            print(f"  {player.name}: {_action_label(action, amount)}")

    def on_hand_complete(self, player_ranks, pot):
        if not player_ranks:
            return
        print()
        # Show cards only when there's a showdown (fold wins have 1 active player)
        active = [p for p, _ in player_ranks if p.is_active]
        if len(active) > 1:
            for p, _ in player_ranks:
                print(f"  {p.name} shows:  {_cards(p.hole_cards)}")

        min_rank = min(r for _, r in player_ranks)
        winners = [p for p, r in player_ranks if r == min_rank]
        winner_str = " & ".join(w.name for w in winners)
        print(f"  → {winner_str} wins {pot} chips")


def _latest_checkpoint(directory="cfr/checkpoints"):
    """Return the .pkl file with the highest iteration count, or None."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return None
    if not files:
        return None
    files.sort()
    return os.path.join(directory, files[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Play heads-up poker against a trained CFR bot"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pkl checkpoint (default: latest in cfr/checkpoints/)",
    )
    parser.add_argument("--stack", type=int, default=1000, help="Starting stack (default 1000)")
    parser.add_argument("--big-blind", type=int, default=10, help="Big blind size (default 10)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint()
    if checkpoint is None:
        print("Error: no checkpoint found. Pass --checkpoint or train a model first.")
        sys.exit(1)
    if not os.path.exists(checkpoint):
        print(f"Error: checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading {checkpoint}...")
    bot_agent = CFRAgent(checkpoint)

    human = Player("You", args.stack, HumanAgent())
    bot = Player("Bot", args.stack, bot_agent)

    observer = DisplayObserver("You", "Bot")
    game = PokerGame([human, bot], small_blind=args.big_blind // 2, observers=[observer])

    print(f"Heads-up  │  {args.stack} chips each  │  BB = {args.big_blind}")
    print("Commands: [f] fold  [c] check/call  [b <n>] bet  [a] allin  [q] quit")

    while human.stack > 0 and bot.stack > 0:
        game.play_hand()
        game.state.button_pos = (game.state.button_pos + 1) % 2

        if human.stack <= 0:
            print("\n  Bot wins — you're out of chips.")
            break
        if bot.stack <= 0:
            print("\n  You win — Bot is out of chips.")
            break

        print()
        try:
            cont = input("  Play next hand? [Enter to continue / q to quit] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cont in ("q", "n", "no"):
            break

    profit = human.stack - args.stack
    print(f"\n  Final:  You {human.stack}  │  Bot {bot.stack}")
    print(f"  Net: {'+'if profit >= 0 else ''}{profit} chips")


if __name__ == "__main__":
    main()
