#!/usr/bin/env python3
"""HU-mode sanity eval: the six-max blueprint (2-player) vs the frozen tabular
and neural bots, in the live game/poker.py engine.

Duplicate deals cancel deal luck: each seeded deck is played twice, with the
hero in seat 0 then seat 1. The deck is shuffled with the global `random`
module and re-created each hand, so seeding `random` before play_hand makes the
deck reproducible; the hero's action RNG uses a separate stream.

Usage:
    uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/blueprint.bin \
        [--tabular cfr/checkpoints/checkpoint_09040000.pkl] \
        [--neural neural_cfr/checkpoints/checkpoint.pt] [--hands 500] [--seed 1]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player import Player
from game.poker import PokerGame


def _play_seeded(hero_agent, villain_agent, hero_seat, seed, bb, stack):
    """One hand at a fixed seed; returns hero's chip delta. bb sets blinds via
    small_blind = bb // 2."""
    random.seed(seed)
    names = ["a", "b"]
    agents = [None, None]
    agents[hero_seat] = hero_agent
    agents[1 - hero_seat] = villain_agent
    players = [Player(names[i], stack, agent=agents[i]) for i in range(2)]
    game = PokerGame(players, small_blind=bb // 2)
    game.play_hand()
    return players[hero_seat].stack - stack


def run_duplicate_match(hero_factory, villain_factory, hands, seed, bb, stack):
    """hero_factory(seat)/villain_factory(seat) build one agent per seat ONCE
    and reuse them across all hands. Agents hold only a loaded strategy + RNG
    and carry no per-hand state, so building them per hand would needlessly
    reload the checkpoint every hand — and, for NeuralAgent, re-run a buck2
    build subprocess every hand. Returns hero BB/100 over `hands` duplicate
    deals (2 games each)."""
    heroes = {seat: hero_factory(seat) for seat in (0, 1)}
    villains = {seat: villain_factory(seat) for seat in (0, 1)}
    total = 0.0
    for h in range(hands):
        s = seed * 1_000_003 + h
        for hero_seat in (0, 1):
            total += _play_seeded(heroes[hero_seat], villains[hero_seat],
                                  hero_seat, s, bb, stack)
    return 100.0 * total / (hands * 2 * bb)  # BB/100


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="HU-mode sanity eval")
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--tabular", default=None)
    parser.add_argument("--neural", default=None)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--stack", type=int, default=200)
    args = parser.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    toml = os.path.join(root, "sixmax", "configs", "default.toml")

    def hero(_):
        return SixmaxAgent(args.blueprint, config_toml=toml)

    if args.tabular:
        from agents.cfr_agent import CFRAgent
        bb100 = run_duplicate_match(hero, lambda _: CFRAgent(args.tabular),
                                    args.hands, args.seed, args.bb, args.stack)
        print(f"blueprint vs tabular: {bb100:+.2f} BB/100 "
              f"({args.hands * 2} hands)")
    if args.neural:
        from agents.neural_agent import NeuralAgent
        bb100 = run_duplicate_match(hero, lambda _: NeuralAgent(args.neural),
                                    args.hands, args.seed, args.bb, args.stack)
        print(f"blueprint vs neural:  {bb100:+.2f} BB/100 "
              f"({args.hands * 2} hands)")


if __name__ == "__main__":
    main()
