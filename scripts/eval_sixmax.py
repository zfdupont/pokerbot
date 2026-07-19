#!/usr/bin/env python3
"""Duplicate-deal A/B evaluation for six-max blueprint checkpoints.

Protocol: for each seeded deck and each hero seat, play the SAME deck with
strategy A in the hero seat and B in every other seat (button = deck index
mod n). Seat rotation over identical decks cancels deal luck; a fixed seed
makes the whole run reproducible. Reports A's win rate in BB/100.

Usage:
    uv run python scripts/eval_sixmax.py --a CKPT [--b CKPT|uniform]
        [--hands 500] [--seed 1] [--config sixmax/configs/default.toml]
"""
import argparse
import importlib.util
import os
import random
import subprocess
import sys


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_and_get_so_dir(repo_root: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            return os.path.join(repo_root, os.path.dirname(line.split()[-1]))
    raise RuntimeError("Could not locate sixmax.so in buck2 output")


def _force_load_sixmax(repo_root: str):
    so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


# When run as a script we must load the extension before touching sixmax
# names; when imported by tests, conftest has already registered it.
if "sixmax" not in sys.modules:
    _force_load_sixmax(_get_repo_root())
import sixmax  # noqa: E402


class UniformStrategy:
    """Baseline: no knowledge; sample_action falls back to uniform-legal."""
    def probs_for(self, state):
        return []


def sample_action(probs, mask, rng):
    """Sample from stored probs re-masked to THIS state's legal actions.

    The abstraction merges states with different masks, so stored mass can
    sit on illegal actions; re-mask and renormalize, falling back to uniform
    over legal when nothing legal has mass (or the infoset is unseen)."""
    legal = [i for i, m in enumerate(mask) if m]
    weights = [probs[i] if i < len(probs) else 0.0 for i in legal]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(legal)
    r = rng.random() * total
    acc = 0.0
    for i, w in zip(legal, weights):
        acc += w
        if r <= acc:
            return i
    return legal[-1]


def _play_hand(deck, button, hero_seat, strat_a, strat_b, vocab, cfg, rng):
    state = sixmax.EngineGameState(cfg, button, deck, vocab, [])
    while not state.is_terminal():
        seat = state.current_player()
        strat = strat_a if seat == hero_seat else strat_b
        action = sample_action(strat.probs_for(state), state.legal_mask(), rng)
        state.apply(action)
    return state.utility(hero_seat)


def run_match(strat_a, strat_b, vocab, cfg, hands, seed):
    """A occupies each seat once per deck; returns A's BB/100."""
    rng = random.Random(seed)
    n = cfg.num_players
    total = 0.0
    for h in range(hands):
        deck = rng.sample(range(52), 2 * n + 5)
        button = h % n
        for hero_seat in range(n):
            total += _play_hand(deck, button, hero_seat, strat_a, strat_b,
                                vocab, cfg, rng)
    return 100.0 * total / (hands * n)


def main() -> None:
    repo_root = _get_repo_root()
    parser = argparse.ArgumentParser(description="Six-max blueprint A/B eval")
    parser.add_argument("--a", required=True, help="checkpoint for strategy A")
    parser.add_argument("--b", default="uniform",
                        help="checkpoint for strategy B, or 'uniform'")
    parser.add_argument("--hands", type=int, default=500,
                        help="decks; each is played once per seat")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str,
                        default=os.path.join(repo_root, "sixmax", "configs",
                                             "default.toml"))
    args = parser.parse_args()

    vc_path = os.path.join(repo_root, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", vc_path)
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    vocab = vc.load_vocab(args.config, "blueprint")

    strat_a = sixmax.BlueprintStrategy.load(args.a, vocab)
    cfg = sixmax.EngineConfig(num_players=strat_a.num_players())
    if args.b == "uniform":
        strat_b = UniformStrategy()
    else:
        strat_b = sixmax.BlueprintStrategy.load(args.b, vocab)
        if strat_b.num_players() != strat_a.num_players():
            raise SystemExit("checkpoints disagree on num_players")

    bb100 = run_match(strat_a, strat_b, vocab, cfg, args.hands, args.seed)
    n_hands = args.hands * cfg.num_players
    print(f"Blueprint A win rate: {bb100:+.2f} BB/100 ({n_hands} hands)")


if __name__ == "__main__":
    main()
