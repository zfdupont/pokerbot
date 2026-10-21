"""200 BB training frame: config, checkpoint metadata, engine, deploy, harness.

Arm C0 (sixmax/configs/hu200.toml) is the leak-A map trained at
starting_stack=200.0. These tests pin the 200 BB frame at every layer the
roadmap's spec 1b names: the TOML config, the trainer's engine, the checkpoint
metadata, the live-deploy all-in translation, and the duplicate-match harness.
"""
import importlib.util
import math
import os
import shutil
import struct

import pytest

import sixmax
from agents.sixmax_agent import SixmaxAgent
from game.poker import PokerGame
from game.pot_manager import PotManager
from models.enums import Action
from models.player import Player
from scripts.eval_hu_sanity import run_duplicate_match

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_HU200_TOML = os.path.join(_ROOT, "sixmax", "configs", "hu200.toml")
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_sixmax", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """argparse.Namespace stand-in; unset flags are None."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _allin_index(vocab):
    idxs = [i for i in range(vocab.size())
            if vocab.at(i).type == sixmax.ActionType.AllIn]
    assert len(idxs) == 1, f"expected exactly one all-in entry, got {idxs}"
    return idxs[0]


def _train_200(tmp_path, vocab, bucket_map, iterations=120):
    """Train a SIXBP002 checkpoint at 200 BB with the map resolvable beside it.

    The map is copied under its own basename into the checkpoint's directory so
    the deploy-side resolver finds it without any environment variables.
    """
    map_copy = tmp_path / os.path.basename(bucket_map)
    shutil.copyfile(bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    trainer = sixmax.BlueprintTrainer(cfg, vocab, abs_, num_threads=1, seed=13)
    trainer.train(iterations)
    ckpt = str(tmp_path / "hu200.bin")
    trainer.save(ckpt, vocab, cfg, abs_)
    return ckpt, cfg, abs_


def test_hu200_config_matches_c0(default_vocab):
    """hu200.toml is C0: hu.toml + the leak-A map at 200 BB, 60M iters."""
    mod = _load_script()
    cfg = mod.resolve_config(_Args(config=_HU200_TOML), _ROOT)
    assert cfg["num_players"] == 2
    assert cfg["starting_stack"] == 200.0
    assert cfg["iterations"] == 60_000_000
    assert cfg["checkpoint_interval"] == 5_000_000
    assert cfg["snapshots"] is True
    assert cfg["checkpoint"] == "sixmax/checkpoints/hu200_c0.bin"
    assert cfg["bucket_map"] == "sixmax/tables/map_pct_50_50_20.bin"

    # The action vocabulary must stay byte-identical to default.toml so the
    # checkpoint's vocab hash matches the deploy bridge (SixmaxAgent).
    vocab_cfg = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    vc = importlib.util.module_from_spec(vocab_cfg)
    vocab_cfg.loader.exec_module(vc)
    hu200_vocab = vc.load_vocab(_HU200_TOML, "blueprint")
    assert hu200_vocab.hash() == default_vocab.hash()


def test_engine_200bb_stacks(default_vocab):
    """Blind arithmetic and the all-in raise-to target land on 200 BB."""
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    deck = list(range(52))
    hand = sixmax.HandState(cfg, 0, deck)
    assert hand.current_player() == 0       # HU: button/SB acts first
    assert hand.stack(0) == pytest.approx(199.5)   # SB posted 0.5 BB
    assert hand.stack(1) == pytest.approx(199.0)   # BB posted 1.0 BB

    state = sixmax.EngineGameState(cfg, 0, deck, default_vocab)
    ctx = state.bet_context()
    assert default_vocab.target_bb(_allin_index(default_vocab), ctx) == \
        pytest.approx(200.0)


def test_checkpoint_records_200(tmp_path, synthetic_bucket_map, default_vocab,
                                monkeypatch):
    ckpt, _cfg, abs_ = _train_200(tmp_path, default_vocab, synthetic_bucket_map)
    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)
    with open(ckpt, "rb") as f:
        header = f.read(32)
    # magic(8) | vocab_hash(8) | num_players(int32) | action_dim(int32) |
    # starting_stack(double) -> the stack sits at byte 24.
    assert header[:8] == b"SIXBP002"
    meta_starting_stack = struct.unpack_from("=d", header, 24)[0]
    assert meta_starting_stack == 200.0

    # The artifact still loads (map resolved from the hint beside it).
    strategy = sixmax.BlueprintStrategy.load(ckpt, default_vocab)
    assert strategy.abstraction().hash() == abs_.hash()


def test_agent_allin_at_200bb(tmp_path, synthetic_bucket_map, default_vocab,
                              monkeypatch):
    ckpt, _cfg, _abs = _train_200(tmp_path, default_vocab, synthetic_bucket_map)
    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)
    agent = SixmaxAgent(ckpt, config_toml=_HU200_TOML)

    # Live 200 BB HU hand: 400 chips each, big blind 2 chips. Post the blinds
    # exactly as PokerGame.play_hand does, leaving the SB (hero) to act.
    hero = Player("hero", 400, agent=agent)
    villain = Player("villain", 400, agent=agent)
    game = PokerGame([hero, villain], small_blind=1)
    pot = PotManager()
    game._reset_for_hand()
    game._assign_positions()
    game.state.deal_hole_cards()
    game._post_blinds(pot)
    game.state.pot = pot.total
    game.state.current_player_idx = game._first_to_act(0)
    assert game.state.players[game.state.current_player_idx] is hero
    assert hero.stack == 399 and hero.current_bet == 1
    assert game.state.big_blind == 2

    bb = game.state.big_blind
    vocab = agent._deploy._vocab
    idx = _allin_index(vocab)
    to_call_chips = game.state.current_bet - hero.current_bet
    ctx = sixmax.BetContext(
        pot=game.state.pot / bb,
        current_bet=game.state.current_bet / bb,
        to_call=to_call_chips / bb,
        stack=(hero.current_bet + hero.stack) / bb,     # 200.0 BB
    )
    assert ctx.stack == pytest.approx(200.0)
    raise_to_bb = vocab.target_bb(idx, ctx)
    assert raise_to_bb == pytest.approx(200.0)

    action, amount = agent._translate(idx, to_call_chips, raise_to_bb, bb, hero)
    assert (action, amount) == (Action.BET, 400)


def test_h2h_runs_at_200bb(tmp_path, synthetic_bucket_map, default_vocab,
                           monkeypatch):
    ckpt, _cfg, _abs = _train_200(tmp_path, default_vocab, synthetic_bucket_map)
    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)

    def factory(_seat):
        return SixmaxAgent(ckpt, config_toml=_HU200_TOML)

    bb100 = run_duplicate_match(factory, factory, hands=2, seed=1, bb=2,
                                stack=400)
    assert isinstance(bb100, float)
    assert math.isfinite(bb100)
    assert bb100 == 0.0     # same strategy in both seats: paired diff is 0
