"""Deployed agent supplies the last-raise key field from a hand replay.

A bet_history checkpoint's keys include the size class of the last raise on
each street. The live engine has no such field, so SixmaxAgent replays the
observed line into an EngineGameState (the search path's replay) and reads the
classes from it, then SixmaxDeployStrategy packs them into every key it looks
up — including the nearest-pot-bucket retries. Off-mode checkpoints never
replay, and an unreplayable (desynced) line plays the safe action instead of a
key with a wrong field.
"""
import os
import random
import shutil

import pytest

import agents.sixmax_agent as sa

sixmax = pytest.importorskip("sixmax")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_C5_TOML = os.path.join(_ROOT, "sixmax", "configs", "hu200_c5.toml")
_VOCAB = sa._load_vocab(_C5_TOML, "blueprint")
_CFG = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
C4_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]


def _train_tiny(directory, bet_history):
    """A tiny HU 200 BB checkpoint beside its bucket map (Task 2's pattern)."""
    abs_ = sixmax.Abstraction.from_bucket_map(str(directory / "idxmod.bin"))
    abs_.set_pot_layout("bb", C4_EDGES)
    abs_.set_bet_history(bet_history)
    trainer = sixmax.BlueprintTrainer(_CFG, _VOCAB, abs_, num_threads=1, seed=13)
    trainer.train(120)
    path = str(directory / f"tiny_{bet_history}.bin")
    trainer.save(path, _VOCAB, _CFG, abs_)
    return path


@pytest.fixture(scope="module")
def c5_checkpoint(tmp_path_factory, synthetic_bucket_map):
    directory = tmp_path_factory.mktemp("deploy_c5")
    shutil.copyfile(synthetic_bucket_map, directory / "idxmod.bin")
    return _train_tiny(directory, "last_raise")


@pytest.fixture(scope="module")
def c4_tiny_ckpt(tmp_path_factory, synthetic_bucket_map):
    """The same C4 key space without the opt-in field (SIXBP003)."""
    directory = tmp_path_factory.mktemp("deploy_c4")
    shutil.copyfile(synthetic_bucket_map, directory / "idxmod.bin")
    return _train_tiny(directory, "off")


@pytest.fixture()
def c5_deploy(c5_checkpoint):
    return sa.SixmaxDeployStrategy.load(c5_checkpoint, _C5_TOML)


@pytest.fixture()
def c5_agent(c5_checkpoint):
    agent = sa.SixmaxAgent(c5_checkpoint, config_toml=_C5_TOML)
    agent._rng.seed(7)
    return agent


def _card(text):
    from models.card import Card
    from models.enums import Suit

    rank = "23456789TJQKA".index(text[0]) + 2
    suit = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS,
            "s": Suit.SPADES}[text[1]]
    return Card(rank, suit)


def _kwargs(**overrides):
    """One postflop decision in the C5 key frame (non-legacy pot layout)."""
    kwargs = dict(
        hole=[sa._card_to_int(_card("As")), sa._card_to_int(_card("Ks"))],
        board=[sa._card_to_int(_card(c)) for c in ("Ah", "7c", "2d")],
        street=1, raises_per_street=[1, 0, 0, 0], pot_bb=6.0,
        current_bet_bb=2.0, to_call_bb=1.0, stack_bb=190.0,
        eff_stack_bb=190.0, live=1, after=0,
        legal=[1] * _VOCAB.size(), rng=random.Random(0))
    kwargs.update(overrides)
    return kwargs


def test_decide_requires_last_raise_when_on(c5_deploy):
    with pytest.raises(ValueError, match="last_raise"):
        c5_deploy.decide(**_kwargs())


class _RecordingAbs:
    """Fake abstraction: an empty table, so every key miss retries."""

    def __init__(self, mode="last_raise"):
        self._mode = mode
        self.calls = []

    def pot_layout(self):
        return ("bb", [3.0, 6.0, 12.0, 24.0])

    def bucket(self, hole, board):
        return 0

    def bet_history(self):
        return self._mode

    def pack_key(self, card, street, raises, pot_bb, eff_stack_bb, live,
                 after, last_raise=None):
        self.calls.append((pot_bb, last_raise))
        return (street, tuple(raises), tuple(last_raise or ()))


class _EmptyTable:
    def __init__(self, abs_):
        self._abs = abs_

    def abstraction(self):
        return self._abs

    def probs(self, key):
        return []


def test_nearest_retries_carry_last_raise():
    abs_ = _RecordingAbs()
    deploy = sa.SixmaxDeployStrategy(_EmptyTable(abs_), _VOCAB)
    classes = [1, 2, 0, 0]
    _idx, raise_to = deploy.decide(**_kwargs(last_raise=classes))
    assert len(abs_.calls) > 1           # primary key + neighbour retries
    assert all(call[1] == classes for call in abs_.calls)
    assert deploy.lookups["safe"] == 1
    assert raise_to == 0.0


def test_off_mode_accepts_no_last_raise(c4_tiny_ckpt):
    deploy = sa.SixmaxDeployStrategy.load(c4_tiny_ckpt, _C5_TOML)
    assert deploy.bet_history == "off"
    kwargs = _kwargs()
    idx, _raise_to = deploy.decide(**kwargs)
    assert kwargs["legal"][idx] == 1


class _RecordingStrategy:
    """Delegates to a BlueprintStrategy, recording every key looked up."""

    def __init__(self, inner):
        self._inner = inner
        self.keys = []

    def abstraction(self):
        return self._inner.abstraction()

    def probs(self, key):
        self.keys.append(key)
        return self._inner.probs(key)

    def num_players(self):
        return self._inner.num_players()

    def num_infosets(self):
        return self._inner.num_infosets()


def _decode(key):
    """Split a C5 (non-legacy pot layout) key into its fields."""
    return {
        "card": key & 0xFF,
        "street": (key >> 8) & 3,
        "raises": [(key >> (10 + 2 * s)) & 3 for s in range(4)],
        "live": (key >> 20) & 7,
        "after": (key >> 23) & 7,
        "pot": (key >> 26) & 0xF,
        "classes": [(key >> (30 + 3 * s)) & 7 for s in range(4)],
    }


def _without_pot(fields):
    return {k: v for k, v in fields.items() if k != "pot"}


def _near_pot_edge(pot_bb, tol):
    return any(abs(pot_bb - edge) <= tol for edge in C4_EDGES)


def test_agent_key_matches_engine_key(c5_checkpoint, monkeypatch):
    """At every live decision, the key the deploy bridge looks up through its
    own pack_key equals the engine's abstract_key for the hand replay.

    The deploy sizes raises with the engine's net-of-call BetContext
    (engine_bet_context), so the replay reproduces the live pot on every
    decision up to whole-chip rounding:

    * at EVERY decision the replay-derived fields (card, street, raises and the
      new last-raise classes) must equal the engine's;
    * at EVERY decision the replayed pot matches the live pot within chip
      rounding, and the whole key matches, live/after included, bar an
      exact pot-edge tie.
    """
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from game.poker import PokerGame
    from models.player import Player

    base = SixmaxDeployStrategy.load(c5_checkpoint, _C5_TOML)
    recorder = _RecordingStrategy(base._strategy)
    deploy = SixmaxDeployStrategy(recorder, base._vocab)
    hero = SixmaxAgent.from_strategy(deploy, _C5_TOML)
    villain = SixmaxAgent.from_strategy(deploy, _C5_TOML)
    hero._rng.seed(1)
    villain._rng.seed(2)
    layout = recorder.abstraction().pot_layout()
    def tol(pot_bb):
        # Chip rounding: the live engine deals whole chips (20-chip big blind
        # below) and later pot-fraction raises compound the rounding, so allow
        # three chips or 1% of the pot. The old gross-pot sizing bug moved the
        # pot by fraction * owe (e.g. 11% of a 7 BB pot), far outside this.
        return max(3.0 / 20.0, 0.01 * pot_bb)

    original_get_action = SixmaxAgent.get_action
    original_decide = SixmaxDeployStrategy.decide
    decisions = []       # dicts: live/engine fields, pots, classes, street

    def get_action(self, player, game_state):
        self._deploy._parity_state = (
            self, game_state,
            [sa._card_to_int(c) for c in player.hole_cards],
            [sa._card_to_int(c) for c in game_state.community_cards])
        return original_get_action(self, player, game_state)

    def decide(self, **kwargs):
        agent, game_state, hole, board = self._parity_state
        engine = agent._replay_to_now(game_state, hole, board)
        recorder.keys.clear()
        result = original_decide(self, **kwargs)
        assert recorder.keys, "decide looked up no key"
        ctx = engine.bet_context()
        decisions.append(dict(
            live=_decode(recorder.keys[0]),
            engine=_decode(engine.abstract_key(recorder.abstraction())),
            live_pot=kwargs["pot_bb"],
            replay_pot=ctx.pot + ctx.to_call,   # gross, as the key packs it
            classes=list(kwargs["last_raise"]),
            street=kwargs["street"],
            eff_stack=kwargs["eff_stack_bb"]))
        return result

    monkeypatch.setattr(SixmaxAgent, "get_action", get_action)
    monkeypatch.setattr(SixmaxDeployStrategy, "decide", decide)

    # 20-chip big blind (200 BB stacks), so chip rounding moves the replayed
    # pot by far less than a bucket.
    for hand in range(300):
        random.seed(hand)
        p0 = Player("hero", 4000, hero)
        p1 = Player("villain", 4000, villain)
        PokerGame([p0, p1], small_blind=10).play_hand()

    assert decisions, "no decisions were compared"
    # The deploy must supply the replay's own classes, and the replay must
    # agree on every action-count-derived field, at every single decision.
    for d in decisions:
        for field in ("card", "street", "raises", "classes"):
            assert d["live"][field] == d["engine"][field], (field, d)
        assert d["live"]["classes"] == d["classes"]
    # The bridge must pack the LIVE pot (the replay only supplies the classes).
    for d in decisions:
        assert d["live"]["pot"] == sa._pot_index(layout, d["live_pot"],
                                                  d["eff_stack"])
    exact = [d for d in decisions
             if abs(d["replay_pot"] - d["live_pot"]) <= tol(d["live_pot"])]
    unequal = [d for d in exact if d["live"] != d["engine"]]
    unexplained = [d for d in unequal
                   if not _near_pot_edge(d["live_pot"], tol(d["live_pot"]))]
    assert not unexplained, (
        f"{len(unexplained)} of {len(exact)} reproduced lines key differently: "
        f"{unexplained[:2]}")
    assert len(exact) > 100, f"only {len(exact)} lines reproduced"
    # With raises sized from the engine's net-of-call pot, the replay must
    # reproduce the live pot on every decision (it diverged on raised lines
    # while the deploy sized from the gross pot).
    assert len(exact) == len(decisions), (
        f"{len(decisions) - len(exact)} decisions' replayed pot diverged")
    earlier = [d for d in decisions if any(d["classes"][:d["street"]])]
    assert earlier, "no decision carried an earlier street's raise class"
    print(f"[parity] decisions={len(decisions)} earlier_street_class={len(earlier)} "
          f"reproduced={len(exact)} pot_edge_ties={len(unequal)} "
          f"pot_diverged={len(decisions) - len(exact)}")


def test_bet_history_is_heads_up_only(c5_agent):
    """The replay is heads-up (hero = engine seat 0, one opponent); a larger
    table must refuse rather than key without the field."""
    from game.poker import PokerGame
    from models.player import Player

    players = [Player(name, 200, c5_agent) for name in ("a", "b", "c")]
    game = PokerGame(players, small_blind=1)
    game.state.deck = game.state._create_deck()
    game.state.deal_hole_cards()
    with pytest.raises(ValueError, match="heads-up only"):
        c5_agent.get_action(players[0], game.state)


def test_replay_desync_plays_safe(c5_agent, monkeypatch):
    from game.poker import PokerGame
    from models.enums import Action
    from models.player import Player

    def desync(self, game_state, hole, board, abstraction=None):
        raise ValueError("replay desync: action illegal at engine state")

    monkeypatch.setattr(sa.SixmaxAgent, "_hand_root", desync)

    player = Player("hero", 200, c5_agent)
    villain = Player("villain", 200, c5_agent)
    game = PokerGame([player, villain], small_blind=1)
    game.state.deck = game.state._create_deck()
    game.state.deal_hole_cards()

    action, amount = c5_agent.get_action(player, game.state)
    assert action in (Action.FOLD, Action.CHECK, Action.CALL)  # never a bet
    assert amount is None
    assert c5_agent._deploy.lookups["desync"] == 1


def test_legacy_checkpoint_ignores_history(c4_tiny_ckpt, monkeypatch):
    """An off-mode checkpoint plays exactly as before: get_action never
    replays, and the decision path still completes."""
    from agents.sixmax_agent import SixmaxAgent
    from game.poker import PokerGame
    from models.player import Player

    def never(self, *args, **kwargs):
        raise AssertionError("off-mode checkpoint must not replay the hand")

    monkeypatch.setattr(SixmaxAgent, "_replay_to_street_root", never)
    monkeypatch.setattr(SixmaxAgent, "_replay_to_now", never)

    agent = SixmaxAgent(c4_tiny_ckpt, config_toml=_C5_TOML)
    agent._rng.seed(3)
    p0 = Player("hero", 200, agent)
    p1 = Player("villain", 200, agent)
    game = PokerGame([p0, p1], small_blind=1)
    game.state.deck = game.state._create_deck()
    game.play_hand()  # raises AssertionError if the replay is called
    assert p0.stack + p1.stack == 400
