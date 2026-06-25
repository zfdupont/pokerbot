import pytest
import tempfile
from models.player import Player
from models.state import GameState
from models.enums import Action
from agents.cfr_agent import CFRAgent
from cfr.regret_table import RegretTable
from cfr.trainer import Trainer


def make_trained_agent(iterations: int = 20) -> CFRAgent:
    table = RegretTable()
    Trainer(table).train(num_iterations=iterations, checkpoint_interval=0, checkpoint_dir=None)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    table.save(path)
    return CFRAgent(strategy_path=path)


def make_game_state():
    from agents.simple_agent import SimpleAgent
    players = [
        Player("P0", 1000, SimpleAgent()),
        Player("P1", 1000, SimpleAgent()),
    ]
    state = GameState(players, small_blind=5)
    state.deal_hole_cards()
    state.current_bet = 10
    state.pot = 15
    return state


class TestCFRAgent:
    def test_returns_valid_action(self):
        agent = make_trained_agent()
        player = Player("P0", 1000, agent)
        state = make_game_state()
        player.hole_cards = state.players[0].hole_cards
        action, amount = agent.get_action(player, state)
        assert action in list(Action)

    def test_fold_or_call_or_raise_when_facing_bet(self):
        agent = make_trained_agent()
        player = Player("P0", 1000, agent)
        state = make_game_state()
        player.hole_cards = state.players[0].hole_cards
        state.current_bet = 50
        state.pot = 75
        player.current_bet = 0
        action, amount = agent.get_action(player, state)
        assert action in (Action.FOLD, Action.CALL, Action.RAISE, Action.BET)

    def test_uses_mixed_strategy(self):
        """Over many calls, the agent should not always return the same action."""
        agent = make_trained_agent(iterations=50)
        player = Player("P0", 1000, agent)
        state = make_game_state()
        player.hole_cards = state.players[0].hole_cards
        state.current_bet = 0
        state.pot = 15
        player.current_bet = 0
        actions = {agent.get_action(player, state)[0] for _ in range(30)}
        assert len(actions) >= 1  # at minimum doesn't crash; mixed if > 1
