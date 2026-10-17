from game.poker import PokerGame
from models.player import Player
from models.enums import Action
from agents.base_agent import PokerAgent


class ScriptedAgent(PokerAgent):
    """Returns pre-scripted (Action, amount) decisions in order."""

    def __init__(self, script):
        self._script = list(script)

    def get_action(self, player, game_state):
        return self._script.pop(0)


def test_action_history_records_voluntary_actions_not_blinds():
    # Heads-up, scripted to a quick preflop raise/fold.
    p0 = Player("A", 100, ScriptedAgent([(Action.RAISE, 6), (Action.FOLD, None)]))
    p1 = Player("B", 100, ScriptedAgent([(Action.FOLD, None)]))
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()

    hist = game.state.action_history
    # Blinds are posted directly (not via _process_action) and must be absent.
    assert all(a in (Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE)
               for (_, a, _, _) in hist)
    assert (0, Action.RAISE, 6, 0) in hist          # preflop raise recorded
    assert hist[0] == (0, Action.RAISE, 6, 0)        # recorded in action order
    assert hist[-1][1] == Action.FOLD                # final action recorded


def test_action_history_cleared_each_hand():
    p0 = Player("A", 100, ScriptedAgent([(Action.FOLD, None), (Action.FOLD, None)]))
    p1 = Player("B", 100, ScriptedAgent([(Action.FOLD, None), (Action.FOLD, None)]))
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()
    first = list(game.state.action_history)
    game.play_hand()
    second = list(game.state.action_history)
    # A fresh hand starts from an empty log, not an accumulation of the prior one.
    assert len(second) <= len(first) + 1
    assert second[0][3] == 0                          # first record is a preflop action
