from models.player import Player
from agents.simple_agent import SimpleAgent


def test_player_default_fields():
    p = Player("Test", 1000, SimpleAgent())
    assert p.is_all_in == False
    assert p.total_contributed == 0
