from models.player import Player
from agents.simple_agent import SimpleAgent
from game.pot_manager import PotManager


def _player(name, stack=1000):
    return Player(name, stack, SimpleAgent())


def test_total_tracks_contributions():
    pm = PotManager()
    a, b = _player("A"), _player("B")
    pm.contribute(a, 50)
    pm.contribute(b, 100)
    assert pm.total == 150


def test_no_allin_single_pot():
    pm = PotManager()
    a, b, c = _player("A"), _player("B"), _player("C")
    pm.contribute(a, 100)
    pm.contribute(b, 100)
    pm.contribute(c, 100)
    pots = pm.calculate_side_pots()
    assert len(pots) == 1
    assert pots[0].amount == 300
    assert set(pots[0].eligible) == {a, b, c}


def test_one_allin_creates_two_pots():
    pm = PotManager()
    a, b, c = _player("A"), _player("B"), _player("C")
    a.is_all_in = True
    pm.contribute(a, 50)
    pm.contribute(b, 200)
    pm.contribute(c, 200)
    pots = pm.calculate_side_pots()
    assert len(pots) == 2
    assert pots[0].amount == 150
    assert set(pots[0].eligible) == {a, b, c}
    assert pots[1].amount == 300
    assert set(pots[1].eligible) == {b, c}


def test_award_gives_chips_to_winner():
    pm = PotManager()
    a, b = _player("A", 900), _player("B", 900)
    pm.contribute(a, 100)
    pm.contribute(b, 100)
    pm.award([(a, 100), (b, 500)])
    assert a.stack == 1100
    assert b.stack == 900


def test_award_splits_on_tie():
    pm = PotManager()
    a, b = _player("A", 900), _player("B", 900)
    pm.contribute(a, 100)
    pm.contribute(b, 100)
    pm.award([(a, 100), (b, 100)])
    assert a.stack == 1000
    assert b.stack == 1000


def test_award_side_pot_allin():
    pm = PotManager()
    a, b, c = _player("A", 950), _player("B", 800), _player("C", 800)
    a.is_all_in = True
    pm.contribute(a, 50)
    pm.contribute(b, 200)
    pm.contribute(c, 200)
    pm.award([(a, 1), (b, 100), (c, 500)])
    assert a.stack == 950 + 150
    assert b.stack == 800 + 300
    assert c.stack == 800
