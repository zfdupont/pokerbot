"""Cross-validate sixmax.settle_pots against the Python engine's PotManager
on randomly generated, betting-consistent scenarios. The Python engine is
the frozen oracle: mismatches mean the C++ settlement is wrong."""
import random

import sixmax
from agents.simple_agent import SimpleAgent
from game.pot_manager import PotManager
from models.player import Player

START = 100_000  # Python Player stacks start here; payout = stack - START


def _python_payouts(contribs, folded, ranks):
    players = [Player(f"p{i}", START, SimpleAgent()) for i in range(len(contribs))]
    pm = PotManager()
    m = max(contribs)
    for p, c, f in zip(players, contribs, folded):
        p.stack -= c  # real engine deducts during betting; contribute() only records
        # Folded players must NOT be marked is_all_in: PotManager builds side-pot
        # levels from every is_all_in player regardless of folding, so flagging a
        # folder would create a spurious level; the unflagged path correctly absorbs
        # their short contribution into the existing pots.
        if not f and c < m:
            p.is_all_in = True
        pm.contribute(p, c)
    pm.award([(p, r) for p, r, f in zip(players, ranks, folded) if not f])
    return [p.stack - START for p in players]


def _scenario(rng):
    """Betting-consistent random scenario. Invariants: >=2 unfolded, >=1
    unfolded player at max contribution M, unfolded non-all-in players
    contribute exactly M, all contributions integer >= 1, distinct ranks."""
    n = rng.randint(2, 6)
    while True:
        m = rng.randint(2, 200)
        contribs, folded = [], []
        for _ in range(n):
            role = rng.choice(["call", "call", "allin", "fold"])
            if role == "call":
                contribs.append(m); folded.append(False)
            elif role == "allin":
                contribs.append(rng.randint(1, m - 1)); folded.append(False)
            else:
                contribs.append(rng.randint(1, m)); folded.append(True)
        unfolded = [i for i in range(n) if not folded[i]]
        callers = [i for i in unfolded if contribs[i] == m]
        if len(unfolded) >= 2 and callers:
            ranks = list(range(n))
            rng.shuffle(ranks)
            return contribs, folded, ranks


def test_settlement_matches_python_engine_on_random_scenarios():
    rng = random.Random(20260718)
    for _ in range(300):
        contribs, folded, ranks = _scenario(rng)
        expected = _python_payouts(contribs, folded, ranks)
        got = sixmax.settle_pots([float(c) for c in contribs], folded, ranks)
        net = [g - c for g, c in zip(got, contribs)]
        assert all(abs(a - b) < 1e-6 for a, b in zip(net, expected)), (
            f"mismatch: contribs={contribs} folded={folded} ranks={ranks} "
            f"cpp_net={net} python={expected}")


def test_uncalled_jam_returns_excess():
    # settle_pots returns GROSS payouts: folded players get 0, not their
    # contribution back; net profit/loss = payout − contribution.
    # A jams 100, B calls all-in for 40, C folds at 10. A wins everything.
    got = sixmax.settle_pots([100.0, 40.0, 10.0], [False, False, True], [0, 1, 2])
    assert got == [150.0, 0.0, 0.0]
    # Same but B wins: B takes the 90-capped pot, A keeps his uncalled 60.
    got = sixmax.settle_pots([100.0, 40.0, 10.0], [False, False, True], [1, 0, 2])
    assert got == [60.0, 90.0, 0.0]


def test_chip_conservation_on_random_scenarios():
    rng = random.Random(99)
    for _ in range(200):
        contribs, folded, ranks = _scenario(rng)
        got = sixmax.settle_pots([float(c) for c in contribs], folded, ranks)
        assert abs(sum(got) - sum(contribs)) < 1e-6
