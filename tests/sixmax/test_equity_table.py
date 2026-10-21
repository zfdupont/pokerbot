"""Exact equity feature store: river enumeration + derived turn/flop equity.

Only the river is enumerated; turn and flop equity are exact means of the
next street's equity over every legal next card (see the design doc,
section 2). Tests therefore exercise the per-hand functions
(`river_equity_exact`, `derive_equity_one`) instead of building the 123M-entry
river table, which is a multi-minute offline job (`scripts/build_equity_tables.py`).
"""
import importlib.util
import os
import random

import pytest

import sixmax

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def card(rank, suit):
    return (rank - 2) * 4 + suit


def _random_deal(street, rng):
    deck = list(range(52))
    rng.shuffle(deck)
    n_board = street + 2
    return deck[:2], deck[2 : 2 + n_board]


def _free_cards(hole, board):
    used = set(hole) | set(board)
    return [c for c in range(52) if c not in used]


def _pair_index(i, j):
    """Position of hole pair (i, j), i < j, in the per-board equity vector."""
    return 46 * i - i * (i - 1) // 2 + (j - i - 1)


def _brute_force_river_equity(hole, board):
    """Pure-Python reference: every non-colliding opponent combo, ties = 0.5."""
    deck = _free_cards(hole, board)
    hero = sixmax.rank7(hole + board)
    score = 0.0
    for i, a in enumerate(deck):
        for b in deck[i + 1 :]:
            opp = sixmax.rank7([a, b] + board)
            if hero.beats(opp):
                score += 1.0
            elif hero.ties(opp):
                score += 0.5
    return score / 990.0


# --------------------------------------------------------------------------
# river_equity_exact
# --------------------------------------------------------------------------
def test_river_exact_matches_brute_force():
    rng = random.Random(20261020)
    for _ in range(50):
        hole, board = _random_deal(3, rng)
        exact = sixmax.river_equity_exact(hole, board)
        reference = _brute_force_river_equity(hole, board)
        assert abs(exact - reference) < 1e-12, (hole, board, exact, reference)


def test_river_exact_known_values():
    # Royal flush in hand: nobody can tie (all five royal cards are used).
    hole = [card(14, 0), card(13, 0)]
    board = [card(12, 0), card(11, 0), card(10, 0), card(2, 1), card(3, 2)]
    assert sixmax.river_equity_exact(hole, board) == 1.0

    # Board plays a royal flush: every legal opponent hand ties.
    board = [card(14, 1), card(13, 1), card(12, 1), card(11, 1), card(10, 1)]
    hole = [card(2, 0), card(3, 0)]
    assert sixmax.river_equity_exact(hole, board) == 0.5


def test_river_exact_close_to_monte_carlo():
    rng = random.Random(7)
    for i in range(100):
        hole, board = _random_deal(3, rng)
        exact = sixmax.river_equity_exact(hole, board)
        mc = sixmax.hand_equity(hole, board, 20000, 1000 + i)
        assert abs(exact - mc) < 0.02, (hole, board, exact, mc)


def test_river_exact_rejects_bad_input():
    hole = [card(14, 0), card(13, 0)]
    with pytest.raises(ValueError):
        sixmax.river_equity_exact(hole, [card(12, 0), card(11, 0), card(10, 0)])
    with pytest.raises(ValueError):
        sixmax.river_equity_exact(
            hole, [card(14, 0), card(11, 0), card(10, 0), card(2, 1), card(3, 2)]
        )
    with pytest.raises(ValueError):
        sixmax.river_equity_exact(
            hole, [card(12, 0), card(11, 0), card(10, 0), card(2, 1), 52]
        )


# --------------------------------------------------------------------------
# derive_equity_one (per-hand, no full tables needed)
# --------------------------------------------------------------------------
def test_derived_river_is_exact():
    rng = random.Random(808)
    hole, board = _random_deal(3, rng)
    assert sixmax.derive_equity_one(3, hole, board) == sixmax.river_equity_exact(
        hole, board
    )


def test_bulk_board_equities_match_brute_force():
    # Exercises the sort-and-count + card-removal correction that the
    # (multi-minute) river build runs for every canonical board, without
    # building the whole 123M-entry table.
    boards = [
        [card(14, 0), card(13, 1), card(7, 2), card(7, 3), card(2, 0)],
        [card(10, 0), card(9, 1), card(8, 0), card(3, 2), card(3, 3)],
    ]
    rng = random.Random(31337)
    pairs = [(i, j) for i in range(47) for j in range(i + 1, 47)]
    for board in boards:
        equities = sixmax._river_equities_for_board(board)
        assert len(equities) == len(pairs) == 1081
        remaining = _free_cards([], board)
        for i, j in rng.sample(pairs, 40):
            expected = _brute_force_river_equity(
                [remaining[i], remaining[j]], board
            )
            assert abs(equities[_pair_index(i, j)] - expected) < 1e-6


def test_derived_turn_equals_mean_of_rivers():
    rng = random.Random(4242)
    hole, board = _random_deal(2, rng)
    derived = sixmax.derive_equity_one(2, hole, board)
    rivers = [
        sixmax.river_equity_exact(hole, board + [c])
        for c in _free_cards(hole, board)
    ]
    assert len(rivers) == 46
    assert derived == pytest.approx(sum(rivers) / 46.0, abs=1e-12)


def test_derived_flop_equals_mean_of_turns():
    rng = random.Random(11)
    hole, board = _random_deal(1, rng)
    derived = sixmax.derive_equity_one(1, hole, board)
    turns = [
        sixmax.derive_equity_one(2, hole, board + [c])
        for c in _free_cards(hole, board)
    ]
    assert len(turns) == 47
    assert derived == pytest.approx(sum(turns) / 47.0, abs=1e-12)


def test_turn_close_to_monte_carlo():
    rng = random.Random(1234)
    for i in range(50):
        hole, board = _random_deal(2, rng)
        derived = sixmax.derive_equity_one(2, hole, board)
        mc = sixmax.hand_equity(hole, board, 20000, 5000 + i)
        assert abs(derived - mc) < 0.02, (hole, board, derived, mc)


def test_derive_equity_one_rejects_bad_input():
    hole = [card(14, 0), card(13, 0)]
    with pytest.raises(ValueError):
        sixmax.derive_equity_one(4, hole, [card(12, 0), card(11, 0), card(10, 0)])
    with pytest.raises(ValueError):
        sixmax.derive_equity_one(2, hole, [card(12, 0), card(11, 0), card(10, 0)])
    with pytest.raises(ValueError):
        # duplicate card across hole and board
        sixmax.derive_equity_one(
            2, hole, [card(14, 0), card(11, 0), card(10, 0), card(2, 1)]
        )


def test_bulk_derive_flop_matches_index_means():
    # The bulk path is a dense pass over the canonical index space. Feed it a
    # synthetic turn table (the real one is 56 MB, so this is cheap) and check
    # sampled flop entries against per-index means computed in Python.
    turn = [(i % 1000) / 1000.0 for i in range(sixmax.hand_index_size(2))]
    flop = sixmax.derive_equity(1, turn)
    assert len(flop) == sixmax.hand_index_size(1)

    rng = random.Random(5)
    for _ in range(200):
        idx = rng.randrange(len(flop))
        hole, board = sixmax.hand_unindex(1, idx)
        free = _free_cards(hole, board)
        assert len(free) == 47
        mean = sum(
            turn[sixmax.hand_index(2, hole, board + [c])] for c in free
        ) / 47.0
        assert abs(flop[idx] - mean) < 1e-6


def test_bulk_derive_rejects_bad_input():
    with pytest.raises(ValueError):
        sixmax.derive_equity(3, [0.0, 0.5, 1.0])  # no street-4 table exists
    with pytest.raises(ValueError):
        sixmax.derive_equity(2, [0.0, 0.5, 1.0])  # next street is the river


# --------------------------------------------------------------------------
# EquityStore file format
# --------------------------------------------------------------------------
def test_store_roundtrip_and_refusals(tmp_path):
    path = str(tmp_path / "tiny.eq")
    flop = [0.0, 0.25, 0.5, 0.75, 1.0]
    turn = [0.125, 0.5]
    river = [0.0, 1.0, 0.5]
    digest = sixmax._write_equity_store_for_test(path, flop, turn, river)

    loaded_flop, loaded_turn, loaded_river, loaded_hash = (
        sixmax._load_equity_store_for_test(path)
    )
    assert loaded_flop == pytest.approx(flop)
    assert loaded_turn == pytest.approx(turn)
    assert loaded_river == pytest.approx(river)
    assert loaded_hash == digest

    # Per-street random access through the read-only lookup helper.
    assert sixmax.equity_store_lookup(path, 1, 2) == pytest.approx(0.5)
    assert sixmax.equity_store_lookup(path, 2, 0) == pytest.approx(0.125)
    assert sixmax.equity_store_lookup(path, 3, 1) == pytest.approx(1.0)
    with pytest.raises(IndexError):
        sixmax.equity_store_lookup(path, 1, len(flop))

    # The production loader is strict about the indexer sizes; the test path
    # only skips that one check.
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(path, strict=True)

    with open(path, "rb") as f:
        original = f.read()

    # One corrupted data byte must be refused by the content hash.
    blob = bytearray(original)
    blob[-1] ^= 0xFF
    corrupt = str(tmp_path / "corrupt.eq")
    with open(corrupt, "wb") as f:
        f.write(blob)
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(corrupt)
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(corrupt, strict=True)

    # Wrong indexer version must be refused even on the non-strict path.
    blob = bytearray(original)
    blob[8:12] = (sixmax.HAND_INDEX_VERSION + 1).to_bytes(4, "little")
    wrong_version = str(tmp_path / "wrong_version.eq")
    with open(wrong_version, "wb") as f:
        f.write(blob)
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(wrong_version)

    # Truncated payload must be refused before anything is allocated.
    truncated = str(tmp_path / "truncated.eq")
    with open(truncated, "wb") as f:
        f.write(original[:-3])
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(truncated)

    # Not our file at all (missing and bad magic).
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(str(tmp_path / "missing.eq"))
    bad_magic = str(tmp_path / "bad_magic.eq")
    blob = bytearray(original)
    blob[0:8] = b"NOTEQ001"
    with open(bad_magic, "wb") as f:
        f.write(blob)
    with pytest.raises(ValueError):
        sixmax._load_equity_store_for_test(bad_magic)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_flags():
    path = os.path.join(_REPO_ROOT, "scripts", "build_equity_tables.py")
    spec = importlib.util.spec_from_file_location("build_equity_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod.parse_args([])
    assert args.threads == 0  # hardware concurrency
    assert args.out.endswith(os.path.join("sixmax", "tables", "equity_v1.bin"))

    args = mod.parse_args(["--out", "x.bin", "--threads", "3"])
    assert (args.out, args.threads) == ("x.bin", 3)
