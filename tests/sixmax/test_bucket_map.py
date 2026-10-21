"""Bucket maps: percentile + potential-aware builders and the SIXBM001 format.

The builders are exercised on small synthetic inputs, so these tests never
need the multi-hundred-MB equity store; the map file format is exercised at
full indexer sizes through the test-only writer. `build_bucket_map` itself is
only checked for its refusals (it needs a real store to produce a map).
"""
import importlib.util
import os
import random
import shutil

import numpy as np
import pytest

import sixmax

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SIZES = (1_286_792, 13_960_050, 123_156_254)


def card(rank, suit):
    return (rank - 2) * 4 + suit


def _random_deal(street, rng):
    deck = list(range(52))
    rng.shuffle(deck)
    return deck[:2], deck[2 : 2 + street + 2]


def _free_cards(hole, board):
    used = set(hole) | set(board)
    return [c for c in range(52) if c not in used]


def _histogram(values, bins):
    hist = [0.0] * bins
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        hist[index] += 1.0 / len(values)
    return hist


# --------------------------------------------------------------------------
# percentile_buckets
# --------------------------------------------------------------------------
def test_percentile_ties_never_split():
    eq = [0.1, 0.5, 0.5, 0.5, 0.9]
    buckets = sixmax._percentile_buckets(eq, [1] * len(eq), 3)
    assert buckets[1] == buckets[2] == buckets[3]
    assert set(buckets) == {0, 1, 2}
    assert buckets[0] < buckets[1] < buckets[4]

    # A heavy tie block that swallows a population boundary still cannot be
    # split; the remaining buckets keep at least one distinct value each.
    eq = [0.2] * 50 + [0.4] * 2 + [0.6] * 2 + [0.8] * 50
    buckets = sixmax._percentile_buckets(eq, [1] * len(eq), 3)
    assert all(b == buckets[0] for b in buckets[:50])
    assert all(b == buckets[-1] for b in buckets[100:])
    assert set(buckets) == {0, 1, 2}


def test_percentile_weighted_equal_population():
    rng = random.Random(20261020)
    n = 10000
    eq = sorted(rng.sample(range(1_000_000), n))  # distinct
    eq = [value / 1_000_000 for value in eq]
    weight = [rng.randint(1, 24) for _ in range(n)]  # real multiplicity range

    buckets = sixmax._percentile_buckets(eq, weight, 50)
    assert set(buckets) == set(range(50))

    population = [0] * 50
    for bucket, w in zip(buckets, weight):
        population[bucket] += w
    assert max(population) / min(population) < 1.1, population


def test_percentile_monotone():
    rng = random.Random(7)
    values = sorted(rng.randrange(0, 40) for _ in range(2000))  # many ties
    eq = [v / 40.0 for v in values]
    weight = [rng.randint(1, 24) for _ in range(len(eq))]

    buckets = sixmax._percentile_buckets(eq, weight, 12)
    assert set(buckets) == set(range(12))
    for left, right in zip(buckets, buckets[1:]):
        assert left <= right
    # identical equity always shares a bucket
    for i in range(1, len(eq)):
        if eq[i] == eq[i - 1]:
            assert buckets[i] == buckets[i - 1]


def test_percentile_rejects_bad_k():
    with pytest.raises(ValueError):
        sixmax._percentile_buckets([0.25, 0.5, 0.75], [1, 1, 1], 0)
    with pytest.raises(ValueError):
        sixmax._percentile_buckets([0.25, 0.5, 0.75], [1, 1, 1], 256)
    with pytest.raises(ValueError):
        sixmax._percentile_buckets([0.25, 0.5], [1, 1], 3)  # too few values
    with pytest.raises(ValueError):
        sixmax._percentile_buckets([0.25, 0.5, 0.75], [1], 2)  # length mismatch


# --------------------------------------------------------------------------
# kmeans_buckets
# --------------------------------------------------------------------------
def _blob(rng, bins, index, count):
    """Histograms concentrated on bin `index`, with a random neighbour split."""
    hists = []
    for _ in range(count):
        split = rng.uniform(0.1, 0.9)
        hist = [0.0] * bins
        hist[index] += split
        hist[index + 1] += 1.0 - split
        hists.append(hist)
    return hists


def test_kmeans_deterministic_and_separates():
    rng = random.Random(11)
    hists = _blob(rng, 10, 0, 100) + _blob(rng, 10, 8, 100)
    weight = [rng.randint(1, 24) for _ in range(len(hists))]

    labels = sixmax._kmeans_buckets(hists, weight, 2, 1234, 20)
    again = sixmax._kmeans_buckets(hists, weight, 2, 1234, 20)
    assert labels == again  # deterministic given the seed
    assert len(set(labels[:100])) == 1
    assert len(set(labels[100:])) == 1
    assert labels[0] != labels[100]  # the blobs land in different buckets


def test_kmeans_thread_count_independent():
    rng = random.Random(4242)
    n, bins = 3000, 8
    hists = []
    for _ in range(n):
        raw = [rng.random() for _ in range(bins)]
        total = sum(raw)
        hists.append([value / total for value in raw])  # proper distributions
    weight = [rng.randint(1, 24) for _ in range(n)]

    single = sixmax._kmeans_buckets(hists, weight, 5, 99, 20, num_threads=1)
    multi = sixmax._kmeans_buckets(hists, weight, 5, 99, 20, num_threads=4)
    assert single == multi
    assert single == sixmax._kmeans_buckets(hists, weight, 5, 99, 20)


def test_kmeans_rejects_bad_input():
    hists = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        sixmax._kmeans_buckets(hists, [1, 1], 0, 0, 20)
    with pytest.raises(ValueError):
        sixmax._kmeans_buckets(hists, [1, 1], 3, 0, 20)  # k > points
    with pytest.raises(ValueError):
        sixmax._kmeans_buckets(hists, [1], 2, 0, 20)  # weight mismatch
    with pytest.raises(ValueError):
        sixmax._kmeans_buckets([[1.0, 0.0], [0.0, 1.0, 0.0]], [1, 1], 2, 0, 1)
    with pytest.raises(ValueError):
        sixmax._kmeans_buckets(hists, [1, 1], 2, 0, 0)  # iters must be >= 1


# --------------------------------------------------------------------------
# potential-aware features (turn histogram / flop distribution)
# --------------------------------------------------------------------------
def test_turn_potential_hist_matches_per_hand_river_equities():
    rng = random.Random(99)
    hole, board = _random_deal(2, rng)
    free = _free_cards(hole, board)
    assert len(free) == 46

    rivers = [sixmax.river_equity_exact(hole, board + [c]) for c in free]
    expected = _histogram(rivers, 50)
    got = sixmax._turn_potential_hist(hole, board, 50)
    assert got == pytest.approx(expected, abs=1e-6)
    assert sum(got) == pytest.approx(1.0)


def test_flop_potential_hist_counts_turn_buckets():
    rng = random.Random(5)
    hole, board = _random_deal(1, rng)
    k = 5
    turn_buckets = (
        np.arange(sixmax.hand_index_size(2), dtype=np.uint32) % k
    ).astype(np.uint8)
    order = [2, 0, 4, 1, 3]  # bucket id -> feature dimension

    got = sixmax._flop_potential_hist(hole, board, turn_buckets, k, order)
    assert len(got) == k

    expected = [0.0] * k
    free = _free_cards(hole, board)
    assert len(free) == 47
    for c in free:
        idx = sixmax.hand_index(2, hole, board + [c])
        expected[order[idx % k]] += 1.0 / 47.0
    assert got == pytest.approx(expected, abs=1e-6)
    assert sum(got) == pytest.approx(1.0)

    identity = sixmax._flop_potential_hist(hole, board, turn_buckets, k, [])
    assert identity == pytest.approx(
        [expected[order[bucket]] for bucket in range(k)], abs=1e-6
    )


# --------------------------------------------------------------------------
# SIXBM001 file format
# --------------------------------------------------------------------------
def _full_bucket_arrays(k):
    arrays = {}
    for street, size in zip((1, 2, 3), _SIZES):
        arrays[street] = np.tile(np.arange(k, dtype=np.uint8), size // k + 1)[
            :size
        ]
    return arrays


@pytest.fixture(scope="module")
def map_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("bucket_map") / "map.bin")
    arrays = _full_bucket_arrays(7)
    sixmax._write_bucket_map_for_test(
        path, arrays[1], arrays[2], arrays[3], "percentile"
    )
    return path


def test_map_roundtrip(map_path):
    m = sixmax.BucketMap.open(map_path)
    for street, size in zip((1, 2, 3), _SIZES):
        assert m.num_buckets(street) == 7
        for idx in (0, 1, 12_345, size - 1):
            assert m.bucket(street, idx) == idx % 7
    assert m.content_hash() != 0
    assert m.equity_hash() == 0  # the test writer records no equity store
    assert m.method() == "percentile"
    assert m.params() == ""
    assert m.path() == os.path.abspath(map_path)
    assert sixmax.BucketMap.open(map_path).content_hash() == m.content_hash()

    with pytest.raises(IndexError):
        m.bucket(1, _SIZES[0])
    with pytest.raises(ValueError):
        m.bucket(4, 0)
    with pytest.raises(ValueError):
        m.num_buckets(0)


@pytest.mark.parametrize("damage", ["truncate", "corrupt", "version"])
def test_map_refuses_damaged_files(tmp_path, map_path, damage):
    damaged = str(tmp_path / f"{damage}.bin")
    shutil.copyfile(map_path, damaged)
    size = os.path.getsize(damaged)
    with open(damaged, "r+b") as f:
        if damage == "truncate":
            f.truncate(size - 1)
        elif damage == "corrupt":
            f.seek(size - 1)
            last = f.read(1)
            f.seek(size - 1)
            f.write(bytes([last[0] ^ 0xFF]))
        else:
            f.seek(8)
            f.write((sixmax.HAND_INDEX_VERSION + 1).to_bytes(4, "little"))
    with pytest.raises(ValueError):
        sixmax.BucketMap.open(damaged)


def test_map_refuses_header_count_mismatch(tmp_path, map_path):
    # The content hash covers only the arrays, so a header bucket count that
    # disagrees with the data (still inside [1, 255]) must be caught by the
    # count cross-check in open().
    damaged = str(tmp_path / "count_mismatch.bin")
    shutil.copyfile(map_path, damaged)
    with open(damaged, "r+b") as f:
        f.seek(36)  # uint32 buckets[0] in the fixed header
        f.write((6).to_bytes(4, "little"))
    with pytest.raises(ValueError):
        sixmax.BucketMap.open(damaged)


def test_map_refuses_bad_magic_and_missing_file(tmp_path, map_path):
    bad_magic = str(tmp_path / "bad_magic.bin")
    shutil.copyfile(map_path, bad_magic)
    with open(bad_magic, "r+b") as f:
        f.seek(0)
        f.write(b"NOTBM001")
    with pytest.raises(ValueError):
        sixmax.BucketMap.open(bad_magic)
    with pytest.raises(ValueError):
        sixmax.BucketMap.open(str(tmp_path / "missing.bin"))


# --------------------------------------------------------------------------
# build_bucket_map refusals (a real build needs the equity store)
# --------------------------------------------------------------------------
def test_build_bucket_map_refuses_bad_arguments(tmp_path):
    equity = str(tmp_path / "tiny.eq")
    sixmax._write_equity_store_for_test(equity, [0.25], [0.5], [0.75])
    out = str(tmp_path / "map.bin")

    # Equity store sizes do not match the indexer: refused by the strict load.
    with pytest.raises(ValueError):
        sixmax.build_bucket_map(equity, out, "percentile", [50, 50, 20])
    with pytest.raises(ValueError):
        sixmax.build_bucket_map(equity, out, "percentile", [0, 50, 20])
    with pytest.raises(ValueError):
        sixmax.build_bucket_map(equity, out, "percentile", [50, 50, 256])
    with pytest.raises(ValueError):
        sixmax.build_bucket_map(equity, out, "nonsense", [50, 50, 20])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_flags():
    path = os.path.join(_REPO_ROOT, "scripts", "build_bucket_map.py")
    spec = importlib.util.spec_from_file_location("build_bucket_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod.parse_args([])
    assert args.method == "percentile"
    assert args.counts == [50, 50, 20]
    assert args.seed == 0
    assert args.threads == 0
    assert args.equity.endswith(
        os.path.join("sixmax", "tables", "equity_v1.bin")
    )
    assert args.out.endswith(os.path.join("sixmax", "tables", "bucket_map.bin"))

    args = mod.parse_args(
        ["--equity", "e.bin", "--out", "o.bin", "--method", "potential",
         "--counts", "3,4,5", "--seed", "9", "--threads", "2"]
    )
    assert (args.equity, args.out, args.method) == (
        "e.bin", "o.bin", "potential"
    )
    assert args.counts == [3, 4, 5]
    assert (args.seed, args.threads) == (9, 2)

    with pytest.raises(SystemExit):
        mod.parse_args(["--counts", "1,2"])  # needs flop,turn,river
    with pytest.raises(SystemExit):
        mod.parse_args(["--counts", "1,2,256"])
