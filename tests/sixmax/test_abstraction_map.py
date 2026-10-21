"""Map-backed Abstraction + SIXBP002 checkpoints.

The bucket map used here is synthetic (`index % k` per street), written once
per module at the real indexer sizes through the test-only writer, so these
tests never touch the (gitignored, multi-hundred-MB) real tables.
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
_COUNTS = {1: 7, 2: 5, 3: 3}  # synthetic idx % k per street
_CFG = sixmax.EngineConfig(num_players=2)


def _vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_REPO_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(
        os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml"),
        "blueprint")


VOCAB = _vocab()


def _write_mod_map(path, counts):
    """Write a SIXBM001 map whose bucket is `index % k` on each street."""
    arrays = []
    for street, k in zip((1, 2, 3), counts):
        size = sixmax.hand_index_size(street)
        values = np.empty(size, dtype=np.uint8)
        chunk = 1 << 22
        for start in range(0, size, chunk):
            stop = min(size, start + chunk)
            values[start:stop] = (
                np.arange(start, stop, dtype=np.uint64) % k).astype(np.uint8)
        arrays.append(values)
    sixmax._write_bucket_map_for_test(str(path), *arrays, "percentile")
    return str(path)


@pytest.fixture(scope="module")
def synthetic_map(tmp_path_factory):
    path = tmp_path_factory.mktemp("maps") / "idxmod.bin"
    return _write_mod_map(path, (_COUNTS[1], _COUNTS[2], _COUNTS[3]))


@pytest.fixture(scope="module")
def other_map(tmp_path_factory):
    path = tmp_path_factory.mktemp("maps_other") / "zeros.bin"
    return _write_mod_map(path, (1, 1, 1))


def _deals(street, n=20, seed=7):
    rng = random.Random(seed + street)
    deals = []
    for _ in range(n):
        cards = rng.sample(range(52), 2 + street + 2)
        deals.append((cards[:2], cards[2:]))
    return deals


def _train(abs_, path, iterations=120):
    trainer = sixmax.BlueprintTrainer(_CFG, VOCAB, abs_, num_threads=1,
                                      seed=13)
    trainer.train(iterations)
    trainer.save(path, VOCAB, _CFG, abs_)
    return trainer


def _magic(path):
    with open(path, "rb") as f:
        return f.read(8)


def test_map_abstraction_bucket_equals_map_lookup(synthetic_map):
    abs_ = sixmax.Abstraction.from_bucket_map(synthetic_map)
    assert abs_.is_map()
    bucket_map = sixmax.BucketMap.open(synthetic_map)
    for street in (1, 2, 3):
        assert abs_.num_buckets(street) == _COUNTS[street]
        assert abs_.num_buckets(street) == bucket_map.num_buckets(street)
        for hole, board in _deals(street):
            expected = bucket_map.bucket(
                street, sixmax.hand_index(street, hole, board))
            assert abs_.bucket(hole, board) == expected

    hole, _ = _deals(3)[0]
    with pytest.raises(ValueError):
        abs_.bucket(hole, [])  # preflop is not bucketed

    legacy = sixmax.Abstraction(flop_buckets=4, turn_buckets=4,
                                river_buckets=2, equity_rollouts=8,
                                quantile_samples=40)
    assert not legacy.is_map()


def test_map_abstraction_hash_differs_by_map(synthetic_map, other_map):
    a1 = sixmax.Abstraction.from_bucket_map(synthetic_map)
    a2 = sixmax.Abstraction.from_bucket_map(synthetic_map)
    b = sixmax.Abstraction.from_bucket_map(other_map)
    assert a1.hash() == a2.hash()
    assert a1.hash() != b.hash()

    legacy = sixmax.Abstraction(flop_buckets=_COUNTS[1],
                                turn_buckets=_COUNTS[2],
                                river_buckets=_COUNTS[3],
                                equity_rollouts=8, quantile_samples=40)
    assert a1.hash() != legacy.hash()


def test_sixbp002_roundtrip(tmp_path, synthetic_map, monkeypatch):
    monkeypatch.setenv("SIXMAX_BUCKET_MAP", synthetic_map)
    abs_ = sixmax.Abstraction.from_bucket_map(synthetic_map)
    path = str(tmp_path / "bp002.bin")
    trainer = _train(abs_, path)
    assert _magic(path) == b"SIXBP002"

    strategy = sixmax.BlueprintStrategy.load(path, VOCAB)
    assert strategy.iterations() == trainer.iterations()
    assert strategy.num_players() == 2
    assert strategy.abstraction().is_map()
    assert strategy.abstraction().hash() == abs_.hash()

    seen = 0
    for key in trainer.keys():
        expected = trainer.average_strategy(key)
        if expected:
            assert strategy.probs(key) == expected
            seen += 1
    assert seen > 0

    # load_abstraction resolves the map the same way
    loaded_abs = sixmax.load_abstraction(path)
    assert loaded_abs.is_map()
    assert loaded_abs.hash() == abs_.hash()


def test_sixbp002_refuses_other_map(tmp_path, synthetic_map, other_map,
                                    monkeypatch):
    abs_ = sixmax.Abstraction.from_bucket_map(synthetic_map)
    path = str(tmp_path / "bp002.bin")
    _train(abs_, path, iterations=30)

    monkeypatch.setenv("SIXMAX_BUCKET_MAP", other_map)
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        sixmax.BlueprintStrategy.load(path, VOCAB)
    with pytest.raises(RuntimeError, match="map hash mismatch"):
        sixmax.load_abstraction(path)


def test_sixbp002_missing_map_lists_paths(tmp_path, synthetic_map,
                                          monkeypatch):
    map_copy = tmp_path / "present.bin"
    shutil.copyfile(synthetic_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    path = str(tmp_path / "bp002.bin")
    _train(abs_, path, iterations=20)
    map_copy.unlink()  # the map the checkpoint needs is gone

    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        sixmax.BlueprintStrategy.load(path, VOCAB)
    message = str(excinfo.value)
    assert "present.bin" in message
    assert str(tmp_path / "present.bin") in message             # ckpt-relative
    assert os.path.join("sixmax", "tables", "present.bin") in message  # default

    tables_dir = tmp_path / "elsewhere"
    monkeypatch.setenv("SIXMAX_TABLES_DIR", str(tables_dir))
    with pytest.raises(RuntimeError) as excinfo:
        sixmax.BlueprintStrategy.load(path, VOCAB)
    assert str(tables_dir / "present.bin") in str(excinfo.value)


def test_sixbp001_still_loads(tmp_path):
    abs_ = sixmax.Abstraction(flop_buckets=6, turn_buckets=6,
                              river_buckets=4, equity_rollouts=8,
                              quantile_samples=80, seed=42)
    path = str(tmp_path / "bp001.bin")
    trainer = _train(abs_, path, iterations=40)
    assert _magic(path) == b"SIXBP001"

    strategy = sixmax.BlueprintStrategy.load(path, VOCAB)
    assert not strategy.abstraction().is_map()
    assert strategy.abstraction().hash() == abs_.hash()
    assert strategy.iterations() == trainer.iterations()
    seen = 0
    for key in trainer.keys():
        expected = trainer.average_strategy(key)
        if expected:
            assert strategy.probs(key) == expected
            seen += 1
    assert seen > 0
