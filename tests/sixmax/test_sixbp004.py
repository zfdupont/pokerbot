"""SIXBP004: a map-backed checkpoint that carries the bet-history setting.

Off-mode artifacts keep their historical SIXBP001/SIXBP002/SIXBP003 bytes
exactly; the opt-in last-raise field gets its own magic, always writes the pot
layout, and is restored by every loader.
"""
import shutil
import struct

import pytest

import sixmax

_CFG = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
LEGACY_EDGES = [7.0, 15.0, 40.0]
C4_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]


def _magic(path):
    with open(path, "rb") as f:
        return f.read(8)


@pytest.fixture()
def map_dir(tmp_path, synthetic_bucket_map):
    """Writable dir holding a copy of the session bucket map.

    Checkpoints resolve their map by a hint beside the artifact, so the copy
    lets `sixmax.load_abstraction(path)` work without an explicit path.
    """
    directory = tmp_path / "ck"
    directory.mkdir()
    shutil.copyfile(synthetic_bucket_map, directory / "idxmod.bin")
    return directory


def _train_tiny(directory, vocab, bet_history="off", pot_layout=None,
                iterations=120):
    abs_ = sixmax.Abstraction.from_bucket_map(str(directory / "idxmod.bin"))
    if pot_layout is not None:
        abs_.set_pot_layout(*pot_layout)
    abs_.set_bet_history(bet_history)
    trainer = sixmax.BlueprintTrainer(_CFG, vocab, abs_, num_threads=1,
                                      seed=13)
    trainer.train(iterations)
    path = str(directory / f"tiny_{bet_history}.bin")
    trainer.save(path, vocab, _CFG, abs_)
    return path, abs_


def _bet_history_offset(path):
    """Byte offset of the V4 bet-history mode: after magic .. pot layout."""
    with open(path, "rb") as f:
        f.seek(44)  # hint length, immediately after the map hash
        hint_len = struct.unpack("=I", f.read(4))[0]
        f.seek(44 + 4 + hint_len)
        mode = struct.unpack("=i", f.read(4))[0]
        assert mode in (0, 1)  # the pot layout's mode
        n_edges = struct.unpack("=I", f.read(4))[0]
    return 44 + 4 + hint_len + 4 + 4 + 8 * n_edges


def test_sixbp004_round_trip(map_dir, default_vocab):
    path, abs_ = _train_tiny(map_dir, default_vocab, bet_history="last_raise")
    assert _magic(path) == b"SIXBP004"

    loaded = sixmax.load_abstraction(path)
    assert loaded.bet_history() == "last_raise"
    assert loaded.hash() == abs_.hash()  # whole artifact contract restored
    # V4 always writes the pot layout, legacy or not.
    assert loaded.pot_layout() == ("bb", LEGACY_EDGES)

    strategy = sixmax.BlueprintStrategy.load(path, default_vocab)
    assert strategy.num_infosets() > 0
    assert strategy.abstraction().bet_history() == "last_raise"


def test_sixbp004_writes_nonlegacy_layout_too(map_dir, default_vocab):
    path, abs_ = _train_tiny(map_dir, default_vocab, bet_history="last_raise",
                             pot_layout=("bb", C4_EDGES))
    assert _magic(path) == b"SIXBP004"
    loaded = sixmax.load_abstraction(path)
    assert loaded.pot_layout() == ("bb", C4_EDGES)
    assert loaded.bet_history() == "last_raise"
    assert loaded.hash() == abs_.hash()


def test_off_checkpoint_keeps_sixbp003(map_dir, default_vocab):
    path, _abs = _train_tiny(map_dir, default_vocab, bet_history="off",
                             pot_layout=("bb", C4_EDGES))
    assert _magic(path) == b"SIXBP003"
    assert sixmax.load_abstraction(path).bet_history() == "off"


def test_bet_history_requires_bucket_map(tmp_path, default_vocab):
    abs_ = sixmax.Abstraction(flop_buckets=3, turn_buckets=3, river_buckets=3,
                              equity_rollouts=8, quantile_samples=30, seed=1)
    abs_.set_bet_history("last_raise")
    trainer = sixmax.BlueprintTrainer(_CFG, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(5)
    with pytest.raises((RuntimeError, ValueError),
                       match="bet_history requires a bucket map"):
        trainer.save(str(tmp_path / "no_map.bin"), default_vocab, _CFG, abs_)


def test_unknown_bet_history_mode_refused(map_dir, default_vocab):
    path, _abs = _train_tiny(map_dir, default_vocab, bet_history="last_raise")
    corrupt = str(map_dir / "bad_mode.bin")
    shutil.copyfile(path, corrupt)
    with open(corrupt, "r+b") as f:
        f.seek(_bet_history_offset(corrupt))
        f.write(struct.pack("=i", 9))
    with pytest.raises(RuntimeError, match="unknown bet history"):
        sixmax.load_abstraction(corrupt)


def test_diagnose_reads_the_v4_layout(map_dir, default_vocab):
    """The plateau autopsy decodes pot bits from a SIXBP004 layout too."""
    from scripts.diagnose_blueprint import _pot_layout_for

    path, _abs = _train_tiny(map_dir, default_vocab, bet_history="last_raise",
                             pot_layout=("bb", C4_EDGES))
    assert _magic(path) == b"SIXBP004"
    assert _pot_layout_for(path) == ("bb", C4_EDGES)


def test_resume_refuses_bet_history_mismatch(map_dir, default_vocab):
    path, abs_ = _train_tiny(map_dir, default_vocab, bet_history="last_raise",
                             pot_layout=("bb", C4_EDGES))
    off = sixmax.load_abstraction(path)
    off.set_bet_history("off")  # same map/layout, wrong key space
    with pytest.raises(RuntimeError, match="bet history mismatch"):
        sixmax.resume_blueprint(path, _CFG, default_vocab, off,
                                num_threads=1, seed=13)

    same = sixmax.load_abstraction(path)
    trainer = sixmax.resume_blueprint(path, _CFG, default_vocab, same,
                                      num_threads=1, seed=13)
    trainer.train(10)
    assert trainer.iterations() == 130
