"""SIXBP003: map-backed checkpoints bound to a non-legacy pot layout.

Legacy layouts keep the SIXBP002/SIXBP001 bytes exactly. A non-legacy layout
writes SIXBP003 (the SIXBP002 prefix plus mode/count/edges) and every consumer
(load, resume, deploy agent, trainer) must agree on it.
"""
import os
import shutil
import struct
import subprocess
import sys

import pytest

import sixmax

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "train_sixmax.py")
_TOML = os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml")
_CFG = sixmax.EngineConfig(num_players=2)

LEGACY_EDGES = [7.0, 15.0, 40.0]
C1_EDGES = [4.0, 8.0, 16.0, 40.0, 100.0]
C2_EDGES = [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]


def _magic(path):
    with open(path, "rb") as f:
        return f.read(8)


def _layout_offset(path):
    """Byte offset of the V3 mode field: after magic .. map hint."""
    with open(path, "rb") as f:
        f.seek(44)  # hint length, immediately after the map hash
        hint_len = struct.unpack("=I", f.read(4))[0]
    return 44 + 4 + hint_len


def _train(abs_, vocab, path, iterations=120):
    trainer = sixmax.BlueprintTrainer(_CFG, vocab, abs_, num_threads=1,
                                      seed=13)
    trainer.train(iterations)
    trainer.save(path, vocab, _CFG, abs_)
    return trainer


@pytest.fixture(scope="module")
def c1_checkpoint(tmp_path_factory, synthetic_bucket_map, default_vocab):
    """A trained C1 checkpoint beside a copy of its bucket map."""
    directory = tmp_path_factory.mktemp("sixbp003_c1")
    map_copy = directory / "idxmod.bin"
    shutil.copyfile(synthetic_bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    abs_.set_pot_layout("bb", C1_EDGES)
    ckpt = str(directory / "c1.bin")
    _train(abs_, default_vocab, ckpt)
    return {"dir": str(directory), "map": str(map_copy), "ckpt": ckpt}


def test_c1_writes_sixbp003(c1_checkpoint):
    path = c1_checkpoint["ckpt"]
    assert _magic(path) == b"SIXBP003"
    assert sixmax.load_abstraction(path).pot_layout() == ("bb", C1_EDGES)

    _iterations, records = sixmax.dump_infosets(path)
    keys = [key for key, _probs, _mass, _reg in records]
    assert keys
    # Non-legacy keys leave the old pot bits (18-19) zero.
    assert all(((key >> 18) & 3) == 0 for key in keys)
    assert any(((key >> 8) & 3) > 0 and ((key >> 26) & 15) > 0
               for key in keys)


def test_legacy_layout_still_sixbp002(tmp_path, synthetic_bucket_map,
                                      default_vocab):
    abs_ = sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)
    path = str(tmp_path / "legacy_map.bin")
    _train(abs_, default_vocab, path, iterations=30)
    assert _magic(path) == b"SIXBP002"


def test_resume_refuses_layout_mismatch(tmp_path, c1_checkpoint,
                                        synthetic_bucket_map, default_vocab):
    path = c1_checkpoint["ckpt"]
    legacy = sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)
    with pytest.raises(RuntimeError, match="pot layout mismatch"):
        sixmax.resume_blueprint(path, _CFG, default_vocab, legacy,
                                num_threads=1, seed=13)

    c2 = sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)
    c2.set_pot_layout("bb", C2_EDGES)
    with pytest.raises(RuntimeError, match="pot layout mismatch"):
        sixmax.resume_blueprint(path, _CFG, default_vocab, c2,
                                num_threads=1, seed=13)

    # Happy path: the abstraction loaded from the checkpoint carries its
    # stored layout, so a normal resume continues in the same key space.
    same = sixmax.load_abstraction(path)
    trainer = sixmax.resume_blueprint(path, _CFG, default_vocab, same,
                                      num_threads=1, seed=13)
    trainer.train(10)
    resumed = str(tmp_path / "resumed.bin")
    trainer.save(resumed, default_vocab, _CFG, same)
    assert _magic(resumed) == b"SIXBP003"
    assert trainer.iterations() == 130
    assert sixmax.load_abstraction(
        resumed, bucket_map=c1_checkpoint["map"]).pot_layout() == ("bb", C1_EDGES)


def test_nonlegacy_without_map_refused(tmp_path, default_vocab):
    abs_ = sixmax.Abstraction(flop_buckets=4, turn_buckets=4,
                              river_buckets=2, equity_rollouts=8,
                              quantile_samples=40)
    abs_.set_pot_layout("bb", C1_EDGES)
    trainer = sixmax.BlueprintTrainer(_CFG, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(5)
    with pytest.raises((RuntimeError, ValueError),
                       match="requires a bucket map"):
        trainer.save(str(tmp_path / "no_map.bin"), default_vocab, _CFG, abs_)


def test_corrupt_layout_field(tmp_path, c1_checkpoint):
    source = c1_checkpoint["ckpt"]

    truncated = str(tmp_path / "truncated_layout.bin")
    shutil.copyfile(source, truncated)
    offset = _layout_offset(truncated)
    with open(truncated, "r+b") as f:
        f.truncate(offset + 4)  # mode present, edge count cut off
    with pytest.raises(RuntimeError, match="truncated"):
        sixmax.load_abstraction(truncated)

    too_long = str(tmp_path / "too_long_layout.bin")
    shutil.copyfile(source, too_long)
    offset = _layout_offset(too_long)
    with open(too_long, "r+b") as f:
        f.seek(offset + 4)  # n sits after the int32 mode
        f.write(struct.pack("=I", 16))
    with pytest.raises(RuntimeError, match="pot layout too long"):
        sixmax.load_abstraction(too_long)


def test_agent_plays_c1_checkpoint(monkeypatch, c1_checkpoint):
    """The deploy bridge keys through the checkpoint's own pot layout."""
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from scripts.eval_hu_sanity import run_duplicate_match

    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)

    seen = []
    original = SixmaxDeployStrategy.decide

    def spy(self, **kwargs):
        result = original(self, **kwargs)
        if kwargs["street"] > 0 and kwargs.get("eff_stack_bb") is not None:
            abs_ = self._strategy.abstraction()
            card = self._card_bucket(kwargs["hole"], kwargs["board"],
                                     kwargs["street"])
            key = abs_.pack_key(
                card, kwargs["street"], list(kwargs["raises_per_street"]),
                kwargs["pot_bb"], kwargs["eff_stack_bb"], kwargs["live"],
                kwargs["after"])
            seen.append(self._strategy.probs(key))
        return result

    monkeypatch.setattr(SixmaxDeployStrategy, "decide", spy)

    ckpt = c1_checkpoint["ckpt"]

    def factory(_seat):
        return SixmaxAgent(ckpt, config_toml=_TOML)

    run_duplicate_match(factory, factory, hands=10, seed=1, bb=2, stack=400)
    assert seen
    assert any(probs for probs in seen)


def test_train_cli_pot_edges_requires_map(tmp_path):
    toml = tmp_path / "no_map.toml"
    toml.write_text(
        "[abstraction]\n"
        "pot_edges = [4.0, 8.0, 16.0, 40.0, 100.0]\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 10\nnum_threads = 1\n")
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(tmp_path / "bp.bin")],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=600)
    assert result.returncode != 0
    assert "pot_edges requires bucket_map" in result.stderr
