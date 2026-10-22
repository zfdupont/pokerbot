"""lbr_sweep.py: arm parsing, snapshot ordering, resume-skipping, and the
per-deck file agreeing with the summary row."""
import argparse
import json
import os
import shutil

import numpy as np
import pytest

import sixmax
from scripts import lbr_sweep

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_parse_arm():
    assert lbr_sweep.parse_arm("c1=cfg.toml:x/c1_*.bin") == (
        "c1", "cfg.toml", "x/c1_*.bin")
    for bad in ("c1", "c1=cfg.toml", "=cfg.toml:g", "c1=:g"):
        with pytest.raises(argparse.ArgumentTypeError):
            lbr_sweep.parse_arm(bad)


def test_snapshots_sorted_and_unsuffixed_ignored(tmp_path):
    for name in ("a_20.bin", "a_5.bin", "a.bin", "a_100.bin"):
        (tmp_path / name).write_bytes(b"")
    got = lbr_sweep.snapshots(str(tmp_path / "a*.bin"))
    assert [i for i, _ in got] == [5, 20, 100]


@pytest.fixture(scope="module")
def two_snapshots(tmp_path_factory, synthetic_bucket_map, default_vocab):
    tmp = tmp_path_factory.mktemp("lbr_sweep")
    shutil.copyfile(synthetic_bucket_map,
                    tmp / os.path.basename(synthetic_bucket_map))
    abs_ = sixmax.Abstraction.from_bucket_map(
        str(tmp / os.path.basename(synthetic_bucket_map)))
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    t = sixmax.BlueprintTrainer(cfg, default_vocab, abs_, num_threads=1,
                                seed=3)
    for iters in (40, 80):
        t.train(40)
        t.save(str(tmp / ("arm_%d.bin" % iters)), default_vocab, cfg, abs_)
    return tmp


def test_sweep_writes_rows_and_resumes(two_snapshots, capsys):
    out = two_snapshots / "sweep.jsonl"
    decks_dir = two_snapshots / "decks"
    argv = ["--arm", "x=%s:%s" % (_CONFIG, two_snapshots / "arm_*.bin"),
            "--decks", "6", "--seed", "2", "--threads", "2",
            "--out", str(out), "--per-deck-dir", str(decks_dir)]
    lbr_sweep.main(argv)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["iterations"] for r in rows] == [40, 80]
    for r in rows:
        assert r["arm"] == "x" and r["decks"] == 6 and r["seed"] == 2
        per_deck = np.load(r["per_deck_file"])
        assert per_deck.shape == (6,)
        assert per_deck.mean() == pytest.approx(r["bb100"])
    capsys.readouterr()
    lbr_sweep.main(argv)  # everything recorded: nothing reruns
    assert len(out.read_text().splitlines()) == 2
    assert capsys.readouterr().out == ""
