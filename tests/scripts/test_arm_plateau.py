"""Plateau/cap tool: snapshot naming, irregular-spacing pairing, the iteration
cap clamps, the convergence rule, and the extracted run_h2h self-play check."""
import os
import shutil
import sys

import pytest

import sixmax
from scripts.arm_plateau import (
    converged,
    iteration_cap,
    pick_pair,
    snapshot_iters,
)
from scripts.eval_blueprint_h2h import run_h2h

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_snapshot_iters():
    assert snapshot_iters("x/hu200_c1_15000000.bin") == 15_000_000


def test_snapshot_iters_rejects_unsuffixed_name():
    with pytest.raises(ValueError):
        snapshot_iters("sixmax/checkpoints/hu_blueprint.bin")


def test_pick_pair_regular():
    paths = [f"x/c0_{i}.bin" for i in (20_000_000, 5_000_000,
                                       15_000_000, 10_000_000)]
    assert pick_pair(paths, 10_000_000) == ("x/c0_20000000.bin",
                                            "x/c0_10000000.bin")


def test_pick_pair_irregular():
    """Newest snapshot at or below N-lag, not N-lag itself (Review Focus 5)."""
    paths = ["x/c0_5000000.bin", "x/c0_12000000.bin", "x/c0_21000000.bin"]
    assert pick_pair(paths, 10_000_000) == ("x/c0_21000000.bin",
                                            "x/c0_5000000.bin")


def test_pick_pair_none():
    with pytest.raises(ValueError, match="no snapshot at or below"):
        pick_pair(["x/c0_5000000.bin", "x/c0_10000000.bin"], 10_000_000)


def test_pick_pair_ignores_unsuffixed_paths():
    """The base checkpoint (and any other name) is not a snapshot."""
    paths = ["sixmax/checkpoints/hu_blueprint.bin", "x/c0_10000000.bin",
             "x/c0_20000000.bin"]
    assert pick_pair(paths, 10_000_000) == ("x/c0_20000000.bin",
                                            "x/c0_10000000.bin")


def test_pick_pair_no_snapshots_at_all():
    with pytest.raises(ValueError, match="no snapshots"):
        pick_pair(["sixmax/checkpoints/hu_blueprint.bin"], 10_000_000)


def test_cap_worked_example():
    assert iteration_cap(25_000_000, 40_000, 120_000) == 75_000_000


def test_cap_clamps():
    assert iteration_cap(25_000_000, 40_000, 10_000) == 20_000_000
    assert iteration_cap(60_000_000, 40_000, 400_000) == 150_000_000


def test_converged_rule():
    assert converged({"mean": 3.0, "t": 2.5}) is False
    assert converged({"mean": 3.0, "t": 1.2}) is True
    assert converged({"mean": -4.0, "t": -3.0}) is True


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, synthetic_bucket_map, default_vocab):
    """A map-backed 200 BB HU checkpoint whose bucket map sits beside it."""
    tmp = tmp_path_factory.mktemp("arm_plateau")
    map_copy = tmp / os.path.basename(synthetic_bucket_map)
    shutil.copyfile(synthetic_bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    trainer = sixmax.BlueprintTrainer(cfg, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(120)
    path = str(tmp / "tiny.bin")
    trainer.save(path, default_vocab, cfg, abs_)
    return path


def test_run_h2h_self_is_zero(tiny_ckpt):
    stats = run_h2h(tiny_ckpt, tiny_ckpt, hands=4, seeds=[1], bb=2, stack=400)
    assert stats["mean"] == 0.0
    assert stats["per_seed"] == [0.0]
    # n=1 has no sample spread: sd/se/t must be 0, not NaN.
    assert stats["sd"] == 0.0
    assert stats["se"] == 0.0
    assert stats["t"] == 0.0


def test_infosets_cli_matches_strategy(monkeypatch, capsys, tiny_ckpt):
    from scripts import arm_plateau
    from agents.sixmax_agent import SixmaxDeployStrategy

    deploy = SixmaxDeployStrategy.load(
        tiny_ckpt, os.path.join(_ROOT, "sixmax", "configs", "default.toml"))
    monkeypatch.setattr(sys, "argv",
                        ["arm_plateau.py", "--infosets", tiny_ckpt])
    arm_plateau.main()
    assert capsys.readouterr().out.strip() == str(deploy.num_infosets)


def test_cap_cli_prints_only_integer(monkeypatch, capsys):
    from scripts import arm_plateau

    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--cap", "25000000", "40000", "120000"])
    arm_plateau.main()
    assert capsys.readouterr().out.strip() == "75000000"


def test_plateau_cli_errors_without_snapshots(monkeypatch, capsys, tmp_path):
    from scripts import arm_plateau

    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", os.path.join(str(tmp_path), "*.bin")])
    with pytest.raises(SystemExit):
        arm_plateau.main()
    assert "no snapshots" in capsys.readouterr().err


def test_plateau_cli_self_snapshots_converge(monkeypatch, capsys, tiny_ckpt):
    """Two identical snapshots at a 100-iteration lag read as converged."""
    from scripts import arm_plateau

    directory = os.path.dirname(tiny_ckpt)
    older = os.path.join(directory, "tiny_100.bin")
    newer = os.path.join(directory, "tiny_200.bin")
    shutil.copyfile(tiny_ckpt, older)
    shutil.copyfile(tiny_ckpt, newer)
    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", os.path.join(directory, "tiny_*.bin"),
        "--lag", "100", "--hands", "2", "--seeds", "1",
        "--bb", "2", "--stack", "400"])
    arm_plateau.main()
    out = capsys.readouterr().out
    assert "iters=200" in out
    assert "iters=100" in out
    assert "VERDICT: converged" in out
