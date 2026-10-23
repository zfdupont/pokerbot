"""Plateau/cap tool: snapshot naming, irregular-spacing pairing, the iteration
cap clamps, the convergence rule, and the extracted run_h2h self-play check."""
import os
import shutil
import sys

import pytest

import sixmax
from scripts.arm_plateau import (
    converged,
    converged_trend,
    iteration_cap,
    pick_pair,
    snapshot_iters,
    trend,
)
from scripts.eval_blueprint_h2h import run_h2h
from scripts.eval_sixmax_lbr import paired_lbr, run_lbr

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


def test_cap_phase3_ceiling():
    # Spec 1d (revised 2026-10-21): Phase 3 arms clamp at 500M, not 150M.
    assert iteration_cap(60_000_000, 33_831, 217_000, phase=3) == 384_854_128
    assert iteration_cap(60_000_000, 40_000, 400_000, phase=3) == 500_000_000
    assert iteration_cap(25_000_000, 40_000, 10_000, phase=3) == 20_000_000


def test_cap_unknown_phase_rejected():
    with pytest.raises(ValueError, match="phase"):
        iteration_cap(60_000_000, 40_000, 400_000, phase=4)


def test_cap_cli_phase3(monkeypatch, capsys):
    from scripts import arm_plateau

    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--cap", "60000000", "40000", "400000",
        "--phase", "3"])
    arm_plateau.main()
    assert capsys.readouterr().out.strip() == "500000000"


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
        "--lag", "100", "--method", "h2h", "--hands", "2", "--seeds", "1",
        "--bb", "2", "--stack", "400"])
    arm_plateau.main()
    out = capsys.readouterr().out
    assert "iters=200" in out
    assert "iters=100" in out
    assert "VERDICT: converged" in out


def test_lag_non_positive_rejected(monkeypatch, capsys, tmp_path):
    """h2h: --lag <= 0 would pair a snapshot with itself; refuse it up front."""
    from scripts import arm_plateau

    for bad in ("0", "-1"):
        monkeypatch.setattr(sys, "argv", [
            "arm_plateau.py", "--snapshots",
            os.path.join(str(tmp_path), "*.bin"), "--method", "h2h",
            "--lag", bad])
        with pytest.raises(SystemExit):
            arm_plateau.main()
        assert "must be positive" in capsys.readouterr().err


def test_lag_still_validated_before_globbing(monkeypatch, capsys, tiny_ckpt):
    """A bad --lag errors even when no snapshot matches the glob."""
    from scripts import arm_plateau

    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", "/nonexistent/*.bin",
        "--method", "h2h", "--lag", "0"])
    with pytest.raises(SystemExit):
        arm_plateau.main()
    assert "must be positive" in capsys.readouterr().err


def test_single_seed_warns_on_stderr(monkeypatch, capsys, tiny_ckpt):
    """One seed has no spread: t is 0, so the verdict always reads converged."""
    from scripts import arm_plateau

    directory = os.path.dirname(tiny_ckpt)
    for iters in (100, 200):
        shutil.copyfile(tiny_ckpt,
                        os.path.join(directory, f"warn_{iters}.bin"))
    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", os.path.join(directory, "warn_*.bin"),
        "--lag", "100", "--method", "h2h", "--hands", "2", "--seeds", "1"])
    arm_plateau.main()
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert ">= 2 seeds" in captured.err
    assert "VERDICT: converged" in captured.out


def test_multi_seed_does_not_warn(monkeypatch, capsys, tiny_ckpt):
    from scripts import arm_plateau

    directory = os.path.dirname(tiny_ckpt)
    for iters in (100, 200):
        shutil.copyfile(tiny_ckpt,
                        os.path.join(directory, f"quiet_{iters}.bin"))
    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", os.path.join(directory, "quiet_*.bin"),
        "--lag", "100", "--method", "h2h", "--hands", "2", "--seeds", "1,2"])
    arm_plateau.main()
    assert "warning" not in capsys.readouterr().err


_CONFIG = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


@pytest.fixture(scope="module")
def older_ckpt(tmp_path_factory, synthetic_bucket_map, default_vocab):
    """Like tiny_ckpt but trained less, so its strategy differs."""
    tmp = tmp_path_factory.mktemp("arm_plateau_older")
    map_copy = tmp / os.path.basename(synthetic_bucket_map)
    shutil.copyfile(synthetic_bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    trainer = sixmax.BlueprintTrainer(cfg, default_vocab, abs_,
                                      num_threads=1, seed=13)
    trainer.train(40)
    path = str(tmp / "older.bin")
    trainer.save(path, default_vocab, cfg, abs_)
    return path


def test_paired_lbr_self_is_zero(tiny_ckpt):
    stats = paired_lbr(tiny_ckpt, tiny_ckpt, _CONFIG, 20, 3, threads=2)
    assert stats["n"] == 20
    assert stats["mean"] == 0.0 and stats["se"] == 0.0 and stats["t"] == 0.0
    assert stats["a_bb100"] == stats["b_bb100"]


def test_paired_lbr_mean_is_difference_of_runs(tiny_ckpt, older_ckpt):
    stats = paired_lbr(tiny_ckpt, older_ckpt, _CONFIG, 30, 5, threads=2)
    a = run_lbr(tiny_ckpt, _CONFIG, 30, 5, threads=1)
    b = run_lbr(older_ckpt, _CONFIG, 30, 5, threads=1)
    # Same seed -> same decks, and runs are thread-count independent.
    assert stats["a_bb100"] == pytest.approx(a["bb100"])
    assert stats["b_bb100"] == pytest.approx(b["bb100"])
    assert stats["mean"] == pytest.approx(a["bb100"] - b["bb100"])
    assert stats["se"] > 0.0
    assert stats["t"] == pytest.approx(stats["mean"] / stats["se"])


def test_paired_lbr_refuses_different_stacks(monkeypatch, tiny_ckpt):
    from scripts import eval_sixmax_lbr as drv

    real = drv.run_lbr
    stacks = iter([200.0, 100.0])

    def run_with_stack(*args, **kwargs):
        report = real(*args, **kwargs)
        report["stack_bb"] = next(stacks)
        return report

    monkeypatch.setattr(drv, "run_lbr", run_with_stack)
    with pytest.raises(ValueError, match="equal stacks"):
        drv.paired_lbr(tiny_ckpt, tiny_ckpt, _CONFIG, 2, 1)


def test_trend_recovers_a_linear_slope():
    # Every deck rises 3 BB/100 per 5M around its own offset: the paired slope
    # is exactly 6 per 10M and deck offsets do not add noise.
    import numpy as np
    rng = np.random.default_rng(0)
    offsets = rng.normal(400, 1600, size=1000)
    iters = [5_000_000, 10_000_000, 15_000_000, 20_000_000]
    per_deck = [offsets + 3.0 * k for k in range(4)]
    stats = trend(iters, per_deck)
    assert stats["slope"] == pytest.approx(6.0)
    assert stats["se"] == pytest.approx(0.0, abs=1e-9)
    assert stats["decks"] == 1000


def test_trend_flat_noise_is_not_significant():
    import numpy as np
    rng = np.random.default_rng(1)
    base = rng.normal(400, 1600, size=4000)
    per_deck = [base + rng.normal(0, 300, size=4000) for _ in range(4)]
    stats = trend([5e6, 10e6, 15e6, 20e6], per_deck)
    assert abs(stats["t"]) < 3.0
    assert stats["se"] > 0


def test_trend_irregular_spacing_uses_real_iterations():
    # Values linear in iterations at uneven spacing (cap snapshot 27.7M).
    iters = [15_000_000, 20_000_000, 25_000_000, 27_705_826]
    per_deck = [[i / 1e6, i / 1e6 + 1.0] for i in iters]
    assert trend(iters, per_deck)["slope"] == pytest.approx(10.0)


def test_trend_refuses_bad_windows():
    with pytest.raises(ValueError, match=">= 2 snapshots"):
        trend([5e6], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="distinct"):
        trend([5e6, 5e6], [[1.0, 2.0], [2.0, 3.0]])


def test_converged_trend_needs_two_flat_windows():
    flat, moving = {"t": 0.5}, {"t": 3.1}
    assert converged_trend([moving, flat, flat]) is True
    assert converged_trend([flat, moving]) is False   # newest window moving
    assert converged_trend([moving, flat]) is False   # one flat window = dip?
    assert converged_trend([flat]) is False           # not enough evidence
    assert converged_trend([{"t": -2.5}, {"t": 0.1}]) is False


def _snapshot_copies(tiny_ckpt, prefix, iters):
    directory = os.path.dirname(tiny_ckpt)
    for i in iters:
        shutil.copyfile(tiny_ckpt, os.path.join(directory, f"{prefix}_{i}.bin"))
    return os.path.join(directory, f"{prefix}_*.bin")


def test_plateau_cli_trend_identical_snapshots_converge(monkeypatch, capsys,
                                                        tiny_ckpt, tmp_path):
    from scripts import arm_plateau

    glob_ = _snapshot_copies(tiny_ckpt, "trend", (100, 200, 300, 400, 500))
    cache = tmp_path / "cache"
    argv = ["arm_plateau.py", "--snapshots", glob_, "--decks", "6",
            "--config", _CONFIG, "--per-deck-dir", str(cache)]
    monkeypatch.setattr(sys, "argv", argv)
    arm_plateau.main()
    out = capsys.readouterr().out
    assert out.count("window ") == 2
    assert "VERDICT: converged" in out
    assert len(list(cache.iterdir())) == 5      # one cached file per snapshot
    # A second check reuses the cache and gives the same verdict.
    monkeypatch.setattr(sys, "argv", argv)
    arm_plateau.main()
    assert "VERDICT: converged" in capsys.readouterr().out


def test_plateau_cli_trend_too_few_snapshots(monkeypatch, capsys, tiny_ckpt,
                                             tmp_path):
    from scripts import arm_plateau

    glob_ = _snapshot_copies(tiny_ckpt, "few", (100, 200, 300, 400))
    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", glob_, "--decks", "4",
        "--config", _CONFIG, "--per-deck-dir", str(tmp_path / "c")])
    arm_plateau.main()
    out = capsys.readouterr().out
    assert "need 5 snapshots for two windows (have 4)" in out
    assert "VERDICT: training" in out


@pytest.mark.parametrize("flag, value, match", [
    ("--decks", "1", "at least 2"), ("--window", "2", "at least 3")])
def test_plateau_cli_trend_rejects_bad_args(monkeypatch, capsys, tiny_ckpt,
                                            flag, value, match):
    from scripts import arm_plateau

    glob_ = _snapshot_copies(tiny_ckpt, "bad", (100, 200))
    monkeypatch.setattr(sys, "argv", [
        "arm_plateau.py", "--snapshots", glob_, "--config", _CONFIG,
        flag, value])
    with pytest.raises(SystemExit):
        arm_plateau.main()
    assert match in capsys.readouterr().err

