"""Argument parsing, CSV shape and driver run for the LBR driver."""
import csv
import importlib.util
import os

import pytest
import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
CSV_HEADER = ("checkpoint,act_streets,fold_call_only,decks,bb100,stderr,"
              "seat0_bb100,seat1_bb100,stack_bb,seed,uniform_opponents,"
              "preflop_wp_samples,iterations,config,opp_unseen_rate")


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(CONFIG, "blueprint")


def _trained_checkpoint(tmp_path, iters=120, seed=13):
    """A tiny heads-up blueprint saved to tmp_path (same recipe as test_lbr)."""
    vocab = _load_vocab()
    cfg = sixmax.EngineConfig(num_players=2)
    absn = sixmax.Abstraction(flop_buckets=10, turn_buckets=10,
                              river_buckets=5, equity_rollouts=40,
                              quantile_samples=300, seed=42)
    t = sixmax.BlueprintTrainer(cfg, vocab, absn, num_threads=1, seed=seed)
    t.train(iters)
    path = str(tmp_path / "bp.bin")
    t.save(path, vocab, cfg, absn)
    return path


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "eval_sixmax_lbr", os.path.join(_ROOT, "scripts", "eval_sixmax_lbr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report():
    return {"decks": 32, "seat_bb100": [1.5, -2.0],
            "seat_stderr": [0.5, 0.7], "bb100": -0.25, "stderr": 0.4,
            "opp_unseen": 0, "opp_decisions": 64,
            "per_deck": [0.0] * 32}


def test_csv_row_shape():
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    row = m.report_row("bp.bin", args, _report(), 100.0, 120)
    assert len(row) == len(CSV_HEADER.split(","))
    assert row[0] == "bp.bin"
    assert row[1] == "2,3"
    assert row[2] is False
    assert row[3] == 32
    assert row[4] == -0.25 and row[5] == 0.4
    assert row[6] == 1.5 and row[7] == -2.0
    assert row[8] == 100.0 and row[9] == 1
    assert row[10] is False and row[11] == 500
    assert row[12] == 120 and row[13] == args.config
    assert row[14] == 0.0  # opp_unseen_rate
    assert m.csv_header() == CSV_HEADER


def test_csv_file_round_trips(tmp_path):
    # The writer must keep "2,3" as one field: a bare join would emit more
    # columns than the header.
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    path = tmp_path / "lbr.csv"
    m.write_csv(str(path), "bp.bin", args, _report(), 100.0, 120)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER.split(",")
    assert len(rows) == 2
    assert len(rows[1]) == len(CSV_HEADER.split(","))
    assert rows[1][1] == "2,3"


def test_parse_args_defaults():
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    assert args.decks == 2000
    assert args.seed == 1
    assert args.act_streets == [2, 3]
    assert args.preflop_wp_samples == 500
    assert args.threads == 0
    assert args.fold_call_only is False
    assert args.uniform_opponents is False


def test_parse_args_overrides():
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin", "--decks", "7",
                         "--act-streets", "0,3", "--fold-call-only",
                         "--threads", "2", "--uniform-opponents",
                         "--preflop-wp-samples", "10"])
    assert args.decks == 7
    assert args.act_streets == [0, 3]
    assert args.fold_call_only is True
    assert args.threads == 2
    assert args.uniform_opponents is True
    assert args.preflop_wp_samples == 10


@pytest.mark.parametrize("bad", ["4", "a", "1,x"])
def test_act_streets_parser_rejects(bad):
    m = _load_driver()
    with pytest.raises(SystemExit):
        m.parse_args(["--checkpoint", "bp.bin", "--act-streets", bad])


def test_act_streets_accepts_empty_list():
    m = _load_driver()
    assert m.parse_args(["--checkpoint", "bp.bin",
                         "--act-streets", ""]).act_streets == []


def test_stack_defaults_to_checkpoint():
    m = _load_driver()
    assert m.parse_args(["--checkpoint", "bp.bin"]).stack_bb is None
    assert m.parse_args(["--checkpoint", "bp.bin",
                         "--stack-bb", "200"]).stack_bb == 200.0


def test_main_prints_and_writes_the_run_facts(tmp_path, capsys):
    # The run's provenance (stack, seed, opponent model, samples, iterations,
    # config) must reach both the header and the CSV row.
    m = _load_driver()
    ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "lbr.csv"
    m.main(["--checkpoint", ckpt, "--config", CONFIG, "--decks", "2",
            "--threads", "1", "--seed", "7", "--uniform-opponents",
            "--act-streets", "", "--csv", str(out)])
    printed = capsys.readouterr().out
    assert "stack: 100 BB" in printed
    assert "seed: 7" in printed
    assert "uniform" in printed
    assert "seat 0 (button/SB)" in printed and "seat 1 (BB)" in printed

    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER.split(",")
    assert len(rows) == 2
    assert len(rows[1]) == len(rows[0])
    row = dict(zip(rows[0], rows[1]))
    assert row["seed"] == "7"
    assert row["uniform_opponents"] == "True"
    assert float(row["stack_bb"]) == 100.0
    assert int(row["preflop_wp_samples"]) == 500
    assert int(row["iterations"]) == 120
    assert row["config"] == CONFIG
    assert float(row["opp_unseen_rate"]) == 0.0
