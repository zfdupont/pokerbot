"""Argument parsing + CSV shape for the LBR driver (no engine run)."""
import csv
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_HEADER = ("checkpoint,act_streets,fold_call_only,decks,bb100,stderr,"
              "seat0_bb100,seat1_bb100")


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "eval_sixmax_lbr", os.path.join(_ROOT, "scripts", "eval_sixmax_lbr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report():
    return {"decks": 32, "seat_bb100": [1.5, -2.0],
            "seat_stderr": [0.5, 0.7], "bb100": -0.25, "stderr": 0.4,
            "per_deck": [0.0] * 32}


def test_csv_row_shape():
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    row = m.report_row("bp.bin", args, _report())
    assert len(row) == len(CSV_HEADER.split(","))
    assert row[0] == "bp.bin"
    assert row[1] == "2,3"
    assert row[2] is False
    assert row[3] == 32
    assert row[4] == -0.25 and row[5] == 0.4
    assert row[6] == 1.5 and row[7] == -2.0
    assert m.csv_header() == CSV_HEADER


def test_csv_file_round_trips(tmp_path):
    # The writer must keep "2,3" as one field: a bare join would emit nine
    # columns under the eight-column header.
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    path = tmp_path / "lbr.csv"
    m.write_csv(str(path), "bp.bin", args, _report())
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER.split(",")
    assert len(rows) == 2
    assert len(rows[1]) == 8
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
