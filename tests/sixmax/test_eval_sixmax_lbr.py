"""Argument parsing + CSV shape for the LBR driver (no engine run)."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "eval_sixmax_lbr", os.path.join(_ROOT, "scripts", "eval_sixmax_lbr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_csv_rows_are_per_seat():
    m = _load_driver()
    report = {"gains": [1.5, -2.0], "stderrs": [0.5, 0.7],
              "max_gain": 1.5, "mean_gain": -0.25, "decks": 32,
              "num_players": 2}
    rows = m.report_rows("bp.bin", report)
    assert rows == [("bp.bin", 0, 1.5, 0.5, 32), ("bp.bin", 1, -2.0, 0.7, 32)]


def test_parse_args_defaults():
    m = _load_driver()
    args = m.parse_args(["--checkpoint", "bp.bin"])
    assert args.decks == 200 and args.seed == 1 and args.samples == 64


def test_stack_defaults_to_checkpoint():
    m = _load_driver()
    assert m.parse_args(["--checkpoint", "bp.bin"]).stack_bb is None
    assert m.parse_args(["--checkpoint", "bp.bin",
                         "--stack-bb", "200"]).stack_bb == 200.0
