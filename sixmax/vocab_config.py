"""Build an ActionVocab from a TOML [actions.<section>] block.

Canonical order (== storage order in every artifact):
fold, check, call, preflop_opens (config order), bet_sizes (config order),
all-in last if enabled. Any change to this ordering rule invalidates
checkpoints — it is part of the artifact contract with vocab.hash().

Import note: This module lives at sixmax/vocab_config.py (a plain Python file),
while the C++ extension sixmax.so is registered as sys.modules["sixmax"] by
conftest.py (importlib force-load). Direct `from sixmax.vocab_config import`
fails because the compiled module is not a package. Tests import this module
by path via importlib.util.spec_from_file_location, mirroring the conftest
pattern for the .so itself.
"""
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import sixmax

_UNITS = {"bb": sixmax.SizeUnit.BB, "pot": sixmax.SizeUnit.Pot}


def load_vocab(toml_path: str, section: str = "blueprint") -> "sixmax.ActionVocab":
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    try:
        block = data["actions"][section]
    except KeyError:
        raise KeyError(f"[actions.{section}] not found in {toml_path}")

    A, T = sixmax.AbstractAction, sixmax.ActionType
    actions = [A(T.Fold, 0.0, sixmax.SizeUnit.BB),
               A(T.Check, 0.0, sixmax.SizeUnit.BB),
               A(T.Call, 0.0, sixmax.SizeUnit.BB)]
    for entry in block.get("preflop_opens", []) + block.get("bet_sizes", []):
        actions.append(A(T.Bet, float(entry["size"]), _UNITS[entry["unit"]]))
    if block.get("include_allin", False):
        actions.append(A(T.AllIn, 0.0, sixmax.SizeUnit.BB))
    return sixmax.ActionVocab(actions)
