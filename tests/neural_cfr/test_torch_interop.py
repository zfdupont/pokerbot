"""
Regression test: neural_cfr's C++ Trainer must work when the Python-level
`torch` package has been imported into the same process.

Root cause it guards against: importing Python torch registers the
`PythonEngine` autograd engine (which forbids being called while the GIL is
held). neural_cfr's bindings invoked `backward()` while holding the GIL, so
any process that imported torch first (e.g. tests/sixmax/test_dream_kuhn.py)
crashed with:

    RuntimeError: The autograd engine was called while holding the GIL.

Fix: release the GIL around the backward-invoking bindings via
`py::call_guard<py::gil_scoped_release>()` (mirroring sixmax/src/bindings).
This test fails before that fix and passes after.
"""
import os
import tempfile

import torch  # noqa: F401  (importing torch is the whole point)

import neural_cfr


def test_trainer_trains_with_python_torch_loaded():
    """Trainer.run() invokes autograd; must not trip the GIL check."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(200)


def test_checkpoint_with_python_torch_loaded():
    """Trainer.checkpoint() retrains the strategy net; must not trip the GIL check."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(200)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.checkpoint(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)
