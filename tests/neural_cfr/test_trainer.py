"""
Integration tests for neural_cfr.Trainer.

sys.path and dylib pre-loading are handled by conftest.py in this directory.
"""
import os
import sys
import tempfile
import pytest

import neural_cfr


def test_buffers_populated_after_run():
    """Trainer.run() should populate internal buffers — indirectly verified via no crash."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(50)  # 50 iterations — fast smoke test


def test_checkpoint_roundtrip():
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(10)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.checkpoint(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        # Load into a fresh trainer — should not raise
        trainer2 = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
        trainer2.load(path)
    finally:
        os.unlink(path)


def test_load_missing_file_raises():
    trainer = neural_cfr.Trainer()
    with pytest.raises(RuntimeError):
        trainer.load("/tmp/definitely_does_not_exist.pt")
