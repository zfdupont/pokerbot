import os
import tempfile
import pytest
from cfr.trainer import Trainer
from cfr.regret_table import RegretTable


class TestTrainer:
    def test_train_runs_n_iterations(self):
        table = RegretTable()
        trainer = Trainer(table)
        trainer.train(num_iterations=10, checkpoint_interval=100, checkpoint_dir=None)
        assert len(table.regrets) > 0

    def test_train_saves_checkpoint(self):
        table = RegretTable()
        trainer = Trainer(table)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(
                num_iterations=50,
                checkpoint_interval=25,
                checkpoint_dir=tmpdir,
            )
            files = os.listdir(tmpdir)
            assert any(f.endswith(".pkl") for f in files)

    def test_trainer_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table1 = RegretTable()
            Trainer(table1).train(
                num_iterations=50,
                checkpoint_interval=50,
                checkpoint_dir=tmpdir,
            )
            # Load latest checkpoint into fresh table
            table2 = RegretTable()
            trainer2 = Trainer(table2)
            trainer2.load_latest(tmpdir)
            assert len(table2.regrets) == len(table1.regrets)
