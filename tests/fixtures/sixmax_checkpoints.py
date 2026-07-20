"""Session-scoped fixtures that train tiny blueprints for deployment tests.

Kept small so the whole downstream suite is independently runnable in well under
a minute. The real full-scale blueprint is a separate operational run; these are
correctness fixtures only (they gate on loadability + a non-empty policy, never
on strategy strength). Checkpoints are written into tmp_path_factory and never
committed.

Cost note: with the abstraction's per-node Monte-Carlo equity bucketing, HU
(n=2) training is ~15x slower per iteration than 6-max, because 2-player
traversals reach the expensive postflop rollouts far more often than 6-max
lines (most 6-max seats fold preflop). So we use a light abstraction (few
rollouts/buckets) and keep the HU iteration count small. See the Phase 1c plan's
Task 2 deviation note."""
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def sixmax_vocab():
    import sixmax  # registered by tests/sixmax/conftest.py
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(
        os.path.join(_ROOT, "sixmax", "configs", "default.toml"), "blueprint")


def _train(tmp_path, sixmax_vocab, num_players, iterations, name):
    import sixmax
    cfg = sixmax.EngineConfig(num_players=num_players)
    # Light abstraction: fewer rollouts/buckets keep per-node cost low so the
    # fixtures train in seconds (see module docstring on HU cost).
    abstraction = sixmax.Abstraction(flop_buckets=6, turn_buckets=6,
                                     river_buckets=4, equity_rollouts=8,
                                     quantile_samples=80, seed=42)
    trainer = sixmax.BlueprintTrainer(cfg, sixmax_vocab, abstraction,
                                      num_threads=1, seed=7)
    trainer.train(iterations)
    path = os.path.join(str(tmp_path), name)
    trainer.save(path, sixmax_vocab, cfg, abstraction)
    return path


@pytest.fixture(scope="session")
def blueprint_hu_ckpt(tmp_path_factory, sixmax_vocab):
    # HU is the expensive case (see docstring), so the iteration count is small;
    # correctness fixtures don't gate on strategy strength.
    tmp = tmp_path_factory.mktemp("sixmax_hu")
    return _train(tmp, sixmax_vocab, 2, 400, "blueprint_hu.bin")


@pytest.fixture(scope="session")
def blueprint_6max_ckpt(tmp_path_factory, sixmax_vocab):
    tmp = tmp_path_factory.mktemp("sixmax_6max")
    return _train(tmp, sixmax_vocab, 6, 300, "blueprint_6max.bin")
