import os
import sys
import importlib

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: tests/sixmax/ -> root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_TOML = os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml")


def test_importing_agent_module_does_not_load_dream():
    # Fresh import of the deploy module must not pull in sixmax_dream/libtorch.
    # Remove cached sixmax_dream so it acts as a sentinel: if sixmax_agent
    # re-imports sixmax_dream at module level, it will re-enter sys.modules here.
    for m in list(sys.modules):
        if m == "sixmax_dream":
            del sys.modules[m]
    # Also evict agents.sixmax_agent (and any cached submodule) so that
    # re-importing it actually re-executes all module-level code.
    for m in list(sys.modules):
        if m == "agents.sixmax_agent" or m.startswith("agents.sixmax_agent."):
            del sys.modules[m]
    importlib.import_module("agents.sixmax_agent")
    assert "sixmax_dream" not in sys.modules, \
        "blueprint deploy path must not import sixmax_dream at module load"


def test_dream_deploy_loads_and_acts(tmp_path, default_vocab):
    # Train a tiny DREAM checkpoint, then deploy it and get a legal action.
    import sixmax
    import sixmax_dream
    abstraction = sixmax.Abstraction(
        flop_buckets=4, turn_buckets=4, river_buckets=4,
        equity_rollouts=5, quantile_samples=100, seed=1)
    cfg = sixmax_dream.DreamConfig()
    cfg.train_interval = 4; cfg.sgd_steps = 1; cfg.batch_size = 8; cfg.reservoir_size = 1000
    tr = sixmax_dream.DreamTrainer(default_vocab.size(), default_vocab, abstraction, cfg, "cpu")
    tr.train(1)
    ckpt = str(tmp_path / "dream.pt")
    tr.save(ckpt, default_vocab.hash())

    from agents.sixmax_agent import DreamDeployStrategy
    strat = DreamDeployStrategy(ckpt, _DEFAULT_TOML)
    assert strat is not None                # constructed from a real DREAM checkpoint
    assert "sixmax_dream" in sys.modules    # loaded lazily on construction, not at import
