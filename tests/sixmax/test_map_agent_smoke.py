"""Deploy smoke: a SIXBP002 map checkpoint plays through SixmaxAgent.

The map is copied next to the checkpoint so the deploy-side resolver finds it
without any environment variables, mirroring how a checkpoint is shipped with
its table.
"""
import os
import shutil

import sixmax
from agents.sixmax_agent import SixmaxAgent
from scripts.eval_hu_sanity import run_duplicate_match

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_sixmax_agent_plays_map_checkpoint(tmp_path, synthetic_bucket_map,
                                           default_vocab, monkeypatch):
    map_copy = tmp_path / "idxmod.bin"
    shutil.copyfile(synthetic_bucket_map, map_copy)
    abs_ = sixmax.Abstraction.from_bucket_map(str(map_copy))
    cfg = sixmax.EngineConfig(num_players=2)
    trainer = sixmax.BlueprintTrainer(cfg, default_vocab, abs_, num_threads=1,
                                      seed=13)
    trainer.train(120)
    ckpt = str(tmp_path / "map_agent.bin")
    trainer.save(ckpt, default_vocab, cfg, abs_)

    monkeypatch.delenv("SIXMAX_BUCKET_MAP", raising=False)
    monkeypatch.delenv("SIXMAX_TABLES_DIR", raising=False)

    def factory(_seat):
        return SixmaxAgent(ckpt, config_toml=_TOML)

    bb100 = run_duplicate_match(factory, factory, hands=20, seed=1, bb=2,
                                stack=200)
    assert bb100 == 0.0
