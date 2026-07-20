"""The dev-scale checkpoint fixtures must produce loadable BlueprintStrategy
artifacts with the expected player counts."""
import sixmax


def test_hu_checkpoint_loads(blueprint_hu_ckpt, sixmax_vocab):
    strat = sixmax.BlueprintStrategy.load(blueprint_hu_ckpt, sixmax_vocab)
    assert strat.num_players() == 2
    assert strat.iterations() > 0
    assert strat.num_infosets() > 0


def test_6max_checkpoint_loads(blueprint_6max_ckpt, sixmax_vocab):
    strat = sixmax.BlueprintStrategy.load(blueprint_6max_ckpt, sixmax_vocab)
    assert strat.num_players() == 6
