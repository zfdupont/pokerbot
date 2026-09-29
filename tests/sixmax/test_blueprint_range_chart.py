"""Blueprint preflop range chart: infoset indexing, position mapping, rendering."""
import glob
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_script():
    path = os.path.join(_ROOT, "scripts", "blueprint_range_chart.py")
    spec = importlib.util.spec_from_file_location("blueprint_range_chart", path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert mod is not None
    spec.loader.exec_module(mod)
    return mod


def test_positions_and_index_shape():
    m = _load_script()
    assert m.POSITIONS == {"BTN": (2, 2), "SB": (1, 1)}
    # Fake records: (key, probs, mass, regret). Build one street-0 unopened key
    # for (live=2, after=2, card=155) using the pinned bit layout.
    key = (155) | (0 << 8) | (2 << 20) | (2 << 23)
    records = [(key, [0.1, 0.0, 0.0, 0.9, 0, 0, 0, 0, 0, 0], 5.0, 1.0)]
    idx = m.build_index(records)
    assert idx[(2, 2, 155)][3] == 0.9
    # a street-1 record must be excluded
    postflop = (7) | (1 << 8) | (2 << 20) | (2 << 23)
    idx2 = m.build_index(records + [(postflop, [1.0] + [0.0] * 9, 1.0, 0.0)])
    assert (2, 2, 7) not in idx2


def test_sample_card_ids_suit_convention():
    m = _load_script()
    # diagonal = pair -> two different suits, same rank
    c1, c2 = m.sample_card_ids(0, 0)          # AA
    assert c1 // 4 == c2 // 4 and c1 % 4 != c2 % 4
    # i<j upper-right = suited -> same suit
    c1, c2 = m.sample_card_ids(0, 1)          # AKs
    assert c1 % 4 == c2 % 4 and c1 // 4 != c2 // 4
    # i>j lower-left = offsuit -> different suits
    c1, c2 = m.sample_card_ids(1, 0)          # AKo
    assert c1 % 4 != c2 % 4


_CKPTS = sorted(glob.glob(os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint_0*.bin")))


@pytest.mark.skipif(not _CKPTS, reason="no blueprint snapshot available")
@pytest.mark.skip(reason="requires trained blueprint (>2M iters); re-enable after cloud run")
def test_position_mapping_discriminates():
    m = _load_script()
    sixmax = m._sixmax()
    ckpt = _CKPTS[-1]
    _iters, records = sixmax.dump_infosets(ckpt)
    vocab = m.load_vocab_for(ckpt)
    roles = m.resolve_action_roles(vocab)
    idx = m.build_index(records)

    def aggr(i, j, pos):
        p = m.cell_probs(idx, pos, i, j)
        return sum(p[k] for k in roles["aggressive"]) if p else None

    # Premiums beat trash at each mapped cell. 72o is offsuit (i>j): 7=index7, 2=index12
    # -> (i=12, j=7). AA=(0,0), AKs=(0,1) suited.
    for pos in ("BTN", "SB"):
        assert aggr(0, 0, pos) > aggr(12, 7, pos)   # AA > 72o
        assert aggr(0, 1, pos) > aggr(12, 7, pos)   # AKs > 72o
    # SB (1,1) opens strictly wider than BTN (2,2): the property that catches a mis-map.
    assert m.role_width(idx, "SB", roles) > m.role_width(idx, "BTN", roles)
