import os
import pytest

import agents.sixmax_agent  # noqa: F401  (force-loads the sixmax .so)
import sixmax

CKPT = "sixmax/checkpoints/blueprint_00100000.bin"


def test_dump_infosets_smoke():
    if not os.path.exists(CKPT):
        pytest.skip(f"checkpoint {CKPT} not present (not committed)")
    iterations, records = sixmax.dump_infosets(CKPT)
    assert iterations > 0
    assert len(records) > 0
    for key, probs, mass, reg in records[:200]:
        assert isinstance(key, int)
        assert mass >= 0.0
        assert reg >= 0.0
        if mass > 0.0:
            assert abs(sum(probs) - 1.0) < 1e-9
