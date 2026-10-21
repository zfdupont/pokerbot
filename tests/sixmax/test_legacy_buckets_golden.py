"""Golden test: the legacy (Monte-Carlo percentile) abstraction is unchanged.

tests/fixtures/legacy_buckets.json records `Abstraction::hash()` and
`bucket()` for a fixed seeded sample of 300 deals (100 each flop/turn/river)
against the two real SIXBP001 checkpoints. It was captured from the build that
preceded the map-backed abstraction work; any drift in the legacy edges, the
equity salt, or the memo cache fails here.

Skips when the (gitignored) checkpoints are not present on this machine.
"""
import json
import os

import pytest

import sixmax

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_FIXTURE = os.path.join(_REPO_ROOT, "tests", "fixtures", "legacy_buckets.json")
_CKPT_DIR = os.path.join(_REPO_ROOT, "sixmax", "checkpoints")


def _fixture():
    with open(_FIXTURE) as f:
        return json.load(f)


def test_fixture_is_well_formed():
    data = _fixture()
    assert len(data["deals"]) == 3 * data["deals_per_street"]
    for name, record in data["checkpoints"].items():
        assert isinstance(record["hash"], str)
        assert len(record["buckets"]) == len(data["deals"])


@pytest.mark.parametrize("name", ["blueprint.bin", "hu_blueprint.bin"])
def test_legacy_hash_and_buckets_unchanged(name):
    path = os.path.join(_CKPT_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"checkpoint not present: {path}")
    data = _fixture()
    record = data["checkpoints"][name]

    abs_ = sixmax.load_abstraction(path)
    assert not abs_.is_map()
    assert str(abs_.hash()) == record["hash"]

    for i, (deal, expected) in enumerate(zip(data["deals"], record["buckets"])):
        got = abs_.bucket(deal["hole"], deal["board"])
        assert got == expected, (
            f"{name}: bucket mismatch at deal {i} (hole={deal['hole']}, "
            f"board={deal['board']}): got {got}, expected {expected}")


def test_every_sixbp001_checkpoint_loads():
    """Every legacy artifact in sixmax/checkpoints/*.bin still loads."""
    if not os.path.isdir(_CKPT_DIR):
        pytest.skip(f"checkpoint directory not present: {_CKPT_DIR}")
    import glob
    paths = sorted(glob.glob(os.path.join(_CKPT_DIR, "*.bin")))
    legacy = []
    for path in paths:
        with open(path, "rb") as f:
            if f.read(8) == b"SIXBP001":
                legacy.append(path)
    if not legacy:
        pytest.skip(f"no SIXBP001 checkpoints in {_CKPT_DIR}")
    for path in legacy:
        abs_ = sixmax.load_abstraction(path)
        assert not abs_.is_map(), path
