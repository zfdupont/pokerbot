import math
from scripts.diagnose_blueprint import (
    entropy_bits, support_size, gini, weighted_mean,
    decode_key, top_tier, probe_policy,
)


def test_entropy_uniform_two_actions_is_one_bit():
    assert abs(entropy_bits([0.5, 0.5]) - 1.0) < 1e-12
    assert abs(entropy_bits([1.0, 0.0])) < 1e-12  # deterministic -> 0


def test_support_counts_nonzero():
    assert support_size([0.0, 0.4, 0.6, 0.0]) == 2


def test_gini_equal_is_zero_skewed_is_high():
    assert abs(gini([1.0, 1.0, 1.0, 1.0])) < 1e-9
    assert gini([0.0, 0.0, 0.0, 100.0]) > 0.7


def test_weighted_mean():
    assert abs(weighted_mean([1.0, 3.0], [1.0, 3.0]) - 2.5) < 1e-12
    assert weighted_mean([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_decode_key_roundtrip_fields():
    # card=5, street=1, raises=[0,2,0,0], pot=1, live=3, after=2
    key = (5) | (1 << 8) | (2 << (10 + 2 * 1)) | (1 << 18) | (3 << 20) | (2 << 23)
    d = decode_key(key)
    assert d["card"] == 5 and d["street"] == 1
    assert d["raises"] == [0, 2, 0, 0]
    assert d["pot"] == 1 and d["live"] == 3 and d["after"] == 2


def test_top_tier_selects_highest_mass():
    recs = [("a", [], 1.0, 0.0), ("b", [], 9.0, 0.0), ("c", [], 5.0, 0.0)]
    tier = top_tier(recs, 0.34)  # ~1 of 3
    assert [r[0] for r in tier] == ["b"]


def test_probe_policy_aggregates_matching_infosets():
    # two infosets, card=5 street=0 raises all 0; masses 1 and 3
    k = (5) | (0 << 8)
    recs = [(k, [1.0, 0.0], 1.0, 0.0), (k, [0.0, 1.0], 3.0, 0.0)]
    probs, mass = probe_policy(recs, card_id=5, street=0, total_raises=0)
    assert mass == 4.0
    assert abs(probs[0] - 0.25) < 1e-12 and abs(probs[1] - 0.75) < 1e-12
