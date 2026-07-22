import math
import os
import pytest
from scripts.diagnose_blueprint import (
    entropy_bits, support_size, gini, weighted_mean,
    decode_key, top_tier, probe_policy, verdict,
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


# ---------------------------------------------------------------------------
# verdict() pure-function tests (no sixmax dependency)
# ---------------------------------------------------------------------------

def _make_report(iterations, entropy_top, avg_regret_top, probe_expected_masses):
    """Build a minimal checkpoint_report dict for verdict() unit tests."""
    probes = [{"expected_mass": m} for m in probe_expected_masses]
    return {
        "iterations": iterations,
        "entropy_top": entropy_top,
        "avg_regret_top": avg_regret_top,
        "probes": probes,
    }


def test_verdict_undertraining_all_three_signals():
    """All three signals trending 'learning' -> undertraining."""
    r0 = _make_report(100000, entropy_top=2.5, avg_regret_top=0.8, probe_expected_masses=[0.2, 0.1])
    r1 = _make_report(1000000, entropy_top=1.5, avg_regret_top=0.4, probe_expected_masses=[0.7, 0.6])
    label, rationale = verdict([r0, r1])
    assert label == "undertraining", f"expected undertraining, got {label}: {rationale}"
    assert rationale  # non-empty rationale


def test_verdict_structural_no_signals():
    """No signals moving -> structural ceiling."""
    r0 = _make_report(100000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.4])
    r1 = _make_report(1000000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.4])
    label, rationale = verdict([r0, r1])
    assert label == "structural", f"expected structural, got {label}: {rationale}"


def test_verdict_mixed_one_signal():
    """Only entropy falling, regret and probe flat -> mixed."""
    r0 = _make_report(100000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.4])
    r1 = _make_report(1000000, entropy_top=1.2, avg_regret_top=0.5, probe_expected_masses=[0.4])
    label, rationale = verdict([r0, r1])
    assert label == "mixed", f"expected mixed, got {label}: {rationale}"
    assert rationale


def test_verdict_mixed_two_signals():
    """Entropy and probe mass moving but regret flat -> mixed."""
    r0 = _make_report(100000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.2])
    r1 = _make_report(1000000, entropy_top=1.0, avg_regret_top=0.5, probe_expected_masses=[0.7])
    label, rationale = verdict([r0, r1])
    assert label == "mixed", f"expected mixed, got {label}: {rationale}"


def test_verdict_insufficient_checkpoints():
    """Single checkpoint -> mixed with a 'need >=2' message."""
    r0 = _make_report(100000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.4])
    label, rationale = verdict([r0])
    assert label == "mixed"
    assert "2" in rationale  # mentions the >=2 requirement


def test_verdict_empty_reports():
    """Empty list -> mixed with a 'need >=2' message."""
    label, rationale = verdict([])
    assert label == "mixed"
    assert "2" in rationale


def test_verdict_sub_noise_wiggle_is_flat():
    """Tiny wiggles below relative tolerance must not flip to undertraining."""
    # 0.1% change is sub-noise relative to the values
    r0 = _make_report(100000, entropy_top=2.0, avg_regret_top=0.5, probe_expected_masses=[0.4])
    r1 = _make_report(1000000, entropy_top=1.999, avg_regret_top=0.4999, probe_expected_masses=[0.4001])
    label, rationale = verdict([r0, r1])
    assert label == "structural", f"expected structural (sub-noise), got {label}: {rationale}"


# ---------------------------------------------------------------------------
# End-to-end smoke test (skipped when checkpoint absent)
# ---------------------------------------------------------------------------

def test_checkpoint_report_smoke():
    ckpt = "sixmax/checkpoints/blueprint_00100000.bin"
    if not os.path.exists(ckpt):
        pytest.skip("checkpoint not present (not committed)")
    import agents.sixmax_agent  # noqa: F401  (force-loads sixmax .so)
    import sixmax
    from scripts.diagnose_blueprint import (
        resolve_action_roles, checkpoint_report, load_vocab_for,
    )
    vocab = load_vocab_for(ckpt)
    roles = resolve_action_roles(vocab)
    assert roles["fold"] and roles["aggressive"]  # non-empty role sets
    iterations, records = sixmax.dump_infosets(ckpt)
    rep = checkpoint_report(iterations, records, roles, probes=[])
    assert rep["iterations"] == iterations
    assert rep["num_infosets"] == len(records)
    assert 0.0 <= rep["gini"] <= 1.0
    assert rep["entropy_all"] >= 0.0
    assert rep["entropy_top"] >= 0.0
