import math
import os
import pytest
from scripts.diagnose_blueprint import (
    entropy_bits, support_size, gini, weighted_mean,
    decode_key, top_tier, probe_policy, verdict, _monotone_trend,
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
# _monotone_trend() helper tests
# ---------------------------------------------------------------------------

def test_monotone_trend_down_perfect():
    frac, net = _monotone_trend([3.0, 2.5, 2.0, 1.5], "down")
    assert frac == 1.0 and abs(net - (-1.5)) < 1e-12


def test_monotone_trend_up_perfect():
    frac, net = _monotone_trend([0.1, 0.2, 0.3, 0.4], "up")
    assert frac == 1.0 and abs(net - 0.3) < 1e-12


def test_monotone_trend_flat_series():
    frac, net = _monotone_trend([1.0, 1.0, 1.0, 1.0], "down")
    assert frac == 0.0 and net == 0.0


# ---------------------------------------------------------------------------
# verdict() pure-function tests (no sixmax dependency)
# ---------------------------------------------------------------------------

def _make_report(iterations, entropy_top, avg_regret_top, probe_expected_masses):
    """Build a minimal checkpoint_report dict for verdict() unit tests.

    Keys match what checkpoint_report() actually produces:
      entropy_top, avg_regret_top, iterations, probes (list of dicts with
      "expected_mass" key).
    """
    probes = [{"expected_mass": m} for m in probe_expected_masses]
    return {
        "iterations": iterations,
        "entropy_top": entropy_top,
        "avg_regret_top": avg_regret_top,
        "probes": probes,
    }


def _make_series(entropy_vals, probe_vals, regret_vals=None):
    """Build a multi-checkpoint series for trend-based verdict tests."""
    n = len(entropy_vals)
    if regret_vals is None:
        # Monotonically rising regret (realistic for linear CFR)
        regret_vals = [100_000 * (i + 1) for i in range(n)]
    reports = []
    for i in range(n):
        reports.append(_make_report(
            iterations=(i + 1) * 100_000,
            entropy_top=entropy_vals[i],
            avg_regret_top=regret_vals[i],
            probe_expected_masses=[probe_vals[i]],
        ))
    return reports


def test_verdict_undertraining_both_gating_signals():
    """Both gating signals (entropy down, probe up) monotonically trending -> undertraining.

    10 checkpoints: entropy_top strictly decreasing, probe mass strictly increasing.
    """
    n = 10
    entropy_vals = [2.5 - i * 0.1 for i in range(n)]   # 2.5 → 1.6
    probe_vals   = [0.2 + i * 0.05 for i in range(n)]  # 0.2 → 0.65
    reports = _make_series(entropy_vals, probe_vals)
    label, rationale = verdict(reports)
    assert label == "undertraining", f"expected undertraining, got {label!r}: {rationale}"
    assert rationale  # non-empty


def test_verdict_structural_both_flat():
    """Both gating signals flat -> structural ceiling."""
    n = 10
    entropy_vals = [2.0] * n
    probe_vals   = [0.4] * n
    reports = _make_series(entropy_vals, probe_vals)
    label, rationale = verdict(reports)
    assert label == "structural", f"expected structural, got {label!r}: {rationale}"
    assert rationale


def test_verdict_mixed_only_entropy_falling():
    """entropy_top strictly decreasing but probe mass flat -> mixed (one gating signal)."""
    n = 10
    entropy_vals = [2.5 - i * 0.1 for i in range(n)]  # strict decrease
    probe_vals   = [0.4] * n                            # completely flat
    reports = _make_series(entropy_vals, probe_vals)
    label, rationale = verdict(reports)
    assert label == "mixed", f"expected mixed, got {label!r}: {rationale}"
    assert rationale


def test_verdict_mixed_only_probe_rising():
    """probe mass strictly increasing but entropy_top flat -> mixed (one gating signal)."""
    n = 10
    entropy_vals = [2.0] * n                            # flat
    probe_vals   = [0.2 + i * 0.05 for i in range(n)]  # strict increase
    reports = _make_series(entropy_vals, probe_vals)
    label, rationale = verdict(reports)
    assert label == "mixed", f"expected mixed, got {label!r}: {rationale}"
    assert rationale


def test_verdict_entropy_not_learning_when_fraction_below_threshold():
    """entropy_top decreases at only ~half the steps (fraction < 0.8) -> that signal is NOT learning.

    Construct a 10-checkpoint series where entropy goes up then down alternately —
    fraction in 'down' direction is ~0.5, well below the 0.8 gate.
    With probe also flat this should be structural.
    """
    # Oscillating pattern: goes down on odd steps, up on even — ~50% "down"
    base = 2.0
    entropy_vals = [base + (0.05 if i % 2 == 0 else -0.05) * (i + 1) for i in range(10)]
    probe_vals   = [0.4] * 10
    reports = _make_series(entropy_vals, probe_vals)
    label, rationale = verdict(reports)
    # entropy_learning should be False (fraction < 0.8)
    assert label in ("structural", "mixed"), (
        f"oscillating entropy should not count as learning; got {label!r}: {rationale}"
    )


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


def test_verdict_real_data_entropy_monotone_should_be_undertraining_or_mixed():
    """Simulate the actual run: entropy_top 0.665→0.642 at all 9 steps (strictly down).

    Probe mass is flat (stuck at 0.67), so we expect 'mixed' (one gating signal).
    The OLD code incorrectly labeled this 'structural' because the net drop (0.023)
    was below the 5% relative tolerance (0.033). The new trend-based code must NOT
    label it structural when entropy is strictly monotone.
    """
    entropy_vals = [0.665, 0.662, 0.660, 0.657, 0.654, 0.651, 0.648, 0.646, 0.644, 0.642]
    probe_vals   = [0.67] * 10   # completely frozen (real data)
    regret_vals  = [2.239e5, 2.458e5, 2.677e5, 2.894e5, 3.113e5,
                    3.324e5, 3.533e5, 3.742e5, 3.952e5, 4.162e5]
    reports = _make_series(entropy_vals, probe_vals, regret_vals)
    label, rationale = verdict(reports)
    assert label != "structural", (
        f"monotone entropy drop must not be labeled 'structural'; got {label!r}: {rationale}"
    )
    assert rationale


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
