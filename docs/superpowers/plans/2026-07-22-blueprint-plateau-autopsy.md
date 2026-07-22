# Blueprint Plateau Autopsy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An offline diagnostic that reads the 10 existing blueprint snapshots and emits a falsifiable verdict — undertraining (scale) vs structural ceiling (redesign) — for the ~−485 BB/100 plateau.

**Architecture:** One new read-only C++ binding (`dump_infosets`) recovers per-infoset visit-weight (`strategy_sum` L1) and regret L1 that the normalized accessors discard. One Python script (`scripts/diagnose_blueprint.py`) decodes keys, computes visit-weight distribution / policy-entropy / probe-trace metrics off that single data source, and prints the verdict. Pure metric functions are stdlib-only and unit-tested; the binding and checkpoint loading are covered by a skip-if-absent smoke test.

**Tech Stack:** C++17 + pybind11 (`sixmax/src/bindings/bindings.cpp`, built with Buck2), Python 3.10 (repo `.venv`), pytest.

## Global Constraints

- Bridging between `sixmax/` and Python lives ONLY in `sixmax/src/bindings/` and `scripts/` — never import `game/poker.py` into `sixmax/`. (This diagnostic touches only those two places.)
- The `sixmax/` action vocabulary is **config-defined**; never assume a fixed index→action mapping. Resolve fold/passive/aggressive indices from `AbstractAction.type` at runtime.
- Mask illegal actions / never reorder storage — N/A here (read-only), but do not filter or re-index the probs vector returned by the binding.
- Checkpoints are native-endian and move only between same-arch machines — run the autopsy on the machine that trained them.
- Never commit checkpoint files (`sixmax/checkpoints/*.bin`) or secrets. Tests that read a checkpoint must `pytest.skip` when it is absent.
- Build the extension with `~/bin/buck2 build //sixmax:sixmax`. The repo-root `sixmax/` package shadows the `.so`; tests force-load it via `import agents.sixmax_agent` (which runs `_ensure_sixmax_extension()`).

---

### Task 1: `dump_infosets` read-only binding

**Files:**
- Modify: `sixmax/src/bindings/bindings.cpp` (add one `m.def` next to `load_abstraction`, ~line 294)
- Test: `tests/test_dump_infosets.py`

**Interfaces:**
- Produces: `sixmax.dump_infosets(path: str) -> tuple[int, list[tuple[int, list[float], float, float]]]`
  returning `(iterations, records)` where each record is `(key, probs, strategy_sum_l1, regret_l1)`.
  `probs` is `strategy_sum` normalized to sum 1 over its support (all-zero when `strategy_sum_l1 == 0`).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_dump_infosets.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dump_infosets.py -v`
Expected: FAIL — `AttributeError: module 'sixmax' has no attribute 'dump_infosets'`.

- [ ] **Step 3: Add the binding**

In `sixmax/src/bindings/bindings.cpp`, immediately after the `load_abstraction` lambda (ends ~line 294), add:

```cpp
    // --- Read-only autopsy dump: recovers per-infoset visit-weight and regret
    //     that the normalized accessors discard (plateau diagnosis). ---
    m.def("dump_infosets", [](const std::string& path) {
        uint64_t h;
        {
            std::ifstream i(path, std::ios::binary);
            i.seekg(8);
            i.read(reinterpret_cast<char*>(&h), sizeof h);
        }
        auto loaded = sixmax::load_blueprint(path, h);
        py::list records;
        for (const auto& [key, data] : loaded.table) {
            double mass = 0.0, reg = 0.0;
            for (double v : data.strategy_sum) mass += v;
            for (double v : data.regret) reg += std::fabs(v);
            std::vector<double> probs(data.strategy_sum.size(), 0.0);
            if (mass > 0.0)
                for (size_t i = 0; i < probs.size(); ++i)
                    probs[i] = data.strategy_sum[i] / mass;
            records.append(py::make_tuple(key, probs, mass, reg));
        }
        return py::make_tuple(loaded.iterations, records);
    }, py::arg("path"));
```

If the file does not already `#include <cmath>` (for `std::fabs`), add it with the other includes at the top.

- [ ] **Step 4: Rebuild the extension**

Run: `~/bin/buck2 build //sixmax:sixmax`
Expected: build succeeds (no compile errors).

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_dump_infosets.py -v`
Expected: PASS (or SKIP if the checkpoint is absent — then run once with a real checkpoint present locally to confirm PASS before committing).

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/bindings/bindings.cpp tests/test_dump_infosets.py
git commit -m "feat(sixmax): read-only dump_infosets binding for blueprint autopsy"
```

---

### Task 2: Pure metric functions

**Files:**
- Create: `scripts/diagnose_blueprint.py` (pure functions only in this task; `main()` added in Task 3)
- Test: `tests/test_diagnose_blueprint.py`

**Interfaces:**
- Produces (all stdlib-only, no `sixmax` import at module load):
  - `entropy_bits(probs: list[float]) -> float`
  - `support_size(probs: list[float], eps: float = 1e-12) -> int`
  - `gini(weights: list[float]) -> float`
  - `weighted_mean(values: list[float], weights: list[float]) -> float`
  - `decode_key(key: int) -> dict` with keys `card, street, raises(list[4]), pot, live, after`
  - `top_tier(records, frac) -> list` — records sorted by mass desc, top `frac` fraction (≥1 element)
  - `probe_policy(records, card_id, street, total_raises) -> tuple[list[float], float]` —
    mass-weighted mean probs vector over matching infosets, and total matched mass (`([], 0.0)` if none)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diagnose_blueprint.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diagnose_blueprint.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'entropy_bits'`.

- [ ] **Step 3: Implement the pure functions**

```python
# scripts/diagnose_blueprint.py
"""Blueprint plateau autopsy — offline diagnostic over checkpoint snapshots.

Reads the visit-weight (strategy_sum L1) and regret L1 that sixmax.dump_infosets
recovers, then decides whether the flat BB/100 curve is undertraining (scale) or
a structural ceiling (redesign). Pure metric functions below are stdlib-only and
unit-tested; checkpoint loading + the vocab-dependent probe resolution live in
main() so importing this module never triggers a Buck2 build.

See docs/superpowers/specs/2026-07-22-blueprint-plateau-autopsy-design.md.
"""
import math


def entropy_bits(probs):
    """Shannon entropy (bits) over the support of a probability vector."""
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def support_size(probs, eps=1e-12):
    """Number of actions with non-negligible probability mass."""
    return sum(1 for p in probs if p > eps)


def gini(weights):
    """Gini coefficient of a non-negative weight vector (0 = equal)."""
    xs = sorted(weights)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0.0:
        return 0.0
    s = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * s) / (n * total) - (n + 1.0) / n


def weighted_mean(values, weights):
    tw = sum(weights)
    if tw == 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / tw


def decode_key(key):
    """Unpack the abstract infoset key per abstract_key.h's documented layout:
    card 0-7, street 8-9, raises 10-17 (2 bits/street), pot 18-19,
    live 20-22, after 23-25."""
    return {
        "card": key & 0xFF,
        "street": (key >> 8) & 0x3,
        "raises": [(key >> (10 + 2 * st)) & 0x3 for st in range(4)],
        "pot": (key >> 18) & 0x3,
        "live": (key >> 20) & 0x7,
        "after": (key >> 23) & 0x7,
    }


def top_tier(records, frac):
    """The top `frac` fraction of records by strategy_sum mass (index 2),
    at least one element."""
    ordered = sorted(records, key=lambda r: r[2], reverse=True)
    k = max(1, int(len(ordered) * frac))
    return ordered[:k]


def probe_policy(records, card_id, street, total_raises):
    """Mass-weighted mean policy over every real infoset matching a probe
    predicate (card bucket, street, and total raises-so-far). Returns
    (mean_probs, total_mass); ([], 0.0) when no infoset matches."""
    acc = None
    total = 0.0
    for key, probs, mass, _reg in records:
        d = decode_key(key)
        if d["card"] != card_id or d["street"] != street:
            continue
        if total_raises is not None and sum(d["raises"]) != total_raises:
            continue
        if mass <= 0.0:
            continue
        if acc is None:
            acc = [0.0] * len(probs)
        for i, p in enumerate(probs):
            acc[i] += p * mass
        total += mass
    if acc is None or total == 0.0:
        return [], 0.0
    return [a / total for a in acc], total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnose_blueprint.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_blueprint.py tests/test_diagnose_blueprint.py
git commit -m "feat(sixmax): pure metric functions for blueprint autopsy"
```

---

### Task 3: Autopsy driver (`main()`), probe resolution, and verdict

**Files:**
- Modify: `scripts/diagnose_blueprint.py` (append vocab/probe resolution + `main()`)
- Test: `tests/test_diagnose_blueprint.py` (add a skip-if-absent end-to-end smoke test)

**Interfaces:**
- Consumes: `sixmax.dump_infosets` (Task 1); pure functions (Task 2); `sixmax.preflop_class`,
  `sixmax.ActionVocab` / `AbstractAction.type` for probe + action-role resolution.
- Produces: `resolve_action_roles(vocab) -> dict` with keys `fold, passive, aggressive`
  (each a list of vocab indices); `checkpoint_report(iterations, records, roles, probes) -> dict`
  (per-checkpoint metrics + probe rows); a CLI `python scripts/diagnose_blueprint.py`.

- [ ] **Step 1: Write the failing end-to-end smoke test**

```python
# add to tests/test_diagnose_blueprint.py
import os
import pytest


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_diagnose_blueprint.py::test_checkpoint_report_smoke -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_action_roles'`.

- [ ] **Step 3: Establish the vocab loader + action roles**

First confirm the `ActionType` enum member names exposed to Python and the config-toml section
name the checkpoints use:

Run: `~/bin/buck2 build //sixmax:sixmax && uv run python -c "import agents.sixmax_agent, sixmax; print([t for t in dir(sixmax.ActionType) if not t.startswith('_')])"`
Expected: prints the enum members (e.g. `['FOLD', 'CHECK', 'CALL', 'BET', 'RAISE', 'ALLIN']` — use the actual names in the code below).

Reuse the vocab-loading helper already in the deploy bridge rather than re-parsing TOML:

```python
# append to scripts/diagnose_blueprint.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sixmax():
    """Lazy import so this module stays Buck2-free until main() runs."""
    import agents.sixmax_agent  # noqa: F401  (force-loads the .so)
    import sixmax
    return sixmax


def load_vocab_for(checkpoint_path):
    """Reconstruct the ActionVocab this checkpoint was trained with, via the
    sibling `<name>.config.toml` the trainer writes (falls back to
    blueprint.bin.config.toml in the same dir)."""
    from agents.sixmax_agent import _load_vocab  # sanctioned bridge helper
    cfg = checkpoint_path + ".config.toml"
    if not os.path.exists(cfg):
        cfg = os.path.join(os.path.dirname(checkpoint_path),
                           "blueprint.bin.config.toml")
    return _load_vocab(cfg, "blueprint")


def resolve_action_roles(vocab):
    """Map vocab indices to fold / passive (check-call) / aggressive
    (bet-raise-allin) roles from AbstractAction.type — the vocab is
    config-defined so indices must never be assumed."""
    sixmax = _sixmax()
    T = sixmax.ActionType
    roles = {"fold": [], "passive": [], "aggressive": []}
    for i in range(vocab.size()):
        t = vocab.at(i).type
        if t == T.FOLD:
            roles["fold"].append(i)
        elif t in (T.CHECK, T.CALL):
            roles["passive"].append(i)
        else:  # BET / RAISE / ALLIN
            roles["aggressive"].append(i)
    return roles
```

If Step 3's probe of `ActionType` shows different member names, adjust the `T.<NAME>`
references (and confirm `_load_vocab`'s signature in `agents/sixmax_agent.py`).

- [ ] **Step 4: Add the per-checkpoint report + verdict, and CLI**

```python
# append to scripts/diagnose_blueprint.py


def _role_mass(probs, roles):
    """Fraction of a policy's mass on each role."""
    return {r: sum(probs[i] for i in idxs) for r, idxs in roles.items()}


def checkpoint_report(iterations, records, roles, probes, top_frac=0.05):
    """Compute the plateau-diagnosis metrics for one checkpoint.

    probes: list of (label, card_id, street, total_raises, expected_role)
    where expected_role in {"fold","aggressive"} names the mass that SHOULD
    grow across checkpoints for that spot.
    """
    masses = [r[2] for r in records]
    ent_all = weighted_mean([entropy_bits(r[1]) for r in records], masses)
    tier = top_tier(records, top_frac)
    tier_masses = [r[2] for r in tier]
    ent_top = weighted_mean([entropy_bits(r[1]) for r in tier], tier_masses)
    supp_top = weighted_mean([support_size(r[1]) for r in tier], tier_masses)
    total_mass = sum(masses) or 1.0
    avg_regret_top = weighted_mean([r[3] / iterations for r in tier], tier_masses)

    probe_rows = []
    for label, card_id, street, total_raises, expected in probes:
        probs, mass = probe_policy(records, card_id, street, total_raises)
        rm = _role_mass(probs, roles) if probs else {}
        probe_rows.append({
            "label": label, "matched_mass": mass,
            "expected_role": expected,
            "expected_mass": rm.get(expected, 0.0),
            "role_mass": rm,
        })
    return {
        "iterations": iterations,
        "num_infosets": len(records),
        "gini": gini(masses),
        "top1pct_share": sum(sorted(masses, reverse=True)[:max(1, len(masses)//100)]) / total_mass,
        "entropy_all": ent_all,
        "entropy_top": ent_top,
        "support_top": supp_top,
        "avg_regret_top": avg_regret_top,
        "probes": probe_rows,
    }


def verdict(reports):
    """Classify the plateau from the metric trend across ordered reports.

    TODO(human): implement the decision rule. Given `reports` (a list of
    checkpoint_report dicts ordered by iterations, ascending), return one of
    "undertraining", "structural", or "mixed" plus a one-line rationale:
        return (label, rationale)
    Signals to weigh (per the design doc's decision rule):
      - entropy_top FALLING across reports  -> policies differentiating (learning)
      - avg_regret_top SHRINKING            -> solver converging where evidence exists
      - probe expected_mass RISING          -> right-direction learning at named spots
    All three trending the "learning" way => "undertraining"; none => "structural";
    partial => "mixed". Use a tolerance so sub-noise wiggles don't flip the verdict.
    """


# Canonical probe spots resolved to card buckets in main(): premium hands should
# grow aggression, trash should grow folds, and the "uniform open-jam" pathology
# should shrink (falling entropy_top captures the last one).
PROBE_HANDS = [
    ("AA UTG unraised", ("A", "A", ""), 0, 0, "aggressive"),
    ("KK UTG unraised", ("K", "K", ""), 0, 0, "aggressive"),
    ("AKs UTG unraised", ("A", "K", "s"), 0, 0, "aggressive"),
    ("72o UTG unraised", ("7", "2", "o"), 0, 0, "fold"),
    ("83o UTG unraised", ("8", "3", "o"), 0, 0, "fold"),
]

_RANKS = "23456789TJQKA"


def _card_int(rank, suit):
    """A concrete card code for preflop_class lookup. CONFIRM the encoding
    against common/src/game/card.h before trusting: this assumes
    code = rank_index*4 + suit_index (rank_index 0='2'..12='A')."""
    return _RANKS.index(rank) * 4 + suit


def _probe_card_id(hand):
    """Resolve a (rank, rank, 's'/'o'/'') hand label to its preflop_class id.
    Suited => same suit index; offsuit and pairs => different suit indices."""
    sixmax = _sixmax()
    r1, r2, suited = hand
    if suited == "s":
        c1, c2 = _card_int(r1, 0), _card_int(r2, 0)  # same suit
    else:
        c1, c2 = _card_int(r1, 0), _card_int(r2, 1)  # offsuit / pair
    return sixmax.preflop_class([c1, c2])


def main(argv=None):
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Blueprint plateau autopsy")
    ap.add_argument("--checkpoints",
                    default="sixmax/checkpoints/blueprint_0*.bin")
    ap.add_argument("--top-frac", type=float, default=0.05)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    sixmax = _sixmax()
    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints}")

    vocab = load_vocab_for(paths[0])
    roles = resolve_action_roles(vocab)
    probes = [(lbl, _probe_card_id(hand), st, tr, exp)
              for (lbl, hand, st, tr, exp) in PROBE_HANDS]

    reports = []
    for p in paths:
        iters, records = sixmax.dump_infosets(p)
        reports.append(checkpoint_report(iters, records, roles, probes,
                                         top_frac=args.top_frac))

    reports.sort(key=lambda r: r["iterations"])
    for r in reports:
        print(f"iters={r['iterations']:>9}  infosets={r['num_infosets']:>7}  "
              f"gini={r['gini']:.3f}  top1%={r['top1pct_share']:.3f}  "
              f"H_all={r['entropy_all']:.3f}  H_top={r['entropy_top']:.3f}  "
              f"supp_top={r['support_top']:.2f}  regret_top={r['avg_regret_top']:.3e}")
    print("\nProbe traces (expected-role mass across checkpoints):")
    for i, (lbl, *_rest) in enumerate(PROBE_HANDS):
        trace = " ".join(f"{r['probes'][i]['expected_mass']:.2f}" for r in reports)
        print(f"  {lbl:<20} [{PROBE_HANDS[i][4]:>10}]: {trace}")

    label, rationale = verdict(reports)
    print(f"\nVERDICT: {label}\n  {rationale}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iters", "num_infosets", "gini", "top1pct_share",
                        "entropy_all", "entropy_top", "support_top",
                        "avg_regret_top"])
            for r in reports:
                w.writerow([r["iterations"], r["num_infosets"], r["gini"],
                            r["top1pct_share"], r["entropy_all"],
                            r["entropy_top"], r["support_top"],
                            r["avg_regret_top"]])


if __name__ == "__main__":
    main()
```

> Note: `verdict()` is intentionally left as `TODO(human)` — see the Learn-by-Doing hand-off at execution time; the surrounding harness (metrics, probes, printing) is complete and testable without it.

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_diagnose_blueprint.py -v`
Expected: PASS (pure-function tests always; the smoke test PASSes with a checkpoint present, else SKIPs). The smoke test does not call `verdict()`, so it passes before the TODO(human) is filled.

- [ ] **Step 6: Confirm the full run end-to-end (with `verdict()` implemented)**

Run: `uv run python scripts/diagnose_blueprint.py --csv sixmax/checkpoints/autopsy.csv`
Expected: prints the per-checkpoint table, probe traces, and a `VERDICT:` line; writes the CSV.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest tests/`
Expected: all green (199 existing + new tests).

- [ ] **Step 8: Commit**

```bash
git add scripts/diagnose_blueprint.py tests/test_diagnose_blueprint.py
git commit -m "feat(sixmax): blueprint plateau autopsy driver + verdict"
```

---

## Self-Review

**Spec coverage:**
- Gap→binding (`dump_infosets`) → Task 1. ✓
- Metric 1 visit-weight distribution (Gini, top-1% share) → `checkpoint_report` (Task 3), pure `gini` (Task 2). ✓
- Metric 2 entropy all + top-tier → `entropy_all`/`entropy_top` (Task 3). ✓
- Metric 3 probe traces → `probe_policy` (Task 2) + `PROBE_HANDS`/`checkpoint_report` (Task 3). ✓
- Average-regret signal → `avg_regret_top` (Task 3). ✓
- Decision rule / verdict → `verdict()` (Task 3, TODO(human)). ✓
- Deliverable script + optional CSV → Task 3 `main()`. ✓
- Pure/bridge split + unit tests + skip-if-absent smoke test → Tasks 2 & 3. ✓
- Threshold risk → resolved by percentile tiers (`top_frac`) instead of an absolute `T`; `top-frac` is a CLI knob so the verdict can be checked at 2–3 fractions. ✓

**Placeholder scan:** No "TBD/handle appropriately". The single `TODO(human)` in `verdict()` is deliberate (learning-style hand-off) and the plan states it explicitly, with the rest of the pipeline testable without it.

**Type consistency:** `records` tuple shape `(key, probs, mass, reg)` is consistent across `dump_infosets`, `top_tier`, `probe_policy`, `checkpoint_report`. `roles` dict shape `{fold,passive,aggressive}` consistent between `resolve_action_roles` and `_role_mass`. `probes` tuple `(label, card_id, street, total_raises, expected)` consistent between `main` and `checkpoint_report`.

**Known confirmations for the implementer** (each has an explicit step): `ActionType` enum member names (Task 3 Step 3), `agents.sixmax_agent._load_vocab` signature/section name (Task 3 Step 3), and the card-int encoding in `common/src/game/card.h` (`_card_int`, Task 3 Step 4). The `_probe_card_id` suited/offsuit helper is deliberately simple — verify suited vs offsuit lands in distinct preflop classes when confirming the encoding.
