# Blueprint Preflop Range Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone script that renders the six-max blueprint's BTN-open and SB-open preflop RFI ranges as colored 13×13 terminal charts with a range-width headline, for eyeball comparison to human-professional opening frequencies.

**Architecture:** Read-only. Reuse `scripts/diagnose_blueprint.py`'s sanctioned helpers (`_sixmax`, `decode_key`, `load_vocab_for`, `resolve_action_roles`, `_card_int`) by import; index the blueprint's unopened-preflop infosets by `(live, after, preflop_class)`; map each of 169 grid cells to its class via `sixmax.preflop_class`; render an ANSI grid reusing `scripts/range_chart.py`'s color scheme.

**Tech Stack:** Python 3.10 (repo `.venv`, extension ABI), `uv run`, pytest, the compiled `sixmax` pybind11 extension.

## Global Constraints

- Language/runner: `uv run python ...`; tests via `uv run pytest`. (Per CLAUDE.md.)
- The `sixmax` extension is imported lazily through `agents.sixmax_agent` — never import `game/poker.py`, `cfr/`, or `neural_cfr/`; bridging stays in `scripts/`/`agents/`. (Hard invariant.)
- Action vocab order is config-defined; never assume indices. Get roles only via `resolve_action_roles(vocab)`. (Hard invariant.)
- Abstract-key bit layout (from `tests/sixmax/test_abstract_key.py`): `card=k&0xFF`, `street=(k>>8)&3`, `raises[st]=(k>>(10+2*st))&3`, `pot=(k>>18)&3`, `live=(k>>20)&7`, `after=(k>>23)&7`. `live` = live *opponents* (hero excluded).
- Position mapping (num_players=6, seat order UTG,HJ,CO,BTN,SB,BB): `BTN=(live=2, after=2)`, `SB=(live=1, after=1)`.
- Grid convention (from `range_chart.py`): `RANKS=[14..2]`; upper-right (i<j) = suited, diagonal (i==j) = pairs, lower-left (i>j) = offsuit.
- Read-only: never open a checkpoint for writing; never commit `.bin`/`.pkl` checkpoint files.

---

## File Structure

- `scripts/blueprint_range_chart.py` — new script. Owns: blueprint loading, infoset indexing, position mapping, per-cell strategy lookup, width math, ANSI rendering, CLI. Contains the one `TODO(human)` (cell classification).
- `tests/sixmax/test_blueprint_range_chart.py` — new test. Owns: position-mapping discrimination test + index/lookup unit tests + render smoke test.

The script is organized so the pure, testable logic (indexing, mapping, width) imports without triggering a Buck2 build only inside functions — mirror `diagnose_blueprint`'s lazy `_sixmax()` pattern.

---

## Task 1: Infoset index + position lookup

**Files:**
- Create: `scripts/blueprint_range_chart.py`
- Test: `tests/sixmax/test_blueprint_range_chart.py`

**Interfaces:**
- Consumes: `scripts.diagnose_blueprint._sixmax`, `decode_key`, `load_vocab_for`, `resolve_action_roles`; `sixmax.dump_infosets(path) -> (iters, [(key, probs, mass, regret), ...])`; `sixmax.preflop_class([c1, c2]) -> int`.
- Produces:
  - `POSITIONS: dict[str, tuple[int,int]] = {"BTN": (2, 2), "SB": (1, 1)}`
  - `build_index(records) -> dict[tuple[int,int,int], list[float]]` — maps `(live, after, card_id)` to the strategy `probs` for street-0 unopened infosets only.
  - `sample_card_ids(i, j) -> tuple[int,int]` — two concrete card codes (`rank*4+suit`) for grid cell `(i,j)` using the suited/pairs/offsuit convention.
  - `cell_probs(index, pos_key, i, j) -> list[float] | None` — strategy for a grid cell at a position, or `None` if absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_blueprint_range_chart.py
"""Blueprint preflop range chart: infoset indexing, position mapping, rendering."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_script():
    path = os.path.join(_ROOT, "scripts", "blueprint_range_chart.py")
    spec = importlib.util.spec_from_file_location("blueprint_range_chart", path)
    mod = importlib.util.module_from_spec(spec)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_positions_and_index_shape tests/sixmax/test_blueprint_range_chart.py::test_sample_card_ids_suit_convention -v`
Expected: FAIL — `No module named 'blueprint_range_chart'` / `ModuleNotFoundError`, then AttributeError once file exists but functions are missing.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Render the six-max blueprint's BTN-open and SB-open preflop RFI ranges as
13x13 terminal range charts, with a range-width headline for eyeballing against
human-professional opening frequencies.

Read-only diagnostic. Backend: sixmax/checkpoints/blueprint*.bin (the only bot
with a real BTN). See docs/superpowers/specs/2026-07-22-blueprint-range-chart-design.md.

Usage:
    uv run python scripts/blueprint_range_chart.py
    uv run python scripts/blueprint_range_chart.py --checkpoint sixmax/checkpoints/blueprint_01500000.bin
    uv run python scripts/blueprint_range_chart.py --position SB
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_blueprint import (  # sanctioned reuse
    _sixmax, decode_key, load_vocab_for, resolve_action_roles,
)

RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
RANK_LABEL = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
              9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}

POSITIONS = {"BTN": (2, 2), "SB": (1, 1)}


def build_index(records):
    """Map (live, after, card_id) -> strategy probs for street-0 unopened infosets."""
    idx = {}
    for key, probs, _mass, _regret in records:
        k = decode_key(key)
        if k["street"] != 0 or any(k["raises"]):
            continue
        idx[(k["live"], k["after"], k["card"])] = list(probs)
    return idx


def sample_card_ids(i, j):
    """Two concrete card codes (rank_index*4 + suit) for grid cell (i, j).
    Grid: RANKS high->low; i<j suited, i==j pair, i>j offsuit."""
    ri, rj = RANKS[i] - 2, RANKS[j] - 2   # rank_index 0='2'..12='A'
    if i == j:                            # pair: same rank, two suits
        return ri * 4 + 0, ri * 4 + 1
    if i < j:                             # suited: same suit
        return ri * 4 + 0, rj * 4 + 0
    return ri * 4 + 0, rj * 4 + 1         # offsuit: different suits


def cell_probs(index, pos_key, i, j):
    """Strategy probs for grid cell (i, j) at POSITIONS[pos_key], or None."""
    sixmax = _sixmax()
    live, after = POSITIONS[pos_key]
    c1, c2 = sample_card_ids(i, j)
    card_id = sixmax.preflop_class([c1, c2])
    return index.get((live, after, card_id))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_positions_and_index_shape tests/sixmax/test_blueprint_range_chart.py::test_sample_card_ids_suit_convention -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/blueprint_range_chart.py tests/sixmax/test_blueprint_range_chart.py
git commit -m "feat(sixmax): blueprint range-chart index + position lookup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Position-mapping discrimination test (validate against the real checkpoint)

**Files:**
- Modify: `tests/sixmax/test_blueprint_range_chart.py` (add one test)
- Create: (none)

**Interfaces:**
- Consumes: Task 1's `build_index`, `sample_card_ids`, `POSITIONS`; `_sixmax().dump_infosets`, `load_vocab_for`, `resolve_action_roles`; adds a helper `role_width(index, pos_key, roles) -> float` to the script (mean aggressive-mass over the 169 grid cells present).
- Produces (added to script): `role_width(index, pos_key, roles) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/sixmax/test_blueprint_range_chart.py
import glob
import pytest

_CKPTS = sorted(glob.glob(os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint_0*.bin")))


@pytest.mark.skipif(not _CKPTS, reason="no blueprint snapshot available")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_position_mapping_discriminates -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'role_width'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/blueprint_range_chart.py`:

```python
def role_width(index, pos_key, roles):
    """Mean aggressive-role mass over the 169 grid cells present for a position."""
    total, agg = 0, 0.0
    for i in range(13):
        for j in range(13):
            p = cell_probs(index, pos_key, i, j)
            if p is None:
                continue
            total += 1
            agg += sum(p[k] for k in roles["aggressive"])
    return agg / total if total else float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_position_mapping_discriminates -v`
Expected: PASS. (Empirically on the 1.5M snapshot: BTN width ≈ 0.346, SB width ≈ 0.530.)

- [ ] **Step 5: Commit**

```bash
git add scripts/blueprint_range_chart.py tests/sixmax/test_blueprint_range_chart.py
git commit -m "test(sixmax): validate BTN/SB position mapping discriminates by width

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rendering + CLI + smoke test

**Files:**
- Modify: `scripts/blueprint_range_chart.py` (add colors, `classify_cell` TODO(human), `render_position`, `latest_checkpoint`, `main`)
- Modify: `tests/sixmax/test_blueprint_range_chart.py` (add render smoke test)

**Interfaces:**
- Consumes: Task 1 + 2 functions; `resolve_action_roles`, `role_width`.
- Produces:
  - `classify_cell(probs, roles) -> str` — returns an ANSI color constant (the `TODO(human)` decision).
  - `render_position(index, pos_key, roles) -> str` — full chart text (headline + grid + summary) for one position.
  - `latest_checkpoint() -> str | None`
  - `main(argv=None)`

- [ ] **Step 1: Write the failing smoke test**

```python
# append to tests/sixmax/test_blueprint_range_chart.py
@pytest.mark.skipif(not _CKPTS, reason="no blueprint snapshot available")
def test_render_position_smoke():
    m = _load_script()
    sixmax = m._sixmax()
    ckpt = _CKPTS[-1]
    _iters, records = sixmax.dump_infosets(ckpt)
    roles = m.resolve_action_roles(m.load_vocab_for(ckpt))
    idx = m.build_index(records)
    out = m.render_position(idx, "BTN", roles)
    assert "BTN" in out
    assert "width" in out.lower()
    # 13 data rows, each rank label present
    for lbl in ("A", "K", "Q", "2"):
        assert lbl in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_render_position_smoke -v`
Expected: FAIL — `AttributeError: ... has no attribute 'render_position'`.

- [ ] **Step 3: Write implementation (color constants, render, CLI) — `classify_cell` is TODO(human)**

Add to `scripts/blueprint_range_chart.py`:

```python
# ANSI cell colors (same scheme as scripts/range_chart.py)
G  = '\033[42m\033[37m'   # green  — strong raise
Y  = '\033[43m\033[30m'   # yellow — mixed raise
B  = '\033[44m\033[37m'   # blue   — limp / call
DM = '\033[2m'            # dim    — fold
RS = '\033[0m'


def classify_cell(probs, roles):
    """Return the ANSI color constant (G/Y/B/DM) for a hand's strategy.

    `probs` is the full strategy vector; `roles` = resolve_action_roles(vocab)
    with keys 'fold', 'passive' (check/call/limp), 'aggressive' (bet/raise/allin).
    """
    raise_pct = sum(probs[i] for i in roles["aggressive"])
    limp_pct = sum(probs[i] for i in roles["passive"])
    # TODO(human): choose the raise/mixed/limp thresholds that make the
    # bot-vs-pro read honest. Return one of G, Y, B, DM.
    raise NotImplementedError


def _hand_label(i, j):
    hi, lo = RANK_LABEL[RANKS[min(i, j)]], RANK_LABEL[RANKS[max(i, j)]]
    if i == j:
        return hi * 2
    return f"{hi}{lo}{'s' if i < j else 'o'}"


def render_position(index, pos_key, roles):
    """Full chart text for one position: headline width, 13x13 grid, summary."""
    width = role_width(index, pos_key, roles)
    lines = [f"\033[1mBlueprint Preflop RFI — {pos_key} open (100BB)\033[0m",
             f"Range width: {width:.1%} raised",
             f"  {G} raise {RS} {Y} mixed {RS} {B} limp {RS} {DM}fold{RS}"
             "   upper-right=suited  diag=pairs  lower-left=offsuit",
             "      " + "".join(f" {RANK_LABEL[r]}  " for r in RANKS)]
    for i in range(13):
        row = f"  {RANK_LABEL[RANKS[i]]}  "
        for j in range(13):
            p = cell_probs(index, pos_key, i, j)
            color = classify_cell(p, roles) if p else DM
            row += f"{color}{_hand_label(i, j).ljust(3)}{RS} "
        lines.append(row)
    return "\n".join(lines)


def latest_checkpoint(directory="sixmax/checkpoints"):
    files = sorted(glob.glob(os.path.join(directory, "blueprint_0*.bin")))
    return files[-1] if files else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blueprint preflop range chart")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--position", choices=list(POSITIONS) + ["all"], default="all")
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        print("Error: no blueprint checkpoint found.")
        sys.exit(1)

    sixmax = _sixmax()
    print(f"Loading {ckpt}...")
    _iters, records = sixmax.dump_infosets(ckpt)
    roles = resolve_action_roles(load_vocab_for(ckpt))
    index = build_index(records)

    positions = list(POSITIONS) if args.position == "all" else [args.position]
    for pos in positions:
        print("\n" + render_position(index, pos, roles) + "\n")


if __name__ == "__main__":
    main()
```

> NOTE FOR IMPLEMENTER: Task 3 stops after adding the code above with `classify_cell` left as the `TODO(human)`. Do NOT implement `classify_cell` — the human writes it. Present the Learn-by-Doing request (per the active output style) and wait. The smoke test in Step 4 will fail until the human fills it in; that is expected and is the handoff point.

- [ ] **Step 4: Hand off `classify_cell` to the human, then run the smoke test**

After the human implements `classify_cell`:
Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py::test_render_position_smoke -v`
Expected: PASS.

Then run the script end-to-end for a visual check:
Run: `uv run python scripts/blueprint_range_chart.py`
Expected: two colored 13×13 charts (BTN then SB) with "Range width: 34.6% / 53.0%" headlines (values depend on the snapshot).

- [ ] **Step 5: Run the full new test file + commit**

Run: `uv run pytest tests/sixmax/test_blueprint_range_chart.py -v`
Expected: all pass (mapping tests + smoke test).

```bash
git add scripts/blueprint_range_chart.py tests/sixmax/test_blueprint_range_chart.py
git commit -m "feat(sixmax): render BTN/SB blueprint range charts + CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** purpose/scope → Tasks 1–3; data flow (index by `(live,after,card_id)`, map via `preflop_class`) → Task 1; position mapping + validating test → Tasks 1–2; rendering (scheme, width headline) → Task 3; `TODO(human)` cell classification → Task 3 handoff; smoke test → Task 3. Out-of-scope items (BB defend, references, exports) intentionally absent.
- **Placeholders:** `classify_cell` is a *deliberate* `TODO(human)` per the spec and the active output style, not a plan placeholder; every other step ships complete code.
- **Type consistency:** `build_index` returns `dict[(live,after,card_id) -> list[float]]`; `cell_probs`/`role_width`/`classify_cell`/`render_position` all consume that shape and the `roles` dict from `resolve_action_roles`. `POSITIONS` values `(2,2)`/`(1,1)` are consistent across tasks and the Global Constraints.
