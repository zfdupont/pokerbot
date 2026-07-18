# Phase 0 De-Quirking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy quirks that would complicate the six-max build: extract the evaluator behind a safe API, create the `sixmax` module skeleton with a first-class action vocabulary, fix the openpoker bridge's scale bug, and harden dev setup + bot runner.

**Architecture:** New top-level `common/` (shared C++, Buck2 target `//common:evaluator`) and `sixmax/` (new pybind11 module `sixmax.so` mirroring the `neural_cfr` build pattern). Bridge and runner fixes land in `scripts/`. `neural_cfr/` sources are never edited — only its BUCK dep list.

**Tech Stack:** C++17, Buck2, pybind11, pytest via `uv run`, zsh scripts.

## Global Constraints

- Evaluator internals move **byte-identical**; the `12 - rank` kicker inversion (lower = better) is preserved. Only the BUCK wiring and file location change.
- All `sixmax` chip quantities are **BB-denominated** (`big_blind = 1.0` frame). No absolute-chip constants.
- No hardcoded action enum or `NUM_ACTIONS`-style constant anywhere under `sixmax/`. Vocab order = config order; legality by masking, never reordering.
- Full suite `uv run pytest tests/` must be green at every commit (currently 102 passed, 1 xfailed, 2 xpassed).
- Rebuild command: `~/bin/buck2 build //neural_cfr:neural_cfr` (and `//sixmax:sixmax` once it exists).
- Never commit secrets or checkpoints.
- Spec: `docs/superpowers/specs/2026-07-17-sixmax-search-design.md`. Spec items 3 (C++ owns sixmax bucketing) and 4's engine half (seat-relative positions) have no Phase 0 deliverable — they are rules that bind Phase 1 code; this plan implements item 4's bridge-isolation half only.

---

### Task 1: Extract the evaluator into `//common:evaluator`

**Files:**
- Create: `common/BUCK`
- Move (git mv, no content edits): `neural_cfr/src/game/card.h` → `common/src/game/card.h`, `neural_cfr/src/game/card.cpp` → `common/src/game/card.cpp`
- Modify: `neural_cfr/BUCK` (add dep)

**Interfaces:**
- Consumes: existing `uint32_t evaluate_5card(std::array<Card,5>)`, `evaluate_7card(std::array<Card,7>)`.
- Produces: Buck2 target `//common:evaluator` exporting include path so `#include "game/card.h"` keeps resolving verbatim inside `neural_cfr/` — zero source edits there.

- [ ] **Step 1: Confirm the only users of card.h are in-tree**

Run: `grep -rn '#include "game/card.h"' neural_cfr/ common/ sixmax/ 2>/dev/null`
Expected: hits only under `neural_cfr/src/`. Note them; no edits needed (include path is preserved below).

- [ ] **Step 2: Move the files and create the target**

```bash
mkdir -p common/src/game
git mv neural_cfr/src/game/card.h common/src/game/card.h
git mv neural_cfr/src/game/card.cpp common/src/game/card.cpp
```

Create `common/BUCK`:

```python
cxx_library(
    name = "evaluator",
    srcs = ["src/game/card.cpp"],
    headers = ["src/game/card.h"],
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O2"],
    exported_preprocessor_flags = ["-Icommon/src"],
    preferred_linkage = "static",
    visibility = ["PUBLIC"],
)
```

In `neural_cfr/BUCK`, add `"//common:evaluator"` to the `deps` list of the `core` library (the moved files fall out of its `glob` automatically):

```python
    deps = ["//third_party:libtorch", "//third_party:indicators", "//common:evaluator"],
```

- [ ] **Step 3: Build and run the full suite**

Run: `~/bin/buck2 build //neural_cfr:neural_cfr && uv run pytest tests/ -q`
Expected: build OK; `102 passed, 1 xfailed, 2 xpassed` (identical to pre-move — this is the refactor's test).

- [ ] **Step 4: Verify byte-identity of the move**

Run: `git log --follow --format='%h %s' -1 -- common/src/game/card.cpp && git diff HEAD --stat`
Expected: rename tracked; diff shows only BUCK changes + renames (no content hunks in card.*).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract hand evaluator into //common:evaluator (byte-identical move)"
```

---

### Task 2: Safe evaluator API + `sixmax` module skeleton

**Files:**
- Create: `common/src/game/safe_eval.h`, `common/src/game/safe_eval.cpp`
- Create: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_safe_eval.py`

**Interfaces:**
- Consumes: `evaluate_7card` from Task 1's target; card int codes `0–51` with `code = (rank-2)*4 + suit` (the project-wide convention from `scripts/openpoker_bot.py::_card_to_int`).
- Produces: C++ `safe_eval::HandRank rank7(const std::array<int,7>& codes)`; `HandRank.beats(other) -> bool`; `HandRank.ties(other) -> bool`. Python module `sixmax` exposing `rank7(list[int]) -> HandRank`. Raw `uint32_t` scores are private — unrepresentable outside `common/`.

- [ ] **Step 1: Read the `Card` constructor before writing the conversion**

Run: `sed -n '1,35p' common/src/game/card.h`
Expected: the `Card` struct's fields/constructor. The conversion below assumes `Card{rank, suit}` with rank 2–14 and suit index 0–3 matching the int-code convention; adapt those two lines if the struct differs (e.g., named fields or a factory function used by the existing bindings).

- [ ] **Step 2: Write the failing test**

`tests/sixmax/test_safe_eval.py`:

```python
import pytest

sixmax = pytest.importorskip("sixmax")

def c(rank, suit):  # rank 2-14, suit 0-3 (c,d,h,s)
    return (rank - 2) * 4 + suit

def test_aces_beat_deuces():
    board = [c(7, 0), c(8, 1), c(2, 2), c(9, 3), c(13, 0)]
    aces = sixmax.rank7([c(14, 0), c(14, 1)] + board)
    deuces = sixmax.rank7([c(3, 0), c(3, 1)] + board)
    assert aces.beats(deuces)
    assert not deuces.beats(aces)

def test_kicker_order_not_inverted():
    # AK high beats AQ high on the same board — the classic 12-rank bug detector.
    board = [c(2, 0), c(7, 1), c(9, 2), c(4, 3), c(11, 0)]
    ak = sixmax.rank7([c(14, 1), c(13, 2)] + board)
    aq = sixmax.rank7([c(14, 2), c(12, 3)] + board)
    assert ak.beats(aq)

def test_ties():
    board = [c(10, 0), c(10, 1), c(4, 2), c(4, 3), c(9, 0)]
    a = sixmax.rank7([c(2, 0), c(3, 1)] + board)   # board plays
    b = sixmax.rank7([c(2, 2), c(3, 3)] + board)
    assert a.ties(b) and not a.beats(b)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_safe_eval.py -v`
Expected: SKIP ("sixmax" not importable) — the skeleton doesn't exist yet. (Skip counts as the failing state here; it flips to real assertions once the module builds.)

- [ ] **Step 4: Implement safe_eval and the module**

`common/src/game/safe_eval.h`:

```cpp
#pragma once
#include <array>

namespace safe_eval {

// Opaque hand rank. The raw evaluator score (lower = better, inverted
// kickers) never leaves common/; comparisons go through beats()/ties().
class HandRank {
public:
    bool beats(const HandRank& o) const { return score_ < o.score_; }
    bool ties(const HandRank& o) const { return score_ == o.score_; }
private:
    explicit HandRank(unsigned int s) : score_(s) {}
    unsigned int score_;
    friend HandRank rank7(const std::array<int, 7>& codes);
};

// codes: 0-51, code = (rank-2)*4 + suit  (project convention)
HandRank rank7(const std::array<int, 7>& codes);

}  // namespace safe_eval
```

`common/src/game/safe_eval.cpp`:

```cpp
#include "game/safe_eval.h"
#include "game/card.h"

namespace safe_eval {

HandRank rank7(const std::array<int, 7>& codes) {
    std::array<Card, 7> cards;
    for (int i = 0; i < 7; ++i)
        cards[i] = Card{codes[i] / 4 + 2, codes[i] % 4};  // adapt per Step 1
    return HandRank(evaluate_7card(cards));
}

}  // namespace safe_eval
```

Add `"src/game/safe_eval.cpp"` to `srcs` and `"src/game/safe_eval.h"` to `headers` in `common/BUCK`.

`sixmax/src/bindings/bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/safe_eval.h"

namespace py = pybind11;

PYBIND11_MODULE(sixmax, m) {
    m.doc() = "Six-max blueprint + search subsystem";
    py::class_<safe_eval::HandRank>(m, "HandRank")
        .def("beats", &safe_eval::HandRank::beats)
        .def("ties", &safe_eval::HandRank::ties);
    m.def("rank7", [](const std::vector<int>& codes) {
        if (codes.size() != 7) throw py::value_error("rank7 expects 7 cards");
        std::array<int, 7> a;
        std::copy(codes.begin(), codes.end(), a.begin());
        return safe_eval::rank7(a);
    });
}
```

`sixmax/BUCK` (mirrors the `neural_cfr` pybind pattern, including the `.buckconfig.local` python include convention):

```python
_python_include = read_config("python", "include_path", "")

cxx_library(
    name = "sixmax",
    srcs = ["src/bindings/bindings.cpp"],
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O2"] +
                     (["-I" + _python_include] if _python_include else []),
    deps = ["//common:evaluator", "//third_party:pybind11"],
    preferred_linkage = "shared",
    soname = "sixmax.so",
    linker_flags = ["-undefined", "dynamic_lookup"],
    visibility = ["PUBLIC"],
)
```

Make the module importable in tests: extend `tests/conftest.py` with the same `.so`-path discovery used for `neural_cfr` (locate `sixmax.so` under `buck-out` via `buck2 build //sixmax:sixmax --show-output` or a cached path, and append its dir to `sys.path` — copy the existing conftest mechanism for `neural_cfr` exactly; read `tests/conftest.py` first and follow its pattern).

- [ ] **Step 5: Build, run test to verify it passes, commit**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_safe_eval.py -v && uv run pytest tests/ -q`
Expected: 3 passed in the new file; full suite green.

```bash
git add -A && git commit -m "feat: safe_eval opaque HandRank API + sixmax module skeleton"
```

---

### Task 3: `ActionVocab` in C++ with translation

**Files:**
- Create: `sixmax/src/vocab/vocab.h`, `sixmax/src/vocab/vocab.cpp`
- Modify: `sixmax/BUCK` (add vocab srcs), `sixmax/src/bindings/bindings.cpp` (bind vocab)
- Test: `tests/sixmax/test_vocab.py`

**Interfaces:**
- Consumes: nothing from other tasks (self-contained).
- Produces (C++ and via bindings):
  - `enum class ActionType { Fold, Check, Call, Bet, AllIn }` and `enum class SizeUnit { BB, Pot }`
  - `struct AbstractAction { ActionType type; double size; SizeUnit unit; }`
  - `struct BetContext { double pot; double current_bet; double to_call; double stack; }` — **all in BB**
  - `class ActionVocab`: ctor from `std::vector<AbstractAction>` (canonical order = given order); `int size()`; `AbstractAction at(int)`; `double target_bb(int idx, const BetContext&)`; `int nearest(double bet_to_bb, const BetContext&, double u)` (`u` ∈ [0,1) supplied by caller — deterministic, testable pseudo-harmonic); `uint64_t hash()` (FNV-1a over type/size/unit sequence).

- [ ] **Step 1: Write the failing tests**

`tests/sixmax/test_vocab.py`:

```python
import pytest

sixmax = pytest.importorskip("sixmax")
A = sixmax.AbstractAction
T = sixmax.ActionType
U = sixmax.SizeUnit


def spec_vocab():
    return sixmax.ActionVocab([
        A(T.Fold, 0.0, U.BB), A(T.Check, 0.0, U.BB), A(T.Call, 0.0, U.BB),
        A(T.Bet, 2.5, U.BB), A(T.Bet, 3.5, U.BB), A(T.Bet, 5.0, U.BB),
        A(T.Bet, 0.33, U.Pot), A(T.Bet, 0.75, U.Pot), A(T.Bet, 1.5, U.Pot),
        A(T.AllIn, 0.0, U.BB),
    ])


def test_order_is_canonical_and_sized():
    v = spec_vocab()
    assert v.size() == 10
    assert v.at(3).size == 2.5 and v.at(3).unit == U.BB


def test_target_bb_units():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=7.5, current_bet=2.0, to_call=1.0, stack=100.0)
    assert v.target_bb(3, ctx) == 2.5                    # BB unit: raise-to 2.5BB
    # Pot unit: raise-to = current_bet + size * (pot + 2*to_call)   [spec formula]
    assert v.target_bb(7, ctx) == pytest.approx(2.0 + 0.75 * (7.5 + 2.0))
    assert v.target_bb(9, ctx) == 100.0                  # all-in = stack


def test_target_bb_capped_by_stack():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=200.0, current_bet=0.0, to_call=0.0, stack=10.0)
    assert v.target_bb(8, ctx) == 10.0                   # 1.5x pot capped to stack


def test_nearest_is_pseudo_harmonic():
    v = spec_vocab()
    ctx = sixmax.BetContext(pot=10.0, current_bet=0.0, to_call=0.0, stack=100.0)
    # Observed bet exactly on a grid point maps there regardless of u.
    on_grid = 0.75 * 10.0
    assert v.nearest(on_grid, ctx, 0.0) == 7
    assert v.nearest(on_grid, ctx, 0.999) == 7
    # Between 0.33x and 0.75x pot: low u -> smaller size, high u -> larger.
    between = 0.5 * 10.0
    assert v.nearest(between, ctx, 0.0) == 6
    assert v.nearest(between, ctx, 0.999) == 7


def test_hash_changes_with_vocab():
    v1, v2 = spec_vocab(), spec_vocab()
    assert v1.hash() == v2.hash()
    v3 = sixmax.ActionVocab([A(T.Fold, 0.0, U.BB), A(T.AllIn, 0.0, U.BB)])
    assert v3.hash() != v1.hash()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_vocab.py -v`
Expected: FAIL/ERROR — `sixmax` imports (Task 2) but has no `ActionVocab` attribute.

- [ ] **Step 3: Implement**

`sixmax/src/vocab/vocab.h`:

```cpp
#pragma once
#include <cstdint>
#include <vector>

namespace sixmax {

enum class ActionType { Fold, Check, Call, Bet, AllIn };
enum class SizeUnit { BB, Pot };

struct AbstractAction {
    ActionType type;
    double size;      // BB unit: raise-to in BB. Pot unit: pot fraction.
    SizeUnit unit;
};

// All quantities in big blinds (big_blind == 1.0 frame).
struct BetContext {
    double pot;
    double current_bet;
    double to_call;
    double stack;
};

class ActionVocab {
public:
    explicit ActionVocab(std::vector<AbstractAction> actions);
    int size() const { return (int)actions_.size(); }
    const AbstractAction& at(int i) const { return actions_[i]; }
    // Bet/raise-to total in BB for action idx in this context; capped by stack.
    double target_bb(int idx, const BetContext& ctx) const;
    // Map an observed bet (raise-to, BB) onto the grid with randomized
    // pseudo-harmonic weighting; u in [0,1) supplied by caller.
    int nearest(double bet_to_bb, const BetContext& ctx, double u) const;
    uint64_t hash() const { return hash_; }

private:
    std::vector<AbstractAction> actions_;
    uint64_t hash_;
};

}  // namespace sixmax
```

`sixmax/src/vocab/vocab.cpp`:

```cpp
#include "vocab/vocab.h"
#include <algorithm>
#include <cmath>

namespace sixmax {

static uint64_t fnv1a(const std::vector<AbstractAction>& as) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    for (const auto& a : as) {
        mix((uint64_t)a.type);
        mix((uint64_t)a.unit);
        uint64_t bits; double s = a.size;
        static_assert(sizeof(bits) == sizeof(s));
        __builtin_memcpy(&bits, &s, sizeof(bits));
        mix(bits);
    }
    return h;
}

ActionVocab::ActionVocab(std::vector<AbstractAction> actions)
    : actions_(std::move(actions)), hash_(fnv1a(actions_)) {}

double ActionVocab::target_bb(int idx, const BetContext& ctx) const {
    const auto& a = actions_[idx];
    double target;
    switch (a.type) {
        case ActionType::AllIn: return ctx.stack;
        case ActionType::Bet:
            target = (a.unit == SizeUnit::BB)
                ? a.size
                : ctx.current_bet + a.size * (ctx.pot + 2.0 * ctx.to_call);
            return std::min(target, ctx.stack);
        default: return 0.0;  // fold/check/call carry no size
    }
}

int ActionVocab::nearest(double bet_to_bb, const BetContext& ctx, double u) const {
    // Candidates: sizeable actions (Bet/AllIn) by their target_bb in this ctx.
    int lo = -1, hi = -1;
    double lo_v = -1e18, hi_v = 1e18;
    for (int i = 0; i < size(); ++i) {
        const auto t = actions_[i].type;
        if (t != ActionType::Bet && t != ActionType::AllIn) continue;
        double v = target_bb(i, ctx);
        if (v <= bet_to_bb && v > lo_v) { lo = i; lo_v = v; }
        if (v >= bet_to_bb && v < hi_v) { hi = i; hi_v = v; }
    }
    if (lo == -1) return hi;
    if (hi == -1) return lo;
    if (lo == hi || lo_v == hi_v) return lo;
    // Pseudo-harmonic (Ganzfried & Sandholm): P(map to lo) for x in [A,B]:
    //   p = (B - x) * (1 + A) / ((B - A) * (1 + x))
    double A = lo_v, B = hi_v, x = bet_to_bb;
    double p = (B - x) * (1.0 + A) / ((B - A) * (1.0 + x));
    return (u < p) ? lo : hi;
}

}  // namespace sixmax
```

Bindings — append inside `PYBIND11_MODULE` in `sixmax/src/bindings/bindings.cpp` (add `#include "vocab/vocab.h"`):

```cpp
    py::enum_<sixmax::ActionType>(m, "ActionType")
        .value("Fold", sixmax::ActionType::Fold).value("Check", sixmax::ActionType::Check)
        .value("Call", sixmax::ActionType::Call).value("Bet", sixmax::ActionType::Bet)
        .value("AllIn", sixmax::ActionType::AllIn);
    py::enum_<sixmax::SizeUnit>(m, "SizeUnit")
        .value("BB", sixmax::SizeUnit::BB).value("Pot", sixmax::SizeUnit::Pot);
    py::class_<sixmax::AbstractAction>(m, "AbstractAction")
        .def(py::init<sixmax::ActionType, double, sixmax::SizeUnit>())
        .def_readonly("type", &sixmax::AbstractAction::type)
        .def_readonly("size", &sixmax::AbstractAction::size)
        .def_readonly("unit", &sixmax::AbstractAction::unit);
    py::class_<sixmax::BetContext>(m, "BetContext")
        .def(py::init([](double pot, double current_bet, double to_call, double stack) {
            return sixmax::BetContext{pot, current_bet, to_call, stack};
        }), py::kw_only(), py::arg("pot"), py::arg("current_bet"),
            py::arg("to_call"), py::arg("stack"));
    py::class_<sixmax::ActionVocab>(m, "ActionVocab")
        .def(py::init<std::vector<sixmax::AbstractAction>>())
        .def("size", &sixmax::ActionVocab::size)
        .def("at", &sixmax::ActionVocab::at)
        .def("target_bb", &sixmax::ActionVocab::target_bb)
        .def("nearest", &sixmax::ActionVocab::nearest)
        .def("hash", &sixmax::ActionVocab::hash);
```

In `sixmax/BUCK`, add `"src/vocab/vocab.cpp"` to `srcs`, `"src/vocab/vocab.h"` to a `headers` glob, and `exported_preprocessor_flags = ["-Isixmax/src"]`.

- [ ] **Step 4: Build, run tests to verify they pass**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_vocab.py -v && uv run pytest tests/ -q`
Expected: 5 passed; full suite green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: ActionVocab with BB/pot units, pseudo-harmonic translation, vocab hash"
```

---

### Task 4: Python vocab config loader

**Files:**
- Create: `sixmax/vocab_config.py`, `sixmax/configs/default.toml`
- Test: `tests/sixmax/test_vocab_config.py`

**Interfaces:**
- Consumes: `sixmax.ActionVocab`, `sixmax.AbstractAction`, `sixmax.ActionType`, `sixmax.SizeUnit` (Task 3).
- Produces: `load_vocab(toml_path: str, section: str = "blueprint") -> sixmax.ActionVocab`. Canonical construction order: Fold, Check, Call, preflop_opens (config order), bet_sizes (config order), AllIn-if-enabled.

- [ ] **Step 1: Write the failing test**

`tests/sixmax/test_vocab_config.py`:

```python
import pytest

sixmax = pytest.importorskip("sixmax")
from sixmax.vocab_config import load_vocab   # noqa: E402


def test_load_spec_default(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text("""
[actions.blueprint]
preflop_opens = [ { size = 2.5, unit = "bb" },
                  { size = 3.5, unit = "bb" },
                  { size = 5.0, unit = "bb" } ]
bet_sizes     = [ { size = 0.33, unit = "pot" },
                  { size = 0.75, unit = "pot" },
                  { size = 1.5,  unit = "pot" } ]
include_allin = true
""")
    v = load_vocab(str(cfg), "blueprint")
    assert v.size() == 10  # fold, check, call, 3 opens, 3 bets, allin
    assert v.at(0).type == sixmax.ActionType.Fold
    assert v.at(3).size == 2.5 and v.at(3).unit == sixmax.SizeUnit.BB
    assert v.at(6).size == 0.33 and v.at(6).unit == sixmax.SizeUnit.Pot
    assert v.at(9).type == sixmax.ActionType.AllIn


def test_repo_default_config_loads():
    v = load_vocab("sixmax/configs/default.toml", "blueprint")
    assert v.size() == 10


def test_missing_section_raises(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text("[actions.blueprint]\ninclude_allin = true\n")
    with pytest.raises(KeyError):
        load_vocab(str(cfg), "search")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_vocab_config.py -v`
Expected: FAIL — `No module named 'sixmax.vocab_config'`. Note: the compiled module is `sixmax.so`; the Python package half lives in `sixmax/` as plain files. If the import collides with the `.so` (both named `sixmax`), the loader moves to `sixmax/vocab_config.py` imported as a top-level file via the tests' path setup — resolve per the conftest pattern and record the choice in the module docstring.

- [ ] **Step 3: Implement**

`sixmax/vocab_config.py`:

```python
"""Build an ActionVocab from a TOML [actions.<section>] block.

Canonical order (== storage order in every artifact):
fold, check, call, preflop_opens (config order), bet_sizes (config order),
all-in last if enabled. Any change to this ordering rule invalidates
checkpoints — it is part of the artifact contract with vocab.hash().
"""
import tomllib

import sixmax

_UNITS = {"bb": sixmax.SizeUnit.BB, "pot": sixmax.SizeUnit.Pot}


def load_vocab(toml_path: str, section: str = "blueprint") -> "sixmax.ActionVocab":
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    try:
        block = data["actions"][section]
    except KeyError:
        raise KeyError(f"[actions.{section}] not found in {toml_path}")

    A, T = sixmax.AbstractAction, sixmax.ActionType
    actions = [A(T.Fold, 0.0, sixmax.SizeUnit.BB),
               A(T.Check, 0.0, sixmax.SizeUnit.BB),
               A(T.Call, 0.0, sixmax.SizeUnit.BB)]
    for entry in block.get("preflop_opens", []) + block.get("bet_sizes", []):
        actions.append(A(T.Bet, float(entry["size"]), _UNITS[entry["unit"]]))
    if block.get("include_allin", False):
        actions.append(A(T.AllIn, 0.0, sixmax.SizeUnit.BB))
    return sixmax.ActionVocab(actions)
```

`sixmax/configs/default.toml`: the exact `[actions.blueprint]` + `[actions.search]` blocks from the spec (copy verbatim from `docs/superpowers/specs/2026-07-17-sixmax-search-design.md`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sixmax/ -v && uv run pytest tests/ -q`
Expected: all sixmax tests pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: vocab TOML loader with canonical ordering contract"
```

---

### Task 5: Bridge fixes — BB-derived scale + HU isolation

**Files:**
- Modify: `scripts/openpoker_bot.py` (`_decide_neural`, `HandTracker.on_hand_start`)
- Test: `tests/scripts/test_openpoker_bridge.py`

**Interfaces:**
- Consumes: existing `HandTracker`.
- Produces: `HandTracker._decide_neural` scales by `self.big_blind` (not `buy_in/100`); `HandTracker.hu_position() -> int` isolates the binary SB/BB quirk with a warning docstring.

- [ ] **Step 1: Write the failing test**

`tests/scripts/test_openpoker_bridge.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.openpoker_bot import HandTracker
from models.card import Card
from models.enums import Suit


class SpyStrategy:
    def __init__(self):
        self.kwargs = None
        self.args = None
    def get_action_probs(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return {"fold": 0.5, "call": 0.5}


def make_tracker(big_blind):
    t = HandTracker()
    t.big_blind = big_blind
    t.my_stack = 50 * big_blind          # 50BB stack in chips
    t.my_committed = big_blind
    t.hole_cards = [Card(14, Suit.HEARTS), Card(13, Suit.HEARTS)]
    t.community_cards = []
    return t


def test_scale_derives_from_big_blind_not_buy_in():
    spy = SpyStrategy()
    t = make_tracker(big_blind=20)
    # buy_in deliberately NOT 100*bb: the old bug scaled by buy_in/100.
    t._decide_neural(spy, to_call_chips=40, pot=60, buy_in=1000,
                     abstract_legal=["fold", "call"])
    # stack must arrive in the training frame: 50BB stack -> 50.0
    assert spy.args[4] == 50.0   # my_stack / scale
    assert spy.args[3] == 3.0    # pot 60 chips / bb20
    assert spy.args[5] == 2.0    # to_call 40 chips / bb20
```

(Argument positions match the existing `get_action_probs(hole, board, street, pot, stack, to_call, ...)` call — verify indices against the call site while writing, and pin them in the test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_openpoker_bridge.py -v`
Expected: FAIL — with `buy_in=1000` the old code uses `scale=10`, so stack arrives as `100.0`, not `50.0`.

- [ ] **Step 3: Fix the scale and isolate HU position**

In `scripts/openpoker_bot.py::_decide_neural`, replace:

```python
        scale      = buy_in / 100.0   # normalize to training scale (stack=100)
```

with:

```python
        # Training frame is big_blind == 1: dividing by the table's big blind
        # is the whole rescale. (buy_in/100 was only correct at exactly 100BB.)
        scale      = float(self.big_blind)
```

Add to `HandTracker`:

```python
    def hu_position(self) -> int:
        """Binary 0=SB/1=BB position — a heads-up-only concept.

        Quarantined here per Phase 0 item 4: the sixmax path uses
        seat-relative-to-button positions and must never consume this.
        """
        return self.my_position
```

and use `self.hu_position()` at the `get_action_probs(..., self.my_position, ...)` call site so the quirk has exactly one named exit point.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_openpoker_bridge.py tests/ -q`
Expected: new test passes; full suite green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix: bridge scale derives from big_blind; quarantine HU position quirk"
```

---

### Task 6: Runner hygiene — graceful shutdown + process management

**Files:**
- Modify: `scripts/openpoker_bot.py` (`run()`, `main()`)
- Create: `scripts/run_openpoker.sh`
- Test: `tests/scripts/test_openpoker_shutdown.py`

**Interfaces:**
- Consumes: existing `run()` async loop.
- Produces: `async def graceful_shutdown(ws) -> None` (sends `leave_table`, closes socket); SIGTERM/SIGINT wired via `loop.add_signal_handler`; `scripts/run_openpoker.sh {start|stop|status}` managing the *python* PID directly with `caffeinate -w` alongside.

- [ ] **Step 1: Write the failing test**

`tests/scripts/test_openpoker_shutdown.py`:

```python
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.openpoker_bot import graceful_shutdown


class FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False
    async def send(self, msg):
        self.sent.append(json.loads(msg))
    async def close(self):
        self.closed = True


def test_graceful_shutdown_leaves_table_then_closes():
    ws = FakeWS()
    asyncio.run(graceful_shutdown(ws))
    assert {"type": "leave_table"} in ws.sent
    assert ws.closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_openpoker_shutdown.py -v`
Expected: FAIL — `cannot import name 'graceful_shutdown'`.

- [ ] **Step 3: Implement shutdown, signal wiring, and the run script**

In `scripts/openpoker_bot.py` add (module level):

```python
async def graceful_shutdown(ws) -> None:
    """Bank the table stack before dying: leave_table, then close.

    Best-effort — errors are swallowed because this runs on the way out."""
    try:
        await ws.send(json.dumps({"type": "leave_table"}))
        await ws.close()
    except Exception:
        pass
```

In `run()`, wire signals once per connection (inside the `async for ws ...: try:` block, before the message loop):

```python
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda: asyncio.ensure_future(_shutdown_and_exit(ws)))
```

with, at module level:

```python
async def _shutdown_and_exit(ws) -> None:
    log.info("Signal received — leaving table and shutting down")
    await graceful_shutdown(ws)
    sys.exit(0)
```

(add `import signal` to the imports). In `main()`, force real-time logs:

```python
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
```

`scripts/run_openpoker.sh`:

```bash
#!/bin/zsh
# Manage the openpoker bot: tracks the *python* PID (not a wrapper).
set -euo pipefail
cd "$(dirname "$0")/.."
PIDFILE=.openpoker.pid
LOG=neural_cfr/checkpoints/openpoker_$(date +%Y%m%d).log

case "${1:-}" in
  start)
    [ -f .env ] && { set -a; source .env; set +a; }
    [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null && \
      { echo "already running ($(cat $PIDFILE))"; exit 1; }
    nohup uv run python scripts/openpoker_bot.py \
        --checkpoint "${CHECKPOINT:-neural_cfr/checkpoints/best_checkpoint.pt}" \
        --buy-in "${BUY_IN:-2000}" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    caffeinate -w "$(cat $PIDFILE)" &
    echo "started $(cat $PIDFILE), log: $LOG" ;;
  stop)
    kill -TERM "$(cat $PIDFILE)" && rm -f "$PIDFILE" && echo "stopped" ;;
  status)
    kill -0 "$(cat $PIDFILE)" 2>/dev/null && echo "running $(cat $PIDFILE)" || echo "not running" ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
```

Run: `chmod +x scripts/run_openpoker.sh`

Note: `nohup` directly on `uv run python` makes `$!` the process the pidfile tracks; `caffeinate -w PID` keeps the machine awake *watching* that PID instead of wrapping it — `stop` signals the bot itself, fixing the killed-the-wrapper incident.

- [ ] **Step 4: Run tests, and manually verify the script's stop path**

Run: `uv run pytest tests/scripts/ -v && uv run pytest tests/ -q`
Expected: green. Script smoke test (no API key needed): `./scripts/run_openpoker.sh status` → "not running".

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: graceful leave_table shutdown, line-buffered logs, run_openpoker.sh"
```

---

### Task 7: Scripted dev setup + scaffold docs

**Files:**
- Create: `scripts/setup_dev.sh`
- Modify: `.mex/context/setup.md` (document the script + new targets), `.mex/context/architecture.md` (add `common/` + `sixmax/` to the module map)

**Interfaces:**
- Consumes: the memory-documented quirks — `.venv` must be Python 3.10 (extension ABI), `third_party/libtorch` symlinks needed in fresh worktrees.
- Produces: idempotent `scripts/setup_dev.sh` that a clean clone/worktree/cloud box runs once.

- [ ] **Step 1: Write the script**

`scripts/setup_dev.sh`:

```bash
#!/bin/zsh
# Idempotent dev setup: run from a fresh clone, worktree, or cloud machine.
set -euo pipefail
cd "$(dirname "$0")/.."
fail() { echo "SETUP FAIL: $1" >&2; exit 1; }

# 1. Python: extension ABI is pinned to 3.10 (sixmax.so / neural_cfr.so).
uv python pin 3.10 >/dev/null 2>&1 || true
uv sync
PYV=$(uv run python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[ "$PYV" = "3.10" ] || fail "venv resolved python $PYV, need 3.10 (extension ABI)"

# 2. third_party symlinks (worktrees don't inherit them).
MAIN_REPO=$(git rev-parse --path-format=absolute --git-common-dir)/..
for d in libtorch pybind11 indicators; do
  [ -e "third_party/$d" ] || ln -s "$MAIN_REPO/third_party/$d" "third_party/$d"
done

# 3. buck2 + python include path for pybind targets.
[ -x ~/bin/buck2 ] || fail "~/bin/buck2 not found (see .mex/context/setup.md)"
if [ ! -f .buckconfig.local ]; then
  INC=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))")
  printf '[python]\n  include_path = %s\n' "$INC" > .buckconfig.local
fi

# 4. Prove it.
~/bin/buck2 build //neural_cfr:neural_cfr //sixmax:sixmax
uv run pytest tests/ -q
echo "SETUP OK"
```

Run: `chmod +x scripts/setup_dev.sh`

- [ ] **Step 2: Run it on this machine (idempotency check)**

Run: `./scripts/setup_dev.sh`
Expected: `SETUP OK` with full suite green — on an already-configured machine it must change nothing and pass.

- [ ] **Step 3: Update the scaffold**

In `.mex/context/setup.md`: add `./scripts/setup_dev.sh` as the first setup step and `./scripts/run_openpoker.sh {start|stop|status}` under deployment commands. In `.mex/context/architecture.md`: add `common/` (shared C++ evaluator, safe API) and `sixmax/` (six-max subsystem, Phase 0 = vocab + skeleton; spec link) to the module map. Follow each file's existing structure and tone.

- [ ] **Step 4: Full suite one more time**

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: idempotent setup_dev.sh; scaffold docs for common/ and sixmax/"
```
