# Six-Max Phase 1c — Blueprint Deployment + HU-Mode Sanity Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the six-max MCCFR blueprint (`sixmax/`) to live play — an openpoker.ai strategy loader over `BlueprintStrategy` with off-tree bet translation via `ActionVocab`, plus a HU-mode sanity eval that pits the blueprint (2-player) against the frozen tabular and neural bots in the live engine.

**Architecture:** The deployment hinge is *infoset-key reconstruction*: to query `BlueprintStrategy::probs(key)` from a live table we must reproduce the exact bit-packed key the trainer used (`EngineGameState::abstract_key`). We extract that packing into one shared C++ function (`pack_abstract_key`) exposed to Python, so the bridge can never drift from the trainer — the same "one translation layer" discipline `ActionVocab` already enforces. A single Python bridge (`agents/sixmax_agent.py`) reconstructs keys from public state and selects/translates actions; two host adapters consume it — a `SixmaxAgent(PokerAgent)` for the live engine (HU eval) and the `HandTracker` in `openpoker_bot.py` (live deployment).

**Tech Stack:** C++17 (libtorch/pybind11/Buck2 subsystem, built with `~/bin/buck2`), Python 3.10 (`.venv` ABI pin), pytest, the `game/poker.py` live engine, `websockets` for the openpoker connector.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from `.mex/AGENTS.md` and the code being extended.

- **Subsystem isolation:** `cfr/`, `neural_cfr/`, and `sixmax/` never import `game/poker.py` or each other. All bridging lives only in `agents/` and `scripts/`; shared C++ lives only in `common/`. (New Python bridge files go in `agents/` and `scripts/`; new C++ stays inside `sixmax/`.)
- **Action vocabulary:** in `sixmax/` the vocab is config-defined; canonical order == config order == storage order; every checkpoint embeds `vocab.hash()` and loaders refuse mismatches. Everywhere: **mask illegal actions; never reorder or filter storage.**
- **Evaluator inversion:** the `12 - rank` kicker inversion (lower = better) is preserved; new consumers use only the opaque `safe_eval` API. This plan touches no evaluator code.
- **Chip frame:** chips crossing an engine boundary are rescaled to the `starting_stack=100, big_blind=1` training frame by **dividing by the table's big blind**. `sixmax/` is BB-denominated natively.
- **Never commit secrets** (`OPENPOKER_API_KEY`) **or checkpoint files** (`*.bin`, `*.pt`, `*.pkl`). Test checkpoints are generated into `tmp_path` at runtime, never committed.
- **Abstraction infoset-key bit layout** (pinned; from `sixmax/src/blueprint/engine_game.cpp`):
  `card` bits 0–7 · `street` bits 8–9 · `raises[st]` (2 bits each, st=0..3) bits 10–17 · `pot_bucket` bits 18–19 · `live` bits 20–22 · `after` bits 23–25.
- **pot_bucket thresholds** (BB): `pot ≤ 7 → 0`, `≤ 15 → 1`, `≤ 40 → 2`, else `3`.
- **Canonical action order** (from `abstract_key`): `start = (street==0) ? (n==2 ? button : (button+3)%n) : (button+1)%n`; `order(seat) = (seat - start + n) % n`. `live` = opponents not folded (excluding hero); `after` = live opponents that are **not all-in** and whose `order` > hero's `order`. Positions are table indices `0..n-1`.
- **Raise-count caps are per-consumer abstraction knobs, applied at consumption, not in the shared engine.** `game/poker.py`'s `raises_per_street` is faithful (uncapped); the sixmax abstraction clamps to 3, the tabular abstraction to 2, and neural to 2 (`inference.cpp:36`). (Task 4 Step 1 performs the uncap + moves the tabular clamp into `cfr_agent.py`.)
- **Build/test commands:** `~/bin/buck2 build //sixmax:sixmax`; `uv run pytest tests/sixmax/ -q`; full suite `uv run pytest tests/`.

---

## File Structure

**Create:**
- `sixmax/src/abstraction/abstract_key.h` — declarations for `pot_bucket(double)` and `pack_abstract_key(...)`.
- `sixmax/src/abstraction/abstract_key.cpp` — the single implementation of the bit packing + pot bucketing.
- `agents/sixmax_agent.py` — the host-agnostic `SixmaxDeployStrategy`, the `canonical_live_after` helper, and `SixmaxAgent(PokerAgent)`.
- `agents/neural_agent.py` — thin `NeuralAgent(PokerAgent)` wrapping `neural_cfr.Strategy` for the live engine.
- `scripts/eval_hu_sanity.py` — duplicate-deal HU eval in `game/poker.py`: blueprint vs tabular and vs neural.
- `tests/sixmax/test_pack_abstract_key.py` — the shared packer reproduces `EngineGameState::abstract_key`.
- `tests/agents/test_sixmax_agent.py` — bridge key packing, legality/translation, `canonical_live_after`, and a full live HU hand.
- `tests/agents/test_neural_agent.py` — neural agent plays a legal live hand.
- `tests/scripts/test_eval_hu_sanity.py` — eval smoke test (few hands, finite BB/100).
- `tests/scripts/test_openpoker_sixmax.py` — `HandTracker` decodes a scripted six-max message stream to a legal action.
- root `conftest.py` (repo root, alongside `pyproject.toml`) — registers `pytest_plugins = ["tests.fixtures.sixmax_checkpoints"]` (must be the top-level conftest under pytest ≥ 9); the session fixtures themselves live in `tests/fixtures/sixmax_checkpoints.py` and train tiny n=2 and n=6 blueprints into `tmp_path_factory`.

**Modify:**
- `sixmax/src/blueprint/engine_game.cpp` — `abstract_key` delegates to `pack_abstract_key`; drop the file-local `pot_bucket`.
- `sixmax/src/bindings/bindings.cpp` — bind `pack_abstract_key` and `pot_bucket`.
- `sixmax/BUCK` — add `src/abstraction/abstract_key.cpp` to `srcs`.
- `game/poker.py` — uncap `raises_per_street` in the game state (make it faithful; consumers cap).
- `models/state.py` — update the `raises_per_street` comment (faithful; consumer-capped).
- `agents/cfr_agent.py` — clamp `raises_per_street` to 2 when building the tabular InfoSet.
- `scripts/openpoker_bot.py` — detect `.bin` checkpoints → `SixmaxDeployStrategy`; extend `HandTracker` for six-max (seat/fold/all-in tracking, raise cap 3, live/after, vocab translation).

---

## Task 1: Shared abstract-key packing in C++

Extract the bit packing and pot bucketing out of `EngineGameState::abstract_key` into a free function usable from Python, so the deployment bridge computes byte-identical keys to the trainer.

**Files:**
- Create: `sixmax/src/abstraction/abstract_key.h`, `sixmax/src/abstraction/abstract_key.cpp`
- Modify: `sixmax/src/blueprint/engine_game.cpp:98-134`, `sixmax/src/bindings/bindings.cpp`, `sixmax/BUCK`
- Test: `tests/sixmax/test_pack_abstract_key.py`

**Interfaces:**
- Produces (C++): `int sixmax::pot_bucket(double pot_bb);` and
  `uint64_t sixmax::pack_abstract_key(int card, int street, const std::array<uint8_t,4>& raises, double pot_bb, int live, int after);`
- Produces (Python binding): `sixmax.pot_bucket(pot_bb: float) -> int` and
  `sixmax.pack_abstract_key(card: int, street: int, raises: list[int], pot_bb: float, live: int, after: int) -> int` (raises must have length 4).

- [ ] **Step 1: Write the failing test**

Create `tests/sixmax/test_pack_abstract_key.py`:

```python
"""The Python-exposed packer must reproduce EngineGameState::abstract_key
exactly, and the pinned bit layout must be stable — this is the contract the
deployment bridge relies on to hit trained infosets."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(
        os.path.join(_ROOT, "sixmax", "configs", "default.toml"), "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)


def card(rank, suit):
    return (rank - 2) * 4 + suit


AKo = [card(14, 3), card(13, 1)]  # preflop class 155


def test_pot_bucket_thresholds():
    assert sixmax.pot_bucket(1.5) == 0
    assert sixmax.pot_bucket(7.0) == 0
    assert sixmax.pot_bucket(7.01) == 1
    assert sixmax.pot_bucket(15.0) == 1
    assert sixmax.pot_bucket(40.0) == 2
    assert sixmax.pot_bucket(40.01) == 3


def test_pack_matches_bit_layout():
    # card=155, street=0, no raises, pot 1.5 BB (bucket 0), live=4, after=4
    expected = 155 | (0 << 8) | (0 << 18) | (4 << 20) | (4 << 23)
    assert sixmax.pack_abstract_key(155, 0, [0, 0, 0, 0], 1.5, 4, 4) == expected


def test_pack_reproduces_engine_key():
    # Rebuild the 6-handed MP spot from test_abstract_key and check the packer
    # reproduces the engine's own key from independently supplied fields.
    cfg = sixmax.EngineConfig(num_players=6)
    deck = list(range(17))
    spares = iter(range(40, 52))
    taken = set(AKo)
    deck = [c if c not in taken else next(spares) for c in deck]
    deck[2 * 4], deck[2 * 4 + 1] = AKo[0], AKo[1]  # hero seat 4
    s = sixmax.EngineGameState(cfg, 0, deck, VOCAB, [], abstraction=ABS)
    s.apply(0)  # UTG (seat 3) folds -> hero (seat 4) to act
    # Fields for the packer: preflop AKo -> card 155, street 0, no raises,
    # pot 1.5 BB, live=4 opponents, after=4 (SB, BB, and two behind).
    packed = sixmax.pack_abstract_key(155, 0, [0, 0, 0, 0], 1.5, 4, 4)
    assert packed == s.infoset_key()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_pack_abstract_key.py -q`
Expected: FAIL — `AttributeError: module 'sixmax' has no attribute 'pot_bucket'`.

- [ ] **Step 3: Create the shared header**

Create `sixmax/src/abstraction/abstract_key.h`:

```cpp
#pragma once
#include <array>
#include <cstdint>

namespace sixmax {

// Pot-size bucket used by the abstraction infoset key. Thresholds are part of
// the artifact contract; changing them invalidates checkpoints.
int pot_bucket(double pot_bb);

// The single source of truth for the abstraction infoset-key bit layout.
// Both EngineGameState::abstract_key (trainer) and the deployment bridge
// (Python, via bindings) route through this so they can never diverge.
//   card   bits 0-7   street bits 8-9   raises[st] bits 10-17 (2 bits each)
//   pot    bits 18-19 live   bits 20-22 after      bits 23-25
uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after);

}  // namespace sixmax
```

- [ ] **Step 4: Create the shared implementation**

Create `sixmax/src/abstraction/abstract_key.cpp`:

```cpp
#include "abstraction/abstract_key.h"

namespace sixmax {

int pot_bucket(double pot_bb) {
    if (pot_bb <= 7.0) return 0;
    if (pot_bb <= 15.0) return 1;
    if (pot_bb <= 40.0) return 2;
    return 3;
}

uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after) {
    uint64_t key = (uint64_t)card;                       // bits 0-7
    key |= (uint64_t)street << 8;                        // bits 8-9
    for (int st = 0; st < 4; ++st)
        key |= (uint64_t)raises[st] << (10 + 2 * st);    // bits 10-17
    key |= (uint64_t)pot_bucket(pot_bb) << 18;           // bits 18-19
    key |= (uint64_t)live << 20;                         // bits 20-22
    key |= (uint64_t)after << 23;                        // bits 23-25
    return key;
}

}  // namespace sixmax
```

- [ ] **Step 5: Refactor `engine_game.cpp` to delegate**

In `sixmax/src/blueprint/engine_game.cpp`, add the include near the top (after the existing `#include "blueprint/engine_game.h"`):

```cpp
#include "abstraction/abstract_key.h"
```

Delete the file-local anonymous-namespace `pot_bucket` (lines 98-105):

```cpp
namespace {
int pot_bucket(double pot) {
    if (pot <= 7.0) return 0;
    if (pot <= 15.0) return 1;
    if (pot <= 40.0) return 2;
    return 3;
}
}  // namespace
```

Replace the tail of `abstract_key` (the manual packing, lines 126-133) with a delegation. The function now ends:

```cpp
    uint64_t key = pack_abstract_key((int)card, street, raises_,
                                     hand_.pot(), live, after);
    return key;
```

(Everything above that line — computing `card`, `start`, `order`, `live`, `after` — is unchanged.)

- [ ] **Step 6: Bind the two functions**

In `sixmax/src/bindings/bindings.cpp`, add `#include "abstraction/abstract_key.h"` with the other abstraction include, then add these definitions inside `PYBIND11_MODULE` (next to the `preflop_class` binding):

```cpp
    m.def("pot_bucket", &sixmax::pot_bucket, py::arg("pot_bb"));
    m.def("pack_abstract_key",
          [](int card, int street, const std::vector<int>& raises,
             double pot_bb, int live, int after) {
              if (raises.size() != 4)
                  throw py::value_error("raises must have length 4");
              std::array<uint8_t, 4> r{(uint8_t)raises[0], (uint8_t)raises[1],
                                       (uint8_t)raises[2], (uint8_t)raises[3]};
              return sixmax::pack_abstract_key(card, street, r, pot_bb, live,
                                               after);
          },
          py::arg("card"), py::arg("street"), py::arg("raises"),
          py::arg("pot_bb"), py::arg("live"), py::arg("after"));
```

- [ ] **Step 7: Add the new source to BUCK**

In `sixmax/BUCK`, add `"src/abstraction/abstract_key.cpp"` to the `srcs` list (alongside `src/abstraction/abstraction.cpp`).

- [ ] **Step 8: Build and run the test to verify it passes**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_pack_abstract_key.py tests/sixmax/test_abstract_key.py -q`
Expected: PASS (the pre-existing `test_abstract_key.py` still passes, proving the refactor is behavior-preserving).

- [ ] **Step 9: Commit**

```bash
git add sixmax/src/abstraction/abstract_key.h sixmax/src/abstraction/abstract_key.cpp \
        sixmax/src/blueprint/engine_game.cpp sixmax/src/bindings/bindings.cpp \
        sixmax/BUCK tests/sixmax/test_pack_abstract_key.py
git commit -m "feat(sixmax): extract pack_abstract_key as the single key-layout source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Dev-scale blueprint checkpoint fixtures

Downstream tasks need loadable blueprints. Train tiny n=2 and n=6 checkpoints into a pytest tmp dir (never committed) as session fixtures, and prove they load through `BlueprintStrategy`.

**Files:**
- Create: `tests/fixtures/__init__.py` (empty), `tests/fixtures/sixmax_checkpoints.py`
- Create: repo-root `conftest.py` with `pytest_plugins = ["tests.fixtures.sixmax_checkpoints"]` (the rootdir conftest — `pytest_plugins` in a deeper conftest is rejected by pytest ≥ 9); `tests/agents/conftest.py` and `tests/scripts/conftest.py` (one-line sixmax force-load). Leave the existing `tests/conftest.py` untouched.
- Test: assertion lives in `tests/sixmax/test_pack_abstract_key.py`? No — add `tests/fixtures/test_checkpoint_fixture.py`.

**Interfaces:**
- Produces (pytest fixtures, session-scoped): `sixmax_vocab` → `sixmax.ActionVocab`; `blueprint_hu_ckpt` → path to a trained n=2 `.bin`; `blueprint_6max_ckpt` → path to a trained n=6 `.bin`. All consume the `sixmax` module already registered by `tests/sixmax/conftest.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/fixtures/test_checkpoint_fixture.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/fixtures/test_checkpoint_fixture.py -q`
Expected: FAIL — `fixture 'blueprint_hu_ckpt' not found`.

- [ ] **Step 3: Write the fixtures**

Create `tests/fixtures/__init__.py` (empty). Create `tests/fixtures/sixmax_checkpoints.py`:

```python
"""Session-scoped fixtures that train tiny blueprints for deployment tests.

Kept small (a few thousand iterations) so the whole downstream suite is
independently runnable in seconds. The real full-scale blueprint is a separate
operational run; these are correctness fixtures only. Checkpoints are written
into tmp_path_factory and never committed."""
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def sixmax_vocab():
    import sixmax  # registered by tests/sixmax/conftest.py
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(
        os.path.join(_ROOT, "sixmax", "configs", "default.toml"), "blueprint")


def _train(tmp_path, sixmax_vocab, num_players, iterations, name):
    import sixmax
    cfg = sixmax.EngineConfig(num_players=num_players)
    abstraction = sixmax.Abstraction(flop_buckets=10, turn_buckets=10,
                                     river_buckets=5, equity_rollouts=40,
                                     quantile_samples=300, seed=42)
    trainer = sixmax.BlueprintTrainer(cfg, sixmax_vocab, abstraction,
                                      num_threads=1, seed=7)
    trainer.train(iterations)
    path = os.path.join(str(tmp_path), name)
    trainer.save(path, sixmax_vocab, cfg, abstraction)
    return path


@pytest.fixture(scope="session")
def blueprint_hu_ckpt(tmp_path_factory, sixmax_vocab):
    tmp = tmp_path_factory.mktemp("sixmax_hu")
    return _train(tmp, sixmax_vocab, 2, 4000, "blueprint_hu.bin")


@pytest.fixture(scope="session")
def blueprint_6max_ckpt(tmp_path_factory, sixmax_vocab):
    tmp = tmp_path_factory.mktemp("sixmax_6max")
    return _train(tmp, sixmax_vocab, 6, 4000, "blueprint_6max.bin")
```

- [ ] **Step 4: Register the fixtures globally**

`pytest_plugins` is only honored in the **top-level (rootdir) conftest** — pytest ≥ 9 (this repo pins `pytest>=9.0`) raises `Defining 'pytest_plugins' in a non-top-level conftest is no longer supported` if it appears deeper. The rootdir here is the repo root (where `pyproject.toml` lives), and there is currently **no** root `conftest.py`; the existing `tests/conftest.py` (the `fast_monte_carlo` autouse fixture) is one level down. So create a new **repo-root** `conftest.py` (NOT `tests/conftest.py`):

```python
# Repo-root conftest: registers session fixtures for the whole suite.
pytest_plugins = ["tests.fixtures.sixmax_checkpoints"]
```

Leave the existing `tests/conftest.py` untouched. Note: the `sixmax` import inside the fixtures resolves because `tests/sixmax/conftest.py` force-loads `sixmax.so` into `sys.modules`; the deployment tests that consume these fixtures live under `tests/agents/` and `tests/scripts/`, so add a one-line `conftest.py` in each of those dirs that force-loads the extension:

Create `tests/agents/conftest.py` and `tests/scripts/conftest.py`, each containing:

```python
# Force-load the sixmax C++ extension the same way tests/sixmax/conftest.py does.
from tests.sixmax.conftest import _mod  # noqa: F401  (side effect: registers sys.modules['sixmax'])
```

(Importing any name from `tests/sixmax/conftest.py` triggers its import-time force-load. Under `--import-mode=importlib` this executes the conftest a second time as `tests.sixmax.conftest`, re-running the buck2 `--show-output` discovery — acceptable but not free; keep discovery cheap.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/fixtures/test_checkpoint_fixture.py -q`
Expected: PASS (training ~4k iters twice takes a few seconds).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/sixmax_checkpoints.py \
        tests/fixtures/test_checkpoint_fixture.py conftest.py \
        tests/agents/conftest.py tests/scripts/conftest.py
git commit -m "test(sixmax): dev-scale blueprint checkpoint fixtures for deployment tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `SixmaxDeployStrategy` — host-agnostic bridge

The core deployment logic: reconstruct the infoset key from public state, look up the blueprint policy, re-mask to legal actions, sample, and translate the chosen vocab index to a raise-to amount in BB. No host coupling — pure BB units in, `(vocab_index, raise_to_bb)` out.

**Files:**
- Create: `agents/sixmax_agent.py` (this task adds `canonical_live_after` and `SixmaxDeployStrategy`; Task 4 adds `SixmaxAgent`)
- Test: `tests/agents/test_sixmax_agent.py`

**Interfaces:**
- Consumes: `sixmax.BlueprintStrategy`, `sixmax.ActionVocab`, `sixmax.pack_abstract_key`, `sixmax.pot_bucket`, `sixmax.preflop_class`, `sixmax.BetContext` (from Task 1 + existing bindings).
- Produces:
  - `canonical_live_after(n, button, hero, folded, all_in, street) -> tuple[int, int]` — `(live, after)` per the pinned canonical order. `folded`/`all_in` are length-`n` bool lists indexed by table position `0..n-1`.
  - `class SixmaxDeployStrategy` with:
    - `__init__(self, strategy, vocab)` (a loaded `BlueprintStrategy` + `ActionVocab`).
    - `classmethod load(cls, path, config_toml, section="blueprint")` — builds the vocab from TOML and loads the strategy.
    - `num_players` property.
    - `decide(self, *, hole, board, street, raises_per_street, pot_bb, current_bet_bb, to_call_bb, stack_bb, live, after, legal, rng) -> tuple[int, float]` — returns `(vocab_index, raise_to_bb)`. `raise_to_bb` is `0.0` for non-bet actions. `hole`/`board` are lists of 0–51 card codes. `legal` is a length-`vocab.size()` list of 0/1 legality flags supplied by the host (the host owns legality, matching `EngineGameState::legal_mask`). `rng` is a `random.Random`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_sixmax_agent.py`:

```python
"""SixmaxDeployStrategy reconstructs trained infoset keys, respects the legal
mask, and translates bets through the vocab."""
import os
import random

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")

from agents.sixmax_agent import SixmaxDeployStrategy, canonical_live_after


def card(rank, suit):
    return (rank - 2) * 4 + suit


def test_canonical_live_after_hu_preflop():
    # HU, button 0, hero is button/seat 0 acting first preflop: one opponent (BB)
    # still to act after.
    live, after = canonical_live_after(2, button=0, hero=0,
                                       folded=[False, False],
                                       all_in=[False, False], street=0)
    assert live == 1 and after == 1


def test_canonical_live_after_6max_after_utg_fold():
    # 6-handed, button 0, hero=HJ(seat 4), UTG(seat 3) folded.
    live, after = canonical_live_after(
        6, button=0, hero=4,
        folded=[False, False, False, True, False, False],
        all_in=[False] * 6, street=0)
    assert live == 4      # everyone but hero and folded UTG
    assert after == 4     # CO, BTN, SB, BB act after HJ preflop


def test_decide_returns_legal_action(blueprint_6max_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_6max_ckpt, _TOML)
    rng = random.Random(0)
    n = strat.num_players
    legal = [0, 0, 1, 1, 1, 1, 1, 1, 1]  # call + all bets/all-in legal
    idx, raise_to = strat.decide(
        hole=[card(14, 3), card(13, 1)], board=[], street=0,
        raises_per_street=[0, 0, 0, 0], pot_bb=1.5, current_bet_bb=1.0,
        to_call_bb=1.0, stack_bb=99.0, live=n - 1, after=n - 1,
        legal=legal, rng=rng)
    assert legal[idx] == 1
    if raise_to > 0.0:
        assert raise_to >= 1.0  # a raise-to is at least the current bet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_sixmax_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.sixmax_agent'`.

- [ ] **Step 3: Write `canonical_live_after` and `SixmaxDeployStrategy`**

Create `agents/sixmax_agent.py`:

```python
"""Deployment bridge for the six-max blueprint.

SixmaxDeployStrategy is host-agnostic: it takes public state in big-blind units
and returns a chosen vocab index plus a raise-to (also in BB). Hosts (the live
engine via SixmaxAgent, the openpoker connector via HandTracker) compute the
legal mask and the (live, after) counts, and translate BB back to table chips.

Key reconstruction routes through sixmax.pack_abstract_key so it is byte-
identical to the trainer's EngineGameState::abstract_key. Bet legality and
translation route through the C++ ActionVocab (target_bb), the single vocab
translation layer.
"""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_vocab(config_toml, section):
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(config_toml, section)


def canonical_live_after(n, button, hero, folded, all_in, street):
    """(live, after) per the pinned canonical action order.

    Mirrors EngineGameState::abstract_key exactly. Positions are table indices
    0..n-1; folded/all_in are length-n bool lists indexed the same way."""
    start = (button if n == 2 else (button + 3) % n) if street == 0 \
        else (button + 1) % n

    def order(seat):
        return (seat - start + n) % n

    live = after = 0
    for s in range(n):
        if s == hero or folded[s]:
            continue
        live += 1
        if not all_in[s] and order(s) > order(hero):
            after += 1
    return live, after


class SixmaxDeployStrategy:
    def __init__(self, strategy, vocab):
        self._strategy = strategy
        self._vocab = vocab

    @classmethod
    def load(cls, path, config_toml, section="blueprint"):
        vocab = _load_vocab(config_toml, section)
        strategy = sixmax.BlueprintStrategy.load(path, vocab)
        return cls(strategy, vocab)

    @property
    def num_players(self):
        return self._strategy.num_players()

    def _card_bucket(self, hole, board, street):
        if street == 0:
            return sixmax.preflop_class(hole)
        return self._strategy.abstraction().bucket(hole, board)

    def decide(self, *, hole, board, street, raises_per_street, pot_bb,
               current_bet_bb, to_call_bb, stack_bb, live, after, legal, rng):
        card = self._card_bucket(hole, board, street)
        key = sixmax.pack_abstract_key(card, street, list(raises_per_street),
                                       pot_bb, live, after)
        probs = self._strategy.probs(key)  # [] if unseen

        legal_idx = [i for i, m in enumerate(legal) if m]
        weights = [probs[i] if i < len(probs) else 0.0 for i in legal_idx]
        total = sum(weights)
        if total <= 0.0:
            idx = rng.choice(legal_idx)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = legal_idx[-1]
            for i, w in zip(legal_idx, weights):
                acc += w
                if r <= acc:
                    idx = i
                    break

        action = self._vocab.at(idx)
        if action.type in (sixmax.ActionType.Bet, sixmax.ActionType.AllIn):
            ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                    to_call=to_call_bb, stack=stack_bb)
            raise_to = self._vocab.target_bb(idx, ctx)
        else:
            raise_to = 0.0
        return idx, raise_to
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/agents/test_sixmax_agent.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/sixmax_agent.py tests/agents/test_sixmax_agent.py
git commit -m "feat(sixmax): host-agnostic SixmaxDeployStrategy bridge + canonical_live_after

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Faithful raise counts + `SixmaxAgent(PokerAgent)` live-engine adapter

Two parts:

**(a) Make `raises_per_street` faithful in the shared engine.** Today `game/poker.py:147` caps the per-street raise count at 2 *in the game state itself* — but that "2" is the *tabular* abstraction's cap, hardcoded into shared code, and it's the reason a sixmax bridge (cap 3) can't read the engine counter. `raises_per_street` is never read for game rules (No-Limit has no raise-count rule — it's only ever written in `game/poker.py`, and consumed by agents as an abstraction feature), so uncapping it changes no game behavior. Instead, the cap becomes each *consumer's* responsibility — which is already how `neural_cfr` does it (`inference.cpp:36` clamps to `[0,2]` at input). After this change: tabular clamps to 2, sixmax to 3, neural is unchanged (already clamps). This is behavior-preserving for every existing consumer because each clamps back to exactly the value it saw before; only sixmax gains access to the true count.

**(b) Wrap `SixmaxDeployStrategy` as a live-engine agent** so the blueprint can play HU in `game/poker.py`. This adapter computes the legal mask, `(live, after)`, and BB rescale from the live `GameState`, clamps the (now faithful) raise counts to 3 at consumption, then translates the returned raise-to (BB) back into an engine `(Action, amount)`.

**Files:**
- Modify: `game/poker.py` (uncap `raises_per_street`), `models/state.py` (comment), `agents/cfr_agent.py` (clamp to 2 at consumption), `agents/sixmax_agent.py` (append `SixmaxAgent`)
- Test: `tests/test_integration.py` (append `test_raises_per_street_is_faithful_past_two` — the guard-test for the uncap) and `tests/agents/test_sixmax_agent.py` (append the SixmaxAgent tests); the guard-test + existing full suite are the regression gate for the engine change.

**Interfaces:**
- Consumes: `SixmaxDeployStrategy` (Task 3); `agents.base_agent.PokerAgent`; `models.enums.Action`; live `GameState`/`Player` fields — `game_state.big_blind`, `game_state.pot`, `game_state.current_bet`, `game_state.community_cards`, `game_state.betting_round`, `game_state.raises_per_street` (now faithful — clamp to 3 at consumption), `game_state.button_pos`, `game_state.players` (each with `.stack`, `.hole_cards`, `.is_active`, `.is_all_in`, `.current_bet`).
- Produces: `class SixmaxAgent(PokerAgent)` with `__init__(self, checkpoint_path, config_toml=<default.toml>)` and `get_action(self, player, game_state) -> tuple[Action, Optional[int]]`. No observer plumbing: the agent reads `game_state.raises_per_street` directly and applies `min(r, 3)` — the engine now supplies the true count.

Note on card encoding: the sixmax card code is `(rank - 2) * 4 + suit_idx` with suit order `clubs=0, diamonds=1, hearts=2, spades=3` — identical to `openpoker_bot._card_to_int`. Reuse that mapping.

- [ ] **Step 1: Make `raises_per_street` faithful in the engine**

First, a guard-test that pins the new contract directly (it fails on the current capped engine — it would read `2` — and the existing suite never exercises a >2-raise line, so nothing else covers this). Append to `tests/test_integration.py`:

```python
def test_raises_per_street_is_faithful_past_two():
    """After the uncap, raises_per_street counts the true number of raises on a
    street (old code capped at 2). Both players use the same order-independent
    rule — min-raise preflop while the live count is below 3 — so exactly 3
    preflop raises go in, then the action closes with a call. Asserting [0]==3
    proves the count passes 2 AND that the closing call didn't bump it; [1]==0
    proves the postflop checks don't count. (On the old capped engine the count
    sticks at 2, so the agents would raise until all-in and [0] reads 2.)"""
    from models.enums import Action
    from models.player import Player
    from game.poker import PokerGame

    class Raiser:
        def get_action(self, player, game_state):
            to_call = game_state.current_bet - player.current_bet
            can_raise = player.stack > to_call + game_state.big_blind
            if (game_state.betting_round == 0
                    and game_state.raises_per_street[0] < 3 and can_raise):
                return Action.BET, game_state.current_bet + 2 * game_state.big_blind
            return (Action.CALL, None) if to_call > 0 else (Action.CHECK, None)

    p0, p1 = Player("a", 200, Raiser()), Player("b", 200, Raiser())
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()
    assert game.state.raises_per_street[0] == 3   # uncapped past 2; call didn't bump
    assert game.state.raises_per_street[1] == 0   # postflop checks don't count
    assert p0.stack + p1.stack == 400             # chips conserved
```

Run it against the *unmodified* engine to see it fail: `uv run pytest tests/test_integration.py::test_raises_per_street_is_faithful_past_two -q` → FAIL (`assert 2 == 3`).

Now make it faithful. In `game/poker.py`, drop the cap where the counter is incremented (inside the `elif action in (Action.BET, Action.RAISE)` handler, currently `:147` — **keep it in that branch** so only raises count):

```python
            # was: self.state.raises_per_street[street] = min(
            #          self.state.raises_per_street[street] + 1, 2)
            self.state.raises_per_street[street] += 1  # faithful; consumers cap
```

In `models/state.py`, update the `:19` comment: `# raise counts per street (faithful; each consumer applies its own cap)`.

In `agents/cfr_agent.py`, the tabular consumer must now apply *its* cap where it builds the InfoSet (`:46`):

```python
            betting_history=tuple(min(r, 2) for r in game_state.raises_per_street),
```

`neural_cfr` already clamps at input (`inference.cpp:36`, `[0,2]`) — no change. Run the guard-test (now PASS) and then the full suite: `uv run pytest tests/ -q`. The suite must still pass unchanged — for any live line with ≤2 raises the values are identical to before, and the tabular consumer clamps back to 2, so no tabular/neural key changes. The guard-test + full suite together are the regression gate for the shared-engine edit.

- [ ] **Step 2: Write the failing test**

Append to `tests/agents/test_sixmax_agent.py`:

```python
from models.card import Card
from models.enums import Action, Suit
from models.player import Player
from game.poker import PokerGame
from agents.sixmax_agent import SixmaxAgent


def test_sixmax_agent_plays_a_legal_hand(blueprint_hu_ckpt):
    # bb=2, stack=200 -> exactly 100 BB, so BB rescale = /2.
    a = SixmaxAgent(blueprint_hu_ckpt, config_toml=_TOML)
    p0 = Player("hero", 200, agent=a)
    p1 = Player("villain", 200, agent=a)
    game = PokerGame([p0, p1], small_blind=1)
    game.play_hand()  # must complete without raising
    assert p0.stack + p1.stack == 400  # chips conserved HU


def test_sixmax_agent_get_action_is_legal(blueprint_hu_ckpt):
    a = SixmaxAgent(blueprint_hu_ckpt, config_toml=_TOML)
    p0 = Player("hero", 200, agent=a)
    p1 = Player("villain", 200, agent=a)  # Player.agent is a required arg
    game = PokerGame([p0, p1], small_blind=1)
    game.state.deck = game.state._create_deck()
    game.state.deal_hole_cards()
    action, amount = a.get_action(p0, game.state)
    assert action in (Action.FOLD, Action.CHECK, Action.CALL, Action.BET)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_sixmax_agent.py -q`
Expected: FAIL — `ImportError: cannot import name 'SixmaxAgent'`.

- [ ] **Step 4: Append `SixmaxAgent` to `agents/sixmax_agent.py`**

Add these imports at the top of `agents/sixmax_agent.py`:

```python
import random

from agents.base_agent import PokerAgent
from models.enums import Action, Suit
```

Append the class:

```python
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_DEFAULT_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_RAISE_CAP = 3  # the sixmax abstraction's per-street raise cap (2 bits)


def _card_to_int(card):
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


class SixmaxAgent(PokerAgent):
    """Live-engine adapter over SixmaxDeployStrategy. Computes the legal mask,
    (live, after), and BB rescale from GameState; translates the blueprint's
    raise-to (BB) into an engine (Action, chips).

    game_state.raises_per_street is now faithful (Step 1 uncapped the engine),
    so we clamp it to the abstraction's cap (3) here at consumption — the same
    pattern neural_cfr uses (it clamps to 2). No observer plumbing needed."""

    def __init__(self, checkpoint_path, config_toml=_DEFAULT_TOML):
        self._deploy = SixmaxDeployStrategy.load(checkpoint_path, config_toml)
        self._rng = random.Random()

    def get_action(self, player, game_state):
        bb = game_state.big_blind
        players = game_state.players
        n = len(players)
        hero = players.index(player)
        street = game_state.betting_round

        hole = [_card_to_int(c) for c in player.hole_cards]
        board = [_card_to_int(c) for c in game_state.community_cards]

        folded = [not p.is_active for p in players]
        all_in = [p.is_all_in for p in players]
        live, after = canonical_live_after(n, game_state.button_pos, hero,
                                           folded, all_in, street)

        to_call_chips = game_state.current_bet - player.current_bet
        pot_bb = game_state.pot / bb
        current_bet_bb = game_state.current_bet / bb
        to_call_bb = to_call_chips / bb
        stack_bb = (player.current_bet + player.stack) / bb  # all-in target

        legal = self._legal_mask(game_state, player, to_call_chips,
                                 current_bet_bb, pot_bb, to_call_bb, stack_bb)

        raises = [min(r, _RAISE_CAP) for r in game_state.raises_per_street]
        idx, raise_to_bb = self._deploy.decide(
            hole=hole, board=board, street=street,
            raises_per_street=raises, pot_bb=pot_bb,
            current_bet_bb=current_bet_bb, to_call_bb=to_call_bb,
            stack_bb=stack_bb, live=live, after=after, legal=legal,
            rng=self._rng)
        return self._translate(idx, to_call_chips, raise_to_bb, bb, player)

    def _legal_mask(self, game_state, player, to_call_chips, current_bet_bb,
                    pot_bb, to_call_bb, stack_bb):
        """Mirror EngineGameState::legal_mask on live state. Bet entries are
        legal when their vocab target lands in [min_raise, all_in) BB."""
        vocab = self._deploy._vocab
        facing = to_call_chips > 1e-9
        can_raise = player.stack > to_call_chips + 1e-9
        street = game_state.betting_round
        unopened_preflop = street == 0 and current_bet_bb <= 1.0 + 1e-9
        min_raise_to_bb = (game_state.current_bet + max(
            game_state.big_blind, to_call_chips)) / game_state.big_blind
        ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                to_call=to_call_bb, stack=stack_bb)
        mask = [0] * vocab.size()
        for i in range(vocab.size()):
            a = vocab.at(i)
            if a.type == sixmax.ActionType.Fold:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Check:
                mask[i] = 0 if facing else 1
            elif a.type == sixmax.ActionType.Call:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Bet:
                bb_unit = a.unit == sixmax.SizeUnit.BB
                if not can_raise or bb_unit != unopened_preflop:
                    continue
                t = vocab.target_bb(i, ctx)
                mask[i] = 1 if (t >= min_raise_to_bb - 1e-9
                                and t < stack_bb - 1e-9) else 0
            elif a.type == sixmax.ActionType.AllIn:
                mask[i] = 1 if can_raise else 0
        return mask

    def _translate(self, idx, to_call_chips, raise_to_bb, bb, player):
        a = self._deploy._vocab.at(idx)
        if a.type == sixmax.ActionType.Fold:
            return Action.FOLD, None
        if a.type == sixmax.ActionType.Check:
            return Action.CHECK, None
        if a.type == sixmax.ActionType.Call:
            return Action.CALL, None
        # Bet / AllIn: game/poker.py:139 treats Action.BET amount as raise-TO
        # (the total street commitment; additional = amount - player.current_bet).
        # This matches CFRAgent._translate, whose all-in returns
        # stack + current_bet and whose bets cap at stack + current_bet.
        max_to = int(round(player.current_bet + player.stack))  # all-in raise-to
        if a.type == sixmax.ActionType.AllIn:
            return Action.BET, max_to
        raise_to = int(round(raise_to_bb * bb))
        return Action.BET, max(0, min(raise_to, max_to))
```

**Note for the implementer:** the `Action.BET` amount is raise-*to*, confirmed against `game/poker.py:139` (`additional = amount - player.current_bet; player.current_bet = amount`) and `agents/cfr_agent.py:_translate` (all-in returns `stack + current_bet`). `_translate` above already returns raise-to — do **not** subtract `current_bet`. The test `test_sixmax_agent_plays_a_legal_hand` guards this: a raise-*by* regression breaks HU chip conservation.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/agents/test_sixmax_agent.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add game/poker.py models/state.py agents/cfr_agent.py \
        agents/sixmax_agent.py tests/test_integration.py \
        tests/agents/test_sixmax_agent.py
git commit -m "feat(sixmax): faithful raises_per_street + SixmaxAgent live-engine adapter

Uncap raises_per_street in game/poker.py; each consumer now applies its own
abstraction cap (tabular 2, sixmax 3; neural already clamps). SixmaxAgent reads
the faithful count and clamps to 3 at consumption.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `NeuralAgent(PokerAgent)` — live-engine neural adapter

The HU sanity eval pits the blueprint against the frozen tabular bot (`CFRAgent`, already a `PokerAgent`) and the frozen neural bot. The neural strategy is only wired into `openpoker_bot.py` and OpenSpiel today; wrap it as a `PokerAgent` so all three share the live engine.

**Files:**
- Create: `agents/neural_agent.py`
- Test: `tests/agents/test_neural_agent.py`

**Interfaces:**
- Consumes: `neural_cfr.Strategy.get_action_probs(hole_ints, board_ints, street, pot, my_stack, to_call, raises_per_street, position, my_street_bet=, opp_street_bet=) -> dict[str,float]` (keys among `fold/check/call/b0.5/b1.0/allin`); `agents.base_agent.PokerAgent`; `models.enums.Action`.
- Produces: `class NeuralAgent(PokerAgent)` with `__init__(self, checkpoint_path)` and `get_action(self, player, game_state)`; module-level `load_neural_strategy(checkpoint)` factored so `openpoker_bot.py` and this agent share one loader (optional refactor — copying the loader is acceptable if a shared import risks import-time buck2 builds in the pure-Python agents).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_neural_agent.py`:

```python
"""NeuralAgent plays legal actions in the live engine (HU-mode eval opponent)."""
import os

import pytest

from models.enums import Action

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _neural_ckpt():
    d = os.path.join(_ROOT, "neural_cfr", "checkpoints")
    if not os.path.isdir(d):
        return None
    pts = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    return os.path.join(d, pts[-1]) if pts else None


@pytest.mark.skipif(_neural_ckpt() is None, reason="no neural checkpoint")
def test_neural_agent_get_action_is_legal():
    from agents.neural_agent import NeuralAgent
    from models.player import Player
    from game.poker import PokerGame

    a = NeuralAgent(_neural_ckpt())
    p0 = Player("hero", 200, agent=a)
    p1 = Player("villain", 200, agent=a)  # Player.agent is a required arg
    game = PokerGame([p0, p1], small_blind=1)
    game.state.deck = game.state._create_deck()
    game.state.deal_hole_cards()
    action, amount = a.get_action(p0, game.state)
    assert action in (Action.FOLD, Action.CHECK, Action.CALL, Action.BET)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_neural_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.neural_agent'` (or SKIP if no `.pt` exists — in that case create a `.pt` first via the neural pipeline, or accept the skip and validate through Task 6's eval smoke).

- [ ] **Step 3: Write `NeuralAgent`**

Create `agents/neural_agent.py`. Reuse the loader logic from `scripts/openpoker_bot.py:_load_neural_strategy` (buck2 build + dylib preload) and the decode from `HandTracker._decide_neural`:

```python
"""Live-engine adapter over neural_cfr.Strategy — a frozen HU eval opponent.

Mirrors scripts/openpoker_bot.py's neural loader and decode. Chips are rescaled
to the 100BB/1BB training frame by dividing by game_state.big_blind."""
import ctypes
import os
import subprocess
import sys

import numpy as np

from agents.base_agent import PokerAgent
from models.enums import Action, Suit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_ABSTRACT = ["fold", "check", "call", "b0.5", "b1.0", "allin"]


def _card_to_int(card):
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


def load_neural_strategy(checkpoint):
    lib_dir = os.path.join(_ROOT, "third_party", "libtorch", "lib")
    for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
        p = os.path.join(lib_dir, lib)
        if os.path.exists(p):
            ctypes.CDLL(p)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//neural_cfr:neural_cfr",
                             "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = None
    for line in result.stdout.splitlines():
        if "neural_cfr.so" in line:
            so_dir = os.path.join(_ROOT, os.path.dirname(line.split()[-1]))
            break
    if so_dir and so_dir not in sys.path:
        sys.path.insert(0, so_dir)
    import neural_cfr
    return neural_cfr.Strategy(checkpoint)


class NeuralAgent(PokerAgent):
    def __init__(self, checkpoint_path):
        self._strategy = load_neural_strategy(checkpoint_path)

    def get_action(self, player, game_state):
        bb = game_state.big_blind
        n = len(game_state.players)
        hero = game_state.players.index(player)
        # HU position: 0=SB/button, 1=BB. Button is SB heads-up.
        position = 0 if hero == game_state.button_pos else 1
        street = game_state.betting_round

        to_call_chips = game_state.current_bet - player.current_bet
        probs = self._strategy.get_action_probs(
            [_card_to_int(c) for c in player.hole_cards],
            [_card_to_int(c) for c in game_state.community_cards],
            street,
            game_state.pot / bb,
            player.stack / bb,
            to_call_chips / bb,
            list(game_state.raises_per_street),
            position,
            my_street_bet=player.current_bet / bb,
            opp_street_bet=(player.current_bet + to_call_chips) / bb,
        )

        facing = to_call_chips > 1e-9
        legal = []
        if facing:
            legal.append("fold")
        else:
            legal.append("check")
        if facing:
            legal.append("call")
        if player.stack > to_call_chips + 1e-9:
            legal.extend(["b0.5", "b1.0", "allin"])
        raw = np.array([probs.get(a, 0.0) for a in legal], dtype=float)
        if raw.sum() <= 0:
            raw = np.ones(len(legal))
        choice = legal[int(np.argmax(raw))]
        return self._translate(choice, game_state, player, to_call_chips)

    def _translate(self, choice, game_state, player, to_call_chips):
        if choice == "fold":
            return Action.FOLD, None
        if choice == "check":
            return Action.CHECK, None
        if choice == "call":
            return Action.CALL, None
        # game/poker.py:139 treats Action.BET amount as raise-TO (total street
        # commitment). All-in raise-to = current_bet + stack; a pot-fraction bet
        # raise-to = current_bet + to_call + size * (pot after the call).
        max_to = int(round(player.current_bet + player.stack))  # all-in raise-to
        if choice == "allin":
            return Action.BET, max_to
        size = float(choice[1:])  # 0.5 or 1.0 pot
        eff_pot = game_state.pot + to_call_chips
        raise_to = int(round(player.current_bet + to_call_chips + size * eff_pot))
        return Action.BET, max(0, min(raise_to, max_to))
```

**Note for the implementer:** `Action.BET` amount is raise-*to* (confirmed against `game/poker.py:139` and `CFRAgent._translate`); `_translate` above already returns raise-to. Take `argmax` (not sampling) here to keep the frozen opponent deterministic per state; this is a fixed eval opponent, not the bot under test.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/agents/test_neural_agent.py -q`
Expected: PASS (or SKIP if no `.pt` — validated via Task 6 instead).

- [ ] **Step 5: Commit**

```bash
git add agents/neural_agent.py tests/agents/test_neural_agent.py
git commit -m "feat(sixmax): NeuralAgent live-engine adapter for HU-mode eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `scripts/eval_hu_sanity.py` — duplicate-deal HU eval

Run the blueprint (2-player) against the tabular and neural bots in `game/poker.py` using duplicate deals (same seeded deck played with hero in each seat) to cancel deal luck. Report the blueprint's BB/100. This is the eval program's HU-mode sanity gate.

**Files:**
- Create: `scripts/eval_hu_sanity.py`
- Test: `tests/scripts/test_eval_hu_sanity.py`

**Interfaces:**
- Consumes: `SixmaxAgent` (Task 4), `NeuralAgent` (Task 5), `agents.cfr_agent.CFRAgent`, `game.poker.PokerGame`, `models.player.Player`. Duplicate deals via `random.seed(s)` before each `play_hand()` (the live deck is shuffled with the global `random` module — `models/state.py:26`, re-created each hand at `game/poker.py:163`).
- Produces: `run_duplicate_match(hero_factory, villain_factory, hands, seed, bb, stack) -> float` (hero BB/100) and a CLI: `--blueprint CKPT [--tabular CKPT] [--neural CKPT] [--hands 500] [--seed 1]`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_eval_hu_sanity.py`:

```python
"""Smoke test: the duplicate-deal HU eval returns a finite BB/100 over a few
hands with the blueprint on both sides (expected ~0 by symmetry, but we only
assert finiteness and duplicate-seat cancellation runs)."""
import math
import os

from scripts.eval_hu_sanity import run_duplicate_match
from agents.sixmax_agent import SixmaxAgent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_run_duplicate_match_finite(blueprint_hu_ckpt):
    def mk(_):
        return SixmaxAgent(blueprint_hu_ckpt, config_toml=_TOML)

    bb100 = run_duplicate_match(mk, mk, hands=20, seed=1, bb=2, stack=200)
    assert math.isfinite(bb100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_eval_hu_sanity.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval_hu_sanity'`.

- [ ] **Step 3: Write the eval script**

Create `scripts/eval_hu_sanity.py`:

```python
#!/usr/bin/env python3
"""HU-mode sanity eval: the six-max blueprint (2-player) vs the frozen tabular
and neural bots, in the live game/poker.py engine.

Duplicate deals cancel deal luck: each seeded deck is played twice, with the
hero in seat 0 then seat 1. The deck is shuffled with the global `random`
module and re-created each hand, so seeding `random` before play_hand makes the
deck reproducible; the hero's action RNG uses a separate stream.

Usage:
    uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/blueprint.bin \
        [--tabular cfr/checkpoints/checkpoint_09040000.pkl] \
        [--neural neural_cfr/checkpoints/checkpoint.pt] [--hands 500] [--seed 1]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player import Player
from game.poker import PokerGame


def _play_seeded(hero_agent, villain_agent, hero_seat, seed, bb, stack):
    """One hand at a fixed seed; returns hero's chip delta. bb sets blinds via
    small_blind = bb // 2."""
    random.seed(seed)
    names = ["a", "b"]
    agents = [None, None]
    agents[hero_seat] = hero_agent
    agents[1 - hero_seat] = villain_agent
    players = [Player(names[i], stack, agent=agents[i]) for i in range(2)]
    game = PokerGame(players, small_blind=bb // 2)
    game.play_hand()
    return players[hero_seat].stack - stack


def run_duplicate_match(hero_factory, villain_factory, hands, seed, bb, stack):
    """hero_factory(seat)/villain_factory(seat) build one agent per seat ONCE
    and reuse them across all hands. Agents hold only a loaded strategy + RNG
    and carry no per-hand state, so building them per hand would needlessly
    reload the checkpoint every hand — and, for NeuralAgent, re-run a buck2
    build subprocess every hand. Returns hero BB/100 over `hands` duplicate
    deals (2 games each)."""
    heroes = {seat: hero_factory(seat) for seat in (0, 1)}
    villains = {seat: villain_factory(seat) for seat in (0, 1)}
    total = 0.0
    for h in range(hands):
        s = seed * 1_000_003 + h
        for hero_seat in (0, 1):
            total += _play_seeded(heroes[hero_seat], villains[hero_seat],
                                  hero_seat, s, bb, stack)
    return 100.0 * total / (hands * 2 * bb)  # BB/100


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="HU-mode sanity eval")
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--tabular", default=None)
    parser.add_argument("--neural", default=None)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--stack", type=int, default=200)
    args = parser.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    toml = os.path.join(root, "sixmax", "configs", "default.toml")

    def hero(_):
        return SixmaxAgent(args.blueprint, config_toml=toml)

    if args.tabular:
        from agents.cfr_agent import CFRAgent
        bb100 = run_duplicate_match(hero, lambda _: CFRAgent(args.tabular),
                                    args.hands, args.seed, args.bb, args.stack)
        print(f"blueprint vs tabular: {bb100:+.2f} BB/100 "
              f"({args.hands * 2} hands)")
    if args.neural:
        from agents.neural_agent import NeuralAgent
        bb100 = run_duplicate_match(hero, lambda _: NeuralAgent(args.neural),
                                    args.hands, args.seed, args.bb, args.stack)
        print(f"blueprint vs neural:  {bb100:+.2f} BB/100 "
              f"({args.hands * 2} hands)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/scripts/test_eval_hu_sanity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_hu_sanity.py tests/scripts/test_eval_hu_sanity.py
git commit -m "feat(sixmax): duplicate-deal HU-mode sanity eval vs tabular/neural

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Wire the blueprint into `scripts/openpoker_bot.py`

Deploy the blueprint to openpoker.ai. Detect a `.bin` checkpoint → build a `SixmaxDeployStrategy`; extend `HandTracker` to track per-seat fold/all-in state, raise counts (cap 3), compute `(live, after)`, query the bridge, and translate the returned raise-to (BB) into the openpoker raise-to convention.

**Files:**
- Modify: `scripts/openpoker_bot.py`
- Test: `tests/scripts/test_openpoker_sixmax.py`

**Interfaces:**
- Consumes: `SixmaxDeployStrategy` and `canonical_live_after` (Task 3). openpoker messages: `hand_start` (seat, blinds, dealer_seat, players), `hole_cards`, `community_cards` (street), `player_action` (seat, action, street), `your_turn` (pot, valid_actions, players).
- Produces: within `openpoker_bot.py`, a `SixmaxHandTracker` (or an extended `HandTracker` branch) whose `decide(msg, strategy, buy_in)` returns an openpoker action dict when `strategy` is a `SixmaxDeployStrategy`; `main()` loads a `SixmaxDeployStrategy` when the checkpoint ends in `.bin`.

Note: openpoker seats may not be contiguous `0..n-1`. Map the seats present at the table (sorted ascending) to positions `0..n-1`; `button` = position of `dealer_seat`, `hero` = position of `my_seat`; `folded`/`all_in` lists indexed by that position. This mapping is the six-max-specific decode work.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_openpoker_sixmax.py`:

```python
"""The six-max openpoker decode path turns a scripted message stream into a
legal action dict via the blueprint bridge."""
import os

from scripts.openpoker_bot import SixmaxHandTracker
from agents.sixmax_agent import SixmaxDeployStrategy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")


def test_sixmax_tracker_decides_preflop(blueprint_6max_ckpt):
    strat = SixmaxDeployStrategy.load(blueprint_6max_ckpt, _TOML)
    t = SixmaxHandTracker()
    t.on_hand_start({
        "hand_id": 1, "seat": 2, "dealer_seat": 0,
        "blinds": {"small_blind": 10, "big_blind": 20},
        "players": [{"seat": s, "stack": 2000} for s in range(6)],
    })
    t.on_hole_cards({"cards": ["Ah", "Kh"]})
    action = t.decide({
        "hand_id": 1, "pot": 30,
        "players": [{"seat": s, "stack": 2000} for s in range(6)],
        "valid_actions": [
            {"action": "fold"}, {"action": "call", "amount": 20},
            {"action": "raise", "min": 40, "max": 2000},
            {"action": "all_in"}],
    }, strat, buy_in=2000)
    assert action["action"] in ("fold", "call", "raise", "all_in")
    if action["action"] == "raise":
        assert 40 <= action["amount"] <= 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_openpoker_sixmax.py -q`
Expected: FAIL — `ImportError: cannot import name 'SixmaxHandTracker'`.

- [ ] **Step 3: Add `SixmaxHandTracker` and loader wiring**

In `scripts/openpoker_bot.py`, add near the imports:

```python
from agents.sixmax_agent import SixmaxDeployStrategy, canonical_live_after
```

Add the tracker class (mirrors `HandTracker` structure; tracks seats, folds, all-in, raise counts capped at 3):

```python
class SixmaxHandTracker:
    """Six-max blueprint deployment tracker. Maintains per-seat fold/all-in
    state and per-street raise counts (cap 3) so it can rebuild the abstraction
    infoset key that the trainer used."""

    def __init__(self):
        self.hole_cards = []
        self.community_cards = []
        self.street = 0
        self.raises_per_street = [0, 0, 0, 0]
        self.big_blind = 20.0
        self.my_seat = None
        self.my_stack = 0.0
        self.my_committed = 0.0
        self.seats = []          # sorted table seat numbers present
        self.button_seat = None
        self.folded = {}         # seat -> bool
        self.all_in = {}         # seat -> bool

    def on_hand_start(self, msg):
        self.hole_cards = []
        self.community_cards = []
        self.street = 0
        self.raises_per_street = [0, 0, 0, 0]
        self.my_seat = msg["seat"]
        self.big_blind = msg["blinds"]["big_blind"]
        self.button_seat = msg.get("dealer_seat")
        self.seats = sorted(p["seat"] for p in msg.get("players", []))
        self.folded = {s: False for s in self.seats}
        self.all_in = {s: False for s in self.seats}
        self.my_committed = 0.0

    def on_hole_cards(self, msg):
        self.hole_cards = [_parse_card(c) for c in msg["cards"]]

    def on_community_cards(self, msg):
        self.community_cards = [_parse_card(c) for c in msg["cards"]]
        self.street = STREET_IDX.get(msg.get("street", "preflop"), self.street)
        self.my_committed = 0.0

    def on_player_action(self, msg):
        seat = msg.get("seat")
        act = msg.get("action")
        if act == "fold" and seat in self.folded:
            self.folded[seat] = True
        if act == "all_in" and seat in self.all_in:
            self.all_in[seat] = True
        if act in ("raise", "all_in"):
            s = STREET_IDX.get(msg.get("street", "preflop"), self.street)
            self.raises_per_street[s] = min(self.raises_per_street[s] + 1, 3)

    def _positions(self):
        n = len(self.seats)
        idx = {s: i for i, s in enumerate(self.seats)}
        return (n, idx[self.button_seat], idx[self.my_seat],
                [self.folded[s] for s in self.seats],
                [self.all_in[s] for s in self.seats])

    def decide(self, msg, strategy, buy_in):
        bb = self.big_blind
        pot = float(msg.get("pot", 0.0))
        valid = {a["action"]: a for a in msg.get("valid_actions", [])}
        for p in msg.get("players", []):
            if p.get("seat") == self.my_seat:
                self.my_stack = float(p["stack"])
                break
        to_call_chips = float(valid.get("call", {}).get("amount") or 0.0)

        n, button, hero, folded, all_in = self._positions()
        live, after = canonical_live_after(n, button, hero, folded, all_in,
                                           self.street)

        current_bet_bb = (self.my_committed + to_call_chips) / bb
        stack_bb = (self.my_committed + self.my_stack) / bb
        legal = self._legal_mask(strategy, valid, current_bet_bb, pot / bb,
                                 to_call_chips / bb, stack_bb)

        import random as _random
        idx, raise_to_bb = strategy.decide(
            hole=[_card_to_int(c) for c in self.hole_cards],
            board=[_card_to_int(c) for c in self.community_cards],
            street=self.street, raises_per_street=self.raises_per_street,
            pot_bb=pot / bb, current_bet_bb=current_bet_bb,
            to_call_bb=to_call_chips / bb, stack_bb=stack_bb, live=live,
            after=after, legal=legal, rng=_random.Random())
        return self._translate(strategy, idx, valid, raise_to_bb, bb)

    def _legal_mask(self, strategy, valid, current_bet_bb, pot_bb, to_call_bb,
                    stack_bb):
        vocab = strategy._vocab
        facing = "call" in valid
        can_raise = "raise" in valid or "all_in" in valid
        unopened_preflop = self.street == 0 and current_bet_bb <= 1.0 + 1e-9
        min_raise_to_bb = float(valid.get("raise", {}).get("min", 0.0)) / self.big_blind
        ctx = __import__("sixmax").BetContext(
            pot=pot_bb, current_bet=current_bet_bb, to_call=to_call_bb,
            stack=stack_bb)
        sm = __import__("sixmax")
        mask = [0] * vocab.size()
        for i in range(vocab.size()):
            a = vocab.at(i)
            if a.type == sm.ActionType.Fold:
                mask[i] = 1 if facing else 0
            elif a.type == sm.ActionType.Check:
                mask[i] = 0 if facing else 1
            elif a.type == sm.ActionType.Call:
                mask[i] = 1 if facing else 0
            elif a.type == sm.ActionType.Bet:
                bb_unit = a.unit == sm.SizeUnit.BB
                if not can_raise or "raise" not in valid \
                        or bb_unit != unopened_preflop:
                    continue
                t = vocab.target_bb(i, ctx)
                mask[i] = 1 if (t >= min_raise_to_bb - 1e-9
                                and t < stack_bb - 1e-9) else 0
            elif a.type == sm.ActionType.AllIn:
                mask[i] = 1 if can_raise else 0
        return mask

    def _translate(self, strategy, idx, valid, raise_to_bb, bb):
        sm = __import__("sixmax")
        a = strategy._vocab.at(idx)
        if a.type == sm.ActionType.Fold:
            return {"action": "fold"} if "fold" in valid else (
                {"action": "check"} if "check" in valid else {"action": "call"})
        if a.type == sm.ActionType.Check:
            return {"action": "check"} if "check" in valid else (
                {"action": "call"} if "call" in valid else {"action": "fold"})
        if a.type == sm.ActionType.Call:
            return {"action": "call"} if "call" in valid else (
                {"action": "check"} if "check" in valid else {"action": "fold"})
        if a.type == sm.ActionType.AllIn:
            if "all_in" in valid:
                return {"action": "all_in"}
            if "raise" in valid:
                return {"action": "raise", "amount": float(valid["raise"]["max"])}
            return {"action": "call"} if "call" in valid else {"action": "check"}
        # Bet: raise-to in chips (openpoker uses the raise-to convention).
        raise_to = round(raise_to_bb * bb, 1)
        min_r = float(valid["raise"].get("min", 0))
        max_r = float(valid["raise"].get("max", self.my_stack))
        return {"action": "raise", "amount": max(min_r, min(max_r, raise_to))}
```

In `main()`, extend checkpoint detection so `.bin` loads the sixmax strategy and uses `SixmaxHandTracker`. Replace the load block:

```python
    log.info(f"Loading {checkpoint} ...")
    if checkpoint.endswith(".bin"):
        toml = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "sixmax", "configs", "default.toml")
        strategy = SixmaxDeployStrategy.load(checkpoint, toml)
        log.info("Six-max blueprint strategy loaded.")
    elif checkpoint.endswith(".pt"):
        strategy = _load_neural_strategy(checkpoint)
        log.info("Neural CFR strategy loaded.")
    else:
        strategy = RegretTable()
        strategy.load(checkpoint)
        log.info("Tabular CFR strategy loaded.")
```

And in `run(...)`, select the tracker by strategy type:

```python
            tracker = (SixmaxHandTracker()
                       if isinstance(strategy, SixmaxDeployStrategy)
                       else HandTracker())
```

**Note for the implementer:** the `SixmaxHandTracker` does not track `my_committed` from `player_action` messages for other-street opponents — for the abstraction key only *our* committed matters (via `current_bet_bb`/`stack_bb`), and pot comes straight from `your_turn`. Update `my_committed` on our own actions if a later `your_turn` in the same street needs it (mirror `HandTracker._update_committed`). Add that only if a same-street second decision test fails.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/scripts/test_openpoker_sixmax.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (all 178 prior tests plus the new ones).

- [ ] **Step 6: Commit**

```bash
git add scripts/openpoker_bot.py tests/scripts/test_openpoker_sixmax.py
git commit -m "feat(sixmax): deploy blueprint via openpoker_bot (.bin + SixmaxHandTracker)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: GROW — update the `.mex` scaffold and run the eval end-to-end

Record the new capability and verify the deployment path against a real (dev-scale) checkpoint end-to-end.

**Files:**
- Modify: `.mex/ROUTER.md`, `.mex/context/architecture.md`, `.mex/context/setup.md` (command reference)

- [ ] **Step 1: Produce a dev checkpoint and run the eval manually**

```bash
uv run python scripts/train_sixmax.py --num-players 2 --iterations 50000 \
    --checkpoint sixmax/checkpoints/blueprint_hu.bin
uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/blueprint_hu.bin \
    --tabular cfr/checkpoints/checkpoint_09040000.pkl --hands 200
```

Expected: prints `blueprint vs tabular: <±X.XX> BB/100 (400 hands)`. (Absolute sign is not gated at dev scale; the point is the pipeline runs end-to-end. Do **not** commit the `.bin`.)

- [ ] **Step 2: Update `ROUTER.md`**

In `.mex/ROUTER.md`, move Phase 1c from "Not yet built" to "Working": add a line under **Working** describing the deployment bridge (`agents/sixmax_agent.py` `SixmaxDeployStrategy` + `SixmaxAgent`, `agents/neural_agent.py`, `scripts/eval_hu_sanity.py`, `.bin` support in `openpoker_bot.py`, shared `pack_abstract_key`). Remove Phase 1c from the "Not yet built" bullet, leaving Phases 2–3. Bump `last_updated` to today.

- [ ] **Step 3: Update `architecture.md`**

In `.mex/context/architecture.md`, under the `sixmax/` module-map entry and the "Trained strategies flow outward" paragraph, note the new bridges: `agents/sixmax_agent.py` and `agents/neural_agent.py` now join `agents/cfr_agent.py` as the sanctioned cross-subsystem bridges; `scripts/eval_hu_sanity.py` is a fourth strategy consumer; `openpoker_bot.py` now auto-detects `.bin`. Bump `last_updated`.

- [ ] **Step 4: Update `setup.md` command reference**

Add to `.mex/context/setup.md`:

```bash
uv run python scripts/eval_hu_sanity.py --blueprint sixmax/checkpoints/blueprint_hu.bin \
    --tabular cfr/checkpoints/checkpoint_09040000.pkl --neural neural_cfr/checkpoints/checkpoint.pt
uv run python scripts/openpoker_bot.py --checkpoint sixmax/checkpoints/blueprint.bin  # deploy blueprint
```

Bump `last_updated`.

- [ ] **Step 5: Log the decision**

```bash
mex log --type decision "Phase 1c: blueprint deployment via shared pack_abstract_key + SixmaxDeployStrategy; HU-mode sanity eval in the live engine (SixmaxAgent/NeuralAgent). Key reconstruction routed through one C++ packer so bridge and trainer cannot drift."
```

- [ ] **Step 6: Commit**

```bash
git add .mex/
git commit -m "docs(mex): GROW pass for Phase 1c — blueprint deployment + HU sanity eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-17-sixmax-search-design.md` Phase 1 deliverable + Evaluation program):
- "deployed via the hardened `openpoker_bot.py` with a new strategy loader" → Task 7 (`.bin` detection + `SixmaxHandTracker`).
- "One translation layer: `to_chips()`/`nearest()` ... used by ... the openpoker bridge" → Tasks 3/4/7 route all bet legality + sizing through `ActionVocab.target_bb`; keys through the single `pack_abstract_key` (Task 1).
- "Artifact contract: every checkpoint embeds the vocab descriptor + hash; loaders refuse mismatches" → reused as-is via `BlueprintStrategy.load(path, vocab)`; no change needed.
- "HU-mode sanity: the sixmax system in 2-player mode vs the frozen tabular and neural bots" → Tasks 4, 5, 6.
- "any failure or timeout falls back to the blueprint action" is a Phase 2 (search) concern — out of scope here; the bridge already falls back to uniform-legal on unseen infosets (Task 3).

**Placeholder scan:** no "TBD"/"handle edge cases"/"similar to Task N" — each code step contains full code. The `Action.BET` convention is resolved in-plan (raise-*to*, per `game/poker.py:139`; Tasks 4/5 `_translate` return raise-to and are guarded by chip-conservation tests). The one remaining **implementer note** (Task 7 `my_committed` same-street update) is a verification instruction with a named test that surfaces a mismatch, not deferred design.

**Raise-count fidelity:** `game/poker.py` no longer caps `raises_per_street` in the game state (Task 4 Step 1) — the cap is now each *consumer's* responsibility: tabular clamps to 2 (`cfr_agent.py`), sixmax to 3 (`SixmaxAgent`), neural already clamps to 2 (`inference.cpp:36`). This matches the pattern `neural_cfr` already used, removes an abstraction decision that had leaked into shared engine code (the hardcoded `2` was the *tabular* cap), and lets the sixmax bridge read the true count with a one-line `min(r, 3)` instead of observer plumbing. It is behavior-preserving for tabular/neural — each clamps back to exactly the value it saw before — and is gated by the existing full suite.

**Eval cost:** `run_duplicate_match` builds each agent once per seat and reuses it across the hand loop — no per-hand checkpoint reloads or `buck2` rebuilds (the latter would otherwise fire once per hand inside `NeuralAgent`).

**Type consistency:** `SixmaxDeployStrategy.decide(...)` keyword signature is identical across its definition (Task 3) and all three call sites (Tasks 4, 7). `canonical_live_after(n, button, hero, folded, all_in, street)` is identical across definition and both callers. `pack_abstract_key(card, street, raises, pot_bb, live, after)` matches between the C++ binding (Task 1) and the Python call (Task 3). Card encoding `(rank-2)*4 + suit_idx` with `clubs=0..spades=3` is consistent across `sixmax_agent`, `neural_agent`, and the existing `openpoker_bot`.

**Known risk to watch during execution:** the abstraction `bucket()` for postflop keys runs Monte-Carlo rollouts per call — acceptable for one decision per turn, but the eval (Task 6) invokes it every postflop action; if the HU eval is slow, cache `bucket` per (hole, board) within a hand. Not pre-optimized (YAGNI); flagged only.

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-19-sixmax-phase1c-deployment.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
