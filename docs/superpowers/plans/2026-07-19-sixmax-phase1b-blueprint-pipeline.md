# Six-Max Phase 1b — Blueprint Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 1a solver core into a runnable blueprint-training pipeline: card/history abstraction, multithreaded MCCFR trainer, self-describing checkpoints, a `train_sixmax.py` entry point, and a duplicate-deal A/B eval harness with best-checkpoint selection.

**Architecture:** A new `sixmax/src/abstraction/` unit provides lossless 169-class preflop indexing and deterministic Monte-Carlo equity-percentile buckets postflop. `EngineGameState` gains an abstraction-based `infoset_key()` (bit-packed, collision-free) that replaces the Phase 1a naive keyer when an `Abstraction` is attached. A new `BlueprintTrainer` (sharded hashmap, per-thread `Game` instances, atomic global iteration counter for linear weighting) is validated against the same Kuhn −1/18 gate as the single-threaded trainer. Checkpoints are binary artifacts embedding the vocab hash and the abstraction (config + quantile edges); `BlueprintStrategy` loads them for eval and later deployment. Python drives everything through two scripts modeled on the proven `neural_cfr` patterns.

**Tech Stack:** C++17 (Buck2 `//sixmax:sixmax`, pybind11), Python 3.10 (uv), pytest.

**Out of scope (Phase 1c):** openpoker deployment loader, live-table translation, HU-mode sanity evals vs the frozen tabular/neural bots.

## Global Constraints

Copied from the spec (`docs/superpowers/specs/2026-07-17-sixmax-search-design.md`) and repo invariants (`CLAUDE.md`, `.mex/AGENTS.md`):

- `sixmax/` never imports `game/poker.py`, `cfr/`, or `neural_cfr/`; shared C++ only via `common/`. Test files under `tests/` may import both sides.
- Chip frame: doubles, `big_blind = 1.0`, `small_blind = 0.5`, `starting_stack = 100.0`, `kChipEps = 1e-9`. Card code = `(rank-2)*4 + suit`.
- Vocab is config-defined; storage order is canonical config order; **legality by masking, never reordering or filtering storage**. Strategy/regret vectors are sized `vocab.size()`.
- **Artifact contract:** every checkpoint embeds the vocab hash and the abstraction descriptor; loaders refuse mismatches.
- Evaluator access only through the opaque `safe_eval::HandRank` API (`rank7`, `beats`, `ties`); raw scores never leave `common/`.
- Spec abstraction parameters (initial values, revisited later with measurements): 169 lossless preflop classes; equity-percentile buckets **50 flop / 50 turn / 20 river**; per-street raise count **capped at 3**; pot-size bucket **4 buckets**.
- Design decision (approved 2026-07-19): the infoset key additionally contains `live_opps` (opponents not folded, 1–5) and `after` (live non-all-in players acting after hero in this street's canonical order, 0–5). This encoding is table-size-agnostic: MP 6-handed and UTG 5-handed share infosets.
- Kuhn gate: the closed-form game value is **−1/18 ≈ −0.0556**, tolerance **±0.01**. Never widen the tolerance.
- Linear CFR weighting: updates at global iteration `t` are multiplied by `t`; the counter is cumulative and monotonic across `train()` calls and resumes (neural_cfr pattern).
- MCCFR update rules (must match the validated Phase 1a `MCCFRTrainer` exactly): traverser explores ALL legal actions via clone and updates regrets; opponents accumulate `weight * sigma` into strategy_sum and sample ONE action; regret matching falls back to uniform-over-legal when no positive regret.
- Never commit checkpoints or secrets. `sixmax/checkpoints/` must be gitignored before any training run.
- Build: `~/bin/buck2 build //sixmax:sixmax` (buck2 is NOT on PATH). Tests: `uv run pytest tests/sixmax/ -v` from the repo root (`tests/sixmax/conftest.py` auto-builds and force-loads the `.so`). Full-suite gate: `uv run pytest tests/ -q` — everything green except known stochastic xfail/xpass drift in `tests/cfr/test_mccfr.py` (never chase it).
- Commit messages end with the trailer line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Infoset key bit layout (fixed by this plan; Tasks 2, 4, 6 all rely on it):**

| bits | field | range |
|---|---|---|
| 0–7 | card (preflop: 169-class; postflop: equity bucket) | 0–168 |
| 8–9 | street | 0–3 |
| 10–17 | per-street raise counts, 2 bits each, capped 3 | 0–3 ×4 |
| 18–19 | pot bucket (edges 7 / 15 / 40 BB) | 0–3 |
| 20–22 | live_opps | 1–5 |
| 23–25 | after | 0–5 |

## File Structure

| File | Responsibility |
|---|---|
| `sixmax/src/abstraction/abstraction.h/.cpp` (new) | `preflop_class`, deterministic `hand_equity`, `Abstraction` (quantile edges, bucket lookup, hash) |
| `sixmax/src/blueprint/engine_game.h/.cpp` (modify) | raise-count tracking, `abstract_key()`, optional `Abstraction*` on state and game |
| `sixmax/src/blueprint/mccfr.h/.cpp` (modify) | extract `regret_matched` free function; `kuhn_exact_value_lookup` |
| `sixmax/src/blueprint/trainer.h/.cpp` (new) | multithreaded `BlueprintTrainer` (sharded table, factory-based, generic over `Game`) |
| `sixmax/src/blueprint/checkpoint.h/.cpp` (new) | binary save/load, `BlueprintStrategy` |
| `sixmax/src/bindings/bindings.cpp` (modify) | bind everything above |
| `sixmax/BUCK` (modify) | add the three new .cpp files |
| `sixmax/configs/default.toml` (modify) | `[abstraction]` and `[train.blueprint]` sections |
| `scripts/train_sixmax.py` (new) | training entry point (config resolution, chunked checkpointing, selection hook) |
| `scripts/eval_sixmax.py` (new) | duplicate-deal seat-rotated A/B match, BB/100 output |
| `scripts/setup_dev.sh` (modify) | pybind11 include-symlink self-heal (ledger-promised hardening) |
| `.gitignore` (modify) | `sixmax/checkpoints/` |
| `tests/sixmax/test_abstraction.py`, `test_abstract_key.py`, `test_blueprint_trainer.py`, `test_blueprint_checkpoint.py`, `test_train_sixmax.py`, `test_eval_sixmax.py` (new) | per-task tests |

---

### Task 1: Card abstraction — 169 preflop classes + equity-percentile buckets

**Files:**
- Create: `sixmax/src/abstraction/abstraction.h`, `sixmax/src/abstraction/abstraction.cpp`
- Modify: `sixmax/BUCK` (add `src/abstraction/abstraction.cpp` to srcs), `sixmax/src/bindings/bindings.cpp` (append bindings)
- Test: `tests/sixmax/test_abstraction.py`

**Interfaces:**
- Consumes: `safe_eval::rank7(std::array<int,7>)` from `common/` (include `"game/safe_eval.h"`).
- Produces (used by Tasks 2–6): `int preflop_class(const std::array<int,2>&)`; `double hand_equity(const std::array<int,2>&, const std::vector<int>&, int rollouts, uint64_t salt)`; `struct AbstractionConfig{int flop_buckets, turn_buckets, river_buckets, equity_rollouts, quantile_samples; uint64_t seed;}`; `class Abstraction` with `Abstraction(const AbstractionConfig&)`, `Abstraction(const AbstractionConfig&, std::array<std::vector<double>,3>)`, `int bucket(hole, board) const`, `int num_buckets(int street) const`, `const AbstractionConfig& config() const`, `const std::array<std::vector<double>,3>& edges() const`, `uint64_t hash() const`.

**Determinism is the load-bearing property:** `hand_equity` seeds its RNG from `(sorted hole, sorted board, salt)`, so the same inputs always give the same estimate. Infoset keys derived from buckets are therefore stable across visits, threads, runs, and machines — no cache or shared state needed.

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_abstraction.py`:

```python
"""169-class preflop indexing + deterministic MC equity-percentile buckets."""
import itertools

import sixmax

# card(r, s) = (r-2)*4 + s ; suits 0=c 1=d 2=h 3=s
def card(rank, suit):
    return (rank - 2) * 4 + suit


TINY = dict(flop_buckets=10, turn_buckets=10, river_buckets=5,
            equity_rollouts=40, quantile_samples=300, seed=42)


def test_preflop_class_covers_exactly_169():
    classes = set()
    for c0, c1 in itertools.combinations(range(52), 2):
        classes.add(sixmax.preflop_class([c0, c1]))
    assert len(classes) == 169
    assert min(classes) == 0 and max(classes) == 168


def test_preflop_class_known_values():
    # pairs on the diagonal: class = r*13 + r
    assert sixmax.preflop_class([card(14, 3), card(14, 2)]) == 12 * 13 + 12  # AA
    assert sixmax.preflop_class([card(2, 0), card(2, 1)]) == 0              # 22
    # suited above the diagonal (hi*13+lo), offsuit below (lo*13+hi)
    assert sixmax.preflop_class([card(14, 3), card(13, 3)]) == 12 * 13 + 11  # AKs
    assert sixmax.preflop_class([card(14, 3), card(13, 1)]) == 11 * 13 + 12  # AKo
    # order of the two cards must not matter
    assert (sixmax.preflop_class([card(13, 1), card(14, 3)])
            == sixmax.preflop_class([card(14, 3), card(13, 1)]))


BOARD = [card(13, 0), card(8, 1), card(3, 2)]  # Kc 8d 3h — dry rainbow


def test_hand_equity_orders_hands_and_is_deterministic():
    aa = [card(14, 3), card(14, 1)]
    trash = [card(7, 0), card(2, 3)]
    e_aa = sixmax.hand_equity(aa, BOARD, 200, 7)
    e_tr = sixmax.hand_equity(trash, BOARD, 200, 7)
    assert 0.0 <= e_tr < e_aa <= 1.0
    assert e_aa > 0.75
    assert e_tr < 0.5
    # deterministic: same inputs -> bitwise-same estimate
    assert e_aa == sixmax.hand_equity(aa, BOARD, 200, 7)
    # hole-card order must not matter (seed uses sorted cards)
    assert e_aa == sixmax.hand_equity([card(14, 1), card(14, 3)], BOARD, 200, 7)


def test_abstraction_buckets_in_range_and_monotone():
    abs_ = sixmax.Abstraction(**TINY)
    aa = [card(14, 3), card(14, 1)]
    trash = [card(7, 0), card(2, 3)]
    b_aa = abs_.bucket(aa, BOARD)
    b_tr = abs_.bucket(trash, BOARD)
    assert 0 <= b_tr <= b_aa < 10
    # river board: bucket range obeys river_buckets
    river = BOARD + [card(9, 3), card(4, 1)]
    assert 0 <= abs_.bucket(aa, river) < 5
    assert abs_.num_buckets(1) == 10
    assert abs_.num_buckets(3) == 5


def test_abstraction_edges_sorted_and_hash_stable():
    a1 = sixmax.Abstraction(**TINY)
    a2 = sixmax.Abstraction(**TINY)
    assert a1.hash() == a2.hash()          # same config -> same edges -> same hash
    a3 = sixmax.Abstraction(**{**TINY, "seed": 43})
    assert a1.hash() != a3.hash()
    for street_edges in a1.edges():
        assert street_edges == sorted(street_edges)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_abstraction.py -v`
Expected: FAIL — `AttributeError: module 'sixmax' has no attribute 'preflop_class'`.

- [ ] **Step 3: Implement the abstraction unit**

Create `sixmax/src/abstraction/abstraction.h`:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace sixmax {

// Lossless 169-class preflop index (0..168): pairs on the diagonal
// (r*13+r), suited above it (hi*13+lo), offsuit below (lo*13+hi).
// Pure function of the two card codes; card order is irrelevant.
int preflop_class(const std::array<int, 2>& hole);

// Deterministic Monte-Carlo equity of hole vs ONE uniform random opponent
// hand, rolling out the remaining board with the shared evaluator. The RNG
// is seeded from (sorted hole, sorted board, salt): identical inputs always
// return the identical estimate, so bucket assignments derived from it are
// stable across visits, threads, runs, and machines — no cache needed.
double hand_equity(const std::array<int, 2>& hole,
                   const std::vector<int>& board, int rollouts, uint64_t salt);

struct AbstractionConfig {
    int flop_buckets = 50;
    int turn_buckets = 50;
    int river_buckets = 20;
    int equity_rollouts = 100;     // MC rollouts per equity estimate
    int quantile_samples = 10000;  // per street, for percentile edges
    uint64_t seed = 20260719;      // edge sampling + equity salt
};

// Postflop equity-percentile buckets. Edges are per-street equity quantiles
// estimated once from quantile_samples random (hole, board) draws, so
// buckets are (approximately) equally populated. Street inferred from board
// size (3=flop, 4=turn, 5=river).
class Abstraction {
public:
    explicit Abstraction(const AbstractionConfig& cfg);  // builds edges
    Abstraction(const AbstractionConfig& cfg,
                std::array<std::vector<double>, 3> edges);  // from artifact
    int bucket(const std::array<int, 2>& hole,
               const std::vector<int>& board) const;
    int num_buckets(int street) const;  // street 1=flop 2=turn 3=river
    const AbstractionConfig& config() const { return cfg_; }
    const std::array<std::vector<double>, 3>& edges() const { return edges_; }
    uint64_t hash() const;  // config + edges; part of the artifact contract

private:
    AbstractionConfig cfg_;
    std::array<std::vector<double>, 3> edges_;  // [0]=flop [1]=turn [2]=river
};

}  // namespace sixmax
```

Create `sixmax/src/abstraction/abstraction.cpp`:

```cpp
#include "abstraction/abstraction.h"
#include <algorithm>
#include <random>
#include "game/safe_eval.h"

namespace sixmax {

int preflop_class(const std::array<int, 2>& hole) {
    int r0 = hole[0] / 4, r1 = hole[1] / 4;
    int hi = std::max(r0, r1), lo = std::min(r0, r1);
    bool suited = (hole[0] % 4) == (hole[1] % 4);
    if (hi == lo) return hi * 13 + hi;
    return suited ? hi * 13 + lo : lo * 13 + hi;
}

namespace {

uint64_t fnv_mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
    return h;
}

uint64_t equity_seed(const std::array<int, 2>& hole,
                     const std::vector<int>& board, uint64_t salt) {
    std::vector<int> bd = board;
    std::sort(bd.begin(), bd.end());
    uint64_t h = 1469598103934665603ull;
    h = fnv_mix(h, (uint64_t)std::min(hole[0], hole[1]));
    h = fnv_mix(h, (uint64_t)std::max(hole[0], hole[1]));
    // +64 namespaces board codes away from hole codes (same trick as the
    // naive keyer in engine_game.cpp).
    for (int c : bd) h = fnv_mix(h, (uint64_t)(c + 64));
    return fnv_mix(h, salt);
}

}  // namespace

double hand_equity(const std::array<int, 2>& hole,
                   const std::vector<int>& board, int rollouts,
                   uint64_t salt) {
    std::mt19937_64 rng(equity_seed(hole, board, salt));
    bool used[52] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int c : board) used[c] = true;
    std::vector<int> deck;
    deck.reserve(52);
    for (int c = 0; c < 52; ++c)
        if (!used[c]) deck.push_back(c);
    const int need = 2 + (5 - (int)board.size());  // opp hole + runout
    double score = 0.0;
    for (int r = 0; r < rollouts; ++r) {
        // Partial Fisher-Yates over the live deck; prior permutations do
        // not bias later draws.
        for (int i = 0; i < need; ++i) {
            std::uniform_int_distribution<int> d(i, (int)deck.size() - 1);
            std::swap(deck[i], deck[d(rng)]);
        }
        std::array<int, 7> mine{}, theirs{};
        mine[0] = hole[0];
        mine[1] = hole[1];
        theirs[0] = deck[0];
        theirs[1] = deck[1];
        int k = 2;
        for (int c : board) { mine[k] = theirs[k] = c; ++k; }
        for (int i = 2; i < need; ++i) { mine[k] = theirs[k] = deck[i]; ++k; }
        auto a = safe_eval::rank7(mine);
        auto b = safe_eval::rank7(theirs);
        if (a.beats(b)) score += 1.0;
        else if (a.ties(b)) score += 0.5;
    }
    return score / rollouts;
}

Abstraction::Abstraction(const AbstractionConfig& cfg) : cfg_(cfg) {
    std::mt19937_64 rng(cfg.seed);
    const int nbuckets[3] = {cfg.flop_buckets, cfg.turn_buckets,
                             cfg.river_buckets};
    for (int s = 0; s < 3; ++s) {
        const int board_n = 3 + s;
        std::vector<double> eqs;
        eqs.reserve(cfg.quantile_samples);
        for (int i = 0; i < cfg.quantile_samples; ++i) {
            std::array<int, 52> deck;
            for (int c = 0; c < 52; ++c) deck[c] = c;
            for (int j = 0; j < 2 + board_n; ++j) {
                std::uniform_int_distribution<int> d(j, 51);
                std::swap(deck[j], deck[d(rng)]);
            }
            std::array<int, 2> hole{deck[0], deck[1]};
            std::vector<int> board(deck.begin() + 2,
                                   deck.begin() + 2 + board_n);
            eqs.push_back(
                hand_equity(hole, board, cfg.equity_rollouts, cfg.seed));
        }
        std::sort(eqs.begin(), eqs.end());
        auto& e = edges_[s];
        for (int k = 1; k < nbuckets[s]; ++k)
            e.push_back(eqs[(size_t)((double)k * eqs.size() / nbuckets[s])]);
    }
}

Abstraction::Abstraction(const AbstractionConfig& cfg,
                         std::array<std::vector<double>, 3> edges)
    : cfg_(cfg), edges_(std::move(edges)) {}

int Abstraction::bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board) const {
    const int s = (int)board.size() - 3;  // 0=flop 1=turn 2=river
    double eq = hand_equity(hole, board, cfg_.equity_rollouts, cfg_.seed);
    const auto& e = edges_[s];
    return (int)(std::upper_bound(e.begin(), e.end(), eq) - e.begin());
}

int Abstraction::num_buckets(int street) const {
    switch (street) {
        case 1: return cfg_.flop_buckets;
        case 2: return cfg_.turn_buckets;
        default: return cfg_.river_buckets;
    }
}

uint64_t Abstraction::hash() const {
    uint64_t h = 1469598103934665603ull;
    auto mix_u = [&](uint64_t v) { h = fnv_mix(h, v); };
    auto mix_d = [&](double v) {
        uint64_t b;
        __builtin_memcpy(&b, &v, sizeof b);
        h = fnv_mix(h, b);
    };
    mix_u((uint64_t)cfg_.flop_buckets);
    mix_u((uint64_t)cfg_.turn_buckets);
    mix_u((uint64_t)cfg_.river_buckets);
    mix_u((uint64_t)cfg_.equity_rollouts);
    mix_u((uint64_t)cfg_.quantile_samples);
    mix_u(cfg_.seed);
    for (const auto& e : edges_) {
        mix_u((uint64_t)e.size());
        for (double v : e) mix_d(v);
    }
    return h;
}

}  // namespace sixmax
```

- [ ] **Step 4: Add to the build and bind**

In `sixmax/BUCK`, extend `srcs`:

```python
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp", "src/blueprint/mccfr.cpp",
            "src/engine/engine.cpp", "src/abstraction/abstraction.cpp"],
```

In `sixmax/src/bindings/bindings.cpp`, add `#include "abstraction/abstraction.h"` to the includes, then append inside `PYBIND11_MODULE` (before the closing brace):

```cpp
    // --- Card abstraction (Phase 1b Task 1) ---
    m.def("preflop_class", [](const std::vector<int>& hole) {
        if (hole.size() != 2) throw py::value_error("expects 2 cards");
        return sixmax::preflop_class({hole[0], hole[1]});
    });
    m.def("hand_equity",
          [](const std::vector<int>& hole, const std::vector<int>& board,
             int rollouts, uint64_t salt) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::hand_equity({hole[0], hole[1]}, board, rollouts,
                                         salt);
          },
          py::arg("hole"), py::arg("board"), py::arg("rollouts"),
          py::arg("salt"));
    py::class_<sixmax::Abstraction>(m, "Abstraction")
        .def(py::init([](int flop_buckets, int turn_buckets, int river_buckets,
                         int equity_rollouts, int quantile_samples,
                         uint64_t seed) {
                 return sixmax::Abstraction(sixmax::AbstractionConfig{
                     flop_buckets, turn_buckets, river_buckets,
                     equity_rollouts, quantile_samples, seed});
             }),
             py::kw_only(), py::arg("flop_buckets") = 50,
             py::arg("turn_buckets") = 50, py::arg("river_buckets") = 20,
             py::arg("equity_rollouts") = 100,
             py::arg("quantile_samples") = 10000,
             py::arg("seed") = 20260719)
        .def("bucket",
             [](const sixmax::Abstraction& a, const std::vector<int>& hole,
                const std::vector<int>& board) {
                 if (hole.size() != 2) throw py::value_error("expects 2 cards");
                 return a.bucket({hole[0], hole[1]}, board);
             })
        .def("num_buckets", &sixmax::Abstraction::num_buckets)
        .def("edges", &sixmax::Abstraction::edges)
        .def("hash", &sixmax::Abstraction::hash);
```

- [ ] **Step 5: Build and run the tests**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_abstraction.py -v`
Expected: 5 passed (a few seconds — the tiny abstraction builds 900 sampled equities).

- [ ] **Step 6: Full sixmax suite, then commit**

Run: `uv run pytest tests/sixmax/ -q` — expected: all pass, no regressions.

```bash
git add sixmax/src/abstraction/ sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_abstraction.py
git commit -m "feat(sixmax): 169 preflop classes + equity-percentile postflop abstraction

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Abstraction-based infoset keys in EngineGameState

**Files:**
- Modify: `sixmax/src/blueprint/engine_game.h`, `sixmax/src/blueprint/engine_game.cpp`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_abstract_key.py`

**Interfaces:**
- Consumes: `preflop_class`, `Abstraction` from Task 1; existing `HandState` accessors (`street()`, `button()`, `num_players()`, `player(i)`, `hole_cards(i)`, `board()`, `pot()`).
- Produces (used by Tasks 3–6): `EngineGameState::abstract_key(const Abstraction&) const` (public, computes the bit-packed key for ANY abstraction — eval strategies with different abstractions each compute their own key from the same state); `EngineGameState` ctor gains trailing `const Abstraction* abstraction = nullptr`; `EngineGame` ctor gains trailing `const Abstraction* abstraction = nullptr` (threads it into every dealt state so `infoset_key()` inside MCCFR uses the abstraction). Bit layout as pinned in Global Constraints.

**Semantics being implemented:**
- `raises_[street]` increments (capped at 3) on every applied Bet/AllIn vocab action, **before** `hand_.apply` (the street may advance when the action closes the round).
- Canonical action order start: preflop = seat after BB, i.e. `(button+3) % n` for `n > 2`, `button` for HU; postflop = `(button+1) % n`. `order(seat) = (seat - start + n) % n`.
- `live_opps` = opponents with `!folded`. `after` = live, non-all-in opponents with `order(seat) > order(hero)`.
- Preflop card field = `preflop_class(hole)` (no equity computation); postflop = `abs.bucket(hole, board)`.
- Pot bucket: `pot <= 7 → 0`, `<= 15 → 1`, `<= 40 → 2`, else `3`.
- `infoset_key()` = `abstract_key(*abstraction_)` when an abstraction is attached, else the Phase 1a naive keyer (unchanged — Kuhn and all existing tests must keep passing).

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_abstract_key.py`:

```python
"""Bit-packed abstraction infoset keys: field packing, and the table-size
invariance that motivated the (live_opps, after) encoding — MP 6-handed and
UTG 5-handed are literally the same infoset."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
FOLD, CHECK, CALL, OPEN25, B33 = 0, 1, 2, 3, 6

def card(rank, suit):
    return (rank - 2) * 4 + suit

# key field extractors (bit layout pinned in the plan's Global Constraints)
def f_card(k):   return k & 0xFF
def f_street(k): return (k >> 8) & 3
def f_raises(k, st): return (k >> (10 + 2 * st)) & 3
def f_potb(k):   return (k >> 18) & 3
def f_live(k):   return (k >> 20) & 7
def f_after(k):  return (k >> 23) & 7

AKo = [card(14, 3), card(13, 1)]  # class 11*13+12 = 155


def _state(n, hero_seat, deck_override):
    cfg = sixmax.EngineConfig(num_players=n)
    deck = list(range(2 * n + 5))
    # place hero's hole cards at deck[2*seat], deck[2*seat+1]; keep the rest
    # distinct by remapping any collision onto high spare codes
    spares = iter(range(40, 52))
    taken = set(deck_override)
    deck = [c if c not in taken else next(spares) for c in deck]
    deck[2 * hero_seat] = deck_override[0]
    deck[2 * hero_seat + 1] = deck_override[1]
    return sixmax.EngineGameState(cfg, 0, deck, VOCAB, [], abstraction=ABS)


def test_mp_6handed_equals_utg_5handed():
    # 6-handed, button 0: preflop order UTG=3, HJ=4, CO=5, BTN=0, SB=1, BB=2.
    s6 = _state(6, hero_seat=4, deck_override=AKo)
    s6.apply(FOLD)                      # UTG (seat 3) folds -> HJ (seat 4) acts
    # 5-handed, button 0: UTG=3 acts first.
    s5 = _state(5, hero_seat=3, deck_override=AKo)
    k6, k5 = s6.infoset_key(), s5.infoset_key()
    assert k6 == k5                     # same spot, different table size
    assert f_card(k6) == 155 and f_street(k6) == 0
    assert f_live(k6) == 4 and f_after(k6) == 4
    assert f_potb(k6) == 0              # pot 1.5 BB


def test_key_fields_track_raises_pot_and_position():
    s = _state(2, hero_seat=0, deck_override=AKo)
    k = s.infoset_key()                 # HU preflop: SB/BTN (seat 0) first
    assert f_live(k) == 1 and f_after(k) == 1 and f_raises(k, 0) == 0
    s.apply(OPEN25)                     # raise count street 0 -> 1
    k = s.infoset_key()                 # BB now acting
    assert f_raises(k, 0) == 1 and f_after(k) == 0
    s.apply(CALL)
    s.apply(CHECK)                      # flop dealt; HU postflop BB first
    k = s.infoset_key()
    assert f_street(k) == 1
    assert f_raises(k, 0) == 1 and f_raises(k, 1) == 0
    assert 0 <= f_card(k) < 10          # flop equity bucket


def test_raise_count_caps_at_three():
    s = _state(2, hero_seat=0, deck_override=AKo)
    s.apply(OPEN25)
    for _ in range(4):                  # raise war beyond the cap
        mask = s.legal_mask()
        raise_idx = next(i for i in range(3, 9) if mask[i])
        s.apply(raise_idx)
        if s.is_terminal():
            break
    assert f_raises(s.infoset_key(), 0) == 3 if not s.is_terminal() else True


def test_naive_keyer_still_default_without_abstraction():
    cfg = sixmax.EngineConfig(num_players=2)
    s = sixmax.EngineGameState(cfg, 0, list(range(52)), VOCAB, [])
    t = sixmax.EngineGameState(cfg, 0, list(range(52)), VOCAB, [],
                               abstraction=ABS)
    assert s.infoset_key() != t.infoset_key()  # naive vs packed differ
    # abstract_key with an explicit abstraction works on a plain state too
    assert s.abstract_key(ABS) == t.infoset_key()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_abstract_key.py -v`
Expected: FAIL — `TypeError` (no `abstraction` kwarg) / `AttributeError: abstract_key`.

- [ ] **Step 3: Implement**

In `sixmax/src/blueprint/engine_game.h`: add `#include "abstraction/abstraction.h"`; change `EngineGameState` to:

```cpp
class EngineGameState : public GameState {
public:
    EngineGameState(HandState hand, const ActionVocab* vocab,
                    const Abstraction* abstraction = nullptr)
        : hand_(std::move(hand)), vocab_(vocab), abstraction_(abstraction) {}
    // ... existing overrides unchanged ...
    // Bit-packed abstraction key (layout in the Phase 1b plan). Public and
    // parameterized so eval-time strategies with their own abstractions can
    // key the same public state independently.
    uint64_t abstract_key(const Abstraction& abs) const;
    BetContext bet_context() const;

private:
    bool size_class_ok(const AbstractAction& a) const;
    HandState hand_;
    const ActionVocab* vocab_;
    const Abstraction* abstraction_ = nullptr;
    std::vector<int> history_;          // naive-keyer input (unchanged)
    std::array<uint8_t, 4> raises_{};   // per-street raise count, capped 3
};
```

and `EngineGame`:

```cpp
class EngineGame : public Game {
public:
    EngineGame(EngineConfig cfg, const ActionVocab* vocab,
               const Abstraction* abstraction = nullptr)
        : cfg_(cfg), vocab_(vocab), abstraction_(abstraction) {}
    // ... unchanged ...
private:
    EngineConfig cfg_;
    const ActionVocab* vocab_;
    const Abstraction* abstraction_ = nullptr;
    int button_ = 0;
};
```

In `sixmax/src/blueprint/engine_game.cpp`:

1. In `apply()`, before the `switch` add nothing; inside the `Bet`/`AllIn` case, before `hand_.apply(...)`:

```cpp
        case ActionType::Bet:
        case ActionType::AllIn: {
            int st = (int)hand_.street();
            if (raises_[st] < 3) ++raises_[st];  // capped per-street count
            double target = vocab_->target_bb(action, bet_context());
            hand_.apply({EngineActionType::RaiseTo, target});
            break;
        }
```

2. Change `infoset_key()` to dispatch:

```cpp
uint64_t EngineGameState::infoset_key() const {
    if (abstraction_) return abstract_key(*abstraction_);
    // FNV-1a over exact private+public information (naive fallback,
    // used by tests and any vocab-only construction).
    ... existing body unchanged ...
}
```

3. Add `abstract_key` and a pot-bucket helper:

```cpp
namespace {
int pot_bucket(double pot) {
    if (pot <= 7.0) return 0;
    if (pot <= 15.0) return 1;
    if (pot <= 40.0) return 2;
    return 3;
}
}  // namespace

uint64_t EngineGameState::abstract_key(const Abstraction& abs) const {
    const int p = hand_.current_player();
    const int n = hand_.num_players();
    const int street = (int)hand_.street();
    auto hole = hand_.hole_cards(p);
    const int card = street == 0 ? preflop_class(hole)
                                 : abs.bucket(hole, hand_.board());
    // Canonical action-order start: preflop = seat after BB (HU: button);
    // postflop = seat after button.
    const int start = street == 0
        ? (n == 2 ? hand_.button() : (hand_.button() + 3) % n)
        : (hand_.button() + 1) % n;
    auto order = [&](int seat) { return (seat - start + n) % n; };
    int live = 0, after = 0;
    for (int s = 0; s < n; ++s) {
        if (s == p || hand_.player(s).folded) continue;
        ++live;
        if (!hand_.player(s).all_in && order(s) > order(p)) ++after;
    }
    uint64_t key = (uint64_t)card;                        // bits 0-7
    key |= (uint64_t)street << 8;                         // bits 8-9
    for (int st = 0; st < 4; ++st)
        key |= (uint64_t)raises_[st] << (10 + 2 * st);    // bits 10-17
    key |= (uint64_t)pot_bucket(hand_.pot()) << 18;       // bits 18-19
    key |= (uint64_t)live << 20;                          // bits 20-22
    key |= (uint64_t)after << 23;                         // bits 23-25
    return key;
}
```

4. `EngineGame::new_hand` passes the abstraction through:

```cpp
std::unique_ptr<GameState> EngineGame::new_hand(std::mt19937_64& rng) {
    button_ = (button_ + 1) % cfg_.num_players;
    return std::make_unique<EngineGameState>(
        HandState::deal(cfg_, button_, rng), vocab_, abstraction_);
}
```

- [ ] **Step 4: Update the bindings**

In `bindings.cpp`, replace the `EngineGameState` init and the `EngineGame` init (keep everything else on those classes):

```cpp
    py::class_<sixmax::EngineGameState, sixmax::GameState>(m, "EngineGameState")
        .def(py::init([](const sixmax::EngineConfig& cfg, int button,
                         std::vector<int> deck, const sixmax::ActionVocab* v,
                         std::vector<double> stacks,
                         const sixmax::Abstraction* abstraction) {
                 return sixmax::EngineGameState(
                     sixmax::HandState(cfg, button, std::move(deck),
                                       std::move(stacks)),
                     v, abstraction);
             }),
             py::arg("cfg"), py::arg("button"), py::arg("deck"),
             py::arg("vocab"), py::arg("stacks") = std::vector<double>{},
             py::arg("abstraction") = nullptr,
             py::keep_alive<1, 5>(),   // state holds ActionVocab*
             py::keep_alive<1, 7>())   // state holds Abstraction*
        .def("bet_context", &sixmax::EngineGameState::bet_context)
        .def("abstract_key", &sixmax::EngineGameState::abstract_key);
    py::class_<sixmax::EngineGame, sixmax::Game>(m, "EngineGame")
        .def(py::init<sixmax::EngineConfig, const sixmax::ActionVocab*,
                      const sixmax::Abstraction*>(),
             py::arg("cfg"), py::arg("vocab"),
             py::arg("abstraction") = nullptr,
             py::keep_alive<1, 3>(),   // game holds ActionVocab*
             py::keep_alive<1, 4>())   // game holds Abstraction*
        .def("new_hand", [](sixmax::EngineGame& g, uint64_t seed) {
            std::mt19937_64 rng(seed);
            return g.new_hand(rng);
        }, py::arg("seed"),
           py::keep_alive<0, 1>());  // returned state holds game-owned ptrs
```

- [ ] **Step 5: Build, run the new tests, then the whole sixmax suite**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_abstract_key.py -v`
Expected: 4 passed.
Run: `uv run pytest tests/sixmax/ -q`
Expected: all pass — the naive keyer, Kuhn, engine, and settlement tests are untouched.

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/blueprint/engine_game.h sixmax/src/blueprint/engine_game.cpp sixmax/src/bindings/bindings.cpp tests/sixmax/test_abstract_key.py
git commit -m "feat(sixmax): bit-packed abstraction infoset keys (card/street/raises/pot/live/after)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Multithreaded BlueprintTrainer (Kuhn-gated)

**Files:**
- Create: `sixmax/src/blueprint/trainer.h`, `sixmax/src/blueprint/trainer.cpp`
- Modify: `sixmax/src/blueprint/mccfr.h`, `sixmax/src/blueprint/mccfr.cpp` (extract `regret_matched`, add `kuhn_exact_value_lookup`), `sixmax/BUCK` (add `src/blueprint/trainer.cpp`), `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_blueprint_trainer.py`

**Interfaces:**
- Consumes: `Game`/`GameState` (Task-agnostic — the factory design lets the SAME trainer run KuhnGame, which is what makes the −1/18 gate apply to the new code); `InfosetData` from `mccfr.h`; `EngineGame(cfg, vocab, abstraction)` from Task 2.
- Produces (used by Tasks 4–6):

```cpp
struct TrainerConfig { int num_threads = 1; uint64_t seed = 1; };
using GameFactory = std::function<std::unique_ptr<Game>()>;
class BlueprintTrainer {
    BlueprintTrainer(GameFactory factory, TrainerConfig cfg);
    void train(uint64_t iterations);            // adds to the global counter
    uint64_t iterations() const;                 // cumulative, resume-safe
    size_t num_infosets() const;
    std::vector<double> average_strategy(uint64_t key) const;  // {} if unseen
    std::vector<uint64_t> keys() const;          // all visited infoset keys
    std::unordered_map<uint64_t, InfosetData> export_table() const;
    void import_table(std::unordered_map<uint64_t, InfosetData> t, uint64_t iters);
};
// mccfr.h additions:
std::vector<double> regret_matched(const std::vector<double>& regret,
                                   const std::vector<uint8_t>& mask);
double kuhn_exact_value_lookup(
    const std::function<std::vector<double>(uint64_t)>& avg);
```

- Python: `sixmax.BlueprintTrainer(cfg, vocab, abstraction, num_threads=1, seed=1)` (EngineGame factory), `sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=1)` (KuhnGame factory), `sixmax.blueprint_kuhn_value(trainer)`.

**Concurrency design:** 64 shards, each `{std::mutex, unordered_map<uint64_t, InfosetData>}`; shard index = top 6 bits of `key * 0x9E3779B97F4A7C15`. Each worker thread owns its own `Game` (from the factory — `EngineGame::new_hand` mutates `button_`, so games must never be shared) and its own `mt19937_64`. The global iteration counter is an atomic; `weight = t` exactly as in the single-threaded trainer. Regret reads take a shard lock, copy what's needed, release; updates re-lock. Slightly stale regards between the two locks are standard MCCFR practice and do not bias the average strategy.

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_blueprint_trainer.py`:

```python
"""BlueprintTrainer: the multithreaded trainer must pass the same Kuhn
-1/18 gate as the Phase 1a single-threaded trainer, single-threaded runs
must be deterministic, and it must drive the abstracted 6-max EngineGame."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          "blueprint")


VOCAB = _load_vocab()
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
KUHN_VALUE = -1.0 / 18.0


def test_kuhn_gate_single_thread():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=3)
    t.train(200_000)
    assert abs(sixmax.blueprint_kuhn_value(t) - KUHN_VALUE) < 0.01


def test_kuhn_gate_four_threads():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=4, seed=5)
    t.train(200_000)
    assert abs(sixmax.blueprint_kuhn_value(t) - KUHN_VALUE) < 0.01


def test_single_thread_determinism():
    a = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=11)
    b = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=11)
    a.train(2000)
    b.train(2000)
    assert sorted(a.keys()) == sorted(b.keys())
    for k in a.keys():
        assert a.average_strategy(k) == b.average_strategy(k)


def test_iterations_accumulate_across_train_calls():
    t = sixmax.BlueprintTrainer.kuhn(num_threads=1, seed=1)
    t.train(100)
    t.train(100)
    assert t.iterations() == 200


def test_drives_abstracted_sixmax_engine():
    cfg = sixmax.EngineConfig(num_players=6)
    t = sixmax.BlueprintTrainer(cfg, VOCAB, ABS, num_threads=2, seed=9)
    t.train(150)
    assert t.num_infosets() > 0
    for k in t.keys():
        probs = t.average_strategy(k)
        if probs:
            assert abs(sum(probs) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_blueprint_trainer.py -v`
Expected: FAIL — `AttributeError: BlueprintTrainer`.

- [ ] **Step 3: Refactor mccfr for shared pieces**

In `sixmax/src/blueprint/mccfr.h`, add above `MCCFRTrainer` (after `InfosetData`):

```cpp
// Regret matching over the masked-legal actions; uniform over legal when no
// positive regret. Shared by MCCFRTrainer and BlueprintTrainer.
std::vector<double> regret_matched(const std::vector<double>& regret,
                                   const std::vector<uint8_t>& mask);
```

and add next to `kuhn_exact_value`:

```cpp
#include <functional>   // at top of file with the other includes
double kuhn_exact_value_lookup(
    const std::function<std::vector<double>(uint64_t)>& avg);
```

In `sixmax/src/blueprint/mccfr.cpp`:

```cpp
std::vector<double> regret_matched(const std::vector<double>& regret,
                                   const std::vector<uint8_t>& mask) {
    int n = (int)regret.size();
    std::vector<double> sigma(n, 0.0);
    double pos = 0.0;
    for (int a = 0; a < n; ++a)
        if (mask[a] && regret[a] > 0.0) pos += regret[a];
    if (pos > 0.0) {
        for (int a = 0; a < n; ++a)
            if (mask[a] && regret[a] > 0.0) sigma[a] = regret[a] / pos;
    } else {
        int legal = 0;
        for (int a = 0; a < n; ++a) legal += mask[a] ? 1 : 0;
        for (int a = 0; a < n; ++a) if (mask[a]) sigma[a] = 1.0 / legal;
    }
    return sigma;
}

std::vector<double> MCCFRTrainer::matched_strategy(
        const InfosetData& d, const std::vector<uint8_t>& mask) const {
    return regret_matched(d.regret, mask);
}
```

and change the Kuhn helper so both trainers share the enumeration (the private `kuhn_ev_p0` takes the lookup function; behavior of the existing `kuhn_exact_value(t)` is unchanged):

```cpp
namespace {
double kuhn_ev_p0(const std::function<std::vector<double>(uint64_t)>& avg,
                  const KuhnState& s) {
    if (s.is_terminal()) return s.utility(0);
    std::vector<double> sigma = avg(s.infoset_key());
    if (sigma.empty()) sigma = {0.5, 0.5};
    double ev = 0.0;
    for (int a = 0; a < 2; ++a) {
        if (sigma[a] <= 0.0) continue;
        KuhnState child = s;
        child.apply(a);
        ev += sigma[a] * kuhn_ev_p0(avg, child);
    }
    return ev;
}
}  // namespace

double kuhn_exact_value_lookup(
        const std::function<std::vector<double>(uint64_t)>& avg) {
    double total = 0.0;
    for (int c0 = 0; c0 < 3; ++c0)
        for (int c1 = 0; c1 < 3; ++c1)
            if (c0 != c1) total += kuhn_ev_p0(avg, KuhnState(c0, c1));
    return total / 6.0;
}

double kuhn_exact_value(const MCCFRTrainer& t) {
    return kuhn_exact_value_lookup(
        [&](uint64_t k) { return t.average_strategy(k); });
}
```

- [ ] **Step 4: Implement the trainer**

Create `sixmax/src/blueprint/trainer.h`:

```cpp
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>
#include "blueprint/game.h"
#include "blueprint/mccfr.h"

namespace sixmax {

struct TrainerConfig {
    int num_threads = 1;  // <=0 means hardware_concurrency
    uint64_t seed = 1;
};

using GameFactory = std::function<std::unique_ptr<Game>()>;

// Multithreaded external-sampling MCCFR with linear weighting. Identical
// update rules to MCCFRTrainer; the table is sharded under mutexes and the
// linear-CFR weight comes from one atomic global iteration counter (the
// counter is cumulative across train() calls and resumes). Each worker
// thread owns its own Game instance from the factory — EngineGame mutates
// per-hand state (button rotation) and must never be shared across threads.
class BlueprintTrainer {
public:
    BlueprintTrainer(GameFactory factory, TrainerConfig cfg)
        : factory_(std::move(factory)), cfg_(cfg) {}
    void train(uint64_t iterations);
    uint64_t iterations() const {
        return iter_.load(std::memory_order_relaxed);
    }
    size_t num_infosets() const;
    std::vector<double> average_strategy(uint64_t key) const;
    std::vector<uint64_t> keys() const;
    // Checkpoint seam: merged copies of the sharded table.
    std::unordered_map<uint64_t, InfosetData> export_table() const;
    void import_table(std::unordered_map<uint64_t, InfosetData> table,
                      uint64_t iterations);

private:
    static constexpr int kShards = 64;
    struct Shard {
        mutable std::mutex mu;
        std::unordered_map<uint64_t, InfosetData> map;
    };
    int shard_of(uint64_t key) const {
        return (int)((key * 0x9E3779B97F4A7C15ull) >> 58);
    }
    double traverse(GameState& s, int traverser, double weight,
                    std::mt19937_64& rng, int num_actions);

    GameFactory factory_;
    TrainerConfig cfg_;
    std::array<Shard, kShards> shards_;
    std::atomic<uint64_t> iter_{0};
};

}  // namespace sixmax
```

Create `sixmax/src/blueprint/trainer.cpp`:

```cpp
#include "blueprint/trainer.h"
#include <thread>

namespace sixmax {

void BlueprintTrainer::train(uint64_t iterations) {
    const uint64_t target = iter_.load() + iterations;
    int T = cfg_.num_threads > 0 ? cfg_.num_threads
                                 : (int)std::thread::hardware_concurrency();
    if (T < 1) T = 1;
    auto worker = [&](int tid) {
        auto game = factory_();
        std::mt19937_64 rng(cfg_.seed * 0x9E3779B97F4A7C15ull +
                            (uint64_t)tid + 1);
        const int n = game->num_actions();
        for (;;) {
            uint64_t t = iter_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (t > target) {
                iter_.fetch_sub(1, std::memory_order_relaxed);
                break;  // each thread over-grabs at most once
            }
            const double w = (double)t;  // linear CFR weight
            for (int p = 0; p < game->num_players(); ++p) {
                auto s = game->new_hand(rng);
                traverse(*s, p, w, rng, n);
            }
        }
    };
    if (T == 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int i = 0; i < T; ++i) threads.emplace_back(worker, i);
    for (auto& th : threads) th.join();
}

double BlueprintTrainer::traverse(GameState& s, int traverser, double w,
                                  std::mt19937_64& rng, int n) {
    if (s.is_terminal()) return s.utility(traverser);
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    const uint64_t key = s.infoset_key();
    Shard& sh = shards_[shard_of(key)];
    std::vector<double> sigma;
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        if (d.regret.empty()) {
            d.regret.assign(n, 0.0);
            d.strategy_sum.assign(n, 0.0);
        }
        sigma = regret_matched(d.regret, mask);
    }
    if (s.current_player() == traverser) {
        std::vector<double> u(n, 0.0);
        double ev = 0.0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            auto child = s.clone();
            child->apply(a);
            u[a] = traverse(*child, traverser, w, rng, n);
            ev += sigma[a] * u[a];
        }
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.regret[a] += w * (u[a] - ev);
        return ev;
    }
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.strategy_sum[a] += w * sigma[a];
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng), acc = 0.0;
    int chosen = -1;
    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        acc += sigma[a];
        chosen = a;
        if (r <= acc) break;
    }
    s.apply(chosen);
    return traverse(s, traverser, w, rng, n);
}

size_t BlueprintTrainer::num_infosets() const {
    size_t total = 0;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        total += sh.map.size();
    }
    return total;
}

std::vector<double> BlueprintTrainer::average_strategy(uint64_t key) const {
    const Shard& sh = shards_[shard_of(key)];
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(key);
    if (it == sh.map.end()) return {};
    const auto& ss = it->second.strategy_sum;
    double total = 0.0;
    for (double v : ss) total += v;
    if (total <= 0.0) return {};
    std::vector<double> out(ss.size());
    for (size_t a = 0; a < ss.size(); ++a) out[a] = ss[a] / total;
    return out;
}

std::vector<uint64_t> BlueprintTrainer::keys() const {
    std::vector<uint64_t> out;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        for (const auto& [k, v] : sh.map) out.push_back(k);
    }
    return out;
}

std::unordered_map<uint64_t, InfosetData> BlueprintTrainer::export_table()
        const {
    std::unordered_map<uint64_t, InfosetData> out;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        out.insert(sh.map.begin(), sh.map.end());
    }
    return out;
}

void BlueprintTrainer::import_table(
        std::unordered_map<uint64_t, InfosetData> table, uint64_t iterations) {
    for (auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.map.clear();
    }
    for (auto& [k, v] : table)
        shards_[shard_of(k)].map.emplace(k, std::move(v));
    iter_.store(iterations, std::memory_order_relaxed);
}

}  // namespace sixmax
```

- [ ] **Step 5: Build entry + bindings**

In `sixmax/BUCK`, add `"src/blueprint/trainer.cpp"` to `srcs`.

In `bindings.cpp`, add `#include "blueprint/trainer.h"` and `#include "blueprint/kuhn.h"` is already present; append:

```cpp
    // --- Multithreaded blueprint trainer (Phase 1b Task 3) ---
    py::class_<sixmax::BlueprintTrainer>(m, "BlueprintTrainer")
        .def(py::init([](const sixmax::EngineConfig& cfg,
                         const sixmax::ActionVocab* vocab,
                         const sixmax::Abstraction* abstraction,
                         int num_threads, uint64_t seed) {
                 sixmax::GameFactory f = [cfg, vocab, abstraction]() {
                     return std::make_unique<sixmax::EngineGame>(cfg, vocab,
                                                                 abstraction);
                 };
                 return new sixmax::BlueprintTrainer(
                     std::move(f),
                     sixmax::TrainerConfig{num_threads, seed});
             }),
             py::arg("cfg"), py::arg("vocab"), py::arg("abstraction"),
             py::arg("num_threads") = 1, py::arg("seed") = 1,
             py::keep_alive<1, 3>(),   // trainer's factory holds vocab*
             py::keep_alive<1, 4>())   // trainer's factory holds abstraction*
        .def_static("kuhn", [](int num_threads, uint64_t seed) {
            sixmax::GameFactory f = []() {
                return std::make_unique<sixmax::KuhnGame>();
            };
            return new sixmax::BlueprintTrainer(
                std::move(f), sixmax::TrainerConfig{num_threads, seed});
        }, py::arg("num_threads") = 1, py::arg("seed") = 1)
        .def("train", &sixmax::BlueprintTrainer::train, py::arg("iterations"),
             py::call_guard<py::gil_scoped_release>())
        .def("iterations", &sixmax::BlueprintTrainer::iterations)
        .def("num_infosets", &sixmax::BlueprintTrainer::num_infosets)
        .def("average_strategy", &sixmax::BlueprintTrainer::average_strategy,
             py::arg("key"))
        .def("keys", &sixmax::BlueprintTrainer::keys);
    m.def("blueprint_kuhn_value", [](const sixmax::BlueprintTrainer& t) {
        return sixmax::kuhn_exact_value_lookup(
            [&](uint64_t k) { return t.average_strategy(k); });
    });
```

- [ ] **Step 6: Build and test**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_blueprint_trainer.py -v`
Expected: 5 passed. The two Kuhn gates take a few seconds each. If a Kuhn gate fails, the trainer's update rules diverge from `MCCFRTrainer` — fix the trainer; **never** widen the 0.01 tolerance or reduce the 200k iterations.
Run: `uv run pytest tests/sixmax/ -q` — all pass (existing Kuhn tests on `MCCFRTrainer` prove the refactor preserved behavior).

- [ ] **Step 7: Commit**

```bash
git add sixmax/src/blueprint/trainer.h sixmax/src/blueprint/trainer.cpp sixmax/src/blueprint/mccfr.h sixmax/src/blueprint/mccfr.cpp sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_blueprint_trainer.py
git commit -m "feat(sixmax): multithreaded BlueprintTrainer passes the Kuhn -1/18 gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Blueprint checkpoints + BlueprintStrategy

**Files:**
- Create: `sixmax/src/blueprint/checkpoint.h`, `sixmax/src/blueprint/checkpoint.cpp`
- Modify: `sixmax/BUCK` (add `src/blueprint/checkpoint.cpp`), `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_blueprint_checkpoint.py`

**Interfaces:**
- Consumes: `BlueprintTrainer::export_table/import_table/iterations` (Task 3), `Abstraction` (Task 1), `EngineGameState::abstract_key` (Task 2), `ActionVocab::hash()`.
- Produces (used by Tasks 5–6 and Phase 1c):

```cpp
struct BlueprintMeta {
    uint64_t vocab_hash;
    int num_players;
    double starting_stack;
    int action_dim;
};
void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer);
struct LoadedBlueprint {
    BlueprintMeta meta;
    AbstractionConfig abs_cfg;
    std::array<std::vector<double>, 3> edges;
    uint64_t iterations;
    std::unordered_map<uint64_t, InfosetData> table;
};
// Throws std::runtime_error on bad magic, truncation, or vocab-hash
// mismatch (artifact contract: loaders refuse mismatches).
LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash);
class BlueprintStrategy {
    static BlueprintStrategy load(const std::string& path,
                                  const ActionVocab& vocab);
    std::vector<double> probs(uint64_t key) const;             // {} if unseen
    std::vector<double> probs_for(const EngineGameState& s) const;
    uint64_t iterations() const;  size_t num_infosets() const;
    int num_players() const;      const Abstraction& abstraction() const;
};
```

- Python: `trainer.save(path, vocab, cfg, abstraction)`; `sixmax.load_abstraction(path) -> Abstraction`; `sixmax.resume_blueprint(path, cfg, vocab, abstraction, num_threads, seed) -> BlueprintTrainer`; `sixmax.BlueprintStrategy` with `load`, `probs`, `probs_for`, `iterations`, `num_infosets`, `num_players`, `abstraction`.

**Binary format** (native-endian; artifacts are same-machine or same-arch cloud — documented in the header):

```
magic "SIXBP001"                              8 bytes
vocab_hash u64 | num_players i32 | action_dim i32 | starting_stack f64
abs config: flop,turn,river,rollouts,qsamples i32 ×5 | seed u64
edges ×3: count u64, then count f64
iterations u64 | num_infosets u64
entries ×num_infosets: key u64, regret f64×action_dim, strategy_sum f64×action_dim
```

Writes go to `path + ".tmp"` then `std::rename` — atomic on POSIX, never a torn artifact (same pattern as neural_cfr best-checkpoint writes).

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_blueprint_checkpoint.py`:

```python
"""Checkpoint round-trip, vocab-hash refusal, resume, and BlueprintStrategy."""
import importlib.util
import os

import pytest
import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _vocab_config():
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VC = _vocab_config()
TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
VOCAB = VC.load_vocab(TOML, "blueprint")
ABS = sixmax.Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5,
                         equity_rollouts=40, quantile_samples=300, seed=42)
CFG = sixmax.EngineConfig(num_players=2)


def _trained(iters=120, seed=13):
    t = sixmax.BlueprintTrainer(CFG, VOCAB, ABS, num_threads=1, seed=seed)
    t.train(iters)
    return t


def test_round_trip_preserves_everything(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    assert not os.path.exists(path + ".tmp")     # atomic write cleaned up

    abs2 = sixmax.load_abstraction(path)
    assert abs2.hash() == ABS.hash()             # edges stored, not rebuilt

    t2 = sixmax.resume_blueprint(path, CFG, VOCAB, abs2, num_threads=1, seed=13)
    assert t2.iterations() == t.iterations()
    assert sorted(t2.keys()) == sorted(t.keys())
    for k in t.keys():
        assert t2.average_strategy(k) == t.average_strategy(k)


def test_resume_continues_the_global_counter(tmp_path):
    t = _trained(iters=100)
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    t2 = sixmax.resume_blueprint(path, CFG, VOCAB, ABS, num_threads=1, seed=13)
    t2.train(50)
    assert t2.iterations() == 150


def test_wrong_vocab_is_refused(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    other = sixmax.ActionVocab([
        sixmax.AbstractAction(sixmax.ActionType.Fold, 0.0, sixmax.SizeUnit.BB),
        sixmax.AbstractAction(sixmax.ActionType.Check, 0.0, sixmax.SizeUnit.BB),
        sixmax.AbstractAction(sixmax.ActionType.Call, 0.0, sixmax.SizeUnit.BB),
    ])
    with pytest.raises(RuntimeError, match="vocab"):
        sixmax.BlueprintStrategy.load(path, other)
    with pytest.raises(RuntimeError, match="vocab"):
        sixmax.resume_blueprint(path, CFG, other, ABS, num_threads=1, seed=1)


def test_strategy_matches_trainer_and_keys_states(tmp_path):
    t = _trained()
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    strat = sixmax.BlueprintStrategy.load(path, VOCAB)
    assert strat.iterations() == t.iterations()
    assert strat.num_players() == 2
    seen = 0
    for k in t.keys():
        expect = t.average_strategy(k)
        if expect:
            assert strat.probs(k) == expect
            seen += 1
    assert seen > 0
    # probs_for keys a fresh state through the strategy's OWN abstraction
    s = sixmax.EngineGameState(CFG, 0, list(range(52)), VOCAB, [])
    assert strat.probs_for(s) == strat.probs(s.abstract_key(strat.abstraction()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sixmax/test_blueprint_checkpoint.py -v`
Expected: FAIL — `AttributeError` on `save` / `load_abstraction`.

- [ ] **Step 3: Implement checkpoint.h/.cpp**

Create `sixmax/src/blueprint/checkpoint.h`:

```cpp
#pragma once
#include <array>
#include <string>
#include <unordered_map>
#include "abstraction/abstraction.h"
#include "blueprint/engine_game.h"
#include "blueprint/trainer.h"
#include "vocab/vocab.h"

namespace sixmax {

// Binary blueprint artifact. Self-describing per the artifact contract:
// embeds the vocab hash and the full abstraction (config + quantile edges),
// so loaders can refuse mismatches and reconstruct the exact keyer without
// re-sampling. Native-endian; artifacts move between same-arch machines.
struct BlueprintMeta {
    uint64_t vocab_hash;
    int num_players;
    double starting_stack;
    int action_dim;
};

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer);

struct LoadedBlueprint {
    BlueprintMeta meta;
    AbstractionConfig abs_cfg;
    std::array<std::vector<double>, 3> edges;
    uint64_t iterations;
    std::unordered_map<uint64_t, InfosetData> table;
};

// Throws std::runtime_error on bad magic, truncation, or vocab-hash
// mismatch against expected_vocab_hash.
LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash);

// Read-only average-strategy view of an artifact, for eval and deployment.
// Owns its Abstraction (reconstructed from the stored edges).
class BlueprintStrategy {
public:
    static BlueprintStrategy load(const std::string& path,
                                  const ActionVocab& vocab);
    std::vector<double> probs(uint64_t key) const;  // {} if unseen/empty
    // Keys the state through THIS strategy's abstraction — two strategies
    // with different abstractions can evaluate the same public state.
    std::vector<double> probs_for(const EngineGameState& s) const;
    uint64_t iterations() const { return iterations_; }
    size_t num_infosets() const { return probs_.size(); }
    int num_players() const { return num_players_; }
    const Abstraction& abstraction() const { return abs_; }

private:
    BlueprintStrategy(Abstraction abs, uint64_t iters, int num_players)
        : abs_(std::move(abs)), iterations_(iters), num_players_(num_players) {}
    Abstraction abs_;
    uint64_t iterations_;
    int num_players_;
    std::unordered_map<uint64_t, std::vector<double>> probs_;
};

}  // namespace sixmax
```

Create `sixmax/src/blueprint/checkpoint.cpp`:

```cpp
#include "blueprint/checkpoint.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sixmax {

namespace {

constexpr char kMagic[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '1'};

template <typename T>
void put(std::ofstream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
T get(std::ifstream& i) {
    T v;
    i.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
    return v;
}

void put_doubles(std::ofstream& o, const std::vector<double>& v) {
    put<uint64_t>(o, v.size());
    o.write(reinterpret_cast<const char*>(v.data()),
            (std::streamsize)(v.size() * sizeof(double)));
}

std::vector<double> get_doubles(std::ifstream& i) {
    auto n = get<uint64_t>(i);
    std::vector<double> v(n);
    i.read(reinterpret_cast<char*>(v.data()),
           (std::streamsize)(n * sizeof(double)));
    if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
    return v;
}

}  // namespace

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open " + tmp);
        o.write(kMagic, 8);
        put<uint64_t>(o, meta.vocab_hash);
        put<int32_t>(o, meta.num_players);
        put<int32_t>(o, meta.action_dim);
        put<double>(o, meta.starting_stack);
        const AbstractionConfig& c = abs.config();
        put<int32_t>(o, c.flop_buckets);
        put<int32_t>(o, c.turn_buckets);
        put<int32_t>(o, c.river_buckets);
        put<int32_t>(o, c.equity_rollouts);
        put<int32_t>(o, c.quantile_samples);
        put<uint64_t>(o, c.seed);
        for (const auto& e : abs.edges()) put_doubles(o, e);
        auto table = trainer.export_table();
        put<uint64_t>(o, trainer.iterations());
        put<uint64_t>(o, table.size());
        for (const auto& [k, d] : table) {
            put<uint64_t>(o, k);
            o.write(reinterpret_cast<const char*>(d.regret.data()),
                    (std::streamsize)(meta.action_dim * sizeof(double)));
            o.write(reinterpret_cast<const char*>(d.strategy_sum.data()),
                    (std::streamsize)(meta.action_dim * sizeof(double)));
        }
        if (!o) throw std::runtime_error("write failed: " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("atomic rename failed: " + path);
}

LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open " + path);
    char magic[8];
    i.read(magic, 8);
    if (!i || std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("not a blueprint checkpoint: " + path);
    LoadedBlueprint out;
    out.meta.vocab_hash = get<uint64_t>(i);
    if (out.meta.vocab_hash != expected_vocab_hash)
        throw std::runtime_error(
            "vocab hash mismatch: checkpoint was trained with a different "
            "action vocabulary");
    out.meta.num_players = get<int32_t>(i);
    out.meta.action_dim = get<int32_t>(i);
    out.meta.starting_stack = get<double>(i);
    out.abs_cfg.flop_buckets = get<int32_t>(i);
    out.abs_cfg.turn_buckets = get<int32_t>(i);
    out.abs_cfg.river_buckets = get<int32_t>(i);
    out.abs_cfg.equity_rollouts = get<int32_t>(i);
    out.abs_cfg.quantile_samples = get<int32_t>(i);
    out.abs_cfg.seed = get<uint64_t>(i);
    for (auto& e : out.edges) e = get_doubles(i);
    out.iterations = get<uint64_t>(i);
    const auto n_infosets = get<uint64_t>(i);
    const int dim = out.meta.action_dim;
    out.table.reserve(n_infosets);
    for (uint64_t k = 0; k < n_infosets; ++k) {
        const uint64_t key = get<uint64_t>(i);
        InfosetData d;
        d.regret.resize(dim);
        d.strategy_sum.resize(dim);
        i.read(reinterpret_cast<char*>(d.regret.data()),
               (std::streamsize)(dim * sizeof(double)));
        i.read(reinterpret_cast<char*>(d.strategy_sum.data()),
               (std::streamsize)(dim * sizeof(double)));
        if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
        out.table.emplace(key, std::move(d));
    }
    return out;
}

BlueprintStrategy BlueprintStrategy::load(const std::string& path,
                                          const ActionVocab& vocab) {
    auto loaded = load_blueprint(path, vocab.hash());
    BlueprintStrategy s(Abstraction(loaded.abs_cfg, std::move(loaded.edges)),
                        loaded.iterations, loaded.meta.num_players);
    for (const auto& [k, d] : loaded.table) {
        double total = 0.0;
        for (double v : d.strategy_sum) total += v;
        if (total <= 0.0) continue;
        std::vector<double> p(d.strategy_sum.size());
        for (size_t a = 0; a < p.size(); ++a) p[a] = d.strategy_sum[a] / total;
        s.probs_.emplace(k, std::move(p));
    }
    return s;
}

std::vector<double> BlueprintStrategy::probs(uint64_t key) const {
    auto it = probs_.find(key);
    return it == probs_.end() ? std::vector<double>{} : it->second;
}

std::vector<double> BlueprintStrategy::probs_for(
        const EngineGameState& s) const {
    return probs(s.abstract_key(abs_));
}

}  // namespace sixmax
```

- [ ] **Step 4: Build entry + bindings**

In `sixmax/BUCK`, add `"src/blueprint/checkpoint.cpp"` to `srcs`.

In `bindings.cpp`, add `#include "blueprint/checkpoint.h"` and append:

```cpp
    // --- Blueprint checkpoints + strategy (Phase 1b Task 4) ---
    m.def("load_abstraction", [](const std::string& path) {
        // Peek only the abstraction block: reuse the loader with the
        // stored hash so it cannot mismatch, then rebuild from edges.
        auto loaded = sixmax::load_blueprint(
            path, [&] {
                std::ifstream i(path, std::ios::binary);
                i.seekg(8);
                uint64_t h;
                i.read(reinterpret_cast<char*>(&h), sizeof h);
                return h;
            }());
        return sixmax::Abstraction(loaded.abs_cfg, std::move(loaded.edges));
    }, py::arg("path"));
    m.def("resume_blueprint",
          [](const std::string& path, const sixmax::EngineConfig& cfg,
             const sixmax::ActionVocab* vocab,
             const sixmax::Abstraction* abstraction, int num_threads,
             uint64_t seed) {
              auto loaded = sixmax::load_blueprint(path, vocab->hash());
              sixmax::GameFactory f = [cfg, vocab, abstraction]() {
                  return std::make_unique<sixmax::EngineGame>(cfg, vocab,
                                                              abstraction);
              };
              auto* t = new sixmax::BlueprintTrainer(
                  std::move(f), sixmax::TrainerConfig{num_threads, seed});
              t->import_table(std::move(loaded.table), loaded.iterations);
              return t;
          },
          py::arg("path"), py::arg("cfg"), py::arg("vocab"),
          py::arg("abstraction"), py::arg("num_threads") = 1,
          py::arg("seed") = 1,
          py::keep_alive<0, 3>(),   // returned trainer holds vocab*
          py::keep_alive<0, 4>());  // returned trainer holds abstraction*
    py::class_<sixmax::BlueprintStrategy>(m, "BlueprintStrategy")
        .def_static("load", &sixmax::BlueprintStrategy::load,
                    py::arg("path"), py::arg("vocab"))
        .def("probs", &sixmax::BlueprintStrategy::probs, py::arg("key"))
        .def("probs_for", &sixmax::BlueprintStrategy::probs_for)
        .def("iterations", &sixmax::BlueprintStrategy::iterations)
        .def("num_infosets", &sixmax::BlueprintStrategy::num_infosets)
        .def("num_players", &sixmax::BlueprintStrategy::num_players)
        .def("abstraction", &sixmax::BlueprintStrategy::abstraction,
             py::return_value_policy::reference_internal);
```

and on the existing `BlueprintTrainer` class binding (from Task 3), add:

```cpp
        .def("save",
             [](const sixmax::BlueprintTrainer& t, const std::string& path,
                const sixmax::ActionVocab* vocab,
                const sixmax::EngineConfig& cfg,
                const sixmax::Abstraction* abstraction) {
                 sixmax::save_blueprint(
                     path,
                     sixmax::BlueprintMeta{vocab->hash(), cfg.num_players,
                                           cfg.starting_stack, vocab->size()},
                     *abstraction, t);
             },
             py::arg("path"), py::arg("vocab"), py::arg("cfg"),
             py::arg("abstraction"))
```

Note for the implementer: the `load_abstraction` lambda needs `#include <fstream>` in bindings.cpp.

- [ ] **Step 5: Build, test, commit**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_blueprint_checkpoint.py -v`
Expected: 4 passed.
Run: `uv run pytest tests/sixmax/ -q` — all pass.

```bash
git add sixmax/src/blueprint/checkpoint.h sixmax/src/blueprint/checkpoint.cpp sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_blueprint_checkpoint.py
git commit -m "feat(sixmax): self-describing blueprint checkpoints + BlueprintStrategy loader

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Config sections + train_sixmax.py entry point + setup hardening

**Files:**
- Modify: `sixmax/configs/default.toml`, `.gitignore`, `scripts/setup_dev.sh`
- Create: `scripts/train_sixmax.py`
- Test: `tests/sixmax/test_train_sixmax.py`

**Interfaces:**
- Consumes: everything bound in Tasks 1–4 (`Abstraction`, `BlueprintTrainer`, `trainer.save`, `resume_blueprint`, `load_abstraction`); `vocab_config.load_vocab`.
- Produces: `scripts/train_sixmax.py` with `resolve_config(args, repo_root) -> dict` and `write_config_snapshot(cfg, path)` importable for tests; CLI `--config --iterations --num-threads --num-players --checkpoint --checkpoint-interval --resume --seed` (Task 6 adds `--selection-*`). Config precedence CLI > TOML > builtin, mirroring `scripts/train_neural.py`.

- [ ] **Step 1: Extend the config and gitignore**

Append to `sixmax/configs/default.toml`:

```toml

[abstraction]
flop_buckets = 50
turn_buckets = 50
river_buckets = 20
equity_rollouts = 100
quantile_samples = 10000
seed = 20260719

[train.blueprint]
num_players = 6
starting_stack = 100.0
iterations = 100000
num_threads = 0          # 0 = hardware_concurrency
checkpoint_interval = 0  # 0 = save at the end only
checkpoint = "sixmax/checkpoints/blueprint.bin"
seed = 7
selection_enabled = false
selection_hands = 2000
```

Append to `.gitignore` (next to the existing checkpoint entries around line 183):

```gitignore
# Sixmax blueprint checkpoints (large binaries)
sixmax/checkpoints/
```

- [ ] **Step 2: Write the failing tests**

Create `tests/sixmax/test_train_sixmax.py`:

```python
"""Config resolution + end-to-end smoke for scripts/train_sixmax.py."""
import importlib.util
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_sixmax", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """argparse.Namespace stand-in; unset flags are None."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __getattr__(self, _):
        return None


def test_resolve_config_precedence(tmp_path):
    mod = _load_script()
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        "[abstraction]\nflop_buckets = 8\n"
        "[train.blueprint]\niterations = 500\nnum_players = 3\n")
    cfg = mod.resolve_config(_Args(config=str(toml), iterations=250), _ROOT)
    assert cfg["iterations"] == 250          # CLI beats TOML
    assert cfg["num_players"] == 3           # TOML beats builtin
    assert cfg["flop_buckets"] == 8          # [abstraction] section merged
    assert cfg["turn_buckets"] == 50         # builtin default survives


def test_resolve_config_rejects_unknown_keys(tmp_path):
    mod = _load_script()
    toml = tmp_path / "bad.toml"
    toml.write_text("[train.blueprint]\nnot_a_key = 1\n")
    try:
        mod.resolve_config(_Args(config=str(toml)), _ROOT)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_end_to_end_smoke(tmp_path):
    """Tiny full run: builds abstraction, trains, saves a loadable artifact."""
    toml = tmp_path / "smoke.toml"
    toml.write_text(
        "[abstraction]\n"
        "flop_buckets = 6\nturn_buckets = 6\nriver_buckets = 4\n"
        "equity_rollouts = 20\nquantile_samples = 150\nseed = 5\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 40\nnum_threads = 1\nseed = 3\n")
    ckpt = tmp_path / "bp.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    assert ckpt.exists()
    assert (tmp_path / "bp.bin.config.toml").exists()   # effective-config snapshot
    import sixmax  # conftest already force-loaded the extension
    abs_ = sixmax.load_abstraction(str(ckpt))
    assert abs_.num_buckets(1) == 6
```

Run: `uv run pytest tests/sixmax/test_train_sixmax.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Write the entry point**

Create `scripts/train_sixmax.py`:

```python
#!/usr/bin/env python3
"""Six-max blueprint training launcher.

Builds //sixmax:sixmax via Buck2, force-loads the extension (the repo-root
`sixmax/` directory would otherwise shadow the .so as a namespace package),
then drives sixmax.BlueprintTrainer with chunked checkpointing.

Config precedence: CLI flag > TOML ([abstraction] + [train.blueprint]) >
builtin defaults. The effective config is snapshotted to
<checkpoint>.config.toml on every save (neural_cfr pattern).
"""
import argparse
import importlib.util
import os
import subprocess
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_and_get_so_dir(repo_root: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            rel_so = line.split()[-1]
            return os.path.join(repo_root, os.path.dirname(rel_so))
    raise RuntimeError(f"Could not locate sixmax.so in buck2 output.\n"
                       f"stdout: {result.stdout}\nstderr: {result.stderr}")


def _force_load_sixmax(repo_root: str):
    """Load the .so and register it as sys.modules['sixmax'] (the repo-root
    sixmax/ directory is a namespace package that would win otherwise)."""
    so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_vocab(repo_root: str, toml_path: str):
    vc_path = os.path.join(repo_root, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", vc_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(toml_path, "blueprint")


BUILTIN_DEFAULTS = {
    # [abstraction]
    "flop_buckets": 50, "turn_buckets": 50, "river_buckets": 20,
    "equity_rollouts": 100, "quantile_samples": 10000,
    "abstraction_seed": 20260719,
    # [train.blueprint]
    "num_players": 6, "starting_stack": 100.0,
    "iterations": 100_000, "num_threads": 0,
    "checkpoint_interval": 0,
    "checkpoint": "sixmax/checkpoints/blueprint.bin",
    "seed": 7,
    "selection_enabled": False, "selection_hands": 2000,
}

# TOML key -> flat config key (only where they differ)
_TOML_RENAME = {"seed": {"abstraction": "abstraction_seed",
                         "train.blueprint": "seed"}}


def resolve_config(args, repo_root: str) -> dict:
    """Merge config sources. Precedence: CLI flag > TOML > builtin."""
    cfg = dict(BUILTIN_DEFAULTS)
    path = args.config or os.path.join(repo_root, "sixmax", "configs",
                                       "default.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        sections = {"abstraction": data.get("abstraction", {}),
                    "train.blueprint": data.get("train", {}).get("blueprint", {})}
        for section, block in sections.items():
            for key, value in block.items():
                flat = _TOML_RENAME.get(key, {}).get(section, key)
                if flat not in BUILTIN_DEFAULTS:
                    raise KeyError(f"Unknown config key [{section}].{key} in {path}")
                cfg[flat] = value
    for key in BUILTIN_DEFAULTS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            cfg[key] = cli_value
    cfg["config_path"] = path
    return cfg


def write_config_snapshot(cfg: dict, checkpoint_path: str) -> None:
    import json
    lines = ["# Effective config — written by train_sixmax.py", "[resolved]"]
    for key, value in sorted(cfg.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = repr(value)
        lines.append(f"{key} = {rendered}")
    with open(checkpoint_path + ".config.toml", "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    repo_root = _get_repo_root()
    sixmax = _force_load_sixmax(repo_root)

    parser = argparse.ArgumentParser(description="Six-max blueprint trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None,
                        dest="num_threads")
    parser.add_argument("--num-players", type=int, default=None,
                        dest="num_players")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        dest="checkpoint_interval")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = resolve_config(args, repo_root)
    print("Effective config: "
          + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))
    vocab = _load_vocab(repo_root, cfg["config_path"])
    engine_cfg = sixmax.EngineConfig(num_players=cfg["num_players"],
                                     starting_stack=cfg["starting_stack"])
    os.makedirs(os.path.dirname(os.path.abspath(cfg["checkpoint"])),
                exist_ok=True)

    if args.resume:
        print(f"Resuming from {args.resume}")
        abstraction = sixmax.load_abstraction(args.resume)
        trainer = sixmax.resume_blueprint(
            args.resume, engine_cfg, vocab, abstraction,
            num_threads=cfg["num_threads"], seed=cfg["seed"])
    else:
        print("Building abstraction (quantile edges) …")
        abstraction = sixmax.Abstraction(
            flop_buckets=cfg["flop_buckets"],
            turn_buckets=cfg["turn_buckets"],
            river_buckets=cfg["river_buckets"],
            equity_rollouts=cfg["equity_rollouts"],
            quantile_samples=cfg["quantile_samples"],
            seed=cfg["abstraction_seed"])
        trainer = sixmax.BlueprintTrainer(
            engine_cfg, vocab, abstraction,
            num_threads=cfg["num_threads"], seed=cfg["seed"])

    ckpt_interval = cfg["checkpoint_interval"] or cfg["iterations"]
    completed = 0
    print(f"Running {cfg['iterations']:,} iterations …")
    while completed < cfg["iterations"]:
        chunk = min(ckpt_interval, cfg["iterations"] - completed)
        trainer.train(chunk)
        completed += chunk
        trainer.save(cfg["checkpoint"], vocab, engine_cfg, abstraction)
        write_config_snapshot(cfg, cfg["checkpoint"])
        print(f"[{completed:,}/{cfg['iterations']:,}] "
              f"{trainer.num_infosets():,} infosets — saved {cfg['checkpoint']}")
    print(f"Done: {trainer.iterations():,} total iterations, "
          f"{trainer.num_infosets():,} infosets.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: setup_dev.sh symlink self-heal (ledger-promised)**

In `scripts/setup_dev.sh`, replace the third_party loop (step 2) with a version that also repairs dead symlinks (the pybind11 include symlink rotted once already — see `.mex/context/setup.md` Common Issues):

```zsh
# 2. third_party symlinks (worktrees don't inherit them; also self-heal
#    dead links — e.g. pybind11/include pointing at a rebuilt venv).
MAIN_REPO=$(git rev-parse --path-format=absolute --git-common-dir)/..
for d in libtorch pybind11 indicators; do
  if [ -L "third_party/$d" ] && [ ! -e "third_party/$d" ]; then
    echo "repairing dead symlink third_party/$d"
    rm "third_party/$d"
  fi
  [ -e "third_party/$d" ] || ln -s "$MAIN_REPO/third_party/$d" "third_party/$d"
done
PB_INC="third_party/pybind11/include"
if [ ! -e "$PB_INC/pybind11/pybind11.h" ]; then
  if [ -L "$PB_INC" ]; then rm "$PB_INC"; fi
  TARGET=$(uv run python -c "import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), 'include'))")
  [ -d "$TARGET" ] || fail "pybind11 include dir not found (uv sync first?)"
  ln -sfn "$TARGET" "$PB_INC"
  echo "repointed $PB_INC -> $TARGET"
fi
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/sixmax/test_train_sixmax.py -v`
Expected: 3 passed (the smoke test takes ~1–2 minutes: cached buck build + tiny abstraction + 40 iterations).
Run: `bash scripts/setup_dev.sh` — expected: ends with `SETUP OK` (verifies the symlink logic on a healthy checkout without breaking anything).
Run: `uv run pytest tests/sixmax/ -q` — all pass.

- [ ] **Step 6: Commit**

```bash
git add sixmax/configs/default.toml .gitignore scripts/train_sixmax.py scripts/setup_dev.sh tests/sixmax/test_train_sixmax.py
git commit -m "feat(sixmax): blueprint config + train_sixmax.py; setup_dev pybind11 self-heal

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Duplicate-deal A/B eval harness + best-checkpoint selection

**Files:**
- Create: `scripts/eval_sixmax.py`
- Modify: `scripts/train_sixmax.py` (selection hook + `--selection-*` flags)
- Test: `tests/sixmax/test_eval_sixmax.py`

**Interfaces:**
- Consumes: `BlueprintStrategy.load/probs_for`, `EngineGameState(..., vocab)` + `legal_mask/apply/payoffs`, `vocab_config.load_vocab`.
- Produces: `scripts/eval_sixmax.py` CLI `--a PATH --b PATH|uniform --hands N --seed S --config TOML`; stdout line `Blueprint A win rate: {bb100:+.2f} BB/100 ({n} hands)` (regex target for selection); importable `run_match(strat_a, strat_b, vocab, cfg, hands, seed) -> float` and `sample_action(probs, mask, rng) -> int`. Selection in `train_sixmax.py`: after each save, first save promotes to `best_checkpoint.bin` unconditionally; later saves run current-vs-best and replace on strictly positive BB/100 (atomic copy + JSON sidecar — the validated neural_cfr pattern).

**Match protocol (variance control):** for each of `hands` decks (seeded `random.Random`), and for each hero seat `0..n-1`, play the same deck with strategy A in the hero seat and B everywhere else; button = deck index mod n. BB/100 = `100 * total_hero_payoff / (hands * n)`. Duplicate seat rotation cancels deal luck; the whole run is deterministic for a fixed seed.

**Action sampling correctness:** the abstraction merges states with different legal masks, so stored probabilities can put mass on actions illegal in the current state. Always re-mask: `p[i] = probs[i] * mask[i]`, renormalize; if the masked sum is 0 (or the infoset is unseen), fall back to uniform over legal.

- [ ] **Step 1: Write the failing tests**

Create `tests/sixmax/test_eval_sixmax.py`:

```python
"""Duplicate-deal A/B harness: masking-safe sampling, determinism, and a
trained-blueprint-beats-uniform smoke. Selection helpers unit-tested."""
import importlib.util
import json
import os
import random

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load(os.path.join(_ROOT, "scripts", "eval_sixmax.py"), "eval_sixmax")
VC = _load(os.path.join(_ROOT, "sixmax", "vocab_config.py"), "vocab_config")
VOCAB = VC.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                      "blueprint")
ABS = sixmax.Abstraction(flop_buckets=6, turn_buckets=6, river_buckets=4,
                         equity_rollouts=20, quantile_samples=150, seed=5)
CFG = sixmax.EngineConfig(num_players=2)


def test_sample_action_remasks_and_falls_back():
    rng = random.Random(1)
    # stored probs put all mass on an illegal action -> uniform over legal
    a = EV.sample_action([1.0, 0.0, 0.0], [0, 1, 1], rng)
    assert a in (1, 2)
    # unseen infoset (empty probs) -> uniform over legal
    a = EV.sample_action([], [1, 0, 1], rng)
    assert a in (0, 2)
    # legal mass is respected
    a = EV.sample_action([0.0, 1.0, 0.0], [1, 1, 0], rng)
    assert a == 1


def test_uniform_selfplay_is_deterministic_and_sane():
    u = EV.UniformStrategy()
    r1 = EV.run_match(u, u, VOCAB, CFG, hands=60, seed=17)
    r2 = EV.run_match(u, u, VOCAB, CFG, hands=60, seed=17)
    assert r1 == r2                       # fully seeded -> reproducible
    assert abs(r1) < 400                  # sanity bound, not a strength claim


def test_trained_blueprint_beats_uniform(tmp_path):
    t = sixmax.BlueprintTrainer(CFG, VOCAB, ABS, num_threads=1, seed=21)
    t.train(3000)
    path = str(tmp_path / "bp.bin")
    t.save(path, VOCAB, CFG, ABS)
    strat = sixmax.BlueprintStrategy.load(path, VOCAB)
    bb100 = EV.run_match(strat, EV.UniformStrategy(), VOCAB, CFG,
                         hands=200, seed=33)
    # Trained HU blueprint vs uniform-random must be clearly positive. The
    # run is deterministic; if this fails, double the training iterations
    # once — if it still fails, STOP and escalate (do not weaken the bound).
    assert bb100 > 0


def test_selection_promote_and_replace(tmp_path):
    TR = _load(os.path.join(_ROOT, "scripts", "train_sixmax.py"), "train_sixmax")
    ckpt = tmp_path / "bp.bin"
    ckpt.write_bytes(b"v1")
    best = tmp_path / "best_checkpoint.bin"
    # first save: unconditional promote
    assert TR.update_best(None, str(ckpt), 100) is True
    assert best.read_bytes() == b"v1"
    # improvement: replace
    ckpt.write_bytes(b"v2")
    assert TR.update_best(5.0, str(ckpt), 200) is True
    assert best.read_bytes() == b"v2"
    side = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert side["iterations"] == 200
    # non-positive: keep
    ckpt.write_bytes(b"v3")
    assert TR.update_best(-1.0, str(ckpt), 300) is False
    assert best.read_bytes() == b"v2"


def test_parse_bb100():
    TR = _load(os.path.join(_ROOT, "scripts", "train_sixmax.py"), "train_sixmax")
    line = "Blueprint A win rate: +12.34 BB/100 (2400 hands)\n"
    assert TR.parse_bb100(line) == 12.34
    assert TR.parse_bb100("Blueprint A win rate: -3 BB/100 (10 hands)") == -3.0
    assert TR.parse_bb100("no match") is None
```

Run: `uv run pytest tests/sixmax/test_eval_sixmax.py -v`
Expected: FAIL — `eval_sixmax.py` does not exist.

- [ ] **Step 2: Write the eval harness**

Create `scripts/eval_sixmax.py`:

```python
#!/usr/bin/env python3
"""Duplicate-deal A/B evaluation for six-max blueprint checkpoints.

Protocol: for each seeded deck and each hero seat, play the SAME deck with
strategy A in the hero seat and B in every other seat (button = deck index
mod n). Seat rotation over identical decks cancels deal luck; a fixed seed
makes the whole run reproducible. Reports A's win rate in BB/100.

Usage:
    uv run python scripts/eval_sixmax.py --a CKPT [--b CKPT|uniform]
        [--hands 500] [--seed 1] [--config sixmax/configs/default.toml]
"""
import argparse
import importlib.util
import os
import random
import subprocess
import sys


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_and_get_so_dir(repo_root: str) -> str:
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run(
        [buck2, "build", "//sixmax:sixmax", "--show-output"],
        capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError("Buck2 build failed")
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            return os.path.join(repo_root, os.path.dirname(line.split()[-1]))
    raise RuntimeError("Could not locate sixmax.so in buck2 output")


def _force_load_sixmax(repo_root: str):
    so_path = os.path.join(_build_and_get_so_dir(repo_root), "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = mod
    spec.loader.exec_module(mod)
    return mod


# When run as a script we must load the extension before touching sixmax
# names; when imported by tests, conftest has already registered it.
if "sixmax" not in sys.modules:
    _force_load_sixmax(_get_repo_root())
import sixmax  # noqa: E402


class UniformStrategy:
    """Baseline: no knowledge; sample_action falls back to uniform-legal."""
    def probs_for(self, state):
        return []


def sample_action(probs, mask, rng):
    """Sample from stored probs re-masked to THIS state's legal actions.

    The abstraction merges states with different masks, so stored mass can
    sit on illegal actions; re-mask and renormalize, falling back to uniform
    over legal when nothing legal has mass (or the infoset is unseen)."""
    legal = [i for i, m in enumerate(mask) if m]
    weights = [probs[i] if i < len(probs) else 0.0 for i in legal]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(legal)
    r = rng.random() * total
    acc = 0.0
    for i, w in zip(legal, weights):
        acc += w
        if r <= acc:
            return i
    return legal[-1]


def _play_hand(deck, button, hero_seat, strat_a, strat_b, vocab, cfg, rng):
    state = sixmax.EngineGameState(cfg, button, deck, vocab, [])
    while not state.is_terminal():
        seat = state.current_player()
        strat = strat_a if seat == hero_seat else strat_b
        action = sample_action(strat.probs_for(state), state.legal_mask(), rng)
        state.apply(action)
    return state.utility(hero_seat)


def run_match(strat_a, strat_b, vocab, cfg, hands, seed):
    """A occupies each seat once per deck; returns A's BB/100."""
    rng = random.Random(seed)
    n = cfg.num_players
    total = 0.0
    for h in range(hands):
        deck = rng.sample(range(52), 2 * n + 5)
        button = h % n
        for hero_seat in range(n):
            total += _play_hand(deck, button, hero_seat, strat_a, strat_b,
                                vocab, cfg, rng)
    return 100.0 * total / (hands * n)


def main() -> None:
    repo_root = _get_repo_root()
    parser = argparse.ArgumentParser(description="Six-max blueprint A/B eval")
    parser.add_argument("--a", required=True, help="checkpoint for strategy A")
    parser.add_argument("--b", default="uniform",
                        help="checkpoint for strategy B, or 'uniform'")
    parser.add_argument("--hands", type=int, default=500,
                        help="decks; each is played once per seat")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str,
                        default=os.path.join(repo_root, "sixmax", "configs",
                                             "default.toml"))
    args = parser.parse_args()

    vc_path = os.path.join(repo_root, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", vc_path)
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    vocab = vc.load_vocab(args.config, "blueprint")

    strat_a = sixmax.BlueprintStrategy.load(args.a, vocab)
    cfg = sixmax.EngineConfig(num_players=strat_a.num_players())
    if args.b == "uniform":
        strat_b = UniformStrategy()
    else:
        strat_b = sixmax.BlueprintStrategy.load(args.b, vocab)
        if strat_b.num_players() != strat_a.num_players():
            raise SystemExit("checkpoints disagree on num_players")

    bb100 = run_match(strat_a, strat_b, vocab, cfg, args.hands, args.seed)
    n_hands = args.hands * cfg.num_players
    print(f"Blueprint A win rate: {bb100:+.2f} BB/100 ({n_hands} hands)")


if __name__ == "__main__":
    main()
```

Note: `_play_hand` uses `state.utility(hero_seat)` — bound on `GameState` since Phase 1a; it returns the hero's net payoff in BB.

- [ ] **Step 3: Add the selection hook to train_sixmax.py**

Add to `scripts/train_sixmax.py` (top-level, after `write_config_snapshot`):

```python
import json
import re
import shutil
from datetime import datetime, timezone

_BB100_RE = re.compile(r"Blueprint A win rate:\s*([+-]?\d+(?:\.\d+)?)\s*BB/100")


def parse_bb100(text: str) -> "float | None":
    m = _BB100_RE.search(text)
    return float(m.group(1)) if m else None


def update_best(bb100: "float | None", ckpt_path: str, iterations: int) -> bool:
    """Promote ckpt to best_checkpoint.bin. bb100=None means 'no best yet:
    promote unconditionally'; otherwise replace on strictly positive BB/100
    vs the current best. Atomic copy + JSON sidecar (neural_cfr pattern)."""
    if bb100 is not None and bb100 <= 0.0:
        return False
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    best = os.path.join(ckpt_dir, "best_checkpoint.bin")
    tmp = best + ".tmp"
    shutil.copy2(ckpt_path, tmp)
    os.replace(tmp, best)
    sidecar = os.path.join(ckpt_dir, "best_checkpoint.json")
    tmp_json = sidecar + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump({
            "bb100_vs_previous_best": bb100,
            "iterations": iterations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": os.path.abspath(ckpt_path),
        }, f, indent=2)
    os.replace(tmp_json, sidecar)
    return True


def run_selection(cfg: dict, repo_root: str, trainer) -> None:
    """Eval current checkpoint vs best; promote on strict improvement.
    Advisory: never raises, never kills training."""
    ckpt = cfg["checkpoint"]
    best = os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                        "best_checkpoint.bin")
    try:
        if not os.path.exists(best):
            update_best(None, ckpt, trainer.iterations())
            print("[selection] first checkpoint promoted to best_checkpoint.bin")
            return
        result = subprocess.run(
            [sys.executable,
             os.path.join(repo_root, "scripts", "eval_sixmax.py"),
             "--a", ckpt, "--b", best,
             "--hands", str(cfg["selection_hands"]),
             "--config", cfg["config_path"]],
            capture_output=True, text=True, cwd=repo_root, timeout=3600)
        bb100 = parse_bb100(result.stdout)
        if result.returncode != 0 or bb100 is None:
            print(f"[selection] eval failed (exit {result.returncode}); "
                  f"skipping. stderr tail: {result.stderr[-300:]}")
            return
        replaced = update_best(bb100, ckpt, trainer.iterations())
        print(f"[selection] {bb100:+.2f} BB/100 vs best — "
              f"{'NEW BEST' if replaced else 'kept existing best'}")
    except Exception as e:  # noqa: BLE001 — advisory path, never fatal
        print(f"[selection] skipped ({type(e).__name__}: {e})")
```

Add the CLI flags in `main()`:

```python
    parser.add_argument("--selection-enabled",
                        action=argparse.BooleanOptionalAction, default=None,
                        dest="selection_enabled")
    parser.add_argument("--selection-hands", type=int, default=None,
                        dest="selection_hands")
```

and inside the training loop, right after `write_config_snapshot(...)`:

```python
        if cfg["selection_enabled"]:
            run_selection(cfg, repo_root, trainer)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/sixmax/test_eval_sixmax.py -v`
Expected: 5 passed (the trained-vs-uniform smoke trains 3000 HU iterations — expect ~10–60 s; the whole file well under the suite's tolerance).
Run: `uv run pytest tests/ -q`
Expected: everything green except the known `tests/cfr/test_mccfr.py` xfail/xpass drift.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_sixmax.py scripts/train_sixmax.py tests/sixmax/test_eval_sixmax.py
git commit -m "feat(sixmax): duplicate-deal A/B eval harness + best-checkpoint selection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Deferred to Phase 1c (do not build here)

- openpoker strategy loader (`scripts/openpoker_bot.py` integration, off-tree bet translation via `ActionVocab::nearest`, live-table state → `EngineGameState` mapping).
- HU-mode sanity evals vs the frozen tabular/neural bots.
- `mccfr.cpp` still includes `kuhn.h` (Task 3 narrows the coupling to the exact-value helper; full extraction stays a cleanup candidate).

## Self-Review Notes

- Spec coverage: 169 preflop (Task 1), equity-percentile 50/50/20 (Task 1), raise cap 3 + 4 pot buckets (Task 2), approved live_opps/after extension (Task 2), multithreaded linear-weighted ES-MCCFR with sparse tables (Task 3), TOML config + checkpointing + best-checkpoint selection on neural_cfr patterns (Tasks 4–6), training entry point (Task 5), in-house A/B BB/100 eval (Task 6). Deployment deliverable explicitly split to 1c per the approved scope decision.
- Type consistency verified across tasks: `AbstractionConfig` field order (Task 1 ctor lambda ↔ Task 4 serialization), `TrainerConfig{num_threads, seed}` brace-init order, `abstract_key(const Abstraction&)` signature (Tasks 2/4/6), `probs_for` naming (Tasks 4/6), bit layout (Tasks 2/6 extractors).
- The Task 6 uniform-selfplay bound (400 BB/100) is deliberately loose: the run is deterministic, so the assertion locks reproducibility, not strength; the trained-vs-uniform smoke carries the strength claim.
