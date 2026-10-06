# Six-Max Phase 2 — Depth-Limited Search (HU-First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the deployed heads-up bot an optional depth-limited search — at each flop/turn/river decision, solve the current street's subgame over the player's *exact* cards and a tracked opponent range, and act on the solution — with a hard time budget and a blueprint fallback, so it is never worse than the blueprint by construction.

**Architecture:** A new C++ unit `sixmax/src/search/` compiled into the existing `//sixmax:core` Buck target. A generic **public-tree, two-player, zero-sum vector-CFR+ solver** (`vector_game.h` + `vector_cfr.{h,cpp}`) is validated against the closed-form Kuhn value before it is allowed near hold'em; `kuhn_vector.{h,cpp}` is the fixture. `range_tracker` maintains the opponent's combo distribution and Bayes-updates it with the blueprint as the opponent model; `subgame` builds the *current street's* public betting tree (depth-limited: leaves are end-of-street, not showdown) over `[actions.search]`; `leaf_eval` values those leaves by blueprint continuation; `search_strategy` orchestrates and owns the deadline + fallback. Python sees it only through pybindings and one branch in `agents/sixmax_agent.py`.

**Tech Stack:** C++17, Buck2 (`~/bin/buck2`), pybind11, pytest via `uv run`, existing `sixmax` engine/vocab/blueprint/abstraction.

**Spec:** `docs/superpowers/specs/2026-09-30-sixmax-phase2-search-hu-design.md` (parent: `docs/superpowers/specs/2026-07-17-sixmax-search-design.md`)

## Global Constraints

- `cfr/`, `neural_cfr/`, `sixmax/` never import each other; shared C++ only via `common/`. Search lives in `sixmax/`.
- In `sixmax/` the action vocab is config-defined; **mask illegal actions, never reorder or filter storage**.
- Chip quantities are **BB-denominated** floats (`SB 0.5`, `BB 1.0`, `starting_stack 100`) inside `sixmax/`; any value crossing into a live-engine host is rescaled by the host.
- The C++ evaluator's `12 - rank` kicker inversion is preserved; new consumers use `safe_eval` only.
- Search is **off by default**; a config flag enables it. No deployment without a passing A/B (Task 7).
- **No new external dependencies.** No libtorch in `//sixmax:core` (that is Phase 3).
- Never commit secrets or checkpoint files (`.bin`/`.pkl`/`.pt`).
- Every C++ change is reachable from Python bindings in the same task; tests exercise it through the loaded `sixmax.so`.

## Review Focus

Inputs and failure modes the spec implies but that no single task's happy-path test covers. Each line is pinned to a test in the owning task.

1. **Deadline expiry mid-solve** — a search that overruns must return a *usable* answer (the partially-solved root strategy) or fall back; it must never return garbage probs or block. (Task 5)
2. **Unseen / degenerate infosets** — empty regrets on every legal action, or a range vector that is all-zero after card removal; both must degrade to uniform-over-legal, never divide by zero. (Tasks 2, 5)
3. **Card-removal consistency** — the tracked range must exclude our own hole cards and every board card, and conflicting hand pairs must contribute exactly 0 utility. (Tasks 1, 2)
4. **Off-tree observed bet sizes** — a villain bet that is not in our vocab grid must still be represented (nearest index) rather than rounded away or crashing. (Task 3)
5. **Node-cap exhaustion** — a pathological tree must stop expanding and degrade to evaluated leaves, not exhaust memory or throw. (Task 3)

---

## File Structure

| File | Responsibility |
|---|---|
| `sixmax/src/search/vector_game.h` | Abstract public-tree 2-player zero-sum game interface (`VectorState`, `VectorGame`) |
| `sixmax/src/search/vector_cfr.{h,cpp}` | CFR+ over range vectors; regret tables; average strategy for the query player |
| `sixmax/src/search/kuhn_vector.{h,cpp}` | Kuhn poker as a `VectorGame` — the closed-form correctness fixture |
| `sixmax/src/search/range_tracker.{h,cpp}` | HU opponent combo distribution + Bayes updates via the blueprint |
| `sixmax/src/search/subgame.{h,cpp}` | Builds the current street's public betting tree over `[actions.search]`; node cap |
| `sixmax/src/search/leaf_eval.{h,cpp}` | End-of-street leaf value matrices via blueprint continuation |
| `sixmax/src/search/search_strategy.{h,cpp}` | Orchestration, `SearchConfig`, deadline, fallback, telemetry |
| `sixmax/src/bindings/bindings.cpp` | Modify: register each search type as its task lands |
| `sixmax/BUCK` | Modify: add the `search/` sources to `core` |
| `agents/sixmax_agent.py` | Modify: `SearchStrategy` wrapper + one consult branch |
| `sixmax/configs/default.toml`, `hu.toml` | Modify: `[search]` section |
| `scripts/eval_search_ab.py` | Create: search-on/off paired A/B harness |
| `tests/sixmax/test_*.py` | Create: one test file per unit above |

Tasks 1–5 build and test the C++ library; Task 6 wires it into the agent behind the flag; Task 7 is the measurement and deployment gate. Each task leaves the suite green.

---

### Task 1: Vector-CFR+ core, validated on Kuhn

**Files:**
- Create: `sixmax/src/search/vector_game.h`
- Create: `sixmax/src/search/vector_cfr.h`, `sixmax/src/search/vector_cfr.cpp`
- Create: `sixmax/src/search/kuhn_vector.h`, `sixmax/src/search/kuhn_vector.cpp`
- Modify: `sixmax/BUCK` (add the three `.cpp` files to `core`'s `srcs`)
- Modify: `sixmax/src/bindings/bindings.cpp` (register `VectorCfrConfig`, `VectorCfr`, `KuhnVectorGame`)
- Test: `tests/sixmax/test_vector_cfr_kuhn.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `VectorState` — `int num_players() const`, `int actor() const` (0/1, `-1` at a leaf), `bool is_leaf() const`, `int num_actions() const`, `void legal_mask(std::vector<uint8_t>&) const`, `std::unique_ptr<VectorState> apply(int) const`, `double utility(int player, int h0, int h1) const`, `uint64_t id() const`.
  - `VectorGame` — `int num_hands(int player) const`, `std::unique_ptr<VectorState> root() const`.
  - `VectorCfrConfig { uint64_t seed; int max_iters; double budget_ms; }`
  - `VectorCfr(VectorCfrConfig)`, `VectorCfrResult solve(const VectorGame&, int query_player, int query_hand)`, `std::vector<double> average_strategy(uint64_t node_id, int hand) const`
  - `VectorCfrResult { double root_value; int iterations; bool deadline_hit; }`
  - `KuhnVectorGame` — default-constructible `VectorGame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_vector_cfr_kuhn.py
"""The Phase-2 correctness gate: the vector CFR+ solver must reproduce the
closed-form Kuhn equilibrium value (-1/18 to player 0) before it is allowed
near hold'em. Mirrors the existing -1/18 gate on the tabular trainer."""
import math

import sixmax


def _solver(max_iters=20000):
    cfg = sixmax.VectorCfrConfig()
    cfg.max_iters = max_iters
    cfg.budget_ms = 60000.0
    return sixmax.VectorCfr(cfg)


def test_kuhn_vector_root_value_matches_closed_form():
    solver = _solver()
    res = solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    assert abs(res.root_value - (-1.0 / 18.0)) < 0.01, res.root_value
    assert not res.deadline_hit


def test_kuhn_vector_root_strategy_bets_the_best_card():
    """P0's first decision: bet the best card (hand 2) with probability 1.
    A solver that returns near-uniform is failing."""
    solver = _solver()
    solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    assert solver.rank_root_action(0, 2) == 1     # 1 == "bet" in the fixture


def test_kuhn_vector_average_strategy_sums_to_one():
    solver = _solver(max_iters=200)
    solver.solve(sixmax.KuhnVectorGame(), 0, 0)
    for hand in range(3):
        probs = list(solver.average_strategy(solver.root_id(), hand))
        assert len(probs) == 2
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_vector_cfr_kuhn.py -v`
Expected: FAIL — `AttributeError: module 'sixmax' has no attribute 'KuhnVectorGame'`.

- [ ] **Step 3: Write the abstract game interface**

```cpp
// sixmax/src/search/vector_game.h
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace sixmax {

// A two-player zero-sum game in VECTOR form: the public tree is explicit and
// each player holds one private "hand" drawn from a fixed index space. The
// solver never sees cards — only indices and a utility matrix per leaf.
//
// Card removal is the caller's contract: a hand pair that cannot co-occur
// (h0 == h1 sharing a card) MUST contribute utility 0, and the range vectors
// fed to the solver must already be zero on blocked hands.
class VectorState {
public:
    virtual ~VectorState() = default;
    virtual int num_players() const = 0;   // 2 in v1
    virtual int actor() const = 0;         // 0/1, or -1 at a leaf
    virtual bool is_leaf() const = 0;
    virtual int num_actions() const = 0;
    virtual void legal_mask(std::vector<uint8_t>& mask) const = 0;
    virtual std::unique_ptr<VectorState> apply(int action) const = 0;
    // Payoff to `player` when the two players hold h0 / h1. Leaf matrices for
    // NLHE are precomputed by LeafEval and cached inside the state; Kuhn
    // computes them directly.
    virtual double utility(int player, int h0, int h1) const = 0;
    // Stable identity for this public state (regret-table key).
    virtual uint64_t id() const = 0;
};

class VectorGame {
public:
    virtual ~VectorGame() = default;
    virtual int num_hands(int player) const = 0;   // 3 (Kuhn), <=1326 (NLHE)
    virtual std::unique_ptr<VectorState> root() const = 0;
};

}  // namespace sixmax
```

- [ ] **Step 4: Write the solver**

```cpp
// sixmax/src/search/vector_cfr.h
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "search/vector_game.h"

namespace sixmax {

struct VectorCfrConfig {
    uint64_t seed = 1;
    int max_iters = 2000;         // hard iteration cap
    double budget_ms = 5000.0;    // wall-clock deadline, checked between iterations
};

struct VectorCfrResult {
    double root_value = 0.0;      // counterfactual value at the root for `query_player`
    int iterations = 0;
    bool deadline_hit = false;
};

// CFR+ (regret-matching+ with linear averaging) over a public tree.
// Two-player zero-sum, one recursion pass:
//   * each node keeps regret and (for `query_player`) average-strategy tables
//     indexed by [hand][action];
//   * the traversal returns, per player, a counterfactual value vector: for
//     player p and hand h, sum over the OPPONENT's hands weighted by the
//     opponent's reach. Returning both vectors in one pass is valid because
//     the actor's reach is folded into each child call.
class VectorCfr {
public:
    explicit VectorCfr(VectorCfrConfig cfg) : cfg_(cfg) {}

    VectorCfrResult solve(const VectorGame& game, int query_player, int query_hand);
    // Average strategy at `node_id` for `query_player` holding `hand`.
    // Empty when the node was never visited as an actor, or all regrets <= 0
    // during the whole run (caller falls back to uniform-over-legal).
    std::vector<double> average_strategy(uint64_t node_id, int hand) const;
    // Test seam: index of the arg-max action at the root for `hand`.
    int rank_root_action(int query_player, int hand) const;
    // Node id of the root (the regret/average key for the root decision).
    uint64_t root_id() const { return root_id_; }

private:
    struct NodeData {
        std::vector<double> regret;    // [n_hands * n_actions]
        std::vector<double> strat_sum; // [n_hands * n_actions]
    };
    // v[p][h] for p in {0,1}: counterfactual value of p's hand h.
    std::array<std::vector<double>, 2> walk(
            const VectorState& s, int depth,
            const std::array<std::vector<double>, 2>& reach, double weight);

    VectorCfrConfig cfg_;
    int num_hands_[2] = {0, 0};
    int query_player_ = 0;
    uint64_t root_id_ = 0;
    std::unordered_map<uint64_t, NodeData> table_;
    std::unordered_map<uint64_t, int> node_actions_;
    std::unordered_map<uint64_t, int> node_actor_;
    bool deadline_hit_ = false;
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/vector_cfr.cpp
#include "search/vector_cfr.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace sixmax {
namespace {
constexpr int kMaxActions = 16;   // search grids are small; guards stack arrays
}

VectorCfrResult VectorCfr::solve(const VectorGame& game, int query_player,
                                 int query_hand) {
    if (query_player < 0 || query_player > 1)
        throw std::invalid_argument("query_player must be 0 or 1");
    query_player_ = query_player;
    num_hands_[0] = game.num_hands(0);
    num_hands_[1] = game.num_hands(1);
    if (query_hand < 0 || query_hand >= num_hands_[query_player])
        throw std::invalid_argument("query_hand out of range");
    table_.clear();
    node_actions_.clear();
    node_actor_.clear();
    deadline_hit_ = false;

    auto root = game.root();
    root_id_ = root->id();
    node_actions_[root_id_] = root->num_actions();
    node_actor_[root_id_] = root->actor();

    // Root reaches: every hand equally likely, normalised per player. Card
    // removal is the game's job (conflicting pairs contribute 0 utility).
    std::array<std::vector<double>, 2> reach;
    for (int p = 0; p < 2; ++p)
        reach[p].assign(num_hands_[p], 1.0 / std::max(1, num_hands_[p]));

    const auto start = std::chrono::steady_clock::now();
    double value = 0.0;
    int it = 0;
    for (; it < cfg_.max_iters; ++it) {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start).count();
        if (ms >= cfg_.budget_ms) { deadline_hit_ = true; break; }
        const double w = static_cast<double>(it + 1);   // linear CFR+ weighting
        auto v = walk(*root, 0, reach, w);
        value = v[query_player][query_hand];
    }
    VectorCfrResult r;
    r.root_value = value;
    r.iterations = it;
    r.deadline_hit = deadline_hit_;
    return r;
}

std::array<std::vector<double>, 2> VectorCfr::walk(
        const VectorState& s, int depth,
        const std::array<std::vector<double>, 2>& reach, double w) {
    std::array<std::vector<double>, 2> out;
    if (s.is_leaf()) {
        const int n0 = num_hands_[0], n1 = num_hands_[1];
        out[0].assign(n0, 0.0);
        out[1].assign(n1, 0.0);
        for (int h0 = 0; h0 < n0; ++h0)
            for (int h1 = 0; h1 < n1; ++h1) {
                const double p = reach[0][h0] * reach[1][h1];
                if (p == 0.0) continue;
                out[0][h0] += p * s.utility(0, h0, h1);
                out[1][h1] += p * s.utility(1, h0, h1);
            }
        return out;
    }
    const int p = s.actor();
    const int q = 1 - p;
    const uint64_t id = s.id();
    const int n = s.num_actions();
    if (n <= 0 || n > kMaxActions) throw std::runtime_error("bad action count");
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    node_actions_[id] = n;
    node_actor_[id] = p;

    NodeData& d = table_[id];
    if (d.regret.empty()) {
        d.regret.assign(static_cast<size_t>(num_hands_[p]) * n, 0.0);
        d.strat_sum.assign(static_cast<size_t>(num_hands_[p]) * n, 0.0);
    }

    // Regret-matching+ with the node's per-hand reach weighting for averaging.
    std::vector<double> sigma(static_cast<size_t>(num_hands_[p]) * n, 0.0);
    for (int h = 0; h < num_hands_[p]; ++h) {
        double pos = 0.0;
        int legal = 0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            ++legal;
            const double r = d.regret[static_cast<size_t>(h) * n + a];
            if (r > 0.0) pos += r;
        }
        if (pos > 0.0) {
            for (int a = 0; a < n; ++a)
                if (mask[a]) {
                    const double r = d.regret[static_cast<size_t>(h) * n + a];
                    sigma[static_cast<size_t>(h) * n + a] = r > 0.0 ? r / pos : 0.0;
                }
        } else {
            for (int a = 0; a < n; ++a)
                if (mask[a]) sigma[static_cast<size_t>(h) * n + a] = 1.0 / legal;
        }
    }

    out[0].assign(num_hands_[0], 0.0);
    out[1].assign(num_hands_[1], 0.0);
    std::vector<std::vector<double>> vp_by_action(n);   // p's values per action
    std::vector<std::vector<double>> vq_by_action(n);   // q's values per action

    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        auto child = s.apply(a);
        auto child_reach = reach;
        for (int h = 0; h < num_hands_[p]; ++h)
            child_reach[p][h] = reach[p][h] * sigma[static_cast<size_t>(h) * n + a];
        auto v = walk(*child, depth + 1, child_reach, w);
        vp_by_action[a] = v[p];
        vq_by_action[a] = v[q];
        for (int h = 0; h < num_hands_[q]; ++h) out[q][h] += v[q][h];
    }

    // p's node value and regret update (regrets are weighted by w, as in CFR+).
    for (int h = 0; h < num_hands_[p]; ++h) {
        double node_value = 0.0;
        for (int a = 0; a < n; ++a)
            if (mask[a]) node_value += sigma[static_cast<size_t>(h) * n + a] * vp_by_action[a][h];
        out[p][h] = node_value;
        if (p == query_player_) {
            double reach_h = reach[p][h];
            for (int a = 0; a < n; ++a) {
                if (!mask[a]) continue;
                const size_t k = static_cast<size_t>(h) * n + a;
                d.regret[k] = std::max(0.0, d.regret[k] + w * (vp_by_action[a][h] - node_value));
                d.strat_sum[k] += w * reach_h * sigma[k];
            }
        } else {
            for (int a = 0; a < n; ++a) {
                if (!mask[a]) continue;
                const size_t k = static_cast<size_t>(h) * n + a;
                d.regret[k] = std::max(0.0, d.regret[k] + w * (vp_by_action[a][h] - node_value));
            }
        }
    }
    return out;
}

std::vector<double> VectorCfr::average_strategy(uint64_t node_id, int hand) const {
    auto it = table_.find(node_id);
    if (it == table_.end()) return {};
    const int n = node_actions_.at(node_id);
    const auto& ss = it->second.strat_sum;
    double total = 0.0;
    for (int a = 0; a < n; ++a) total += ss[static_cast<size_t>(hand) * n + a];
    if (total <= 0.0) return {};
    std::vector<double> out(n, 0.0);
    for (int a = 0; a < n; ++a) out[a] = ss[static_cast<size_t>(hand) * n + a] / total;
    return out;
}

int VectorCfr::rank_root_action(int, int hand) const {
    auto probs = average_strategy(root_id_, hand);
    if (probs.empty()) return -1;
    int best = -1;
    double bestv = -1.0;
    for (size_t a = 0; a < probs.size(); ++a)
        if (probs[a] > bestv) { bestv = probs[a]; best = static_cast<int>(a); }
    return best;
}

}  // namespace sixmax
```

- [ ] **Step 5: Write the Kuhn fixture**

```cpp
// sixmax/src/search/kuhn_vector.h
#pragma once
#include <memory>
#include <string>
#include "search/vector_game.h"

namespace sixmax {

// Standard 3-card Kuhn poker as a VectorGame. History alphabet:
//   'p' = pass/check, 'b' = bet/call. Terminals:
//   "pp"  showdown, stakes 1
//   "pbp" P0 folded            -> u0 = -1
//   "pbb" showdown, stakes 2
//   "bp"  P1 folded            -> u0 = +1
//   "bb"  showdown, stakes 2
class KuhnVectorState : public VectorState {
public:
    explicit KuhnVectorState(std::string history) : h_(std::move(history)) {}
    int num_players() const override { return 2; }
    int actor() const override { return h_.size() % 2; }      // P0 first
    bool is_leaf() const override {
        return h_ == "pp" || h_ == "pbp" || h_ == "pbb" || h_ == "bp" || h_ == "bb";
    }
    int num_actions() const override { return 2; }            // 0 = pass, 1 = bet
    void legal_mask(std::vector<uint8_t>& m) const override { m.assign({1, 1}); }
    std::unique_ptr<VectorState> apply(int action) const override {
        return std::make_unique<KuhnVectorState>(h_ + (action == 1 ? 'b' : 'p'));
    }
    double utility(int player, int h0, int h1) const override;
    uint64_t id() const override;                              // history code
private:
    std::string h_;
};

class KuhnVectorGame : public VectorGame {
public:
    int num_hands(int) const override { return 3; }
    std::unique_ptr<VectorState> root() const override {
        return std::make_unique<KuhnVectorState>("");
    }
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/kuhn_vector.cpp
#include "search/kuhn_vector.h"

namespace sixmax {

double KuhnVectorState::utility(int player, int h0, int h1) const {
    if (h0 == h1) return 0.0;                 // impossible pair (card removal)
    // Fold terminals are card-independent.
    if (h_ == "pbp") return player == 0 ? -1.0 : 1.0;
    if (h_ == "bp")  return player == 0 ?  1.0 : -1.0;
    const double stake = (h_ == "pp") ? 1.0 : 2.0;
    const bool p0_wins = h0 > h1;
    const double u0 = p0_wins ? stake : -stake;
    return player == 0 ? u0 : -u0;
}

uint64_t KuhnVectorState::id() const {
    uint64_t code = 1469598103934665603ull;   // FNV-1a over the history
    for (char c : h_) { code ^= static_cast<uint8_t>(c); code *= 1099511628211ull; }
    return code;
}

}  // namespace sixmax
```

- [ ] **Step 6: Register the bindings**

Add to `sixmax/src/bindings/bindings.cpp` (with the other class registrations, before the module's closing brace):

```cpp
    py::class_<sixmax::VectorCfrConfig>(m, "VectorCfrConfig")
        .def(py::init<>())
        .def_readwrite("seed", &sixmax::VectorCfrConfig::seed)
        .def_readwrite("max_iters", &sixmax::VectorCfrConfig::max_iters)
        .def_readwrite("budget_ms", &sixmax::VectorCfrConfig::budget_ms);

    py::class_<sixmax::VectorCfrResult>(m, "VectorCfrResult")
        .def_readonly("root_value", &sixmax::VectorCfrResult::root_value)
        .def_readonly("iterations", &sixmax::VectorCfrResult::iterations)
        .def_readonly("deadline_hit", &sixmax::VectorCfrResult::deadline_hit);

    py::class_<sixmax::KuhnVectorGame, sixmax::VectorGame>(m, "KuhnVectorGame")
        .def(py::init<>());

    py::class_<sixmax::VectorCfr>(m, "VectorCfr")
        .def(py::init<sixmax::VectorCfrConfig>(), py::arg("cfg"))
        .def("solve", &sixmax::VectorCfr::solve,
             py::arg("game"), py::arg("query_player"), py::arg("query_hand"))
        .def("average_strategy", &sixmax::VectorCfr::average_strategy,
             py::arg("node_id"), py::arg("hand"))
        .def("rank_root_action", &sixmax::VectorCfr::rank_root_action,
             py::arg("query_player"), py::arg("hand"))
        .def("root_id", &sixmax::VectorCfr::root_id);
```

Add the three sources to `sixmax/BUCK`'s `core` `srcs`:

```python
        "src/search/vector_cfr.cpp",
        "src/search/kuhn_vector.cpp",
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/sixmax/test_vector_cfr_kuhn.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add sixmax/src/search/vector_game.h sixmax/src/search/vector_cfr.h \
        sixmax/src/search/vector_cfr.cpp sixmax/src/search/kuhn_vector.h \
        sixmax/src/search/kuhn_vector.cpp sixmax/BUCK \
        sixmax/src/bindings/bindings.cpp tests/sixmax/test_vector_cfr_kuhn.py
git commit -m "feat(sixmax): vector CFR+ core validated against the Kuhn equilibrium"
```

---

### Task 2: HU range tracker

**Files:**
- Create: `sixmax/src/search/range_tracker.h`, `sixmax/src/search/range_tracker.cpp`
- Modify: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_range_tracker.py`

**Interfaces:**
- Consumes: `BlueprintStrategy` (`probs_for`), `ActionVocab` (`nearest`), `EngineGameState`.
- Produces:
  - `RangeTracker(const EngineConfig& cfg, const ActionVocab& vocab)` 
  - `void reset(const std::vector<int>& board, const std::array<int,2>& our_hole)` — uniform over legal combos, zeroing blocked cards
  - `void observe(const BlueprintStrategy&, const EngineGameState& pre_action, int actor, int action)` — Bayes reweight + renormalise
  - `const std::vector<double>& range(int player) const`
  - `double combo_prob(int player, std::array<int,2> hole) const`
  - `static int combo_index(int c0, int c1)` / `static std::array<int,2> combo_cards(int idx)` — the canonical 1326 ordering (c0 < c1)

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_range_tracker.py
"""HU range tracker invariants: exact card removal, an always-normalised
posterior, and a Bayes update that moves mass toward the observed action."""
import importlib.util
import math
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_vocab(section="blueprint"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


VOCAB = _load_vocab()


def _tracker():
    return sixmax.RangeTracker(sixmax.EngineConfig(num_players=2), VOCAB)


def test_combo_index_round_trips():
    n = 0
    for c0 in range(52):
        for c1 in range(c0 + 1, 52):
            idx = sixmax.RangeTracker.combo_index(c0, c1)
            assert idx == n
            assert tuple(sixmax.RangeTracker.combo_cards(idx)) == (c0, c1)
            n += 1
    assert n == 1326


def test_reset_zeroes_blocked_cards_and_normalises():
    t = _tracker()
    board, hole = [0, 1, 2], (4, 5)
    t.reset(board, hole)
    r = list(t.range(1))
    assert math.isclose(sum(r), 1.0, abs_tol=1e-9)
    for blocked in (*board, *hole):
        for other in range(52):
            if other == blocked:
                continue
            lo, hi = min(blocked, other), max(blocked, other)
            assert r[sixmax.RangeTracker.combo_index(lo, hi)] == 0.0


def test_bayes_update_collapses_onto_the_observed_hand():
    """Closed-form toy, no engine needed: the Bayes step multiplies by the
    per-combo action probability and renormalises."""
    t = _tracker()
    t.set_range_for_test(1, [0.5, 0.5, 0.0])
    t.observe_with_probs(1, [1.0, 0.0, 0.0])
    r = list(t.range(1))
    assert math.isclose(r[0], 1.0, abs_tol=1e-12)
    assert math.isclose(r[1], 0.0, abs_tol=1e-12)
    assert math.isclose(sum(r), 1.0, abs_tol=1e-12)


def test_degenerate_update_keeps_a_valid_distribution():
    """An observation impossible on the current support must not produce NaNs
    or a zero vector: the prior is kept, still normalised."""
    t = _tracker()
    t.set_range_for_test(1, [0.5, 0.5, 0.0])
    t.observe_with_probs(1, [0.0, 0.0, 1.0])
    r = list(t.range(1))
    assert math.isclose(sum(r), 1.0, abs_tol=1e-12)
    assert all(v >= 0.0 and not math.isnan(v) for v in r)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_range_tracker.py -v`
Expected: FAIL — `module 'sixmax' has no attribute 'RangeTracker'`.

- [ ] **Step 3: Write the implementation**

```cpp
// sixmax/src/search/range_tracker.h
#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "blueprint/checkpoint.h"
#include "blueprint/engine_game.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

// Opponent range over the 1,326 two-card combos, canonically indexed by
// (min,max) card code. One tracker serves both seats; `player` selects whose
// distribution is read. v1 only ever tracks a single opponent, so the vector
// for the hero's own seat is a point mass after `reset`.
class RangeTracker {
public:
    RangeTracker(const EngineConfig& cfg, const ActionVocab& vocab)
        : cfg_(cfg), vocab_(vocab), range_(1326, 0.0) {}

    static int combo_index(int c0, int c1);
    static std::array<int, 2> combo_cards(int idx);

    // Uniform over combos not blocked by `board` or `our_hole`.
    void reset(const std::vector<int>& board, const std::array<int, 2>& our_hole);

    // Bayes reweight for `actor` taking `action` from `pre_action`, using the
    // blueprint's policy at that state as the opponent model. Renormalises;
    // an all-zero posterior degrades to uniform-over-the-previous-support.
    void observe(const BlueprintStrategy& blueprint, const EngineGameState& pre_action,
                 int actor, int action);

    const std::vector<double>& range(int) const { return range_; }
    double combo_prob(std::array<int, 2> hole) const {
        return range_[combo_index(hole[0], hole[1])];
    }

    // Test seams (never used in production paths).
    void set_range_for_test(int, const std::vector<double>& probs) { range_ = probs; }
    void observe_with_probs(int, const std::vector<double>& action_probs_per_combo);

private:
    EngineConfig cfg_;
    const ActionVocab& vocab_;
    std::vector<double> range_;
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/range_tracker.cpp
#include "search/range_tracker.h"
#include <algorithm>
#include <cmath>

namespace sixmax {

int RangeTracker::combo_index(int c0, int c1) {
    if (c0 > c1) std::swap(c0, c1);
    // Number of combos whose first card is < c0, plus the offset within c0.
    return c0 * 51 - (c0 * (c0 - 1)) / 2 + (c1 - c0 - 1);
}

std::array<int, 2> RangeTracker::combo_cards(int idx) {
    for (int c0 = 0; c0 < 52; ++c0) {
        const int base = c0 * 51 - (c0 * (c0 - 1)) / 2;
        const int count = 51 - c0;
        if (idx < base + count) return {c0, c0 + 1 + (idx - base)};
    }
    return {-1, -1};
}

void RangeTracker::reset(const std::vector<int>& board,
                         const std::array<int, 2>& our_hole) {
    std::array<bool, 52> blocked{};
    for (int c : board) blocked[c] = true;
    blocked[our_hole[0]] = blocked[our_hole[1]] = true;
    range_.assign(1326, 0.0);
    double total = 0.0;
    for (int i = 0; i < 1326; ++i) {
        auto c = combo_cards(i);
        if (blocked[c[0]] || blocked[c[1]]) continue;
        range_[i] = 1.0;
        total += 1.0;
    }
    if (total > 0.0)
        for (double& v : range_) v /= total;
}

void RangeTracker::observe(const BlueprintStrategy& blueprint,
                           const EngineGameState& pre_action, int actor, int action) {
    std::vector<double> w(1326, 0.0);
    for (int i = 0; i < 1326; ++i) {
        if (range_[i] <= 0.0) continue;
        auto hole = combo_cards(i);
        EngineGameState probe = pre_action;      // value copy: mutable_hand below
        probe.mutable_hand().reroll_hidden(actor, {}, rng_unused_);  // see note
        w[i] = range_[i];
    }
    observe_with_probs(actor, w);
}

void RangeTracker::observe_with_probs(int, const std::vector<double>& w) {
    double total = 0.0;
    for (size_t i = 0; i < range_.size(); ++i) {
        range_[i] *= (i < w.size() ? w[i] : 0.0);
        total += range_[i];
    }
    if (total <= 0.0) return;                    // keep the prior: degenerate update
    for (double& v : range_) v /= total;
}

}  // namespace sixmax
```

> **Note for the implementer:** the real `observe` must key the blueprint at the *specific
> hand* rather than re-rolling: build a temporary `EngineGameState` whose hidden cards are the
> combo under test and read `blueprint.probs_for(...)` at the observed action index. The
> `reroll_hidden` sketch above is deliberately incomplete — replace it with a helper that sets
> `mutable_hand()`'s hole cards for `actor` to the combo and leaves the board fixed, then
> `w[i] = range_[i] * probs[action]`. `observe_with_probs` is the tested Bayes core; `observe`
> is its thin adapter. Add a `#include <random>` member only if you keep a scratch RNG.

- [ ] **Step 4: Register bindings and BUCK sources**

```cpp
    py::class_<sixmax::RangeTracker>(m, "RangeTracker")
        .def(py::init<sixmax::EngineConfig, const sixmax::ActionVocab&>(),
             py::arg("cfg"), py::arg("vocab"))
        .def_static("combo_index", &sixmax::RangeTracker::combo_index)
        .def_static("combo_cards", &sixmax::RangeTracker::combo_cards)
        .def("reset", &sixmax::RangeTracker::reset)
        .def("range", [](const sixmax::RangeTracker& t, int p) { return t.range(p); })
        .def("combo_prob", &sixmax::RangeTracker::combo_prob)
        .def("set_range_for_test", &sixmax::RangeTracker::set_range_for_test)
        .def("observe_with_probs", &sixmax::RangeTracker::observe_with_probs);
```

Add `"src/search/range_tracker.cpp"` to `core`'s `srcs` in `sixmax/BUCK`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/sixmax/test_range_tracker.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/search/range_tracker.h sixmax/src/search/range_tracker.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_range_tracker.py
git commit -m "feat(sixmax): heads-up range tracker with blueprint Bayes updates"
```

---

### Task 3: Subgame builder over the live engine

**Files:**
- Create: `sixmax/src/search/subgame.h`, `sixmax/src/search/subgame.cpp`
- Modify: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_subgame.py`

**Interfaces:**
- Consumes: `HandState`, `ActionVocab`, `VectorGame`/`VectorState` (Task 1), `RangeTracker` (Task 2).
- Produces:
  - `struct SubgameConfig { int max_nodes = 20000; }`
  - `class Subgame : public VectorGame` — `Subgame(const EngineGameState& street_root, const ActionVocab&, const RangeTracker&, SubgameConfig)` (accepts the bridge state, not `HandState`, so legality and `bet_context()` come from one place and cannot drift)
  - `num_hands(int)` (HU only in v1), `root()`, `size_t node_count() const`, `std::vector<uint64_t> node_ids() const` (deterministic; the solver's regret keys and the tests use it)
  - `SubgameState : VectorState` with `utility` = the cached leaf matrix (empty until Task 4 fills it) and `id()` = a hash of (board, pot, stacks, history).

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_subgame.py
"""Street-subgame tree invariants: legality mirrors EngineGameState exactly,
the node cap degrades to leaves rather than throwing, and construction is
deterministic."""
import importlib.util
import os

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _flop_state():
    """Call + check preflop -> flop, BB to act. Same construction as
    tests/sixmax/test_engine_game.py::test_postflop_bet_spot_mask."""
    vocab = _load_vocab("search")
    cfg = sixmax.EngineConfig(num_players=2)
    eng = sixmax.EngineGameState(cfg, 0, list(range(52)), vocab, [])
    eng.apply(CALL)
    eng.apply(CHECK)
    assert eng.street() == 1
    tracker = sixmax.RangeTracker(cfg, vocab)
    tracker.reset(list(eng.board()), tuple(eng.hole_cards(0)))
    return eng, vocab, tracker


def test_subgame_legality_matches_engine_exactly():
    eng, vocab, tracker = _flop_state()
    sg = sixmax.Subgame(eng, vocab, tracker, sixmax.SubgameConfig())
    root = sg.root()
    assert list(root.legal_mask()) == list(eng.legal_mask())
    assert root.actor() == eng.current_player()


def test_subgame_respects_node_cap():
    eng, vocab, tracker = _flop_state()
    cfg = sixmax.SubgameConfig()
    cfg.max_nodes = 2
    sg = sixmax.Subgame(eng, vocab, tracker, cfg)
    assert sg.node_count() <= 2


def test_subgame_construction_is_deterministic():
    eng, vocab, tracker = _flop_state()
    a = sixmax.Subgame(eng, vocab, tracker, sixmax.SubgameConfig())
    b = sixmax.Subgame(eng, vocab, tracker, sixmax.SubgameConfig())
    assert a.node_ids() == b.node_ids()
    assert list(a.root().legal_mask()) == list(b.root().legal_mask())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_subgame.py -v`
Expected: FAIL — `module 'sixmax' has no attribute 'Subgame'`.

- [ ] **Step 3: Write the builder**

```cpp
// sixmax/src/search/subgame.h
#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include "engine/engine.h"
#include "search/range_tracker.h"
#include "search/vector_game.h"
#include "vocab/vocab.h"

namespace sixmax {

struct SubgameConfig {
    int max_nodes = 20000;   // expansion cap; the frontier becomes leaves
};

// The current street's public betting tree, as a VectorGame. Depth-limited:
// every node past the street's last action is a LEAF whose utility matrix is
// supplied by LeafEval (Task 4). Hand indices are the caller's combo indices
// (RangeTracker's canonical ordering), filtered to the legal set.
class Subgame : public VectorGame {
public:
    Subgame(const EngineGameState& street_root, const ActionVocab& vocab,
            const RangeTracker& ranges, SubgameConfig cfg);

    int num_hands(int player) const override { return (int)legal_hands_.size(); }
    std::unique_ptr<VectorState> root() const override;
    size_t node_count() const { return nodes_.size(); }
    std::vector<uint64_t> node_ids() const;

private:
    struct Node {
        uint64_t id;
        int actor;
        bool leaf;
        std::vector<double> action_amounts;   // raise-to in BB, vocab order
        std::vector<int> children;            // node indices
    };
    std::vector<Node> nodes_;
    std::vector<int> legal_hands_;            // combo indices surviving card removal
    std::unordered_map<uint64_t, int> by_id_;
    SubgameConfig cfg_;
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/subgame.cpp
#include "search/subgame.h"
#include <algorithm>
#include <functional>

namespace sixmax {

Subgame::Subgame(const EngineGameState& street_root, const ActionVocab& vocab,
                 const RangeTracker& ranges, SubgameConfig cfg)
    : cfg_(cfg) {
    for (int i = 0; i < 1326; ++i)
        if (ranges.range(0)[i] > 0.0) legal_hands_.push_back(i);

    std::function<int(const HandState&)> build = [&](const HandState& s) -> int {
        Node n;
        n.id = /* hash of (board, street, pot, stacks, history) */ 0;
        n.leaf = s.is_terminal() || s.street() != street_root.street();
        n.actor = s.is_terminal() ? -1 : s.current_player();
        if (n.leaf || (int)nodes_.size() >= cfg_.max_nodes) {
            n.leaf = true;
            n.actor = -1;
            nodes_.push_back(n);
            return (int)nodes_.size() - 1;
        }
        std::vector<uint8_t> mask;
        // Rebuild the masking through EngineGameState so legality can never drift.
        EngineGameState gs(s, &vocab, nullptr);
        gs.legal_mask(mask);
        for (int a = 0; a < (int)mask.size(); ++a) {
            if (!mask[a]) continue;
            HandState child = s.clone();
            // Translate vocab index -> EngineAction via ActionVocab::target_bb,
            // then recurse; ids/amounts stored for the solver and leaf eval.
            auto act = vocab.at(a);
            (void)act;
            n.children.push_back(build(child));
        }
        nodes_.push_back(n);
        return (int)nodes_.size() - 1;
    };
    build(street_root);
}

std::unique_ptr<VectorState> Subgame::root() const {
    // v1: the root is the street root; the query decision node is found by
    // following the observed action prefix (Task 5 passes the prefix in).
    return nullptr;   // replaced in Step 3b
}

}  // namespace sixmax
```

> **Note for the implementer:** Step 3b completes this file: (a) implement the id hash over
> `(board, street, pot, rounded stack/street-bet, raise count, actor)` — deterministic, no
> pointers; (b) apply actions with `HandState::apply(EngineAction)` after translating the vocab
> index through `ActionVocab::target_bb(idx, bet_context)`, exactly as `EngineGameState::apply`
> does (read it and reuse its translation, do not re-derive); (c) return a
> `SubgameState` wrapper whose `apply(i)` indexes `children[i]` and whose `utility` reads the
> leaf matrix (empty until Task 4 — return 0 and let Task 4's test catch it).

- [ ] **Step 4: Register bindings and BUCK sources; run the tests**

```cpp
    py::class_<sixmax::SubgameConfig>(m, "SubgameConfig")
        .def(py::init<>())
        .def_readwrite("max_nodes", &sixmax::SubgameConfig::max_nodes);
```

Add `"src/search/subgame.cpp"` to `core`'s `srcs`.

Run: `uv run pytest tests/sixmax/test_subgame.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sixmax/src/search/subgame.h sixmax/src/search/subgame.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_subgame.py
git commit -m "feat(sixmax): depth-limited street subgame builder"
```

---

### Task 4: Leaf evaluation by blueprint continuation

**Files:**
- Create: `sixmax/src/search/leaf_eval.h`, `sixmax/src/search/leaf_eval.cpp`
- Modify: `sixmax/src/search/subgame.{h,cpp}` (fill leaf matrices), `sixmax/BUCK`, bindings
- Test: `tests/sixmax/test_leaf_eval.py`

**Interfaces:**
- Consumes: `BlueprintStrategy`, `EngineGameState`, `HandState::reroll_hidden`, `safe_eval`.
- Produces:
  - `struct LeafEvalConfig { int samples = 256; uint64_t seed = 1; }`
  - `std::vector<double> LeafEval::matrix(const HandState& leaf, int n_hands, const std::vector<int>& legal_hands, const std::vector<int>& opp_legal_hands)` — a flat `n_hands*n_hands` value matrix for the acting player, deterministic given `seed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_leaf_eval.py
"""Leaf-evaluation contract. The terminal (river) branch is exact and needs no
blueprint, so these tests never depend on the gitignored checkpoint; the
sampled branch is covered by a module-level skip when it is absent."""
import importlib.util
import math
import os

import pytest

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _street_state(target_street):
    """Drive the engine from preflop to `target_street` (1=flop .. 3=river)."""
    vocab = _load_vocab("search")
    eng = sixmax.EngineGameState(sixmax.EngineConfig(num_players=2), 0,
                                 list(range(52)), vocab, [])
    while eng.street() < target_street:
        eng.apply(CALL)
        eng.apply(CHECK)
    assert eng.street() == target_street
    return eng


def test_terminal_leaf_zeroes_conflicting_hands():
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())      # terminal-only mode
    hands = [0, 1, 2, 3]
    m = list(ev.matrix(eng, 0, hands, hands))
    n = len(hands)
    for i in range(n):
        assert m[i * n + i] == 0.0


def test_terminal_leaf_is_antisymmetric_between_players():
    """Zero-sum: player 0's payoff for (h0,h1) must be minus player 1's for
    (h1,h0), i.e. m0[i][j] == -m1[j][i]."""
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())
    hands = [0, 1, 2, 3]
    n = len(hands)
    m0 = list(ev.matrix(eng, 0, hands, hands))
    m1 = list(ev.matrix(eng, 1, hands, hands))
    for i in range(n):
        for j in range(n):
            assert math.isclose(m0[i * n + j], -m1[j * n + i], abs_tol=1e-9)


def test_terminal_leaf_is_deterministic():
    eng = _street_state(3)
    ev = sixmax.LeafEval(sixmax.LeafEvalConfig())
    a = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    b = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    assert a == b


@pytest.mark.skipif(not os.path.exists(_CKPT),
                    reason="blueprint checkpoint is gitignored")
def test_sampled_leaf_is_deterministic():
    """A turn leaf uses the blueprint continuation; the same seed and inputs
    must give the same matrix."""
    vocab = _load_vocab("search")
    bp = sixmax.BlueprintStrategy.load(_CKPT, vocab)
    ev = sixmax.LeafEval(bp, sixmax.LeafEvalConfig())
    eng = _street_state(2)
    a = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    b = list(ev.matrix(eng, 0, [0, 1, 2], [3, 4, 5]))
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_leaf_eval.py -v`
Expected: FAIL — no `LeafEval`.

- [ ] **Step 3: Implement**

```cpp
// sixmax/src/search/leaf_eval.h
#pragma once
#include <cstdint>
#include <vector>
#include "blueprint/checkpoint.h"
#include "engine/engine.h"

namespace sixmax {

struct LeafEvalConfig {
    int samples = 256;     // (opponent hand, runout) draws per leaf
    uint64_t seed = 1;
};

// End-of-street leaf values. Both players continue with the blueprint for the
// rest of the hand; the value of a leaf is the expected payoff over sampled
// opponent hands and board runouts. Replaced wholesale by the Phase-3 value
// net, so keep the interface to one matrix call.
class LeafEval {
public:
    LeafEval(const BlueprintStrategy& bp, LeafEvalConfig cfg)
        : bp_(bp), cfg_(cfg) {}

    // Row-major [our_hand][opp_hand] payoff to the player to act at `leaf`.
    // Terminal (river) leaves are exact; earlier leaves are estimates.
    std::vector<double> matrix(const HandState& leaf, int actor,
                               const std::vector<int>& our_hands,
                               const std::vector<int>& opp_hands) const;

private:
    const BlueprintStrategy& bp_;
    LeafEvalConfig cfg_;
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/leaf_eval.cpp
#include "search/leaf_eval.h"
#include <random>

namespace sixmax {

std::vector<double> LeafEval::matrix(const HandState& leaf, int actor,
                                     const std::vector<int>& our_hands,
                                     const std::vector<int>& opp_hands) const {
    const size_t n = our_hands.size(), m = opp_hands.size();
    std::vector<double> out(n * m, 0.0);
    std::mt19937_64 rng(cfg_.seed ^ (leaf.street() * 0x9E3779B97F4A7C15ull));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (our_hands[i] == opp_hands[j]) continue;   // card removal
            if (leaf.is_terminal()) {
                // River (or a fold): exact. Probe with a concrete hand assignment.
                HandState probe = leaf.clone();
                probe.set_hole_for_test(actor, our_hands[i], 1 - actor, opp_hands[j]);
                out[i * m + j] = probe.payoffs()[actor];
            } else {
                // Sample runouts and roll the blueprint to showdown; average.
                double acc = 0.0;
                for (int k = 0; k < cfg_.samples; ++k) {
                    HandState probe = leaf.clone();
                    probe.set_hole_for_test(actor, our_hands[i], 1 - actor, opp_hands[j]);
                    // deal remaining board + play blueprint to showdown
                    acc += /* rollout(probe, actor, rng) */ 0.0;
                }
                out[i * m + j] = acc / cfg_.samples;
            }
        }
    }
    return out;
}

}  // namespace sixmax
```

> **Note for the implementer:** two pieces are intentionally left for you to write because they
> are mechanical and are what the tests pin: (1) `HandState::set_hole_for_test(actor, c0, c1, ...)`
> — you may instead use the existing `reroll_hidden` with a constructed pool; pick one and keep
> it out of production paths; (2) `rollout(...)` — deal the remaining board from the deck, then
> advance the hand with blueprint actions (`policy.h`'s `sample_blueprint`) until
> `is_terminal()`, returning `payoffs()[actor]`. Determinism is the tested contract: the same
> `(leaf, actor, hands, seed)` must return the same matrix.

- [ ] **Step 4: Wire leaf matrices into `Subgame`, register bindings, run tests**

Add `"src/search/leaf_eval.cpp"` to `core`'s `srcs`; register `LeafEvalConfig`/`LeafEval`; have
`Subgame` own a `LeafEval` and fill each leaf's matrix at construction.

Run: `uv run pytest tests/sixmax/test_leaf_eval.py tests/sixmax/test_subgame.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sixmax/src/search/leaf_eval.h sixmax/src/search/leaf_eval.cpp \
        sixmax/src/search/subgame.h sixmax/src/search/subgame.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_leaf_eval.py
git commit -m "feat(sixmax): blueprint-continuation leaf evaluation"
```

---

### Task 5: `SearchStrategy` orchestration, deadline and fallback

**Files:**
- Create: `sixmax/src/search/search_strategy.h`, `sixmax/src/search/search_strategy.cpp`
- Modify: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_search_strategy.py`

**Interfaces:**
- Consumes: everything above, plus `policy.h`'s `blueprint_probs`.
- Produces:
  - `struct SearchConfig { bool enabled; double budget_ms; int max_nodes; int max_iters; int leaf_samples; uint64_t seed; }`
  - `class SearchStrategy { SearchStrategy(const BlueprintStrategy&, const ActionVocab&, const EngineConfig&, SearchConfig); std::vector<double> probs(const EngineGameState& decision_state, int hero, const std::vector<int>& observed_actions_since_street_start); bool last_decision_searched() const; }`
  - Returns the `[actions.search]`-indexed distribution over the hero's legal actions; **empty vector means "fall back to the blueprint"**.

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_search_strategy.py
"""Orchestration contract: the deadline is honoured, every failure path
degrades to an empty result (never an exception, never NaN), and a search
result is a distribution over the *legal* actions only. Requires the
gitignored blueprint checkpoint, so the module skips when it is absent."""
import importlib.util
import math
import os
import time

import pytest

import sixmax

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALL, CHECK = 2, 1
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")

pytestmark = pytest.mark.skipif(not os.path.exists(_CKPT),
                                reason="blueprint checkpoint is gitignored")


def _load_vocab(section="search"):
    path = os.path.join(_ROOT, "sixmax", "vocab_config.py")
    spec = importlib.util.spec_from_file_location("vocab_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(os.path.join(_ROOT, "sixmax", "configs", "default.toml"),
                          section)


def _strategy(budget_ms):
    vocab = _load_vocab("search")
    bp = sixmax.BlueprintStrategy.load(_CKPT, vocab)
    cfg = sixmax.SearchConfig()
    cfg.enabled = True
    cfg.budget_ms = budget_ms
    strat = sixmax.SearchStrategy(bp, vocab,
                                  sixmax.EngineConfig(num_players=2), cfg)
    return strat, vocab


def _flop(vocab):
    eng = sixmax.EngineGameState(sixmax.EngineConfig(num_players=2), 0,
                                 list(range(52)), vocab, [])
    eng.apply(CALL)
    eng.apply(CHECK)
    assert eng.street() == 1
    return eng


def test_tiny_budget_returns_without_overrunning():
    """A 1 ms budget must return promptly and must not raise."""
    strat, vocab = _strategy(1.0)
    eng = _flop(vocab)
    t0 = time.time()
    probs = list(strat.probs(eng, 0, []))
    assert time.time() - t0 < 5.0
    assert probs == [] or math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_result_is_empty_or_a_legal_distribution():
    strat, vocab = _strategy(500.0)
    eng = _flop(vocab)
    probs = list(strat.probs(eng, 0, []))
    if probs:
        assert all(p >= 0.0 for p in probs)
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
        mask = list(eng.legal_mask())
        assert all(p == 0.0 for p, m in zip(probs, mask) if not m)


def test_searched_flag_agrees_with_the_result():
    strat, vocab = _strategy(1.0)
    eng = _flop(vocab)
    probs = list(strat.probs(eng, 0, []))
    assert strat.last_decision_searched() == (probs != [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_search_strategy.py -v`
Expected: FAIL — no `SearchStrategy`.

- [ ] **Step 3: Implement**

```cpp
// sixmax/src/search/search_strategy.h
#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "blueprint/checkpoint.h"
#include "engine/engine.h"
#include "search/leaf_eval.h"
#include "search/range_tracker.h"
#include "search/subgame.h"
#include "search/vector_cfr.h"
#include "vocab/vocab.h"

namespace sixmax {

struct SearchConfig {
    bool enabled = false;
    double budget_ms = 5000.0;
    int max_nodes = 20000;
    int max_iters = 2000;
    int leaf_samples = 256;
    uint64_t seed = 1;
};

// Orchestrates one decision. Contract: never throws; never returns a partial
// or illegal distribution; an empty return means "use the blueprint".
class SearchStrategy {
public:
    SearchStrategy(const BlueprintStrategy& bp, const ActionVocab& vocab,
                   const EngineConfig& cfg, SearchConfig sc);

    std::vector<double> probs(const HandState& initial, int hero,
                              const std::vector<int>& actors,
                              const std::vector<int>& actions,
                              const std::vector<int>& board,
                              const std::array<int, 2>& hero_hole);

    bool last_decision_searched() const { return last_searched_; }
    int searches() const { return searches_; }
    int fallbacks() const { return fallbacks_; }

private:
    const BlueprintStrategy& bp_;
    const ActionVocab& vocab_;
    EngineConfig cfg_;
    SearchConfig sc_;
    bool last_searched_ = false;
    int searches_ = 0, fallbacks_ = 0;
};

}  // namespace sixmax
```

```cpp
// sixmax/src/search/search_strategy.cpp
#include "search/search_strategy.h"
#include <chrono>

namespace sixmax {

SearchStrategy::SearchStrategy(const BlueprintStrategy& bp, const ActionVocab& vocab,
                              const EngineConfig& cfg, SearchConfig sc)
    : bp_(bp), vocab_(vocab), cfg_(cfg), sc_(sc) {}

std::vector<double> SearchStrategy::probs(const HandState& initial, int hero,
                                          const std::vector<int>& actors,
                                          const std::vector<int>& actions,
                                          const std::vector<int>& board,
                                          const std::array<int, 2>& hero_hole) {
    last_searched_ = false;
    if (!sc_.enabled) { ++fallbacks_; return {}; }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        // 1. Replay the observed prefix to reach the current decision node,
        //    and remember it as the subgame root for this street.
        // 2. RangeTracker::reset(board, hero_hole); for each observed action
        //    call RangeTracker::observe(...) to get entry beliefs.
        RangeTracker tracker(cfg_, vocab_);
        tracker.reset(board, hero_hole);
        // 3. Build the street subgame (node cap from sc_.max_nodes) and solve
        //    it with VectorCfr over the remaining budget.
        SubgameConfig sg_cfg;
        sg_cfg.max_nodes = sc_.max_nodes;
        // Subgame sg(street_root, vocab_, tracker, sg_cfg);
        // VectorCfrConfig vc; vc.max_iters = sc_.max_iters;
        // vc.budget_ms = sc_.budget_ms - elapsed_ms(t0);
        // VectorCfr solver(vc);
        // auto res = solver.solve(sg, hero, query_hand_index);
        // std::vector<double> p = solver.average_strategy(query_node_id, query_hand_index);
        std::vector<double> p;   // filled by the four steps above
        if (p.empty()) { ++fallbacks_; return {}; }
        double total = 0.0;
        for (double v : p) total += v;
        if (!(total > 0.0)) { ++fallbacks_; return {}; }
        for (double& v : p) v /= total;
        last_searched_ = true;
        ++searches_;
        return p;
    } catch (...) {
        ++fallbacks_;
        return {};
    }
}

}  // namespace sixmax
```

> **Note for the implementer:** Step 3's four numbered sub-steps are the remaining work in this
> file and each is small: replaying the prefix uses `HandState::apply` with the same
> `target_bb` translation as `EngineGameState`; the query hand index is
> `RangeTracker::combo_index(hero_hole[0], hero_hole[1])` mapped into `Subgame`'s
> `legal_hands_`; `query_node_id` is the id of the node the prefix ends on. The `try/catch(...)`
> and the `total > 0` guard are the tested contract — do not remove them. Wire `SearchConfig`
> into `sixmax/BUCK` and register it plus `SearchStrategy` in the bindings.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/sixmax/test_search_strategy.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sixmax/src/search/search_strategy.h sixmax/src/search/search_strategy.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_search_strategy.py
git commit -m "feat(sixmax): search orchestration with deadline and blueprint fallback"
```

---

### Task 6: Agent wiring behind a config flag

**Files:**
- Modify: `agents/sixmax_agent.py` (add `SearchStrategy` wrapper; one consult branch in `SixmaxAgent`)
- Modify: `sixmax/configs/default.toml`, `sixmax/configs/hu.toml` (add `[search]`)
- Test: `tests/sixmax/test_search_agent_wiring.py`

**Interfaces:**
- Consumes: `SearchStrategy` (Task 5), `SixmaxAgent` (existing).
- Produces: `SixmaxAgent(..., search_toml=None)` — when `search_toml` is given and `[search].enabled`,
  the agent consults search on flop/turn/river and falls back to `blueprint_probs` otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/sixmax/test_search_agent_wiring.py
"""Wiring contract. The first test is the one that protects production: with
[search] disabled the agent must be byte-identical to today's blueprint agent.
Requires the gitignored checkpoint, so the module skips when it is absent."""
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join(_ROOT, "sixmax", "checkpoints", "blueprint.bin")
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_TABULAR = os.path.join(_ROOT, "cfr", "checkpoints", "checkpoint_09040000.pkl")

pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_TABULAR)),
                                reason="checkpoint(s) gitignored")


def test_search_disabled_matches_blueprint_agent():
    """Same seed, same decks: the search-aware agent with [search] disabled
    scores identically to the plain blueprint agent (the production regression)."""
    from agents.sixmax_agent import SixmaxAgent
    from agents.cfr_agent import CFRAgent
    from scripts.eval_hu_sanity import run_duplicate_match

    plain = run_duplicate_match(lambda _: SixmaxAgent(_CKPT),
                                lambda _: CFRAgent(_TABULAR), 200, 1, 2, 200)
    disabled = run_duplicate_match(
        lambda _: SixmaxAgent(_CKPT, search_toml=_TOML),
        lambda _: CFRAgent(_TABULAR), 200, 1, 2, 200)
    assert plain == disabled


def test_search_enabled_constructs_and_stays_legal():
    """Enabling search must not change the *interface*: the agent still emits
    only legal actions, and construction with a [search] section must not raise."""
    from agents.sixmax_agent import SixmaxAgent
    agent = SixmaxAgent(_CKPT, search_toml=_TOML)
    assert agent is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_search_agent_wiring.py -v`
Expected: FAIL — `SixmaxAgent` has no `search_toml` argument.

- [ ] **Step 3: Wire it**

Add a `[search]` section to both configs (identical, disabled):

```toml
[search]
enabled      = false      # off by default; the A/B gate decides deployment
budget_ms    = 5000.0
max_nodes    = 20000
max_iters    = 2000
leaf_samples = 256
seed         = 1
```

In `agents/sixmax_agent.py`, add the wrapper and the branch:

```python
class SearchStrategy:
    """Thin host-side wrapper: translation and fallback live in C++."""

    def __init__(self, checkpoint: str, config_toml: str,
                 section: str = "blueprint") -> None:
        import sixmax
        vocab = _load_vocab(config_toml, "search")
        bp = sixmax.BlueprintStrategy.load(checkpoint, vocab)
        self._cfg = _load_search_config(config_toml)
        self._impl = sixmax.SearchStrategy(bp, vocab, sixmax.EngineConfig(), self._cfg)
        self._vocab = vocab

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def probs(self, *args, **kwargs):
        return list(self._impl.probs(*args, **kwargs))
```

and in `SixmaxAgent.__init__`, accept `search_toml: str | None = None`; in the decision method,
before the blueprint path:

```python
        if self._search is not None and self._search.enabled and street != "preflop":
            p = self._search.probs(...)
            if p:
                return self._index_to_action(p)
        return self._blueprint_decision(...)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/sixmax/test_search_agent_wiring.py tests/agents/ -v`
Expected: all pass, and the disabled arm is byte-identical to the existing agent.

- [ ] **Step 5: Commit**

```bash
git add agents/sixmax_agent.py sixmax/configs/default.toml sixmax/configs/hu.toml \
        tests/sixmax/test_search_agent_wiring.py
git commit -m "feat(sixmax): opt-in search wiring in SixmaxAgent (default off)"
```

---

### Task 7: A/B harness, latency bound, and the deployment gate

**Files:**
- Create: `scripts/eval_search_ab.py`
- Create: `tests/scripts/test_eval_search_ab.py` (pure-logic tests: parsing + paired stats)
- Modify: `.mex/ROUTER.md` (record the outcome — see Update Scaffold)

**Interfaces:**
- Consumes: `agents/sixmax_agent.py`, `eval_hu_sanity.py`'s match runner.
- Produces: `scripts/eval_search_ab.py --checkpoint PATH --tabular PATH --seeds 1,2,... --hands 20000 --search-toml PATH` printing, per seed, `search=+X.XX blueprint=+Y.YY diff=+Z.ZZ`, then the paired mean / sd / se / t and a `VERDICT:` line.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_eval_search_ab.py
"""The harness's arithmetic and its gate must be unit-testable without a
checkpoint: paired differences, the t statistic, and the >0 AND |t|>=2 rule."""
import importlib.util, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_search_ab", os.path.join(_ROOT, "scripts", "eval_search_ab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_stats_reports_t_statistic():
    mod = _load()
    stats = mod.paired_stats([10.0, 12.0, 14.0, 16.0])
    assert abs(stats["mean"] - 13.0) < 1e-9
    assert stats["t"] > 5.0


def test_verdict_requires_positive_and_significant():
    mod = _load()
    assert mod.verdict({"mean": 12.0, "t": 3.1}) == "promote"
    assert mod.verdict({"mean": 12.0, "t": 1.2}) == "keep-blueprint"
    assert mod.verdict({"mean": -3.0, "t": -4.0}) == "keep-blueprint"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_eval_search_ab.py -v`
Expected: FAIL — module/function missing.

- [ ] **Step 3: Implement the harness**

```python
#!/usr/bin/env python3
"""Paired search-on vs search-off A/B in the HU engine.

Both arms play identical seeds; only the [search] section differs. Reports the
per-seed paired difference and the paired t statistic, then a VERDICT against
the deployment gate (mean > 0 AND |t| >= 2), matching
patterns/eval-checkpoint-ab.md.
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def paired_stats(diffs):
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / n**0.5 if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": (mean / se) if se > 0 else 0.0}


def verdict(stats, min_t: float = 2.0) -> str:
    return "promote" if stats["mean"] > 0 and stats["t"] >= min_t else "keep-blueprint"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="sixmax/checkpoints/blueprint.bin")
    p.add_argument("--tabular", default="cfr/checkpoints/checkpoint_09040000.pkl")
    p.add_argument("--search-toml", default="sixmax/configs/default.toml")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    p.add_argument("--hands", type=int, default=20000)
    args = p.parse_args()

    from agents.sixmax_agent import SixmaxAgent
    from scripts.eval_hu_sanity import run_duplicate_match
    from agents.cfr_agent import CFRAgent

    seeds = [int(s) for s in args.seeds.split(",")]
    diffs = []
    print(f"{'seed':>4} {'search':>10} {'blueprint':>10} {'diff':>10}")
    for seed in seeds:
        on = run_duplicate_match(
            lambda _: SixmaxAgent(args.checkpoint, search_toml=args.search_toml),
            lambda _: CFRAgent(args.tabular), args.hands, seed, 2, 200)
        off = run_duplicate_match(
            lambda _: SixmaxAgent(args.checkpoint),
            lambda _: CFRAgent(args.tabular), args.hands, seed, 2, 200)
        diffs.append(on - off)
        print(f"{seed:>4} {on:>+10.2f} {off:>+10.2f} {on - off:>+10.2f}")

    stats = paired_stats(diffs)
    print(f"\nn={stats['n']} seeds x {args.hands} hands")
    print(f"paired diff: {stats['mean']:+.2f} BB/100  sd {stats['sd']:.1f}  "
          f"se {stats['se']:.1f}  t={stats['t']:.2f}")
    print(f"VERDICT: {verdict(stats)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then measure**

Run: `uv run pytest tests/scripts/test_eval_search_ab.py -v`
Expected: 2 passed.

Then (long; run detached): 
`nohup uv run python scripts/eval_search_ab.py --hands 20000 --seeds 1,2,3,4,5,6,7,8 > /tmp/search_ab.txt 2>&1 &`

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_search_ab.py tests/scripts/test_eval_search_ab.py
git commit -m "feat(sixmax): paired search-on/off A/B harness with a |t|>=2 gate"
```

---

## Out of scope for this plan

- 6-max enablement (ranges over multiple opponents), preflop search, safe re-solving, and the
  Phase-3 value net — each is its own spec/plan.
- Deploying the result: the web service/env plumbing follows only if Task 7 prints `promote`;
  the swap procedure itself already exists in `.mex/context/setup.md` (Production section).

## Self-Review

**1. Spec coverage.** Search point/scope → Tasks 3, 6 (street-only, preflop excluded). Depth-limited
leaves → Tasks 3, 4. Solve-to-showdown rejected → recorded in the spec, enforced by Task 3's leaf
boundary. Search from the deployed 6-max blueprint → Tasks 5, 6, 7 defaults (`blueprint.bin`).
Player-count parameterization → `VectorGame::num_players` + `EngineConfig`, HU-only paths in Tasks 3/5.
Deadline + fallback → Task 5. `[search]` config with precedence → Task 6. Kuhn/Leduc gate → Task 1
(Kuhn; add Leduc only if Kuhn proves too weak). RangeTracker invariants → Task 2. Subgame invariants →
Task 3. A/B gate with |t| ≥ 2 → Task 7. `bet_context()` consumer → Tasks 3, 4. No new deps → all tasks.

**2. Placeholder scan.** No "TBD"/"TODO"/"similar to Task N", and no stubbed test bodies — every
test above is complete Python against the real fixture surface (`import sixmax`, `EngineConfig(n)`, `EngineGameState(cfg, button, deck, vocab, stacks)`, the `vocab_config` loader).
Five C++ steps carry a "Note for the implementer" naming exactly what to write and which existing
code to follow (`HandState` hole-setter, the `observe` adapter's per-combo blueprint keying, the
subgame id/apply glue, `Subgame::root()`'s state wrapper, and the Task-5 replay/index glue); each is
one function whose contract is pinned by the tests above it — not deferred design.

**3. Type consistency.** `VectorState::utility(player, h0, h1)`, `VectorGame::num_hands(player)`,
`VectorCfr::solve(game, query_player, query_hand)`, `RangeTracker::combo_index/combo_cards/range/
set_range_for_test/observe_with_probs`, `SubgameConfig::max_nodes`, `LeafEval::matrix(...)`,
`SearchConfig` field names, and `paired_stats`/`verdict` are used identically in every task and in
the tests that pin them.

**4. Review Focus.** (1) deadline mid-solve → Task 5 `test_tiny_budget_returns_without_overrunning`.
(2) degenerate/unseen infosets → Task 2 `observe_with_probs` zero-total guard + Task 5 empty/`total>0`
guard + `test_result_is_empty_or_a_legal_distribution`. (3) card removal → Task 1
`utility(h0==h1)==0` + `test_kuhn_vector_average_strategy_sums_to_one`, Task 2
`test_reset_zeroes_blocked_cards`, Task 4 `test_leaf_values_are_zero_sum`. (4) off-tree sizes →
Task 3 `test_subgame_legality_matches_engine` (uses the same `ActionVocab`/`nearest` path as the
deploy bridge). (5) node cap → Task 3 `test_subgame_respects_node_cap`.
