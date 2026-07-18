# Six-Max Phase 1a — Solver Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The correctness core of the six-max blueprint: a generic MCCFR game interface, a Kuhn-poker fixture that reproduces the closed-form equilibrium, a 2–6 player NLHE engine validated against the Python engine's side-pot semantics, and the engine↔ActionVocab bridge with legality masking.

**Architecture:** Everything is C++ under `sixmax/src/` (Buck2 target `//sixmax:sixmax`, pybind11 module `sixmax`), tested from Python via `tests/sixmax/`. The MCCFR trainer works against an abstract `Game`/`GameState` interface; Kuhn poker and the NLHE engine are two implementations of it. Phase 1b adds card/history abstraction, multithreading, and checkpoints on top of these interfaces — the naive infoset keyer in Task 5 is the one deliberate placeholder seam.

**Tech Stack:** C++17, Buck2 (`~/bin/buck2`), pybind11, pytest via `uv run`.

**Parent spec:** `docs/superpowers/specs/2026-07-17-sixmax-search-design.md` (Phase 1 section). Phase 1b (abstraction + scaled training + deployment) is a separate plan, written after this one lands.

## Global Constraints

- Chip frame: all chip quantities are `double` in big blinds — `big_blind = 1.0`, `small_blind = 0.5`, default `starting_stack = 100.0`. Comparisons use `kChipEps = 1e-9`.
- Action vocabulary: canonical order is config order (`fold, check, call, preflop_opens…, bet_sizes…, allin`); legality is expressed **only by masking** — never reorder or filter storage vectors. Strategy/regret vectors are sized `vocab.size()`.
- The default blueprint vocab (`sixmax/configs/default.toml`, section `blueprint`) has 10 entries: indices `0=fold, 1=check, 2=call, 3=open2.5bb, 4=open3.5bb, 5=open5.0bb, 6=bet0.33pot, 7=bet0.75pot, 8=bet1.5pot, 9=allin`.
- `BetContext` conventions (fixed by Phase 0 `vocab.cpp`, do not change): `stack` = the actor's **maximum raise-to total** for this street (`street_bet + remaining stack`); `to_call` = unmatched amount faced (`current_bet − street_bet`, uncapped); `pot` = total chips on the table **minus** `to_call`. `target_bb` for Pot-unit bets is `current_bet + size*(pot + 2*to_call)`.
- Hand evaluation only via the opaque `safe_eval` API (`rank7`, `HandRank::beats/ties`); raw scores are never exposed. Include as `#include "game/safe_eval.h"` (exported from `//common:evaluator`).
- Module isolation: `sixmax/` never imports `game/poker.py`, `cfr/`, or `neural_cfr/`. Cross-validation against the Python engine happens only in `tests/` (allowed — tests are the designated bridge zone).
- Linear MCCFR weighting: regret and strategy-sum updates at iteration `t` are multiplied by `t`.
- Build: `~/bin/buck2 build //sixmax:sixmax`. Test: `uv run pytest tests/sixmax/ -v` from the repo root (`tests/sixmax/conftest.py` auto-builds and force-loads the `.so` — never `import sixmax` from a script run outside pytest without replicating that loader).
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
sixmax/src/
  blueprint/game.h            # Task 1: abstract Game/GameState (header-only)
  blueprint/kuhn.h / .cpp     # Task 1: Kuhn poker fixture
  blueprint/mccfr.h / .cpp    # Task 2: external-sampling MCCFR, linear weighting
  engine/engine.h / .cpp      # Task 3: 2–6 player NLHE engine + settle_pots
  blueprint/engine_game.h/.cpp# Task 5: engine ↔ ActionVocab bridge
  bindings/bindings.cpp       # modified in Tasks 1, 2, 3, 5
sixmax/BUCK                   # srcs list grows in Tasks 1, 2, 3, 5
tests/sixmax/
  test_kuhn.py                # Task 1
  test_mccfr_kuhn.py          # Task 2
  test_engine.py              # Task 3
  test_settlement.py          # Task 4 (vs Python PotManager)
  test_engine_game.py         # Task 5
```

---

### Task 1: Game interface + Kuhn poker fixture

**Files:**
- Create: `sixmax/src/blueprint/game.h`
- Create: `sixmax/src/blueprint/kuhn.h`, `sixmax/src/blueprint/kuhn.cpp`
- Modify: `sixmax/BUCK` (add `src/blueprint/kuhn.cpp` to `srcs`)
- Modify: `sixmax/src/bindings/bindings.cpp` (bind `GameState`, `Game`, `KuhnState`, `KuhnGame`, `kuhn_infoset_key`)
- Test: `tests/sixmax/test_kuhn.py`

**Interfaces:**
- Consumes: nothing new (pybind/Buck patterns from Phase 0).
- Produces: `sixmax::GameState` (virtuals `is_terminal, current_player, legal_mask, infoset_key, apply, utility, clone`) and `sixmax::Game` (`num_players, num_actions, new_hand(rng)`) — Tasks 2 and 5 build directly on these exact signatures. Python: `sixmax.KuhnState(card0, card1)`, `sixmax.KuhnGame()`, `sixmax.kuhn_infoset_key(card, history_code)`.

- [ ] **Step 1: Write the failing test**

Create `tests/sixmax/test_kuhn.py`:

```python
"""Kuhn poker fixture: 2 players, cards {0=J,1=Q,2=K}, ante 1 BB, bet 1 BB.
Actions: 0 = check/fold ("pass"), 1 = bet/call."""
import sixmax


def test_check_check_showdown():
    s = sixmax.KuhnState(2, 0)  # P0 has K, P1 has J
    assert not s.is_terminal()
    assert s.current_player() == 0
    s.apply(0)
    assert s.current_player() == 1
    s.apply(0)
    assert s.is_terminal()
    assert s.utility(0) == 1.0   # showdown for the antes
    assert s.utility(1) == -1.0


def test_bet_fold_pays_bettor():
    s = sixmax.KuhnState(0, 2)  # P0 has J (bluffs), P1 has K but folds
    s.apply(1)
    s.apply(0)
    assert s.is_terminal()
    assert s.utility(0) == 1.0


def test_bet_call_showdown_for_two():
    s = sixmax.KuhnState(1, 2)  # P0 Q bets, P1 K calls
    s.apply(1)
    s.apply(1)
    assert s.is_terminal()
    assert s.utility(0) == -2.0
    assert s.utility(1) == 2.0


def test_check_bet_fold_and_call():
    s = sixmax.KuhnState(2, 1)
    s.apply(0)
    s.apply(1)
    assert not s.is_terminal()
    assert s.current_player() == 0
    fold = sixmax.KuhnState(2, 1)
    fold.apply(0); fold.apply(1); fold.apply(0)
    assert fold.is_terminal() and fold.utility(0) == -1.0
    call = sixmax.KuhnState(2, 1)
    call.apply(0); call.apply(1); call.apply(1)
    assert call.is_terminal() and call.utility(0) == 2.0


def test_legal_mask_always_both():
    s = sixmax.KuhnState(0, 1)
    assert s.legal_mask() == [1, 1]


def test_infoset_keys_hide_opponent_card():
    # History codes: 0="" 1="check" 2="bet" 3="check,bet".
    a = sixmax.KuhnState(1, 0)
    b = sixmax.KuhnState(1, 2)
    assert a.infoset_key() == b.infoset_key() == sixmax.kuhn_infoset_key(1, 0)
    a.apply(0); b.apply(0)
    # P1's key depends on P1's card, which differs.
    assert a.infoset_key() == sixmax.kuhn_infoset_key(0, 1)
    assert b.infoset_key() == sixmax.kuhn_infoset_key(2, 1)
    assert a.infoset_key() != b.infoset_key()


def test_game_deals_valid_hands():
    g = sixmax.KuhnGame()
    assert g.num_players() == 2
    assert g.num_actions() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_kuhn.py -v`
Expected: FAIL / ERROR with `AttributeError: module 'sixmax' has no attribute 'KuhnState'`

- [ ] **Step 3: Implement the interface and Kuhn**

Create `sixmax/src/blueprint/game.h`:

```cpp
#pragma once
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace sixmax {

// Abstract sequential game for MCCFR. Chance (the deal) happens once in
// Game::new_hand(); states after that are deterministic in applied actions.
class GameState {
public:
    virtual ~GameState() = default;
    virtual bool is_terminal() const = 0;
    virtual int current_player() const = 0;      // valid iff !is_terminal()
    // mask is assigned to size Game::num_actions(); 1 = legal.
    virtual void legal_mask(std::vector<uint8_t>& mask) const = 0;
    virtual uint64_t infoset_key() const = 0;    // for current_player()
    virtual void apply(int action) = 0;          // action must be legal
    virtual double utility(int player) const = 0;  // BB; valid iff terminal
    virtual std::unique_ptr<GameState> clone() const = 0;
};

class Game {
public:
    virtual ~Game() = default;
    virtual int num_players() const = 0;
    virtual int num_actions() const = 0;         // fixed action-vector width
    virtual std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) = 0;
};

}  // namespace sixmax
```

Create `sixmax/src/blueprint/kuhn.h`:

```cpp
#pragma once
#include "blueprint/game.h"

namespace sixmax {

// Kuhn poker: cards {0=J,1=Q,2=K}, ante 1 BB each, bet size 1 BB.
// Actions: 0 = check/fold ("pass"), 1 = bet/call. Closed-form equilibrium:
// P0 bets J with alpha in [0,1/3], K with 3*alpha, never Q; value -1/18.
class KuhnState : public GameState {
public:
    KuhnState(int card0, int card1) { cards_[0] = card0; cards_[1] = card1; }
    bool is_terminal() const override;
    int current_player() const override { return (int)history_.size() % 2; }
    void legal_mask(std::vector<uint8_t>& mask) const override { mask.assign(2, 1); }
    uint64_t infoset_key() const override;
    void apply(int action) override { history_.push_back(action); }
    double utility(int player) const override;
    std::unique_ptr<GameState> clone() const override {
        return std::make_unique<KuhnState>(*this);
    }
    // History codes for keys: 0="" 1="check" 2="bet" 3="check,bet".
    static uint64_t key_for(int card, int history_code) {
        return (uint64_t)card * 4 + (uint64_t)history_code;
    }

private:
    int cards_[2];
    std::vector<int> history_;
};

class KuhnGame : public Game {
public:
    int num_players() const override { return 2; }
    int num_actions() const override { return 2; }
    std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) override;
};

}  // namespace sixmax
```

Create `sixmax/src/blueprint/kuhn.cpp`:

```cpp
#include "blueprint/kuhn.h"

namespace sixmax {

bool KuhnState::is_terminal() const {
    const auto& h = history_;
    if (h.size() < 2) return false;
    if (h.size() == 2) return !(h[0] == 0 && h[1] == 1);  // "check,bet" continues
    return true;  // length 3: "check,bet,{fold|call}"
}

uint64_t KuhnState::infoset_key() const {
    int code = 0;
    if (history_.size() == 1) code = history_[0] == 0 ? 1 : 2;
    else if (history_.size() == 2) code = 3;  // must be "check,bet"
    return key_for(cards_[current_player()], code);
}

double KuhnState::utility(int player) const {
    const auto& h = history_;
    int winner = cards_[0] > cards_[1] ? 0 : 1;
    double p0;
    if (h.size() == 2 && h[0] == 0 && h[1] == 0) p0 = winner == 0 ? 1 : -1;  // cc
    else if (h.size() == 2 && h[0] == 1 && h[1] == 0) p0 = 1;                // b,f
    else if (h.size() == 2) p0 = winner == 0 ? 2 : -2;                       // b,c
    else if (h[2] == 0) p0 = -1;                                             // c,b,f
    else p0 = winner == 0 ? 2 : -2;                                          // c,b,c
    return player == 0 ? p0 : -p0;
}

std::unique_ptr<GameState> KuhnGame::new_hand(std::mt19937_64& rng) {
    int c0 = (int)(rng() % 3);
    int c1 = (int)(rng() % 2);
    if (c1 >= c0) ++c1;  // uniform ordered pair without replacement
    return std::make_unique<KuhnState>(c0, c1);
}

}  // namespace sixmax
```

In `sixmax/BUCK`, change the `srcs` line to:

```python
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp"],
```

In `sixmax/src/bindings/bindings.cpp`, add after the existing includes:

```cpp
#include "blueprint/game.h"
#include "blueprint/kuhn.h"
```

and append inside `PYBIND11_MODULE(sixmax, m)` (before the closing brace):

```cpp
    // --- MCCFR game interface (Task 1) ---
    py::class_<sixmax::GameState>(m, "GameState")
        .def("is_terminal", &sixmax::GameState::is_terminal)
        .def("current_player", &sixmax::GameState::current_player)
        .def("legal_mask", [](const sixmax::GameState& s) {
            std::vector<uint8_t> mask;
            s.legal_mask(mask);
            return std::vector<int>(mask.begin(), mask.end());
        })
        .def("infoset_key", &sixmax::GameState::infoset_key)
        .def("apply", &sixmax::GameState::apply)
        .def("utility", &sixmax::GameState::utility);
    py::class_<sixmax::Game>(m, "Game")
        .def("num_players", &sixmax::Game::num_players)
        .def("num_actions", &sixmax::Game::num_actions);
    py::class_<sixmax::KuhnState, sixmax::GameState>(m, "KuhnState")
        .def(py::init<int, int>(), py::arg("card0"), py::arg("card1"));
    py::class_<sixmax::KuhnGame, sixmax::Game>(m, "KuhnGame")
        .def(py::init<>());
    m.def("kuhn_infoset_key", &sixmax::KuhnState::key_for,
          py::arg("card"), py::arg("history_code"));
```

- [ ] **Step 4: Build and run test to verify it passes**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_kuhn.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the existing sixmax tests for regressions**

Run: `uv run pytest tests/sixmax/ -v`
Expected: all pass (existing safe_eval/vocab tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/blueprint/game.h sixmax/src/blueprint/kuhn.h \
        sixmax/src/blueprint/kuhn.cpp sixmax/BUCK \
        sixmax/src/bindings/bindings.cpp tests/sixmax/test_kuhn.py
git commit -m "feat(sixmax): MCCFR game interface + Kuhn poker fixture"
```

---

### Task 2: External-sampling MCCFR trainer + Kuhn equilibrium validation

**Files:**
- Create: `sixmax/src/blueprint/mccfr.h`, `sixmax/src/blueprint/mccfr.cpp`
- Modify: `sixmax/BUCK` (add `src/blueprint/mccfr.cpp`)
- Modify: `sixmax/src/bindings/bindings.cpp` (bind `MCCFRTrainer`, `kuhn_exact_value`)
- Test: `tests/sixmax/test_mccfr_kuhn.py`

**Interfaces:**
- Consumes: `sixmax::Game` / `sixmax::GameState` from Task 1 (exact virtuals listed there); `KuhnState::key_for`.
- Produces: `sixmax::MCCFRTrainer(Game&, uint64_t seed)` with `void train(uint64_t iterations)`, `uint64_t iterations() const`, `size_t num_infosets() const`, `std::vector<double> average_strategy(uint64_t key) const` (empty vector if key unseen or unaccumulated). Task 5's smoke test and all of Phase 1b use this exact class. Also `double kuhn_exact_value(const MCCFRTrainer&)`.

- [ ] **Step 1: Write the failing test**

Create `tests/sixmax/test_mccfr_kuhn.py`:

```python
"""External-sampling MCCFR must reproduce Kuhn poker's closed-form
equilibrium: game value -1/18 for P0; P0 bets J with alpha in [0,1/3],
K with 3*alpha, never Q; P1 folds J to a bet, always calls K."""
import sixmax

J, Q, K = 0, 1, 2
ROOT, AFTER_CHECK, AFTER_BET = 0, 1, 2


def _trained(seed=7, iters=200_000):
    g = sixmax.KuhnGame()
    t = sixmax.MCCFRTrainer(g, seed)
    t.train(iters)
    return t


def _bet_prob(t, card, hist):
    sigma = t.average_strategy(sixmax.kuhn_infoset_key(card, hist))
    assert len(sigma) == 2 and abs(sum(sigma) - 1.0) < 1e-9
    return sigma[1]


def test_game_value_converges():
    t = _trained()
    assert abs(sixmax.kuhn_exact_value(t) - (-1.0 / 18.0)) < 0.01


def test_equilibrium_strategy_structure():
    t = _trained()
    alpha = _bet_prob(t, J, ROOT)
    assert 0.0 <= alpha <= 1.0 / 3.0 + 0.05
    assert abs(_bet_prob(t, K, ROOT) - 3.0 * alpha) < 0.10
    assert _bet_prob(t, Q, ROOT) < 0.05          # never bet Q first
    assert _bet_prob(t, J, AFTER_BET) < 0.05     # P1 folds J to a bet
    assert _bet_prob(t, K, AFTER_BET) > 0.95     # P1 always calls K
    assert abs(_bet_prob(t, Q, AFTER_BET) - 1.0 / 3.0) < 0.07  # call Q 1/3


def test_deterministic_given_seed():
    a, b = _trained(seed=3, iters=5_000), _trained(seed=3, iters=5_000)
    assert a.num_infosets() == b.num_infosets() == 12
    for card in (J, Q, K):
        for hist in (ROOT, AFTER_CHECK, AFTER_BET, 3):
            key = sixmax.kuhn_infoset_key(card, hist)
            assert a.average_strategy(key) == b.average_strategy(key)


def test_unseen_key_returns_empty():
    g = sixmax.KuhnGame()
    t = sixmax.MCCFRTrainer(g, 1)
    assert t.average_strategy(12345) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_mccfr_kuhn.py -v`
Expected: ERROR with `AttributeError: module 'sixmax' has no attribute 'MCCFRTrainer'`

- [ ] **Step 3: Implement the trainer**

Create `sixmax/src/blueprint/mccfr.h`:

```cpp
#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include "blueprint/game.h"

namespace sixmax {

struct InfosetData {
    std::vector<double> regret;
    std::vector<double> strategy_sum;
};

// External-sampling MCCFR with linear weighting: regret and strategy-sum
// updates at iteration t are multiplied by t. Single-threaded; the
// multithreaded trainer with abstraction-based keys is Phase 1b.
class MCCFRTrainer {
public:
    MCCFRTrainer(Game& game, uint64_t seed) : game_(game), rng_(seed) {}
    void train(uint64_t iterations);  // one traversal per player per iteration
    uint64_t iterations() const { return iter_; }
    size_t num_infosets() const { return table_.size(); }
    // Normalized average strategy for a visited infoset; empty if unseen
    // or never accumulated. Illegal actions hold probability 0 because
    // strategy_sum only ever accumulates on masked-legal actions.
    std::vector<double> average_strategy(uint64_t key) const;

private:
    double traverse(GameState& s, int traverser);
    std::vector<double> matched_strategy(const InfosetData& d,
                                         const std::vector<uint8_t>& mask) const;
    Game& game_;
    std::unordered_map<uint64_t, InfosetData> table_;
    std::mt19937_64 rng_;
    uint64_t iter_ = 0;
    double weight_ = 0.0;
};

// Exact EV for player 0 under the trainer's average strategy, by
// enumerating all 6 Kuhn deals (uniform strategy where unaccumulated).
// Test helper for the closed-form value -1/18.
double kuhn_exact_value(const MCCFRTrainer& t);

}  // namespace sixmax
```

Create `sixmax/src/blueprint/mccfr.cpp`:

```cpp
#include "blueprint/mccfr.h"
#include "blueprint/kuhn.h"

namespace sixmax {

std::vector<double> MCCFRTrainer::matched_strategy(
        const InfosetData& d, const std::vector<uint8_t>& mask) const {
    int n = (int)d.regret.size();
    std::vector<double> sigma(n, 0.0);
    double pos = 0.0;
    for (int a = 0; a < n; ++a)
        if (mask[a] && d.regret[a] > 0.0) pos += d.regret[a];
    if (pos > 0.0) {
        for (int a = 0; a < n; ++a)
            if (mask[a] && d.regret[a] > 0.0) sigma[a] = d.regret[a] / pos;
    } else {
        int legal = 0;
        for (int a = 0; a < n; ++a) legal += mask[a] ? 1 : 0;
        for (int a = 0; a < n; ++a) if (mask[a]) sigma[a] = 1.0 / legal;
    }
    return sigma;
}

void MCCFRTrainer::train(uint64_t iterations) {
    for (uint64_t i = 0; i < iterations; ++i) {
        ++iter_;
        weight_ = (double)iter_;  // linear CFR weighting
        for (int t = 0; t < game_.num_players(); ++t) {
            auto s = game_.new_hand(rng_);
            traverse(*s, t);
        }
    }
}

double MCCFRTrainer::traverse(GameState& s, int traverser) {
    if (s.is_terminal()) return s.utility(traverser);
    int n = game_.num_actions();
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    InfosetData& d = table_[s.infoset_key()];
    if (d.regret.empty()) {
        d.regret.assign(n, 0.0);
        d.strategy_sum.assign(n, 0.0);
    }
    std::vector<double> sigma = matched_strategy(d, mask);

    if (s.current_player() == traverser) {
        std::vector<double> u(n, 0.0);
        double ev = 0.0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            auto child = s.clone();
            child->apply(a);
            u[a] = traverse(*child, traverser);
            ev += sigma[a] * u[a];
        }
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.regret[a] += weight_ * (u[a] - ev);
        return ev;
    }
    // Opponent/chance-free node: accumulate average strategy, sample one.
    for (int a = 0; a < n; ++a)
        if (mask[a]) d.strategy_sum[a] += weight_ * sigma[a];
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng_), acc = 0.0;
    int chosen = -1;
    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        acc += sigma[a];
        chosen = a;
        if (r <= acc) break;
    }
    s.apply(chosen);
    return traverse(s, traverser);
}

std::vector<double> MCCFRTrainer::average_strategy(uint64_t key) const {
    auto it = table_.find(key);
    if (it == table_.end()) return {};
    const auto& ss = it->second.strategy_sum;
    double total = 0.0;
    for (double v : ss) total += v;
    if (total <= 0.0) return {};
    std::vector<double> out(ss.size());
    for (size_t a = 0; a < ss.size(); ++a) out[a] = ss[a] / total;
    return out;
}

namespace {
double kuhn_ev_p0(const MCCFRTrainer& t, const KuhnState& s) {
    if (s.is_terminal()) return s.utility(0);
    std::vector<double> sigma = t.average_strategy(s.infoset_key());
    if (sigma.empty()) sigma = {0.5, 0.5};
    double ev = 0.0;
    for (int a = 0; a < 2; ++a) {
        if (sigma[a] <= 0.0) continue;
        KuhnState child = s;
        child.apply(a);
        ev += sigma[a] * kuhn_ev_p0(t, child);
    }
    return ev;
}
}  // namespace

double kuhn_exact_value(const MCCFRTrainer& t) {
    double total = 0.0;
    for (int c0 = 0; c0 < 3; ++c0)
        for (int c1 = 0; c1 < 3; ++c1)
            if (c0 != c1) total += kuhn_ev_p0(t, KuhnState(c0, c1));
    return total / 6.0;
}

}  // namespace sixmax
```

In `sixmax/BUCK`, change `srcs` to:

```python
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp", "src/blueprint/mccfr.cpp"],
```

In `sixmax/src/bindings/bindings.cpp`, add `#include "blueprint/mccfr.h"` and append inside the module body:

```cpp
    // --- MCCFR trainer (Task 2) ---
    py::class_<sixmax::MCCFRTrainer>(m, "MCCFRTrainer")
        .def(py::init<sixmax::Game&, uint64_t>(),
             py::arg("game"), py::arg("seed"),
             py::keep_alive<1, 2>())  // trainer holds Game&; keep game alive
        .def("train", &sixmax::MCCFRTrainer::train, py::arg("iterations"))
        .def("iterations", &sixmax::MCCFRTrainer::iterations)
        .def("num_infosets", &sixmax::MCCFRTrainer::num_infosets)
        .def("average_strategy", &sixmax::MCCFRTrainer::average_strategy,
             py::arg("key"));
    m.def("kuhn_exact_value", &sixmax::kuhn_exact_value);
```

- [ ] **Step 4: Build and run test to verify it passes**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_mccfr_kuhn.py -v`
Expected: 4 passed (the 200k-iteration tests take a few seconds). If `test_game_value_converges` fails marginally, the trainer has a bug — do **not** widen the 0.01 tolerance; the value test is the correctness gate.

- [ ] **Step 5: Full sixmax suite**

Run: `uv run pytest tests/sixmax/ -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/blueprint/mccfr.h sixmax/src/blueprint/mccfr.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp \
        tests/sixmax/test_mccfr_kuhn.py
git commit -m "feat(sixmax): external-sampling MCCFR reproduces Kuhn equilibrium"
```

---

### Task 3: 6-max NLHE engine (dealing, blinds, betting, settlement)

**Files:**
- Create: `sixmax/src/engine/engine.h`, `sixmax/src/engine/engine.cpp`
- Modify: `sixmax/BUCK` (add `src/engine/engine.cpp`)
- Modify: `sixmax/src/bindings/bindings.cpp` (bind `EngineConfig`, `Street`, `HandState`, `settle_pots`)
- Test: `tests/sixmax/test_engine.py`

**Interfaces:**
- Consumes: `safe_eval::rank7` / `HandRank::beats/ties` (`#include "game/safe_eval.h"`).
- Produces (Task 4 and Task 5 rely on these exact signatures):
  - `std::vector<double> settle_pots(const std::vector<double>& total_bets, const std::vector<uint8_t>& folded, const std::vector<int>& rank_order)` — gross payout per player; `rank_order[i]` is a dense showdown rank, 0 = best, ties share a value, entries for folded players ignored.
  - `HandState(const EngineConfig&, int button, std::vector<int> deck, std::vector<double> stacks = {})` and `HandState::deal(cfg, button, rng)`; methods `is_terminal, current_player, street, pot, to_call, current_bet, min_raise_to, can_raise, player(i), hole_cards(i), board, apply(EngineAction), payoffs`.
  - Python: `sixmax.HandState(cfg, button, deck, stacks=[])` with `apply_fold() / apply_check_call() / apply_raise_to(x)` plus read accessors; `sixmax.settle_pots(...)`; `sixmax.EngineConfig(num_players, starting_stack=100.0)`.

**Rules implemented (and two documented simplifications):**
- Blinds SB=0.5/BB=1.0; HU: button posts SB and acts first preflop, other seat is BB and acts first postflop; 3+: SB=button+1, BB=button+2, UTG (BB+1) opens preflop; postflop first actor = first live player after the button. Min-raise = current bet + last raise size (initially 1 BB); an all-in below the min-raise is always allowed. Deal order from `deck`: player `i` gets `deck[2i], deck[2i+1]`; board is `deck[2n..2n+4]`; no burn cards.
- Simplification 1: a short all-in raise does **not** re-close betting rights (no "action reopened" bookkeeping). Standard in abstraction training frames; revisit only if search-phase legality audits demand it.
- Simplification 2: betting rounds where fewer than 2 players can still act are skipped (board runs out to showdown). A lone live player facing only all-ins cannot bet — bets no one can call are pruned from the tree.

- [ ] **Step 1: Write the failing test**

Create `tests/sixmax/test_engine.py`:

```python
"""NLHE engine: BB frame (SB=0.5, BB=1.0, stacks in BB). Card code =
(rank-2)*4 + suit. Deck layout: player i holds deck[2i], deck[2i+1];
board = deck[2n .. 2n+4]."""
import sixmax


def _card(rank, suit):
    return (rank - 2) * 4 + suit


def _deck_hu(p0, p1, board):
    d = list(p0) + list(p1) + list(board)
    d += [c for c in range(52) if c not in d]
    return d


AA = [_card(14, 0), _card(14, 1)]
KK = [_card(13, 0), _card(13, 1)]
QQ = [_card(12, 0), _card(12, 1)]
DRY_BOARD = [_card(2, 0), _card(7, 1), _card(11, 3), _card(3, 2), _card(9, 0)]


def _hu(deck=None, stacks=()):
    cfg = sixmax.EngineConfig(num_players=2)
    deck = deck or _deck_hu(AA, KK, DRY_BOARD)
    return sixmax.HandState(cfg, 0, deck, list(stacks))


def test_preflop_setup_hu():
    s = _hu()
    assert s.current_player() == 0          # button/SB acts first HU preflop
    assert s.street() == sixmax.Street.Preflop
    assert abs(s.pot() - 1.5) < 1e-9
    assert abs(s.to_call() - 0.5) < 1e-9
    assert abs(s.min_raise_to() - 2.0) < 1e-9


def test_fold_preflop_awards_blinds():
    s = _hu()
    s.apply_fold()
    assert s.is_terminal()
    assert s.payoffs() == [-0.5, 0.5]
    assert s.board() == []                  # no cards revealed on a fold


def test_full_hand_raise_call_bet_call_showdown():
    s = _hu()
    s.apply_raise_to(2.5)
    assert abs(s.to_call() - 1.5) < 1e-9
    s.apply_check_call()                    # BB calls, pot 5.0
    assert s.street() == sixmax.Street.Flop
    assert s.current_player() == 1          # BB first postflop HU
    assert len(s.board()) == 3
    s.apply_check_call()                    # BB checks
    s.apply_raise_to(3.75)                  # button bets 0.75 pot
    s.apply_check_call()                    # BB calls, pot 12.5
    s.apply_check_call(); s.apply_check_call()   # turn checks through
    s.apply_check_call(); s.apply_check_call()   # river checks through
    assert s.is_terminal()
    assert s.payoffs() == [6.25, -6.25]     # aces win


def test_chopped_pot_on_played_board():
    board = [_card(10, 3), _card(11, 3), _card(12, 1), _card(13, 0), _card(14, 2)]
    lo0, lo1 = [_card(2, 0), _card(3, 1)], [_card(4, 0), _card(6, 1)]
    s = _hu(deck=_deck_hu(lo0, lo1, board))
    s.apply_check_call()                    # SB limps
    s.apply_check_call()                    # BB checks
    for _ in range(3):
        s.apply_check_call(); s.apply_check_call()
    assert s.is_terminal()
    assert s.payoffs() == [0.0, 0.0]        # broadway on board, chop


def test_min_raise_tracking():
    s = _hu()
    s.apply_raise_to(3.0)                   # raise size 2.0
    assert abs(s.min_raise_to() - 5.0) < 1e-9
    s.apply_raise_to(9.0)                   # 3-bet, raise size 6.0
    assert abs(s.min_raise_to() - 15.0) < 1e-9


def test_three_way_allin_side_pots():
    cfg = sixmax.EngineConfig(num_players=3)
    deck = QQ + KK + AA + [_card(2, 0), _card(7, 1), _card(9, 3),
                           _card(3, 2), _card(11, 1)]
    deck += [c for c in range(52) if c not in deck]
    s = sixmax.HandState(cfg, 0, deck, [100.0, 40.0, 10.0])
    # button=0 => SB=1, BB=2, UTG=0 opens.
    assert s.current_player() == 0
    s.apply_raise_to(100.0)                 # UTG jams QQ
    s.apply_check_call()                    # SB calls all-in for 40 (KK)
    s.apply_check_call()                    # BB calls all-in for 10 (AA)
    assert s.is_terminal()
    assert s.street() == sixmax.Street.River
    assert len(s.board()) == 5
    # Main pot 30 -> AA; side pot 60 -> KK; 60 uncalled back to QQ.
    assert s.payoffs() == [-40.0, 20.0, 20.0]
    assert abs(sum(s.payoffs())) < 1e-9


def test_bet_when_opponents_allin_is_skipped():
    cfg = sixmax.EngineConfig(num_players=2)
    s = sixmax.HandState(cfg, 0, _deck_hu(AA, KK, DRY_BOARD), [100.0, 20.0])
    s.apply_raise_to(30.0)
    s.apply_check_call()                    # BB all-in for 20
    assert s.is_terminal()                  # board runs out, no more betting
    assert s.payoffs() == [20.0, -20.0]


def test_settle_pots_ties_split_exactly():
    assert sixmax.settle_pots([100.0, 100.0], [False, False], [0, 0]) == [100.0, 100.0]
    # Main-pot tie plus a side pot: p0 all-in 50, p1/p2 at 200, p0 ties p1.
    got = sixmax.settle_pots([50.0, 200.0, 200.0], [False, False, False], [0, 0, 1])
    assert got == [75.0, 375.0, 0.0]
    # Odd amounts split exactly in the BB double frame.
    assert sixmax.settle_pots([3.0, 3.0, 3.0], [False, False, False],
                              [0, 0, 1]) == [4.5, 4.5, 0.0]


def test_settle_pots_folded_max_contributor_gets_nothing():
    got = sixmax.settle_pots([60.0, 60.0, 60.0], [False, True, False], [2, 0, 1])
    assert got == [0.0, 0.0, 180.0]


def test_deal_is_seed_deterministic():
    cfg = sixmax.EngineConfig(num_players=6)
    a = sixmax.HandState.deal(cfg, 2, 99)
    b = sixmax.HandState.deal(cfg, 2, 99)
    assert [a.hole_cards(i) for i in range(6)] == [b.hole_cards(i) for i in range(6)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_engine.py -v`
Expected: ERROR with `AttributeError: module 'sixmax' has no attribute 'EngineConfig'`

- [ ] **Step 3: Implement the engine**

Create `sixmax/src/engine/engine.h`:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace sixmax {

// All chip quantities are doubles denominated in big blinds:
// big_blind = 1.0, small_blind = 0.5. No absolute-chip frame anywhere.
constexpr double kChipEps = 1e-9;

struct EngineConfig {
    int num_players = 6;             // 2..6
    double starting_stack = 100.0;   // BB; per-seat override via HandState ctor
};

enum class Street { Preflop = 0, Flop = 1, Turn = 2, River = 3 };

enum class EngineActionType { Fold, CheckCall, RaiseTo };
struct EngineAction {
    EngineActionType type;
    double amount = 0.0;  // RaiseTo: total street commitment ("raise to"), BB
};

struct PlayerState {
    double stack = 0.0;        // remaining behind
    double street_bet = 0.0;   // committed this street
    double total_bet = 0.0;    // committed this hand
    bool folded = false;
    bool all_in = false;
};

// Pure side-pot settlement from total contributions. rank_order[i] is the
// dense showdown rank of player i (0 = best, ties share a value); entries
// for folded players are ignored. Returns gross payouts (winnings only).
std::vector<double> settle_pots(const std::vector<double>& total_bets,
                                const std::vector<uint8_t>& folded,
                                const std::vector<int>& rank_order);

class HandState {
public:
    // deck: >= 2n+5 card codes, code = (rank-2)*4 + suit. Deal order:
    // player i holds deck[2i], deck[2i+1]; board = deck[2n..2n+4]; no burn.
    // stacks: per-seat starting stacks; empty = cfg.starting_stack for all.
    // Precondition: every stack > 1.0 BB (blind posting cannot go all-in).
    HandState(const EngineConfig& cfg, int button, std::vector<int> deck,
              std::vector<double> stacks = {});
    static HandState deal(const EngineConfig& cfg, int button,
                          std::mt19937_64& rng);

    bool is_terminal() const { return terminal_; }
    int current_player() const { return next_; }
    Street street() const { return street_; }
    int num_players() const { return (int)players_.size(); }
    int button() const { return button_; }
    double pot() const;                // total contributed by everyone
    double current_bet() const { return current_bet_; }
    double to_call() const;            // current player's owe, capped by stack
    double min_raise_to() const { return current_bet_ + last_raise_; }
    bool can_raise() const;            // current player may bet/raise
    const PlayerState& player(int i) const { return players_[i]; }
    std::array<int, 2> hole_cards(int i) const;
    std::vector<int> board() const;    // cards revealed so far (0/3/4/5)

    void apply(const EngineAction& a);
    // Net result per player (final stack minus starting stack), BB.
    // Valid iff is_terminal().
    const std::vector<double>& payoffs() const { return payoffs_; }

private:
    void commit(int seat, double amount);
    void finish_action(int seat);
    void close_round_or_advance();
    int next_active_after(int seat) const;  // next seat that can still act
    int num_can_act() const;
    std::vector<int> showdown_order() const;
    void settle();

    EngineConfig cfg_;
    int button_;
    std::vector<int> deck_;
    std::vector<PlayerState> players_;
    std::vector<double> start_stacks_;
    Street street_ = Street::Preflop;
    int next_ = -1;
    int to_act_ = 0;               // players still owed an action this round
    double current_bet_ = 0.0;     // highest street_bet on the table
    double last_raise_ = 1.0;      // last raise increment (min-raise basis)
    bool terminal_ = false;
    std::vector<double> payoffs_;
};

}  // namespace sixmax
```

Create `sixmax/src/engine/engine.cpp`:

```cpp
#include "engine/engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include "game/safe_eval.h"

namespace sixmax {

std::vector<double> settle_pots(const std::vector<double>& total_bets,
                                const std::vector<uint8_t>& folded,
                                const std::vector<int>& rank_order) {
    int n = (int)total_bets.size();
    std::vector<double> payout(n, 0.0);
    std::vector<double> levels;
    for (int i = 0; i < n; ++i)
        if (total_bets[i] > kChipEps) levels.push_back(total_bets[i]);
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end(),
                             [](double a, double b) {
                                 return std::abs(a - b) < kChipEps;
                             }),
                 levels.end());
    double prev = 0.0;
    for (double level : levels) {
        double amount = 0.0;
        for (int i = 0; i < n; ++i)
            amount += std::max(0.0, std::min(total_bets[i], level) - prev);
        int best = std::numeric_limits<int>::max();
        for (int i = 0; i < n; ++i)
            if (!folded[i] && total_bets[i] >= level - kChipEps)
                best = std::min(best, rank_order[i]);
        std::vector<int> winners;
        if (best != std::numeric_limits<int>::max()) {
            for (int i = 0; i < n; ++i)
                if (!folded[i] && total_bets[i] >= level - kChipEps &&
                    rank_order[i] == best)
                    winners.push_back(i);
        } else {
            // Defensive: every contributor at this level folded. Unreachable
            // from valid play (the closing aggressor never folds); dead money
            // goes to all live players so chips are conserved.
            for (int i = 0; i < n; ++i)
                if (!folded[i]) winners.push_back(i);
        }
        for (int w : winners) payout[w] += amount / (double)winners.size();
        prev = level;
    }
    return payout;
}

HandState::HandState(const EngineConfig& cfg, int button, std::vector<int> deck,
                     std::vector<double> stacks)
    : cfg_(cfg), button_(button), deck_(std::move(deck)) {
    int n = cfg_.num_players;
    assert(n >= 2 && n <= 6);
    assert((int)deck_.size() >= 2 * n + 5);
    players_.resize(n);
    start_stacks_.resize(n);
    for (int i = 0; i < n; ++i) {
        double st = stacks.empty() ? cfg_.starting_stack : stacks[i];
        assert(st > 1.0 + kChipEps);  // blinds may not force an all-in
        players_[i].stack = st;
        start_stacks_[i] = st;
    }
    int sb = (n == 2) ? button_ : (button_ + 1) % n;
    int bb = (n == 2) ? (button_ + 1) % n : (button_ + 2) % n;
    commit(sb, 0.5);
    commit(bb, 1.0);
    current_bet_ = 1.0;
    last_raise_ = 1.0;
    next_ = (n == 2) ? button_ : (bb + 1) % n;
    to_act_ = num_can_act();  // everyone, including the BB option
}

HandState HandState::deal(const EngineConfig& cfg, int button,
                          std::mt19937_64& rng) {
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    return HandState(cfg, button, std::move(deck));
}

double HandState::pot() const {
    double p = 0.0;
    for (const auto& pl : players_) p += pl.total_bet;
    return p;
}

double HandState::to_call() const {
    const PlayerState& p = players_[next_];
    return std::min(std::max(0.0, current_bet_ - p.street_bet), p.stack);
}

bool HandState::can_raise() const {
    const PlayerState& p = players_[next_];
    if (p.stack <= (current_bet_ - p.street_bet) + kChipEps) return false;
    for (int i = 0; i < (int)players_.size(); ++i)
        if (i != next_ && !players_[i].folded && !players_[i].all_in)
            return true;  // someone can respond, so a bet has meaning
    return false;
}

std::array<int, 2> HandState::hole_cards(int i) const {
    return {deck_[2 * i], deck_[2 * i + 1]};
}

std::vector<int> HandState::board() const {
    static const int reveal[4] = {0, 3, 4, 5};
    int n = (int)players_.size();
    int count = reveal[(int)street_];
    return std::vector<int>(deck_.begin() + 2 * n,
                            deck_.begin() + 2 * n + count);
}

void HandState::commit(int seat, double amount) {
    PlayerState& p = players_[seat];
    amount = std::min(amount, p.stack);
    p.stack -= amount;
    p.street_bet += amount;
    p.total_bet += amount;
    if (p.stack < kChipEps) {
        p.stack = 0.0;
        p.all_in = true;
    }
}

int HandState::next_active_after(int seat) const {
    int n = (int)players_.size();
    for (int k = 1; k <= n; ++k) {
        int i = (seat + k) % n;
        if (!players_[i].folded && !players_[i].all_in) return i;
    }
    return -1;
}

int HandState::num_can_act() const {
    int c = 0;
    for (const auto& p : players_)
        if (!p.folded && !p.all_in) ++c;
    return c;
}

void HandState::apply(const EngineAction& a) {
    assert(!terminal_);
    int seat = next_;
    PlayerState& p = players_[seat];
    switch (a.type) {
        case EngineActionType::Fold:
            p.folded = true;
            --to_act_;
            break;
        case EngineActionType::CheckCall:
            commit(seat, std::max(0.0, current_bet_ - p.street_bet));
            --to_act_;
            break;
        case EngineActionType::RaiseTo: {
            double target = std::min(a.amount, p.street_bet + p.stack);
            assert(target > current_bet_ + kChipEps);
            double raise_size = target - current_bet_;
            if (raise_size > last_raise_ - kChipEps) last_raise_ = raise_size;
            current_bet_ = target;
            commit(seat, target - p.street_bet);
            to_act_ = 0;  // everyone else live owes a response
            for (int i = 0; i < (int)players_.size(); ++i)
                if (i != seat && !players_[i].folded && !players_[i].all_in)
                    ++to_act_;
            break;
        }
    }
    finish_action(seat);
}

void HandState::finish_action(int seat) {
    int unfolded = 0;
    for (const auto& p : players_)
        if (!p.folded) ++unfolded;
    if (unfolded == 1) {
        terminal_ = true;
        settle();
        return;
    }
    if (to_act_ > 0) {
        next_ = next_active_after(seat);
        return;
    }
    close_round_or_advance();
}

void HandState::close_round_or_advance() {
    while (true) {
        if (street_ == Street::River) {
            terminal_ = true;
            settle();
            return;
        }
        street_ = (Street)((int)street_ + 1);
        for (auto& p : players_) p.street_bet = 0.0;
        current_bet_ = 0.0;
        last_raise_ = 1.0;  // min bet is 1 BB on a fresh street
        if (num_can_act() >= 2) {
            next_ = next_active_after(button_);
            to_act_ = num_can_act();
            return;
        }
        // Fewer than 2 players can act: no betting, run out the board.
    }
}

std::vector<int> HandState::showdown_order() const {
    int n = (int)players_.size();
    std::vector<int> order(n, 0);
    std::vector<int> alive;
    for (int i = 0; i < n; ++i)
        if (!players_[i].folded) alive.push_back(i);
    if (alive.size() < 2) return order;  // fold-win: ranks are irrelevant
    std::vector<int> bd = board();       // river reached: 5 cards
    std::vector<safe_eval::HandRank> hr;
    hr.reserve(alive.size());
    for (int seat : alive) {
        auto hc = hole_cards(seat);
        std::array<int, 7> codes = {hc[0], hc[1], bd[0], bd[1],
                                    bd[2], bd[3], bd[4]};
        hr.push_back(safe_eval::rank7(codes));
    }
    std::vector<int> pos(alive.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::sort(pos.begin(), pos.end(),
              [&](int a, int b) { return hr[a].beats(hr[b]); });
    int rank = 0;
    for (size_t k = 0; k < pos.size(); ++k) {
        if (k > 0 && !hr[pos[k - 1]].ties(hr[pos[k]])) rank = (int)k;
        order[alive[pos[k]]] = rank;
    }
    return order;
}

void HandState::settle() {
    int n = (int)players_.size();
    std::vector<double> totals(n);
    std::vector<uint8_t> folded(n);
    for (int i = 0; i < n; ++i) {
        totals[i] = players_[i].total_bet;
        folded[i] = players_[i].folded ? 1 : 0;
    }
    std::vector<double> payout = settle_pots(totals, folded, showdown_order());
    payoffs_.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        players_[i].stack += payout[i];
        payoffs_[i] = players_[i].stack - start_stacks_[i];
    }
    next_ = -1;
}

}  // namespace sixmax
```

In `sixmax/BUCK`, change `srcs` to:

```python
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp", "src/blueprint/mccfr.cpp",
            "src/engine/engine.cpp"],
```

In `sixmax/src/bindings/bindings.cpp`, add `#include "engine/engine.h"` and append inside the module body:

```cpp
    // --- NLHE engine (Task 3) ---
    py::enum_<sixmax::Street>(m, "Street")
        .value("Preflop", sixmax::Street::Preflop)
        .value("Flop", sixmax::Street::Flop)
        .value("Turn", sixmax::Street::Turn)
        .value("River", sixmax::Street::River);
    py::class_<sixmax::EngineConfig>(m, "EngineConfig")
        .def(py::init([](int n, double stack) {
                 return sixmax::EngineConfig{n, stack};
             }),
             py::arg("num_players"), py::arg("starting_stack") = 100.0)
        .def_readonly("num_players", &sixmax::EngineConfig::num_players)
        .def_readonly("starting_stack", &sixmax::EngineConfig::starting_stack);
    py::class_<sixmax::HandState>(m, "HandState")
        .def(py::init<const sixmax::EngineConfig&, int, std::vector<int>,
                      std::vector<double>>(),
             py::arg("cfg"), py::arg("button"), py::arg("deck"),
             py::arg("stacks") = std::vector<double>{})
        .def_static("deal",
                    [](const sixmax::EngineConfig& c, int button, uint64_t seed) {
                        std::mt19937_64 rng(seed);
                        return sixmax::HandState::deal(c, button, rng);
                    },
                    py::arg("cfg"), py::arg("button"), py::arg("seed"))
        .def("is_terminal", &sixmax::HandState::is_terminal)
        .def("current_player", &sixmax::HandState::current_player)
        .def("street", &sixmax::HandState::street)
        .def("button", &sixmax::HandState::button)
        .def("pot", &sixmax::HandState::pot)
        .def("to_call", &sixmax::HandState::to_call)
        .def("current_bet", &sixmax::HandState::current_bet)
        .def("min_raise_to", &sixmax::HandState::min_raise_to)
        .def("can_raise", &sixmax::HandState::can_raise)
        .def("board", &sixmax::HandState::board)
        .def("hole_cards",
             [](const sixmax::HandState& s, int i) {
                 auto hc = s.hole_cards(i);
                 return std::vector<int>{hc[0], hc[1]};
             })
        .def("stack", [](const sixmax::HandState& s, int i) {
            return s.player(i).stack;
        })
        .def("folded", [](const sixmax::HandState& s, int i) {
            return s.player(i).folded;
        })
        .def("all_in", [](const sixmax::HandState& s, int i) {
            return s.player(i).all_in;
        })
        .def("apply_fold", [](sixmax::HandState& s) {
            s.apply({sixmax::EngineActionType::Fold, 0.0});
        })
        .def("apply_check_call", [](sixmax::HandState& s) {
            s.apply({sixmax::EngineActionType::CheckCall, 0.0});
        })
        .def("apply_raise_to", [](sixmax::HandState& s, double amount) {
            s.apply({sixmax::EngineActionType::RaiseTo, amount});
        })
        .def("payoffs", &sixmax::HandState::payoffs);
    m.def("settle_pots",
          [](const std::vector<double>& totals, const std::vector<bool>& folded,
             const std::vector<int>& ranks) {
              std::vector<uint8_t> f(folded.begin(), folded.end());
              return sixmax::settle_pots(totals, f, ranks);
          },
          py::arg("total_bets"), py::arg("folded"), py::arg("rank_order"));
```

- [ ] **Step 4: Build and run test to verify it passes**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_engine.py -v`
Expected: 10 passed

- [ ] **Step 5: Full sixmax suite**

Run: `uv run pytest tests/sixmax/ -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/engine/engine.h sixmax/src/engine/engine.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp \
        tests/sixmax/test_engine.py
git commit -m "feat(sixmax): 2-6 player NLHE engine with side-pot settlement"
```

---

### Task 4: Settlement cross-validation against the Python engine

**Files:**
- Test: `tests/sixmax/test_settlement.py` (test-only task; no C++ changes expected)

**Interfaces:**
- Consumes: `sixmax.settle_pots(total_bets, folded, rank_order)` from Task 3; `game.pot_manager.PotManager`, `models.player.Player`, `agents.simple_agent.SimpleAgent` from the Python engine (the frozen oracle — this import is allowed only because it lives in `tests/`).
- Produces: nothing new — this is the spec's "side-pot payouts property-tested against the Python engine" gate. If a mismatch is found, the C++ side is presumed wrong unless the scenario violates the generator invariants below; fixes go into `sixmax/src/engine/engine.cpp` and the whole file re-runs.

**Oracle semantics (why the generator has invariants):** `PotManager` derives levels only from players flagged `is_all_in`, its `award` filters winners through the passed rank list (folded players are simply not passed), and it gives remainder chips to `winners[0]`. To make the comparison exact: contributions are integers, ranks are distinct (no splits, so no remainder-chip divergence), every scenario has ≥ 2 unfolded players, at least one unfolded player at the maximum contribution `M`, and every unfolded non-all-in player contributes exactly `M` (the end-of-hand invariant real betting guarantees).

- [ ] **Step 1: Write the property test (this is the deliverable)**

Create `tests/sixmax/test_settlement.py`:

```python
"""Cross-validate sixmax.settle_pots against the Python engine's PotManager
on randomly generated, betting-consistent scenarios. The Python engine is
the frozen oracle: mismatches mean the C++ settlement is wrong."""
import random

import sixmax
from agents.simple_agent import SimpleAgent
from game.pot_manager import PotManager
from models.player import Player

START = 100_000  # Python Player stacks start here; payout = stack - START


def _python_payouts(contribs, folded, ranks):
    players = [Player(f"p{i}", START, SimpleAgent()) for i in range(len(contribs))]
    pm = PotManager()
    m = max(contribs)
    for p, c, f in zip(players, contribs, folded):
        if not f and c < m:
            p.is_all_in = True
        pm.contribute(p, c)
    pm.award([(p, r) for p, r, f in zip(players, ranks, folded) if not f])
    return [p.stack - START for p in players]


def _scenario(rng):
    """Betting-consistent random scenario. Invariants: >=2 unfolded, >=1
    unfolded player at max contribution M, unfolded non-all-in players
    contribute exactly M, all contributions integer >= 1, distinct ranks."""
    n = rng.randint(2, 6)
    while True:
        m = rng.randint(2, 200)
        contribs, folded = [], []
        for _ in range(n):
            role = rng.choice(["call", "call", "allin", "fold"])
            if role == "call":
                contribs.append(m); folded.append(False)
            elif role == "allin":
                contribs.append(rng.randint(1, m - 1)); folded.append(False)
            else:
                contribs.append(rng.randint(1, m)); folded.append(True)
        unfolded = [i for i in range(n) if not folded[i]]
        callers = [i for i in unfolded if contribs[i] == m]
        if len(unfolded) >= 2 and callers:
            ranks = list(range(n))
            rng.shuffle(ranks)
            return contribs, folded, ranks


def test_settlement_matches_python_engine_on_random_scenarios():
    rng = random.Random(20260718)
    for _ in range(300):
        contribs, folded, ranks = _scenario(rng)
        expected = _python_payouts(contribs, folded, ranks)
        got = sixmax.settle_pots([float(c) for c in contribs], folded, ranks)
        net = [g - c for g, c in zip(got, contribs)]
        assert all(abs(a - b) < 1e-6 for a, b in zip(net, expected)), (
            f"mismatch: contribs={contribs} folded={folded} ranks={ranks} "
            f"cpp_net={net} python={expected}")


def test_uncalled_jam_returns_excess():
    # A jams 100, B calls all-in for 40, C folds at 10. A wins everything.
    got = sixmax.settle_pots([100.0, 40.0, 10.0], [False, False, True], [0, 1, 2])
    assert got == [150.0, 0.0, 0.0]
    # Same but B wins: B takes the 90-capped pot, A keeps his uncalled 60.
    got = sixmax.settle_pots([100.0, 40.0, 10.0], [False, False, True], [1, 0, 2])
    assert got == [60.0, 90.0, 0.0]


def test_chip_conservation_on_random_scenarios():
    rng = random.Random(99)
    for _ in range(200):
        contribs, folded, ranks = _scenario(rng)
        got = sixmax.settle_pots([float(c) for c in contribs], folded, ranks)
        assert abs(sum(got) - sum(contribs)) < 1e-6
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/sixmax/test_settlement.py -v`
Expected: 3 passed. If the property test fails, minimize the printed scenario by hand, fix `settle_pots` in `sixmax/src/engine/engine.cpp`, rebuild, and re-run — do not weaken the generator invariants or tolerances.

- [ ] **Step 3: Full suite (both engines untouched elsewhere)**

Run: `uv run pytest tests/ -q`
Expected: everything passes except the known stochastic xfail/xpass drift in `tests/cfr/test_mccfr.py` (x-counts vary run to run; not caused by this task).

- [ ] **Step 4: Commit**

```bash
git add tests/sixmax/test_settlement.py
git commit -m "test(sixmax): property-test side-pot settlement vs Python engine"
```

---

### Task 5: EngineGame bridge — ActionVocab masking over the engine

**Files:**
- Create: `sixmax/src/blueprint/engine_game.h`, `sixmax/src/blueprint/engine_game.cpp`
- Modify: `sixmax/BUCK` (add `src/blueprint/engine_game.cpp`)
- Modify: `sixmax/src/bindings/bindings.cpp` (bind `EngineGameState`, `EngineGame`)
- Test: `tests/sixmax/test_engine_game.py`

**Interfaces:**
- Consumes: `Game`/`GameState` (Task 1), `MCCFRTrainer` (Task 2), `HandState`/`EngineConfig` (Task 3), `ActionVocab`/`BetContext`/`AbstractAction` (Phase 0 — signatures in Global Constraints).
- Produces: `sixmax::EngineGameState(HandState, const ActionVocab*)` (a `GameState`) and `sixmax::EngineGame(EngineConfig, const ActionVocab*)` (a `Game`; button rotates each `new_hand`). Python: `sixmax.EngineGameState(cfg, button, deck, vocab, stacks=[])`, `sixmax.EngineGame(cfg, vocab)` with `new_hand(seed)`. Phase 1b replaces only `infoset_key()` (the naive exact-information keyer) with the card/history abstraction.

**Masking rules (the design core of this task):**
1. `Fold` legal iff facing a bet (`to_call > eps`); `Check` iff not facing; `Call` iff facing.
2. Size-class rule from the spec: BB-unit `Bet` entries (preflop opens) are legal **only** preflop in an unopened pot (`current_bet ≤ 1.0 + eps` — only blinds posted, limpers included); Pot-unit `Bet` entries are legal everywhere else (3-bet+ and all postflop). This implements "every raise after the open shares one pot-fraction grid".
3. A `Bet` entry is legal only if the player `can_raise()` and its `target_bb` is `≥ min_raise_to` (no clamping small bets up) and **strictly below** the all-in target (the jam entry owns stack-offs; no duplicate actions).
4. `AllIn` legal iff `can_raise()`.
5. `BetContext` is filled exactly as pinned in Global Constraints: `{pot() − owe, current_bet, owe, street_bet + stack}` with `owe = current_bet − street_bet`.

- [ ] **Step 1: Write the failing test**

Create `tests/sixmax/test_engine_game.py`:

```python
"""Engine <-> ActionVocab bridge. Uses the default blueprint vocab
(10 entries): 0=fold 1=check 2=call 3=open2.5 4=open3.5 5=open5.0
6=bet0.33pot 7=bet0.75pot 8=bet1.5pot 9=allin."""
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
FOLD, CHECK, CALL = 0, 1, 2
OPEN25, OPEN35, OPEN50 = 3, 4, 5
B33, B75, B150 = 6, 7, 8
ALLIN = 9


def _hu_state(stacks=()):
    cfg = sixmax.EngineConfig(num_players=2)
    deck = list(range(52))  # deterministic deal; cards don't matter for masks
    return sixmax.EngineGameState(cfg, 0, deck, VOCAB, list(stacks))


def test_preflop_open_spot_mask():
    s = _hu_state()
    # SB facing the blind: fold/call legal, check not; BB opens legal,
    # pot-fraction entries masked (unopened preflop); jam legal.
    assert s.legal_mask() == [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]


def test_postflop_bet_spot_mask():
    s = _hu_state()
    s.apply(CALL)
    s.apply(CHECK)
    # Flop, pot 2.0, BB to act, no bet yet. 0.33 pot = 0.66 BB is below the
    # 1 BB min bet -> masked. BB-unit opens are preflop-only -> masked.
    assert s.legal_mask() == [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]


def test_preflop_3bet_spot_uses_pot_grid():
    s = _hu_state()
    s.apply(OPEN25)
    # BB facing a 2.5 open: pot-fraction 3-bets now legal, BB opens masked.
    mask = s.legal_mask()
    assert mask[FOLD] and mask[CALL] and not mask[CHECK]
    assert mask[OPEN25] == mask[OPEN35] == mask[OPEN50] == 0
    assert mask[B33] and mask[B75] and mask[B150] and mask[ALLIN]
    # ctx: pot=3.5-1.5=2.0, current_bet=2.5, to_call=1.5 =>
    # 0.33-pot 3-bet raises to 2.5 + 0.33*(2.0+3.0) = 4.15 >= min_raise 4.0.
    ctx = s.bet_context()
    assert abs(ctx.pot - 2.0) < 1e-9
    assert abs(ctx.to_call - 1.5) < 1e-9
    assert abs(VOCAB.target_bb(B33, ctx) - 4.15) < 1e-9


def test_short_stack_dedupes_bet_against_jam():
    # 4 BB stacks: every open's target caps at/over the 4 BB jam -> only
    # the AllIn entry may represent the stack-off.
    s = _hu_state(stacks=(4.0, 4.0))
    mask = s.legal_mask()
    assert mask[OPEN25] == 1          # 2.5 < 4.0 all-in target: distinct raise
    assert mask[OPEN35] == 1          # 3.5 < 4.0
    assert mask[OPEN50] == 0          # would cap to 4.0 == jam -> deduped
    assert mask[ALLIN] == 1


def test_apply_translates_and_terminal_utility_matches_engine():
    s = _hu_state()
    s.apply(OPEN25)
    s.apply(CALL)
    s.apply(CHECK)      # flop: BB checks
    s.apply(B75)        # button bets 0.75 * 5.0 = 3.75
    s.apply(FOLD)
    assert s.is_terminal()
    assert abs(s.utility(0) - 2.5) < 1e-9
    assert abs(s.utility(1) + 2.5) < 1e-9


def test_random_playouts_are_zero_sum_and_legal():
    import random
    rng = random.Random(5)
    for players in (2, 3, 6):
        g = sixmax.EngineGame(sixmax.EngineConfig(num_players=players), VOCAB)
        for trial in range(60):
            s = g.new_hand(seed=rng.randrange(2**60))
            steps = 0
            while not s.is_terminal():
                mask = s.legal_mask()
                assert len(mask) == VOCAB.size()
                legal = [i for i, ok in enumerate(mask) if ok]
                assert legal, "no legal action"
                s.apply(rng.choice(legal))
                steps += 1
                assert steps < 200, "hand failed to terminate"
            total = sum(s.utility(i) for i in range(players))
            assert abs(total) < 1e-6


def test_mccfr_smoke_on_hu_engine():
    g = sixmax.EngineGame(sixmax.EngineConfig(num_players=2), VOCAB)
    t = sixmax.MCCFRTrainer(g, seed=11)
    t.train(300)
    assert t.num_infosets() > 200
    g2 = sixmax.EngineGame(sixmax.EngineConfig(num_players=2), VOCAB)
    u2 = sixmax.MCCFRTrainer(g2, seed=11)
    u2.train(300)
    # Fresh game + same seed reproduces exactly (determinism end to end).
    assert u2.num_infosets() == t.num_infosets()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sixmax/test_engine_game.py -v`
Expected: ERROR with `AttributeError: module 'sixmax' has no attribute 'EngineGameState'`

- [ ] **Step 3: Implement the bridge**

Create `sixmax/src/blueprint/engine_game.h`:

```cpp
#pragma once
#include "blueprint/game.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

// Adapts HandState + ActionVocab to the MCCFR Game interface. Legality is
// masking only — vocab order is never filtered or reordered. infoset_key()
// is a NAIVE exact-information hash (hole cards + board + action history);
// Phase 1b replaces it with the card/history abstraction. Everything else
// here (masking, translation) is final.
class EngineGameState : public GameState {
public:
    EngineGameState(HandState hand, const ActionVocab* vocab)
        : hand_(std::move(hand)), vocab_(vocab) {}
    bool is_terminal() const override { return hand_.is_terminal(); }
    int current_player() const override { return hand_.current_player(); }
    void legal_mask(std::vector<uint8_t>& mask) const override;
    uint64_t infoset_key() const override;
    void apply(int action) override;
    double utility(int player) const override {
        return hand_.payoffs()[player];
    }
    std::unique_ptr<GameState> clone() const override {
        return std::make_unique<EngineGameState>(*this);
    }
    BetContext bet_context() const;  // exposed for tests and Phase 2 search

private:
    bool size_class_ok(const AbstractAction& a) const;
    HandState hand_;
    const ActionVocab* vocab_;
    std::vector<int> history_;  // applied vocab indices, all seats (public)
};

class EngineGame : public Game {
public:
    EngineGame(EngineConfig cfg, const ActionVocab* vocab)
        : cfg_(cfg), vocab_(vocab) {}
    int num_players() const override { return cfg_.num_players; }
    int num_actions() const override { return vocab_->size(); }
    std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) override;

private:
    EngineConfig cfg_;
    const ActionVocab* vocab_;
    int button_ = 0;  // rotates every hand for positional balance
};

}  // namespace sixmax
```

Create `sixmax/src/blueprint/engine_game.cpp`:

```cpp
#include "blueprint/engine_game.h"

namespace sixmax {

BetContext EngineGameState::bet_context() const {
    const PlayerState& p = hand_.player(hand_.current_player());
    double owe = hand_.current_bet() - p.street_bet;  // uncapped
    return BetContext{hand_.pot() - owe, hand_.current_bet(), owe,
                      p.street_bet + p.stack};
}

bool EngineGameState::size_class_ok(const AbstractAction& a) const {
    bool unopened_preflop = hand_.street() == Street::Preflop &&
                            hand_.current_bet() <= 1.0 + kChipEps;
    // BB-unit sizes are preflop opens; Pot-unit sizes are everything after.
    return (a.unit == SizeUnit::BB) == unopened_preflop;
}

void EngineGameState::legal_mask(std::vector<uint8_t>& mask) const {
    int n = vocab_->size();
    mask.assign(n, 0);
    BetContext ctx = bet_context();
    bool facing = ctx.to_call > kChipEps;
    for (int i = 0; i < n; ++i) {
        const AbstractAction& a = vocab_->at(i);
        switch (a.type) {
            case ActionType::Fold:
                mask[i] = facing ? 1 : 0;
                break;
            case ActionType::Check:
                mask[i] = facing ? 0 : 1;
                break;
            case ActionType::Call:
                mask[i] = facing ? 1 : 0;
                break;
            case ActionType::Bet: {
                if (!hand_.can_raise() || !size_class_ok(a)) break;
                double t = vocab_->target_bb(i, ctx);
                mask[i] = (t >= hand_.min_raise_to() - kChipEps &&
                           t < ctx.stack - kChipEps)
                              ? 1
                              : 0;  // jam entry owns the stack-off
                break;
            }
            case ActionType::AllIn:
                mask[i] = hand_.can_raise() ? 1 : 0;
                break;
        }
    }
}

uint64_t EngineGameState::infoset_key() const {
    // FNV-1a over exact private+public information. Placeholder keyer:
    // Phase 1b substitutes the 169-class/equity-bucket abstraction here.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    int p = hand_.current_player();
    auto hc = hand_.hole_cards(p);
    mix((uint64_t)std::min(hc[0], hc[1]));
    mix((uint64_t)std::max(hc[0], hc[1]));
    for (int c : hand_.board()) mix((uint64_t)(c + 64));
    mix(0xFFFFull);  // separator: board cards vs action history
    for (int a : history_) mix((uint64_t)(a + 128));
    return h;
}

void EngineGameState::apply(int action) {
    const AbstractAction& a = vocab_->at(action);
    switch (a.type) {
        case ActionType::Fold:
            hand_.apply({EngineActionType::Fold, 0.0});
            break;
        case ActionType::Check:
        case ActionType::Call:
            hand_.apply({EngineActionType::CheckCall, 0.0});
            break;
        case ActionType::Bet:
        case ActionType::AllIn: {
            double target = vocab_->target_bb(action, bet_context());
            hand_.apply({EngineActionType::RaiseTo, target});
            break;
        }
    }
    history_.push_back(action);
}

std::unique_ptr<GameState> EngineGame::new_hand(std::mt19937_64& rng) {
    button_ = (button_ + 1) % cfg_.num_players;
    return std::make_unique<EngineGameState>(
        HandState::deal(cfg_, button_, rng), vocab_);
}

}  // namespace sixmax
```

In `sixmax/BUCK`, change `srcs` to:

```python
    srcs = ["src/bindings/bindings.cpp", "src/vocab/vocab.cpp",
            "src/blueprint/kuhn.cpp", "src/blueprint/mccfr.cpp",
            "src/engine/engine.cpp", "src/blueprint/engine_game.cpp"],
```

In `sixmax/src/bindings/bindings.cpp`, add `#include "blueprint/engine_game.h"` and append inside the module body:

```cpp
    // --- Engine <-> vocab bridge (Task 5) ---
    py::class_<sixmax::EngineGameState, sixmax::GameState>(m, "EngineGameState")
        .def(py::init([](const sixmax::EngineConfig& cfg, int button,
                         std::vector<int> deck, const sixmax::ActionVocab* v,
                         std::vector<double> stacks) {
                 return sixmax::EngineGameState(
                     sixmax::HandState(cfg, button, std::move(deck),
                                       std::move(stacks)),
                     v);
             }),
             py::arg("cfg"), py::arg("button"), py::arg("deck"),
             py::arg("vocab"), py::arg("stacks") = std::vector<double>{},
             py::keep_alive<1, 5>())  // state holds ActionVocab*
        .def("bet_context", &sixmax::EngineGameState::bet_context);
    py::class_<sixmax::EngineGame, sixmax::Game>(m, "EngineGame")
        .def(py::init<sixmax::EngineConfig, const sixmax::ActionVocab*>(),
             py::arg("cfg"), py::arg("vocab"),
             py::keep_alive<1, 3>())  // game holds ActionVocab*
        .def("new_hand", [](sixmax::EngineGame& g, uint64_t seed) {
            std::mt19937_64 rng(seed);
            return g.new_hand(rng);
        }, py::arg("seed"));
```

- [ ] **Step 4: Build and run test to verify it passes**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_engine_game.py -v`
Expected: 7 passed

- [ ] **Step 5: Full suite**

Run: `uv run pytest tests/ -q`
Expected: everything passes (same known stochastic x-count drift in `tests/cfr/test_mccfr.py` as always)

- [ ] **Step 6: Commit**

```bash
git add sixmax/src/blueprint/engine_game.h sixmax/src/blueprint/engine_game.cpp \
        sixmax/BUCK sixmax/src/bindings/bindings.cpp \
        tests/sixmax/test_engine_game.py
git commit -m "feat(sixmax): ActionVocab bridge over the engine + MCCFR smoke"
```

---

## Out of scope for 1a (goes in the Phase 1b plan)

Card abstraction (169 preflop + equity-percentile buckets), history abstraction (raise cap + pot buckets) and the real infoset keyer, multithreaded training, TOML config + checkpointing with the embedded vocab hash, best-checkpoint selection, `scripts/train_sixmax.py`, the openpoker strategy loader, and the 6-max eval harness.
