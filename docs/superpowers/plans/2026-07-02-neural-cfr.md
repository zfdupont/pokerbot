# Neural CFR (Deep CFR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Deep CFR (Brown et al. 2019) as a self-contained `neural_cfr/` C++ module with libtorch networks, exposed to Python via pybind11, replacing `RegretTable` with two MLPs.

**Architecture:** C++ owns the hot path — game state, feature encoding, MLP networks, reservoir buffers, CFR traversal, and training loop. A thin pybind11 layer exposes `Trainer` and `Strategy` classes to Python. Existing eval and play scripts swap `RegretTable` for `neural_cfr.Strategy` with one constructor change.

**Tech Stack:** C++17, libtorch (PyTorch C++ frontend), pybind11, Google Test, Buck2, Python 3 / pytest

## Global Constraints

- C++17 throughout (`-std=c++17`)
- libtorch CPU build (pytorch.org prebuilt tarball); CUDA not required
- Action vocabulary: `["fold", "check", "call", "b0.5", "b1.0", "allin"]` (6 actions, indices 0–5)
- Feature vector: 134 dims — see spec at `docs/superpowers/specs/2026-07-02-neural-cfr-design.md`
- Card encoding: `int` 0–51 where `card = rank_index * 4 + suit_index`, `rank_index = rank - 2` (0=2 … 12=A), `suit_index` 0–3 matching Python `Suit` enum order
- Game constants: `STARTING_STACK = 100.0f`, `BIG_BLIND = 1.0f`, `SMALL_BLIND = 0.5f`
- Reference Python implementations: `cfr/abstract_state.py`, `cfr/mccfr.py`, `cfr/abstraction.py`
- pybind11 module name: `neural_cfr`
- All C++ tests in `neural_cfr/tests/` (gtest), all Python integration tests in `tests/neural_cfr/` (pytest)
- Commit after every task

---

## File Map

```
third_party/
  BUCK                             # prebuilt_cxx_library for libtorch, pybind11, gtest

neural_cfr/
  BUCK                             # cxx_library targets + pybind11 .so target
  src/
    game/
      card.h                       # Card type, deck, hand evaluation
      abstract_state.h             # AbstractState struct declaration
      abstract_state.cpp           # C++ port of cfr/abstract_state.py
    net/
      features.h                   # encode_features() → torch::Tensor [134]
      features.cpp
      mlp.h                        # MLP : torch::nn::Module (shared for adv + strat)
      mlp.cpp
    cfr/
      reservoir_buffer.h           # ReservoirBuffer<T> template (header-only)
      traversal.h                  # external_sample() declaration
      traversal.cpp                # C++ port of cfr/mccfr.py external_sample()
      trainer.h                    # Trainer class declaration
      trainer.cpp                  # alternating traversal → buffer → train loop
    bindings/
      bindings.cpp                 # pybind11 module: Trainer + Strategy classes
  tests/
    BUCK
    test_abstract_state.cpp
    test_features.cpp
    test_reservoir_buffer.cpp
    test_traversal.cpp

scripts/
  train_neural.py                  # ~30-line Python launcher
  eval_openspiel_neural.py         # eval_openspiel.py adapted for neural_cfr.Strategy

tests/
  neural_cfr/
    test_trainer.py
    test_strategy_compat.py
    test_eval_compat.py
```

---

## Task 1: Buck2 Build Scaffolding

**Files:**
- Create: `third_party/BUCK`
- Create: `neural_cfr/BUCK`
- Create: `neural_cfr/src/game/card.h` (stub — `Card` type only)
- Create: `neural_cfr/tests/BUCK`

**Goal:** `buck2 build //neural_cfr:core` compiles successfully.

**Setup before starting:**

```bash
# Download libtorch CPU (macOS ARM):
cd third_party
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip -o libtorch.zip
unzip libtorch.zip  # extracts to third_party/libtorch/

# pybind11 (header-only):
pip install pybind11
# find header path:
python3 -c "import pybind11; print(pybind11.get_include())"
# typical: /path/to/site-packages/pybind11/include
# symlink or copy to third_party/pybind11/include/

# gtest (macOS):
brew install googletest
# headers: /opt/homebrew/include/gtest/
# libs:    /opt/homebrew/lib/libgtest.a, libgtest_main.a
```

- [ ] **Step 1: Write `third_party/BUCK`**

```python
prebuilt_cxx_library(
    name = "libtorch",
    header_dirs = [
        "libtorch/include",
        "libtorch/include/torch/csrc/api/include",
    ],
    shared_libs = {
        "libtorch.dylib":     "libtorch/lib/libtorch.dylib",
        "libtorch_cpu.dylib": "libtorch/lib/libtorch_cpu.dylib",
        "libc10.dylib":       "libtorch/lib/libc10.dylib",
    },
    visibility = ["PUBLIC"],
)

prebuilt_cxx_library(
    name = "pybind11",
    header_dirs = ["pybind11/include"],
    visibility = ["PUBLIC"],
)

prebuilt_cxx_library(
    name = "gtest",
    header_dirs = ["/opt/homebrew/include"],
    static_libs = [
        "/opt/homebrew/lib/libgtest.a",
        "/opt/homebrew/lib/libgtest_main.a",
    ],
    visibility = ["PUBLIC"],
)
```

- [ ] **Step 2: Write `neural_cfr/src/game/card.h` (Card type stub)**

```cpp
#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

// card = rank_index * 4 + suit_index
// rank_index: 0=2, 1=3, ..., 12=A
// suit_index: 0=clubs, 1=diamonds, 2=hearts, 3=spades
using Card = int;

constexpr int NUM_CARDS = 52;

inline int card_rank(Card c) { return c / 4; }   // 0–12
inline int card_suit(Card c) { return c % 4; }   // 0–3

inline std::vector<Card> make_deck() {
    std::vector<Card> deck(NUM_CARDS);
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
    return deck;
}

inline void shuffle_deck(std::vector<Card>& deck) {
    static std::mt19937 rng{std::random_device{}()};
    std::shuffle(deck.begin(), deck.end(), rng);
}
```

- [ ] **Step 3: Write `neural_cfr/BUCK`**

```python
cxx_library(
    name = "core",
    srcs = glob(["src/**/*.cpp"]),
    headers = glob(["src/**/*.h"]),
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O2"],
    deps = ["//third_party:libtorch"],
    preferred_linkage = "static",
    visibility = ["PUBLIC"],
)

cxx_library(
    name = "neural_cfr",
    srcs = ["src/bindings/bindings.cpp"],
    headers = [],
    header_namespace = "",
    compiler_flags = [
        "-std=c++17",
        "-O2",
        # Add Python headers: python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
        # e.g. "-I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13"
    ],
    deps = [":core", "//third_party:pybind11", "//third_party:libtorch"],
    preferred_linkage = "shared",
    soname = "neural_cfr.so",
    visibility = ["PUBLIC"],
)
```

- [ ] **Step 4: Write `neural_cfr/tests/BUCK`**

```python
cxx_test(
    name = "test_abstract_state",
    srcs = ["test_abstract_state.cpp"],
    deps = ["//neural_cfr:core", "//third_party:gtest"],
    compiler_flags = ["-std=c++17"],
)

cxx_test(
    name = "test_features",
    srcs = ["test_features.cpp"],
    deps = ["//neural_cfr:core", "//third_party:gtest"],
    compiler_flags = ["-std=c++17"],
)

cxx_test(
    name = "test_reservoir_buffer",
    srcs = ["test_reservoir_buffer.cpp"],
    deps = ["//neural_cfr:core", "//third_party:gtest"],
    compiler_flags = ["-std=c++17"],
)

cxx_test(
    name = "test_traversal",
    srcs = ["test_traversal.cpp"],
    deps = ["//neural_cfr:core", "//third_party:gtest"],
    compiler_flags = ["-std=c++17"],
)
```

- [ ] **Step 5: Verify build compiles**

```bash
buck2 build //neural_cfr:core
```

Expected: build succeeds (card.h is the only source, no .cpp yet).

- [ ] **Step 6: Commit**

```bash
git add third_party/BUCK neural_cfr/BUCK neural_cfr/src/game/card.h neural_cfr/tests/BUCK
git commit -m "build: Buck2 scaffolding for neural_cfr — libtorch + pybind11 + gtest"
```

---

## Task 2: C++ Game Layer (AbstractState)

**Files:**
- Create: `neural_cfr/src/game/abstract_state.h`
- Create: `neural_cfr/src/game/abstract_state.cpp`
- Update: `neural_cfr/src/game/card.h` (add `evaluate_7card`)
- Create: `neural_cfr/tests/test_abstract_state.cpp`

**Reference:** Port logic from `cfr/abstract_state.py`. Preserve exact game semantics:
- Preflop: p0 (SB) acts first, `to_act = {0, 1}`
- Postflop: p1 (BB/OOP) acts first, `to_act = {1, 0}`
- `betting_history`: raise count per street, capped at 2
- `payoff`: pot × win fraction − amount invested

**Interfaces:**
- Produces: `AbstractState`, `deal_heads_up()`, `legal_abstract_actions()`

- [ ] **Step 1: Write failing test `neural_cfr/tests/test_abstract_state.cpp`**

```cpp
#include <gtest/gtest.h>
#include "game/abstract_state.h"

TEST(AbstractState, DealHeadsUpInitialState) {
    auto state = deal_heads_up();
    EXPECT_EQ(state.street, 0);
    EXPECT_FLOAT_EQ(state.pot, 1.5f);           // SB(0.5) + BB(1.0)
    EXPECT_FLOAT_EQ(state.stacks[0], 99.5f);    // SB posted
    EXPECT_FLOAT_EQ(state.stacks[1], 99.0f);    // BB posted
    EXPECT_FLOAT_EQ(state.current_bet, 1.0f);
    EXPECT_EQ(state.to_act.size(), 2u);
    EXPECT_EQ(state.to_act[0], 0);              // SB acts first preflop
    EXPECT_FALSE(state.is_terminal());
}

TEST(AbstractState, FoldIsTerminal) {
    auto state = deal_heads_up();
    auto next = state.apply_action("fold");
    EXPECT_TRUE(next.is_terminal());
    // p0 folded: p1 wins pot minus p1's investment
    float p1_invest = 100.0f - next.stacks[1];
    EXPECT_FLOAT_EQ(next.payoff(1), next.pot - p1_invest);
    EXPECT_FLOAT_EQ(next.payoff(0), -(100.0f - next.stacks[0]));
}

TEST(AbstractState, CallAdvancesStreet) {
    auto state = deal_heads_up();
    // p0 calls, p1 checks → to_act empty → advance_street
    auto s1 = state.apply_action("call");
    EXPECT_EQ(s1.to_act.size(), 1u);  // p1 still to act
    auto s2 = s1.apply_action("check");
    EXPECT_EQ(s2.to_act.size(), 0u);  // street over
    auto s3 = s2.advance_street();
    EXPECT_EQ(s3.street, 1);           // flop
    EXPECT_EQ(s3.board.size(), 3u);
    EXPECT_EQ(s3.to_act[0], 1);        // BB acts first postflop
}

TEST(AbstractState, BettingHistoryTracksRaises) {
    auto state = deal_heads_up();
    auto s1 = state.apply_action("b0.5");
    EXPECT_EQ(s1.betting_history[0], 1);  // one raise preflop
    auto s2 = s1.apply_action("b1.0");
    EXPECT_EQ(s2.betting_history[0], 2);  // two raises (cap)
    auto s3 = s2.apply_action("b0.5");
    EXPECT_EQ(s3.betting_history[0], 2);  // still capped at 2
}

TEST(AbstractState, LegalActionsPreflop) {
    auto state = deal_heads_up();
    auto actions = state.legal_actions();
    // Facing BB=1.0, p0 can fold/call/raise/allin
    EXPECT_TRUE(std::find(actions.begin(), actions.end(), "fold") != actions.end());
    EXPECT_TRUE(std::find(actions.begin(), actions.end(), "call") != actions.end());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
buck2 test //neural_cfr/tests:test_abstract_state
```

Expected: compile error — `abstract_state.h` not found.

- [ ] **Step 3: Write `neural_cfr/src/game/abstract_state.h`**

```cpp
#pragma once
#include <array>
#include <vector>
#include <string>
#include "card.h"

struct AbstractState {
    std::array<std::array<Card, 2>, 2> hole_cards;  // [player][card]
    std::vector<Card> board;                          // 0–5 cards
    std::vector<Card> deck;                           // remaining runout cards
    int street;                                       // 0=preflop … 3=river
    float pot;
    std::array<float, 2> stacks;
    float current_bet;
    std::array<float, 2> player_bets;               // bets placed this street
    std::vector<int> to_act;                          // action queue
    std::array<int, 4> betting_history;              // raise counts per street, capped at 2
    std::array<bool, 2> folded;

    bool is_terminal() const;
    int acting_player() const;
    float payoff(int player) const;
    std::vector<std::string> legal_actions() const;
    AbstractState apply_action(const std::string& action) const;
    AbstractState advance_street() const;
};

std::vector<std::string> legal_abstract_actions(
    float to_call, float pot, float stack,
    float current_bet, float player_bet);

AbstractState deal_heads_up(
    float starting_stack = 100.0f,
    float big_blind = 1.0f);
```

- [ ] **Step 4: Add `evaluate_7card` to `neural_cfr/src/game/card.h`**

```cpp
// Append to card.h after existing content.
// Returns hand rank: lower = better hand. Brute-force 7C5 selection.

#include <limits>
#include <tuple>

// 5-card hand rank: returns (category, tiebreaker) as uint32_t.
// category 1=straight_flush … 9=high_card (lower = better).
// Internal helper — use evaluate_7card for 7-card hands.
uint32_t evaluate_5card(std::array<Card,5> cards);

// Returns best 5-card rank from 7 cards (lower = better).
uint32_t evaluate_7card(std::array<Card,7> cards);
```

- [ ] **Step 5: Write `neural_cfr/src/game/abstract_state.cpp`**

Port logic from `cfr/abstract_state.py` directly. Key sections:

```cpp
#include "game/abstract_state.h"
#include "game/card.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>

static constexpr float STARTING_STACK = 100.0f;

// ---------- legal_abstract_actions ----------
// Port of cfr/abstraction.py:legal_abstract_actions()
std::vector<std::string> legal_abstract_actions(
    float to_call, float pot, float stack,
    float current_bet, float player_bet)
{
    std::vector<std::string> actions;
    if (to_call > 0.0f) {
        actions.push_back("fold");
        if (stack >= to_call) actions.push_back("call");
        float eff_pot = pot + to_call * 2.0f;
        for (float size : {0.5f, 1.0f}) {
            if (stack > to_call + size * eff_pot)
                actions.push_back(size == 0.5f ? "b0.5" : "b1.0");
        }
        if (stack >= to_call) actions.push_back("allin");
    } else {
        actions.push_back("check");
        for (float size : {0.5f, 1.0f}) {
            if (stack > size * pot)
                actions.push_back(size == 0.5f ? "b0.5" : "b1.0");
        }
        if (stack > 0.0f) actions.push_back("allin");
    }
    return actions;
}

// ---------- AbstractState methods ----------

bool AbstractState::is_terminal() const {
    if (folded[0] || folded[1]) return true;
    if (to_act.empty() && street >= 3) return true;
    return false;
}

int AbstractState::acting_player() const { return to_act[0]; }

float AbstractState::payoff(int player) const {
    float invest = STARTING_STACK - stacks[player];
    if (folded[0]) return player == 1 ? (pot - invest) : -invest;
    if (folded[1]) return player == 0 ? (pot - invest) : -invest;
    // Showdown
    std::array<Card, 7> p0h = {
        hole_cards[0][0], hole_cards[0][1],
        board[0], board[1], board[2], board[3], board[4]
    };
    std::array<Card, 7> p1h = {
        hole_cards[1][0], hole_cards[1][1],
        board[0], board[1], board[2], board[3], board[4]
    };
    uint32_t r0 = evaluate_7card(p0h);
    uint32_t r1 = evaluate_7card(p1h);
    if (r0 < r1)       return player == 0 ? (pot - invest) : -invest;
    if (r1 < r0)       return player == 1 ? (pot - invest) : -invest;
    return pot / 2.0f - invest;  // split
}

std::vector<std::string> AbstractState::legal_actions() const {
    int p = acting_player();
    float to_call = current_bet - player_bets[p];
    return legal_abstract_actions(to_call, pot, stacks[p], current_bet, player_bets[p]);
}

AbstractState AbstractState::apply_action(const std::string& action) const {
    AbstractState next = *this;
    int p = acting_player();

    if (action == "b0.5" || action == "b1.0" || action == "allin") {
        next.betting_history[street] = std::min(next.betting_history[street] + 1, 2);
    }

    if (action == "fold") {
        next.folded[p] = true;
        next.to_act.clear();
    } else if (action == "check") {
        next.to_act.erase(next.to_act.begin());
    } else if (action == "call") {
        float to_call = std::min(current_bet - player_bets[p], stacks[p]);
        next.stacks[p] -= to_call;
        next.player_bets[p] += to_call;
        next.pot += to_call;
        next.to_act.erase(next.to_act.begin());
    } else if (action == "b0.5" || action == "b1.0" || action == "allin") {
        float to_call = current_bet - player_bets[p];
        float additional;
        if (action == "allin") {
            additional = stacks[p];
        } else {
            float size = (action == "b0.5") ? 0.5f : 1.0f;
            float eff_pot = pot + to_call * 2.0f;
            additional = std::min(to_call + size * eff_pot, stacks[p]);
        }
        next.stacks[p] -= additional;
        next.player_bets[p] += additional;
        next.pot += additional;
        next.current_bet = std::max(next.current_bet, next.player_bets[p]);
        // Other non-folded players with chips still to act
        next.to_act.clear();
        for (int i = 0; i < 2; ++i)
            if (!next.folded[i] && next.stacks[i] > 0.0f && i != p)
                next.to_act.push_back(i);
    } else {
        throw std::invalid_argument("Unknown action: " + action);
    }
    return next;
}

AbstractState AbstractState::advance_street() const {
    assert(to_act.empty() && !all_of(folded.begin(), folded.end(), [](bool b){ return b; }));
    AbstractState next = *this;
    next.street = street + 1;
    int cards_to_deal = (next.street == 1) ? 3 : 1;
    for (int i = 0; i < cards_to_deal; ++i) {
        next.board.push_back(next.deck.back());
        next.deck.pop_back();
    }
    next.current_bet = 0.0f;
    next.player_bets = {0.0f, 0.0f};
    // Postflop: p1 (BB/OOP) acts first
    next.to_act.clear();
    for (int i : {1, 0})
        if (!next.folded[i] && next.stacks[i] > 0.0f)
            next.to_act.push_back(i);
    return next;
}

// ---------- deal_heads_up ----------
AbstractState deal_heads_up(float starting_stack, float big_blind) {
    auto deck = make_deck();
    shuffle_deck(deck);

    AbstractState s;
    s.hole_cards[0] = {deck[0], deck[1]};
    s.hole_cards[1] = {deck[2], deck[3]};
    // Pre-deal 5 runout cards (indices 4–8), stored reversed for pop_back
    s.deck = std::vector<Card>(deck.begin() + 4, deck.begin() + 9);
    std::reverse(s.deck.begin(), s.deck.end());

    float sb = std::min(big_blind * 0.5f, starting_stack);
    float bb = std::min(big_blind, starting_stack);

    s.board = {};
    s.street = 0;
    s.pot = sb + bb;
    s.stacks = {starting_stack - sb, starting_stack - bb};
    s.current_bet = bb;
    s.player_bets = {sb, bb};
    s.to_act = {0, 1};  // SB acts first preflop
    s.betting_history = {0, 0, 0, 0};
    s.folded = {false, false};
    return s;
}
```

- [ ] **Step 6: Write `evaluate_5card` and `evaluate_7card` in a new file `neural_cfr/src/game/card.cpp`**

```cpp
#include "game/card.h"
#include <algorithm>
#include <limits>

// Encode 5-card hand rank as uint32_t — lower = better hand.
// Category (bits 28–31): 1=str_flush, 2=quads, 3=full_house,
//   4=flush, 5=straight, 6=trips, 7=two_pair, 8=pair, 9=high_card
uint32_t evaluate_5card(std::array<Card, 5> cards) {
    std::array<int, 5> ranks, suits;
    for (int i = 0; i < 5; ++i) {
        ranks[i] = card_rank(cards[i]);
        suits[i] = card_suit(cards[i]);
    }
    std::sort(ranks.begin(), ranks.end(), std::greater<int>());  // descending

    bool flush = (suits[0]==suits[1] && suits[1]==suits[2] &&
                  suits[2]==suits[3] && suits[3]==suits[4]);

    // Straight detection (including A-2-3-4-5 wheel)
    bool straight = false;
    int straight_high = ranks[0];
    if (ranks[0]-ranks[4] == 4 &&
        std::set<int>(ranks.begin(), ranks.end()).size() == 5) {
        straight = true;
    } else if (ranks[0]==12 && ranks[1]==3 && ranks[2]==2 &&
               ranks[3]==1 && ranks[4]==0) {
        straight = true; straight_high = 3; // wheel: A-2-3-4-5, high=5
    }

    // Count rank frequencies
    std::array<int,13> freq{};
    for (int r : ranks) freq[r]++;
    std::vector<int> counts;
    std::vector<int> rank_by_count;
    for (int r = 12; r >= 0; --r) if (freq[r]) {
        counts.push_back(freq[r]);
        rank_by_count.push_back(r);
    }
    // Sort by count desc, then rank desc
    std::vector<int> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        return counts[a] != counts[b] ? counts[a] > counts[b] : rank_by_count[a] > rank_by_count[b];
    });

    auto cat_rank = [&](uint32_t cat, std::initializer_list<int> kickers) -> uint32_t {
        uint32_t v = cat << 20;
        int shift = 16;
        for (int k : kickers) { v |= (k << shift); shift -= 4; }
        return v;
    };

    if (flush && straight) return cat_rank(1, {straight_high});
    if (counts[order[0]] == 4) return cat_rank(2, {rank_by_count[order[0]], rank_by_count[order[1]]});
    if (counts[order[0]] == 3 && counts[order[1]] == 2)
        return cat_rank(3, {rank_by_count[order[0]], rank_by_count[order[1]]});
    if (flush) return cat_rank(4, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
    if (straight) return cat_rank(5, {straight_high});
    if (counts[order[0]] == 3) return cat_rank(6, {rank_by_count[order[0]],
        rank_by_count[order[1]], rank_by_count[order[2]]});
    if (counts[order[0]] == 2 && counts[order[1]] == 2)
        return cat_rank(7, {rank_by_count[order[0]], rank_by_count[order[1]], rank_by_count[order[2]]});
    if (counts[order[0]] == 2) return cat_rank(8, {rank_by_count[order[0]],
        rank_by_count[order[1]], rank_by_count[order[2]], rank_by_count[order[3]]});
    return cat_rank(9, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
}

uint32_t evaluate_7card(std::array<Card, 7> cards) {
    uint32_t best = std::numeric_limits<uint32_t>::max();
    // All C(7,5) = 21 combinations
    for (int i = 0; i < 7; ++i)
        for (int j = i+1; j < 7; ++j) {
            std::array<Card,5> hand;
            int k = 0;
            for (int x = 0; x < 7; ++x)
                if (x != i && x != j) hand[k++] = cards[x];
            best = std::min(best, evaluate_5card(hand));
        }
    return best;
}
```

- [ ] **Step 7: Run tests**

```bash
buck2 test //neural_cfr/tests:test_abstract_state
```

Expected: all 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add neural_cfr/src/game/
git commit -m "feat(neural_cfr): C++ AbstractState + hand evaluator"
```

---

## Task 3: Feature Encoder

**Files:**
- Create: `neural_cfr/src/net/features.h`
- Create: `neural_cfr/src/net/features.cpp`
- Create: `neural_cfr/tests/test_features.cpp`

**Interfaces:**
- Consumes: `AbstractState` from Task 2
- Produces: `torch::Tensor encode_features(const AbstractState&, int player)` — shape `[134]`, dtype `float32`

Feature layout (indices):
- `[0–33]` — hole cards 2 × 17 (rank one-hot 13 + suit one-hot 4)
- `[34–118]` — board cards 5 × 17, zero-padded
- `[119–122]` — street one-hot
- `[123]` — pot / 200.0f
- `[124]` — stack / 200.0f
- `[125–132]` — betting_history[0..3] each as float, repeated for both players → 4 values
- `[133]` — position (player as float)

- [ ] **Step 1: Write failing test `neural_cfr/tests/test_features.cpp`**

```cpp
#include <gtest/gtest.h>
#include "net/features.h"
#include "game/abstract_state.h"
#include <torch/torch.h>

TEST(Features, ShapeIs134) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    EXPECT_EQ(t.size(0), 134);
    EXPECT_EQ(t.dtype(), torch::kFloat32);
}

TEST(Features, StreetOneHotPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // street=0: index 119 = 1.0, 120–122 = 0.0
    EXPECT_FLOAT_EQ(acc[119], 1.0f);
    EXPECT_FLOAT_EQ(acc[120], 0.0f);
}

TEST(Features, CardRankOneHot) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // hole card 0: exactly one of indices 0–12 should be 1.0
    float rank_sum = 0.0f;
    for (int i = 0; i < 13; ++i) rank_sum += acc[i];
    EXPECT_FLOAT_EQ(rank_sum, 1.0f);
}

TEST(Features, PositionEncoding) {
    auto state = deal_heads_up();
    auto t0 = encode_features(state, 0);
    auto t1 = encode_features(state, 1);
    EXPECT_FLOAT_EQ(t0.accessor<float,1>()[133], 0.0f);
    EXPECT_FLOAT_EQ(t1.accessor<float,1>()[133], 1.0f);
}

TEST(Features, BoardPaddedAtPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // Board cards [34–118] should all be zero at preflop
    for (int i = 34; i < 119; ++i)
        EXPECT_FLOAT_EQ(acc[i], 0.0f) << "index " << i;
}
```

- [ ] **Step 2: Run to verify failure**

```bash
buck2 test //neural_cfr/tests:test_features
```

Expected: compile error — `features.h` not found.

- [ ] **Step 3: Write `neural_cfr/src/net/features.h`**

```cpp
#pragma once
#include <torch/torch.h>
#include "game/abstract_state.h"

constexpr int FEATURE_DIM = 134;
constexpr int NUM_ACTIONS = 6;

// Encode game state from perspective of `player` into a [134] float tensor.
torch::Tensor encode_features(const AbstractState& state, int player);
```

- [ ] **Step 4: Write `neural_cfr/src/net/features.cpp`**

```cpp
#include "net/features.h"
#include "game/card.h"
#include <cstring>

static void encode_card(float* buf, Card c) {
    // 17 floats: 13 rank one-hot + 4 suit one-hot
    std::memset(buf, 0, 17 * sizeof(float));
    if (c >= 0) {
        buf[card_rank(c)] = 1.0f;
        buf[13 + card_suit(c)] = 1.0f;
    }
    // if c < 0: padding card — stays zero
}

torch::Tensor encode_features(const AbstractState& state, int player) {
    auto t = torch::zeros({FEATURE_DIM}, torch::kFloat32);
    float* d = t.data_ptr<float>();

    // [0–33] hole cards (player's own cards first)
    encode_card(d + 0,  state.hole_cards[player][0]);
    encode_card(d + 17, state.hole_cards[player][1]);

    // [34–118] board cards, zero-padded to 5
    for (int i = 0; i < 5; ++i) {
        Card c = (i < (int)state.board.size()) ? state.board[i] : -1;
        encode_card(d + 34 + i * 17, c);
    }

    // [119–122] street one-hot
    d[119 + state.street] = 1.0f;

    // [123] pot normalized, [124] stack normalized
    d[123] = state.pot / 200.0f;
    d[124] = state.stacks[player] / 200.0f;

    // [125–128] betting history (raise counts per street, normalized by 2)
    for (int i = 0; i < 4; ++i)
        d[125 + i] = state.betting_history[i] / 2.0f;

    // [129–132] player bets per street (current street's bet, normalized)
    // Use player_bets[player] for current street contribution
    d[129] = state.player_bets[player] / 200.0f;
    d[130] = state.player_bets[1-player] / 200.0f;
    d[131] = 0.0f;  // reserved — prior street bet info not tracked in state
    d[132] = 0.0f;

    // [133] position
    d[133] = static_cast<float>(player);

    return t;
}
```

- [ ] **Step 5: Run tests**

```bash
buck2 test //neural_cfr/tests:test_features
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add neural_cfr/src/net/features.h neural_cfr/src/net/features.cpp \
        neural_cfr/tests/test_features.cpp
git commit -m "feat(neural_cfr): 134-dim feature encoder — raw card one-hots, no bucketing"
```

---

## Task 4: MLP + Reservoir Buffer

**Files:**
- Create: `neural_cfr/src/net/mlp.h`
- Create: `neural_cfr/src/net/mlp.cpp`
- Create: `neural_cfr/src/cfr/reservoir_buffer.h`
- Create: `neural_cfr/tests/test_reservoir_buffer.cpp`

**Interfaces:**
- Produces: `MLP` (libtorch Module), `ReservoirBuffer<T>`, `BufferEntry`

- [ ] **Step 1: Write failing test `neural_cfr/tests/test_reservoir_buffer.cpp`**

```cpp
#include <gtest/gtest.h>
#include "cfr/reservoir_buffer.h"
#include <vector>
#include <numeric>
#include <cmath>

TEST(ReservoirBuffer, SizeStaysBounded) {
    ReservoirBuffer<int> buf(100);
    for (int i = 0; i < 1000; ++i) buf.add(i);
    EXPECT_EQ(buf.size(), 100u);
}

TEST(ReservoirBuffer, AllItemsBeforeCapacity) {
    ReservoirBuffer<int> buf(50);
    for (int i = 0; i < 50; ++i) buf.add(i);
    EXPECT_EQ(buf.size(), 50u);
    for (int v : buf.data()) EXPECT_GE(v, 0);
}

TEST(ReservoirBuffer, ApproximatelyUniformSampling) {
    // Insert 10x capacity. Each of 10 slots should appear ~equally often
    // across 1000 repeated experiments (chi-squared at loose tolerance).
    const int max_size = 10;
    const int total = 10 * max_size;
    std::vector<int> slot_counts(max_size, 0);
    const int trials = 2000;
    for (int t = 0; t < trials; ++t) {
        ReservoirBuffer<int> buf(max_size);
        for (int i = 0; i < total; ++i) buf.add(i);
        for (int v : buf.data()) slot_counts[v % max_size]++;
    }
    // Each value should appear in ~trials*max_size/total = trials/1 of slots
    // Just check no single value dominates (> 3× expected)
    float expected = (float)(trials * max_size) / total;
    for (int c : slot_counts)
        EXPECT_LT((float)c, expected * 3.0f);
}
```

- [ ] **Step 2: Write `neural_cfr/src/cfr/reservoir_buffer.h`**

```cpp
#pragma once
#include <vector>
#include <random>
#include <cstddef>

template<typename T>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(size_t max_size)
        : max_size_(max_size), n_seen_(0),
          rng_(std::random_device{}()) {
        data_.reserve(max_size);
    }

    void add(T item) {
        ++n_seen_;
        if (data_.size() < max_size_) {
            data_.push_back(std::move(item));
        } else {
            // Uniform random replacement
            size_t idx = std::uniform_int_distribution<size_t>(0, n_seen_ - 1)(rng_);
            if (idx < max_size_)
                data_[idx] = std::move(item);
        }
    }

    const std::vector<T>& data() const { return data_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void clear() { data_.clear(); n_seen_ = 0; }

private:
    size_t max_size_;
    size_t n_seen_;
    std::vector<T> data_;
    std::mt19937_64 rng_;
};

// Entry stored in advantage and strategy buffers.
struct BufferEntry {
    std::vector<float> features;  // 134 floats (stored as vector for easy stacking)
    std::array<float, 6> targets; // advantages or strategy probs, per action
    float weight;                 // iteration number t (linear CFR weighting)
};
```

- [ ] **Step 3: Write `neural_cfr/src/net/mlp.h`**

```cpp
#pragma once
#include <torch/torch.h>

// Shared MLP architecture for both advantage and strategy networks.
// Output activation differs: advantage net uses raw logits,
// strategy net applies softmax at inference (handled by caller).
struct MLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    MLP(int64_t input_dim = 134, int64_t hidden_dim = 256, int64_t output_dim = 6);
    torch::Tensor forward(torch::Tensor x);  // returns raw logits
};
```

- [ ] **Step 4: Write `neural_cfr/src/net/mlp.cpp`**

```cpp
#include "net/mlp.h"

MLP::MLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
    fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, hidden_dim));
    fc4 = register_module("fc4", torch::nn::Linear(hidden_dim, output_dim));
}

torch::Tensor MLP::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    return fc4->forward(x);  // raw logits — no output activation
}
```

- [ ] **Step 5: Run reservoir buffer tests**

```bash
buck2 test //neural_cfr/tests:test_reservoir_buffer
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add neural_cfr/src/net/mlp.h neural_cfr/src/net/mlp.cpp \
        neural_cfr/src/cfr/reservoir_buffer.h \
        neural_cfr/tests/test_reservoir_buffer.cpp
git commit -m "feat(neural_cfr): MLP module + typed reservoir buffer"
```

---

## Task 5: CFR Traversal

**Files:**
- Create: `neural_cfr/src/cfr/traversal.h`
- Create: `neural_cfr/src/cfr/traversal.cpp`
- Create: `neural_cfr/tests/test_traversal.cpp`

**Reference:** Port `external_sample()` from `cfr/mccfr.py`. Replace `RegretTable` calls with:
- `table.get_strategy(infoset, legal)` → regret-match on `adv_net.forward(features)`
- `table.update_regrets(...)` → store `BufferEntry` in `adv_buffer`
- `table.accumulate_strategy(...)` → store `BufferEntry` in `strat_buffer`

**Interfaces:**
- Consumes: `AbstractState`, `MLP`, `ReservoirBuffer<BufferEntry>`, `encode_features()`
- Produces: `float external_sample(state, traversing_player, adv_net, strat_net, adv_buf, strat_buf, iteration)`

- [ ] **Step 1: Write failing test `neural_cfr/tests/test_traversal.cpp`**

```cpp
#include <gtest/gtest.h>
#include "cfr/traversal.h"
#include "cfr/reservoir_buffer.h"
#include "net/mlp.h"
#include "game/abstract_state.h"

TEST(Traversal, BuffersPopulatedAfterTraversal) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(10000), mv1(10000), mpi(10000);

    // Run 20 traversals for each player
    for (int t = 1; t <= 20; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, strat, mv0, mpi, t);
        s = deal_heads_up();
        external_sample(s, 1, adv1, strat, mv1, mpi, t);
    }

    EXPECT_GT(mv0.size(), 0u);
    EXPECT_GT(mv1.size(), 0u);
    EXPECT_GT(mpi.size(), 0u);
}

TEST(Traversal, ReturnedEVIsFinite) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    auto s = deal_heads_up();
    float ev = external_sample(s, 0, adv0, strat, mv0, mpi, 1);
    EXPECT_TRUE(std::isfinite(ev));
}

TEST(Traversal, FeatureDimInBuffer) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    for (int t = 1; t <= 5; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, strat, mv0, mpi, t);
    }
    ASSERT_GT(mv0.size(), 0u);
    EXPECT_EQ(mv0.data()[0].features.size(), 134u);
    EXPECT_EQ(mv0.data()[0].targets.size(), 6u);
}
```

- [ ] **Step 2: Write `neural_cfr/src/cfr/traversal.h`**

```cpp
#pragma once
#include "game/abstract_state.h"
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"

float external_sample(
    const AbstractState& state,
    int traversing_player,
    MLP& adv_net,           // advantage net for traversing player
    MLP& strat_net,         // strategy net (used for opponent sampling)
    ReservoirBuffer<BufferEntry>& adv_buffer,   // M_v[traversing_player]
    ReservoirBuffer<BufferEntry>& strat_buffer, // M_π
    int iteration            // t — used as linear CFR weight
);
```

- [ ] **Step 3: Write `neural_cfr/src/cfr/traversal.cpp`**

Port `external_sample()` from `cfr/mccfr.py`:

```cpp
#include "cfr/traversal.h"
#include "net/features.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>

static const std::array<std::string, 6> ALL_ACTIONS =
    {"fold", "check", "call", "b0.5", "b1.0", "allin"};

// Regret matching: given raw advantage logits and legal action indices,
// return a probability distribution via ReLU + normalize.
static std::vector<float> regret_match(
    const std::array<float, 6>& advantages,
    const std::vector<int>& legal_indices)
{
    std::vector<float> pos(legal_indices.size());
    float total = 0.0f;
    for (size_t i = 0; i < legal_indices.size(); ++i) {
        pos[i] = std::max(0.0f, advantages[legal_indices[i]]);
        total += pos[i];
    }
    if (total > 0.0f)
        for (auto& p : pos) p /= total;
    else
        for (auto& p : pos) p = 1.0f / pos.size();
    return pos;
}

// Sample index from probability distribution
static int sample_action(const std::vector<float>& probs) {
    static std::mt19937 rng{std::random_device{}()};
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Map action string to index in ALL_ACTIONS
static int action_idx(const std::string& a) {
    for (int i = 0; i < 6; ++i)
        if (ALL_ACTIONS[i] == a) return i;
    return -1;
}

float external_sample(
    const AbstractState& state,
    int traversing_player,
    MLP& adv_net,
    MLP& strat_net,
    ReservoirBuffer<BufferEntry>& adv_buffer,
    ReservoirBuffer<BufferEntry>& strat_buffer,
    int iteration)
{
    if (state.is_terminal())
        return state.payoff(traversing_player);

    // Chance node: advance street
    if (state.to_act.empty())
        return external_sample(state.advance_street(), traversing_player,
                               adv_net, strat_net, adv_buffer, strat_buffer, iteration);

    int acting = state.acting_player();
    auto legal_strs = state.legal_actions();
    std::vector<int> legal_idx;
    for (auto& a : legal_strs) legal_idx.push_back(action_idx(a));

    // Encode features for acting player
    auto feat_tensor = encode_features(state, acting);
    auto feat_vec = std::vector<float>(
        feat_tensor.data_ptr<float>(),
        feat_tensor.data_ptr<float>() + FEATURE_DIM);

    if (acting == traversing_player) {
        // Query advantage net → regret match → traverse ALL actions
        torch::NoGradGuard no_grad;
        auto logits = adv_net.forward(feat_tensor.unsqueeze(0)).squeeze(0);
        std::array<float, 6> advantages{};
        for (int i = 0; i < 6; ++i) advantages[i] = logits[i].item<float>();

        auto strategy = regret_match(advantages, legal_idx);

        // Traverse all legal actions
        std::array<float, 6> action_values{};
        float node_value = 0.0f;
        for (size_t i = 0; i < legal_strs.size(); ++i) {
            float v = external_sample(state.apply_action(legal_strs[i]),
                                      traversing_player, adv_net, strat_net,
                                      adv_buffer, strat_buffer, iteration);
            action_values[legal_idx[i]] = v;
            node_value += strategy[i] * v;
        }

        // Compute advantages and store in M_v
        std::array<float, 6> adv_targets{};
        for (size_t i = 0; i < legal_strs.size(); ++i)
            adv_targets[legal_idx[i]] = action_values[legal_idx[i]] - node_value;

        adv_buffer.add({feat_vec, adv_targets, static_cast<float>(iteration)});

        // Also accumulate strategy for M_π
        std::array<float, 6> strat_targets{};
        for (size_t i = 0; i < legal_strs.size(); ++i)
            strat_targets[legal_idx[i]] = strategy[i];
        strat_buffer.add({feat_vec, strat_targets, static_cast<float>(iteration)});

        return node_value;

    } else {
        // Opponent: query strategy net → sample ONE action
        torch::NoGradGuard no_grad;
        auto logits = strat_net.forward(feat_tensor.unsqueeze(0)).squeeze(0);
        std::array<float, 6> raw{};
        for (int i = 0; i < 6; ++i) raw[i] = logits[i].item<float>();

        // Softmax over legal actions
        std::vector<float> legal_logits(legal_idx.size());
        for (size_t i = 0; i < legal_idx.size(); ++i)
            legal_logits[i] = raw[legal_idx[i]];
        float max_l = *std::max_element(legal_logits.begin(), legal_logits.end());
        float sum = 0.0f;
        for (auto& l : legal_logits) { l = std::exp(l - max_l); sum += l; }
        for (auto& l : legal_logits) l /= sum;

        // Accumulate strategy for M_π
        std::array<float, 6> strat_targets{};
        for (size_t i = 0; i < legal_idx.size(); ++i)
            strat_targets[legal_idx[i]] = legal_logits[i];
        strat_buffer.add({feat_vec, strat_targets, static_cast<float>(iteration)});

        int chosen = sample_action(legal_logits);
        return external_sample(state.apply_action(legal_strs[chosen]),
                               traversing_player, adv_net, strat_net,
                               adv_buffer, strat_buffer, iteration);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
buck2 test //neural_cfr/tests:test_traversal
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add neural_cfr/src/cfr/traversal.h neural_cfr/src/cfr/traversal.cpp \
        neural_cfr/tests/test_traversal.cpp
git commit -m "feat(neural_cfr): external sampling MCCFR traversal with network queries"
```

---

## Task 6: Trainer

**Files:**
- Create: `neural_cfr/src/cfr/trainer.h`
- Create: `neural_cfr/src/cfr/trainer.cpp`

**Interfaces:**
- Consumes: `MLP`, `ReservoirBuffer<BufferEntry>`, `external_sample()`
- Produces: `Trainer` class with `run()`, `checkpoint()`, `load()`

- [ ] **Step 1: Write `neural_cfr/src/cfr/trainer.h`**

```cpp
#pragma once
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"
#include <string>
#include <memory>
#include <torch/optim.h>

class Trainer {
public:
    explicit Trainer(
        size_t reservoir_size = 2'000'000,
        size_t batch_size     = 4096,
        float  lr             = 1e-4f);

    void run(int iterations);
    void checkpoint(const std::string& path);
    void load(const std::string& path);

private:
    MLP adv0_, adv1_, strat_;
    ReservoirBuffer<BufferEntry> mv0_, mv1_, mpi_;
    torch::optim::Adam opt_adv0_, opt_adv1_, opt_strat_;
    size_t batch_size_;

    // Train a network on its buffer using weighted loss.
    // mode: "advantage" uses weighted MSE; "strategy" uses weighted cross-entropy.
    void train_step(MLP& net, torch::optim::Adam& opt,
                    ReservoirBuffer<BufferEntry>& buffer,
                    const std::string& mode);
};
```

- [ ] **Step 2: Write `neural_cfr/src/cfr/trainer.cpp`**

```cpp
#include "cfr/trainer.h"
#include "cfr/traversal.h"
#include "game/abstract_state.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>

Trainer::Trainer(size_t reservoir_size, size_t batch_size, float lr)
    : mv0_(reservoir_size), mv1_(reservoir_size), mpi_(reservoir_size),
      opt_adv0_(adv0_.parameters(), torch::optim::AdamOptions(lr)),
      opt_adv1_(adv1_.parameters(), torch::optim::AdamOptions(lr)),
      opt_strat_(strat_.parameters(), torch::optim::AdamOptions(lr)),
      batch_size_(batch_size)
{}

void Trainer::train_step(MLP& net, torch::optim::Adam& opt,
                         ReservoirBuffer<BufferEntry>& buffer,
                         const std::string& mode)
{
    if (buffer.size() < batch_size_) return;  // not enough data yet

    // Sample a random batch
    const auto& data = buffer.data();
    static std::mt19937 rng{std::random_device{}()};
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(batch_size_);

    // Stack features, targets, weights into tensors
    auto feat_t   = torch::zeros({(int64_t)batch_size_, FEATURE_DIM});
    auto target_t = torch::zeros({(int64_t)batch_size_, 6});
    auto weight_t = torch::zeros({(int64_t)batch_size_});

    auto fa = feat_t.accessor<float, 2>();
    auto ta = target_t.accessor<float, 2>();
    auto wa = weight_t.accessor<float, 1>();

    for (size_t i = 0; i < batch_size_; ++i) {
        const auto& e = data[indices[i]];
        for (int j = 0; j < FEATURE_DIM; ++j) fa[i][j] = e.features[j];
        for (int j = 0; j < 6; ++j)           ta[i][j] = e.targets[j];
        wa[i] = e.weight;
    }

    // Normalize weights
    weight_t = weight_t / weight_t.sum();

    opt.zero_grad();
    auto pred = net.forward(feat_t);

    torch::Tensor loss;
    if (mode == "advantage") {
        // Weighted MSE
        auto diff = (pred - target_t).pow(2).sum(1);  // [batch]
        loss = (diff * weight_t).sum();
    } else {
        // Weighted cross-entropy (strategy net)
        auto log_softmax = torch::log_softmax(pred, 1);
        auto ce = -(target_t * log_softmax).sum(1);  // [batch]
        loss = (ce * weight_t).sum();
    }

    loss.backward();
    opt.step();
}

void Trainer::run(int iterations) {
    for (int t = 1; t <= iterations; ++t) {
        // Player 0 traversal
        {
            auto s = deal_heads_up();
            external_sample(s, 0, adv0_, strat_, mv0_, mpi_, t);
        }
        train_step(adv0_, opt_adv0_, mv0_, "advantage");
        train_step(strat_, opt_strat_, mpi_, "strategy");

        // Player 1 traversal
        {
            auto s = deal_heads_up();
            external_sample(s, 1, adv1_, strat_, mv1_, mpi_, t);
        }
        train_step(adv1_, opt_adv1_, mv1_, "advantage");
        train_step(strat_, opt_strat_, mpi_, "strategy");

        if (t % 1000 == 0)
            std::cout << "Iteration " << t << " / " << iterations
                      << "  |  buffers: mv0=" << mv0_.size()
                      << " mv1=" << mv1_.size()
                      << " mpi=" << mpi_.size() << "\n";
    }
}

void Trainer::checkpoint(const std::string& path) {
    torch::serialize::OutputArchive archive;
    adv0_.save(archive);
    adv1_.save(archive);
    strat_.save(archive);
    archive.save_to(path);
    std::cout << "Checkpoint saved to " << path << "\n";
}

void Trainer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("Checkpoint not found: " + path);
    f.close();
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    adv0_.load(archive);
    adv1_.load(archive);
    strat_.load(archive);
    std::cout << "Checkpoint loaded from " << path << "\n";
}
```

- [ ] **Step 3: Quick smoke test (no gtest needed — compile only)**

```bash
buck2 build //neural_cfr:core
```

Expected: builds without errors.

- [ ] **Step 4: Commit**

```bash
git add neural_cfr/src/cfr/trainer.h neural_cfr/src/cfr/trainer.cpp
git commit -m "feat(neural_cfr): Trainer — alternating traversal + buffer + Adam training loop"
```

---

## Task 7: pybind11 Bindings

**Files:**
- Create: `neural_cfr/src/bindings/bindings.cpp`

**Interfaces:**
- Produces: `neural_cfr.Trainer`, `neural_cfr.Strategy` — Python-callable classes
- `Strategy.get_action_probs` takes raw game state (not InfoSet buckets) — see signature below

**Note on eval_openspiel.py:** `eval_openspiel.py` creates `InfoSet` objects using bucket abstraction. The neural strategy takes raw card ints. A separate `scripts/eval_openspiel_neural.py` (Task 8) adapts the interface.

- [ ] **Step 1: Write `neural_cfr/src/bindings/bindings.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include "cfr/trainer.h"
#include "net/mlp.h"
#include "net/features.h"
#include "game/abstract_state.h"

namespace py = pybind11;

// Strategy wrapper: loads a strat_ network from checkpoint, provides inference.
class Strategy {
public:
    explicit Strategy(const std::string& checkpoint_path) {
        std::ifstream f(checkpoint_path);
        if (!f.good())
            throw std::runtime_error("Checkpoint not found: " + checkpoint_path);
        torch::serialize::InputArchive archive;
        archive.load_from(checkpoint_path);
        // The checkpoint saves adv0, adv1, strat_ in that order.
        // Skip adv0 and adv1, load strat_.
        MLP skip1, skip2;
        skip1.load(archive);
        skip2.load(archive);
        net_.load(archive);
        net_.eval();
    }

    // Returns action→probability map for legal actions.
    // hole_cards: [c0, c1] as 0-51 ints (player's own cards)
    // board_cards: 0-5 cards as 0-51 ints
    // street: 0-3
    // pot, stack: raw float values (normalized internally)
    // raises_per_street: list of 4 ints
    // position: 0 or 1
    py::dict get_action_probs(
        std::vector<int> hole_cards,
        std::vector<int> board_cards,
        int street, float pot, float stack,
        std::vector<int> raises_per_street,
        int position)
    {
        // Build a synthetic AbstractState for feature encoding
        AbstractState s{};
        s.hole_cards[position][0] = hole_cards[0];
        s.hole_cards[position][1] = hole_cards[1];
        s.board = std::vector<Card>(board_cards.begin(), board_cards.end());
        s.street = street;
        s.pot = pot;
        s.stacks[position] = stack;
        s.stacks[1 - position] = stack;  // approximation
        s.player_bets = {0.0f, 0.0f};
        s.current_bet = 0.0f;
        s.betting_history = {0, 0, 0, 0};
        for (int i = 0; i < 4 && i < (int)raises_per_street.size(); ++i)
            s.betting_history[i] = raises_per_street[i];
        s.folded = {false, false};
        s.to_act = {position};

        auto feat = encode_features(s, position);

        torch::NoGradGuard no_grad;
        auto logits = net_.forward(feat.unsqueeze(0)).squeeze(0);

        // Get legal actions for this state
        auto legal = s.legal_actions();

        // Softmax over legal actions
        std::vector<float> legal_logits;
        for (auto& a : legal)
            legal_logits.push_back(logits[action_idx(a)].item<float>());
        float maxl = *std::max_element(legal_logits.begin(), legal_logits.end());
        float sum = 0.0f;
        for (auto& l : legal_logits) { l = std::exp(l - maxl); sum += l; }
        for (auto& l : legal_logits) l /= sum;

        py::dict result;
        for (size_t i = 0; i < legal.size(); ++i)
            result[py::str(legal[i])] = legal_logits[i];
        return result;
    }

private:
    MLP net_;

    static int action_idx(const std::string& a) {
        static const std::array<std::string, 6> ALL =
            {"fold","check","call","b0.5","b1.0","allin"};
        for (int i = 0; i < 6; ++i) if (ALL[i] == a) return i;
        return -1;
    }
};

PYBIND11_MODULE(neural_cfr, m) {
    m.doc() = "Deep CFR neural network strategy — C++ core via libtorch";

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<size_t, size_t, float>(),
             py::arg("reservoir_size") = 2'000'000,
             py::arg("batch_size")     = 4096,
             py::arg("lr")             = 1e-4f)
        .def("run",        &Trainer::run,        py::arg("iterations"))
        .def("checkpoint", &Trainer::checkpoint, py::arg("path"))
        .def("load",       &Trainer::load,       py::arg("path"));

    py::class_<Strategy>(m, "Strategy")
        .def(py::init<const std::string&>(), py::arg("checkpoint_path"))
        .def("get_action_probs", &Strategy::get_action_probs,
             py::arg("hole_cards"),
             py::arg("board_cards"),
             py::arg("street"),
             py::arg("pot"),
             py::arg("stack"),
             py::arg("raises_per_street"),
             py::arg("position"));
}
```

- [ ] **Step 2: Build the .so**

```bash
buck2 build //neural_cfr:neural_cfr
```

Expected: `neural_cfr.so` produced in Buck2 output directory.

- [ ] **Step 3: Smoke test the import**

```bash
# Add Buck2 output dir to PYTHONPATH (adjust path as needed)
PYTHONPATH=$(buck2 root)/buck-out/gen/neural_cfr python3 -c "import neural_cfr; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add neural_cfr/src/bindings/bindings.cpp
git commit -m "feat(neural_cfr): pybind11 bindings — Trainer + Strategy Python interface"
```

---

## Task 8: Python Launcher + Integration Tests

**Files:**
- Create: `scripts/train_neural.py`
- Create: `scripts/eval_openspiel_neural.py`
- Create: `tests/neural_cfr/test_trainer.py`
- Create: `tests/neural_cfr/test_strategy_compat.py`
- Create: `tests/neural_cfr/test_eval_compat.py`

**Interfaces:**
- Consumes: `neural_cfr.Trainer`, `neural_cfr.Strategy` from Task 7

- [ ] **Step 1: Write `scripts/train_neural.py`**

```python
#!/usr/bin/env python3
"""Neural CFR training launcher. Calls into C++ Trainer via pybind11."""
import argparse
import os
import sys

# Adjust to actual Buck2 output path for neural_cfr.so
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'buck-out', 'gen', 'neural_cfr'))

import neural_cfr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations',      type=int,   default=100_000)
    parser.add_argument('--reservoir-size',  type=int,   default=2_000_000)
    parser.add_argument('--batch-size',      type=int,   default=4096)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--checkpoint',      type=str,   default='neural_cfr/checkpoints/checkpoint.pt')
    parser.add_argument('--resume',          type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    trainer = neural_cfr.Trainer(
        reservoir_size=args.reservoir_size,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if args.resume:
        trainer.load(args.resume)

    trainer.run(args.iterations)
    trainer.checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Write `tests/neural_cfr/test_trainer.py`**

```python
import os, sys, pytest, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'buck-out', 'gen', 'neural_cfr'))
import neural_cfr


def test_buffers_populated_after_run():
    """Trainer.run() should populate internal buffers — indirectly verified via no crash."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(50)  # 50 iterations — fast smoke test


def test_checkpoint_roundtrip():
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(10)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        trainer.checkpoint(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        # Load into a fresh trainer — should not raise
        trainer2 = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
        trainer2.load(path)
    finally:
        os.unlink(path)


def test_load_missing_file_raises():
    trainer = neural_cfr.Trainer()
    with pytest.raises(RuntimeError):
        trainer.load("/tmp/definitely_does_not_exist.pt")
```

- [ ] **Step 3: Write `tests/neural_cfr/test_strategy_compat.py`**

```python
import os, sys, pytest, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'buck-out', 'gen', 'neural_cfr'))
import neural_cfr


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    """Train briefly and save a checkpoint for strategy tests."""
    trainer = neural_cfr.Trainer(reservoir_size=500, batch_size=64, lr=1e-3)
    trainer.run(20)
    path = str(tmp_path_factory.mktemp("ckpt") / "test.pt")
    trainer.checkpoint(path)
    return path


def test_strategy_loads(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    assert strat is not None


def test_get_action_probs_valid_distribution(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    probs = strat.get_action_probs(
        hole_cards=[0, 1],       # 2c, 2d
        board_cards=[],
        street=0,
        pot=1.5,
        stack=99.0,
        raises_per_street=[0, 0, 0, 0],
        position=0,
    )
    assert isinstance(probs, dict)
    assert len(probs) > 0
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"
    for v in probs.values():
        assert 0.0 <= v <= 1.0


def test_get_action_probs_postflop(checkpoint_path):
    strat = neural_cfr.Strategy(checkpoint_path)
    # Flop: AhKhQh board
    board = [48, 44, 40]  # Ah=12*4+3, Kh=11*4+3, Qh=10*4+3
    probs = strat.get_action_probs(
        hole_cards=[2, 6],   # 3c, 4c
        board_cards=board,
        street=1,
        pot=4.0,
        stack=96.0,
        raises_per_street=[1, 0, 0, 0],
        position=1,
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-5
```

- [ ] **Step 4: Write `tests/neural_cfr/test_eval_compat.py`**

```python
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'buck-out', 'gen', 'neural_cfr'))
import neural_cfr


def test_strategy_interface_matches_contract():
    """Verify Strategy exposes the interface expected by eval_openspiel_neural.py."""
    import inspect
    strat_cls = neural_cfr.Strategy
    # get_action_probs must exist
    assert hasattr(strat_cls, 'get_action_probs')


def test_missing_checkpoint_raises():
    import pytest
    with pytest.raises(RuntimeError, match="not found"):
        neural_cfr.Strategy("/tmp/nonexistent.pt")
```

- [ ] **Step 5: Write `scripts/eval_openspiel_neural.py`**

This is a minimal adaptation of `scripts/eval_openspiel.py`. Copy `eval_openspiel.py` and change only the strategy construction and `CFRBotPolicy.action_probabilities`:

```python
"""
eval_openspiel_neural.py — eval_openspiel.py adapted for neural_cfr.Strategy.
Usage: uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'buck-out', 'gen', 'neural_cfr'))

import neural_cfr
# Import everything from eval_openspiel except the strategy-related parts
from scripts.eval_openspiel import (
    parse_args, run_evaluation,
    _parse_info_state,       # extracts hole_cards, board_cards, raises_per_street, street
    _is_check_action,
)
import pyspiel


class NeuralCFRPolicy(pyspiel.Policy):
    """Wraps neural_cfr.Strategy as an OpenSpiel Policy."""

    def __init__(self, checkpoint_path: str):
        self.strategy = neural_cfr.Strategy(checkpoint_path)

    def action_probabilities(self, state, player=None):
        info_str = state.information_state_string(player or state.current_player())
        parsed = _parse_info_state(info_str)
        probs = self.strategy.get_action_probs(
            hole_cards=parsed['hole_cards'],
            board_cards=parsed['board_cards'],
            street=parsed['street'],
            pot=parsed['pot'],
            stack=parsed['stack'],
            raises_per_street=parsed['raises_per_street'],
            position=parsed['position'],
        )
        # Map abstract action probs to OpenSpiel action indices
        legal = state.legal_actions(state.current_player())
        result = {}
        for action in legal:
            action_str = state.action_to_string(state.current_player(), action)
            result[action] = probs.get(action_str, 0.0)
        # Renormalize (in case of unmapped actions)
        total = sum(result.values()) or 1.0
        return {k: v / total for k, v in result.items()}


if __name__ == '__main__':
    args = parse_args()
    policy = NeuralCFRPolicy(args.checkpoint)
    run_evaluation(policy, args)
```

- [ ] **Step 6: Run Python integration tests**

```bash
uv run pytest tests/neural_cfr/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_neural.py scripts/eval_openspiel_neural.py \
        tests/neural_cfr/
git commit -m "feat(neural_cfr): Python launcher, eval adapter, integration tests"
```

---

## Self-Review

**Spec coverage:**
- ✅ Two networks (adv + strat) — Tasks 4, 5
- ✅ External sampling MCCFR traversal — Task 5
- ✅ Reservoir sampling buffers — Task 4
- ✅ Linear CFR weighting (iteration t as weight) — Task 5, 6
- ✅ 134-dim raw card feature encoding — Task 3
- ✅ 6-action vocabulary — Global Constraints + Task 5
- ✅ Buck2 build with libtorch + pybind11 + gtest — Task 1
- ✅ pybind11 Trainer + Strategy — Task 7
- ✅ Python launcher — Task 8
- ✅ eval_openspiel adaptation — Task 8
- ✅ C++ unit tests (gtest) in `neural_cfr/tests/` — Tasks 2–5
- ✅ Python integration tests in `tests/neural_cfr/` — Task 8
- ✅ Error handling: missing checkpoint raises RuntimeError — Tasks 6, 7, 8
- ✅ Uniform distribution fallback at inference for unknown infoset — traversal.cpp regret_match()

**Type consistency:**
- `BufferEntry.features` is `std::vector<float>` (size 134) — consistent Tasks 4, 5, 6
- `BufferEntry.targets` is `std::array<float, 6>` — consistent Tasks 4, 5, 6
- `FEATURE_DIM = 134` constant used in features.h, trainer.cpp, bindings.cpp
- `external_sample()` signature consistent across traversal.h, traversal.cpp, trainer.cpp
- `MLP` used as value type in Trainer — ensure copy constructors work or use `shared_ptr` if linker errors arise
