# DREAM Blueprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tabular `BlueprintTrainer` in `sixmax/` with a DREAM neural blueprint (outcome-sampling MCCFR + advantage/strategy nets) that generalizes across variable stack sizes (20–250 BB) and player counts (2–6).

**Architecture:** New `sixmax/src/dream/` directory (features, nets, reservoir, trainer, checkpoint) alongside the frozen `src/blueprint/`. `DreamTrainer` runs outcome-sampling MCCFR with IS-weighted advantage targets on randomized EngineGame configurations. `DreamStrategy` (inference-only) loads the strategy net from `SIXDM001` checkpoints.

**Tech Stack:** C++17 + libtorch (same as `neural_cfr/`), Buck2 build, pybind11 Python surface, pytest.

## Global Constraints

- All chip values in BB; training frame `big_blind=1.0`.
- Module boundary: `sixmax/` never imports `cfr/` or `neural_cfr/`; shared C++ only via `//common:evaluator`.
- Action vocabulary order is immutable: fold/check/call/opens/bets/allin — never reorder.
- Always mask illegal actions; never skip or reorder storage.
- `raises_` uncapped in engine, clipped to 5 in feature encoding, capped at 3 inside `abstract_key()` only (tabular compatibility during transition).
- Checkpoint magic `SIXDM001`; refuse load on vocab hash mismatch.
- `FEATURE_DIM=154`, `CHIP_NORM=100.0f`, `RAISE_NORM=5.0f`.
- Build command: `~/bin/buck2 build //sixmax:sixmax`
- Test command: `uv run pytest tests/sixmax/ -x -q`
- Never commit checkpoint `.bin`/`.pt` files or secrets.

---

## File Map

**New files:**
- `sixmax/src/dream/features.h` / `features.cpp` — 154-dim feature encoder
- `sixmax/src/dream/nets.h` / `nets.cpp` — libtorch MLP definition
- `sixmax/src/dream/reservoir.h` / `reservoir.cpp` — weighted reservoir buffers
- `sixmax/src/dream/trainer.h` / `trainer.cpp` — DreamTrainer (outcome-sampling traversal + retraining loop)
- `sixmax/src/dream/checkpoint.h` / `checkpoint.cpp` — SIXDM001 format + DreamStrategy
- `tests/sixmax/test_dream_features.py`
- `tests/sixmax/test_dream_nets.py`
- `tests/sixmax/test_dream_reservoir.py`
- `tests/sixmax/test_dream_checkpoint.py`
- `tests/sixmax/test_dream_kuhn.py`
- `scripts/train_dream.py`

**Modified files:**
- `sixmax/src/blueprint/engine_game.h` — add public accessors for feature encoder
- `sixmax/src/blueprint/engine_game.cpp` — uncap `raises_[st]`; cap inside `abstract_key()` at 3
- `sixmax/src/blueprint/kuhn.h` — add `card(int p)` and `history_code()` to `KuhnState`
- `sixmax/BUCK` — split into `core` + `sixmax` targets; add libtorch to `core`
- `sixmax/src/bindings/bindings.cpp` — expose `DreamStrategy`
- `sixmax/configs/default.toml` — add `[train.dream]` section
- `scripts/openpoker_bot.py` — update auto-detect for `dream_*.pt`

---

## Task 1: Engine raise cap removal + EngineGameState accessors

**Files:**
- Modify: `sixmax/src/blueprint/engine_game.cpp`
- Modify: `sixmax/src/blueprint/engine_game.h`

**Interfaces:**
- Produces: `EngineGameState::hole_cards(int p)`, `board()`, `player_state(int p)`, `num_players()`, `street()`, `pot()`, `to_call()`; raises tracked without cap

- [ ] **Step 1: Uncap raises in apply(), cap inside abstract_key()**

In `sixmax/src/blueprint/engine_game.cpp`, find the apply() method. The current line (around line 90):
```cpp
if (raises_[st] < 3) ++raises_[st];  // capped per-street count
```
Change to:
```cpp
++raises_[st];
```

Then in `abstract_key()`, the call to `pack_abstract_key` passes `raises_` directly. Wrap it with a capped copy so the tabular key is unchanged:
```cpp
// before the pack_abstract_key call, add:
std::array<uint8_t, 4> capped_raises;
for (int i = 0; i < 4; ++i)
    capped_raises[i] = std::min(raises_[i], uint8_t(3));
// then change the pack_abstract_key call from raises_ to capped_raises:
uint64_t key = pack_abstract_key((int)card, street, capped_raises,
                                 hand_.pot(), live, after);
```

- [ ] **Step 2: Add public accessors to EngineGameState**

In `sixmax/src/blueprint/engine_game.h`, inside the `EngineGameState` class (public section), add:

```cpp
// Accessors for neural feature encoding
std::array<int, 2> hole_cards(int p) const { return hand_.hole_cards(p); }
std::vector<int>   board()           const { return hand_.board(); }
const PlayerState& player_state(int p) const { return hand_.player(p); }
int  num_players() const { return hand_.num_players(); }
Street street()    const { return hand_.street(); }
double pot()       const { return hand_.pot(); }
double to_call()   const { return hand_.to_call(); }
```

- [ ] **Step 3: Verify the change compiles**

```bash
~/bin/buck2 build //sixmax:sixmax 2>&1 | tail -20
```
Expected: build succeeds (no new errors). If the test suite was passing before, run:
```bash
uv run pytest tests/sixmax/ -x -q 2>&1 | tail -10
```
Expected: same pass count as before.

- [ ] **Step 4: Commit**

```bash
git add sixmax/src/blueprint/engine_game.h sixmax/src/blueprint/engine_game.cpp
git commit -m "feat(dream): uncap raises_ in engine; add EngineGameState accessors for feature encoder"
```

---

## Task 2: Feature encoding

**Files:**
- Create: `sixmax/src/dream/features.h`
- Create: `sixmax/src/dream/features.cpp`

**Interfaces:**
- Produces: `sixmax::encode_state(const EngineGameState&) -> torch::Tensor` (shape [154], float32, CPU)
- Constants: `FEATURE_DIM=154`, `CHIP_NORM=100.0f`, `RAISE_NORM=5.0f`

- [ ] **Step 1: Create features.h**

```cpp
// sixmax/src/dream/features.h
#pragma once
#include <torch/torch.h>
#include "blueprint/engine_game.h"

namespace sixmax {

inline constexpr int   FEATURE_DIM = 154;
inline constexpr float CHIP_NORM   = 100.0f;
inline constexpr float RAISE_NORM  = 5.0f;

// Encode the current EngineGameState into a 154-dim float tensor (CPU).
// Layout (see spec for full table):
//   0–33:   hole cards × 17 (13 rank + 4 suit one-hot)
//   34–118: board cards × 17, zero-padded
//   119–122: street one-hot
//   123:    pot / CHIP_NORM
//   124–129: stack[seat 0-5] / CHIP_NORM (0 if absent or all-in)
//   130–135: street_bet[seat 0-5] / CHIP_NORM
//   136–141: live_mask[seat 0-5] in {0,1}
//   142–147: acting player one-hot
//   148:    n_active / 6.0
//   149:    to_call / CHIP_NORM
//   150–153: min(raises_per_street[0-3], 5) / RAISE_NORM
torch::Tensor encode_state(const EngineGameState& state);

}  // namespace sixmax
```

- [ ] **Step 2: Create features.cpp**

```cpp
// sixmax/src/dream/features.cpp
#include "dream/features.h"
#include <algorithm>

namespace sixmax {

namespace {

void encode_card(float* out, int card_code) {
    // card_code = (rank-2)*4 + suit; rank 0-12, suit 0-3
    for (int i = 0; i < 17; ++i) out[i] = 0.0f;
    out[card_code / 4] = 1.0f;      // rank one-hot (0-12)
    out[13 + card_code % 4] = 1.0f; // suit one-hot (0-3)
}

void encode_card_or_pad(float* out, int card_code) {
    if (card_code < 0) { for (int i = 0; i < 17; ++i) out[i] = 0.0f; }
    else encode_card(out, card_code);
}

}  // namespace

torch::Tensor encode_state(const EngineGameState& state) {
    auto t = torch::zeros({FEATURE_DIM}, torch::kFloat32);
    float* d = t.data_ptr<float>();

    int p = state.current_player();

    // 0–33: hole cards (2 × 17)
    auto hc = state.hole_cards(p);
    encode_card(d + 0,  hc[0]);
    encode_card(d + 17, hc[1]);

    // 34–118: board (5 × 17), zero-pad missing cards
    auto board = state.board();
    board.resize(5, -1);  // pad to 5
    for (int i = 0; i < 5; ++i)
        encode_card_or_pad(d + 34 + i * 17, board[i]);

    // 119–122: street one-hot
    d[119 + (int)state.street()] = 1.0f;

    // 123: pot
    d[123] = (float)(state.pot() / CHIP_NORM);

    // 124–141: per-seat stacks, street bets, live mask
    int n = state.num_players();
    int n_active = 0;
    for (int s = 0; s < 6; ++s) {
        if (s < n) {
            const auto& ps = state.player_state(s);
            bool live = !ps.folded && !ps.all_in;
            d[124 + s] = (float)(ps.stack / CHIP_NORM);
            d[130 + s] = (float)(ps.street_bet / CHIP_NORM);
            d[136 + s] = live ? 1.0f : 0.0f;
            if (!ps.folded) ++n_active;
        }
        // seats >= n stay 0
    }

    // 142–147: acting player one-hot
    if (p < 6) d[142 + p] = 1.0f;

    // 148: n_active / 6
    d[148] = (float)n_active / 6.0f;

    // 149: to_call
    d[149] = (float)(state.to_call() / CHIP_NORM);

    // 150–153: raises_per_street clipped at 5 / RAISE_NORM
    // raises_ is private; access via bet_context or a dedicated accessor.
    // Add raises_per_street() to EngineGameState (see note below).
    // For now, use state.raises_per_street() which returns std::array<uint8_t,4>.
    auto raises = state.raises_per_street();
    for (int i = 0; i < 4; ++i)
        d[150 + i] = std::min((int)raises[i], 5) / RAISE_NORM;

    return t;
}

}  // namespace sixmax
```

- [ ] **Step 3: Add raises_per_street() accessor to EngineGameState**

The feature encoder needs `raises_`. Add to `engine_game.h` (public section):

```cpp
std::array<uint8_t, 4> raises_per_street() const { return raises_; }
```

- [ ] **Step 4: Commit**

```bash
git add sixmax/src/dream/features.h sixmax/src/dream/features.cpp \
        sixmax/src/blueprint/engine_game.h
git commit -m "feat(dream): 154-dim feature encoder for EngineGameState"
```

---

## Task 3: MLP nets

**Files:**
- Create: `sixmax/src/dream/nets.h`
- Create: `sixmax/src/dream/nets.cpp`

**Interfaces:**
- Produces: `sixmax::DreamMLP` — libtorch module, `forward(tensor) -> tensor`
- Constructor: `DreamMLP(int input_dim, int hidden_size, int n_layers, int output_dim)`

- [ ] **Step 1: Create nets.h**

```cpp
// sixmax/src/dream/nets.h
#pragma once
#include <torch/torch.h>

namespace sixmax {

// Fully-connected MLP with ReLU activations.
// Architecture: input_dim -> [hidden_size]*n_layers -> output_dim
// Output is raw logits (no softmax/sigmoid).
struct DreamMLPImpl : torch::nn::Module {
    DreamMLPImpl(int input_dim, int hidden_size, int n_layers, int output_dim);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::Sequential layers_{nullptr};
};
TORCH_MODULE(DreamMLP);

}  // namespace sixmax
```

- [ ] **Step 2: Create nets.cpp**

```cpp
// sixmax/src/dream/nets.cpp
#include "dream/nets.h"

namespace sixmax {

DreamMLPImpl::DreamMLPImpl(int input_dim, int hidden_size,
                           int n_layers, int output_dim) {
    torch::nn::Sequential seq;
    int in = input_dim;
    for (int i = 0; i < n_layers; ++i) {
        seq->push_back(torch::nn::Linear(in, hidden_size));
        seq->push_back(torch::nn::ReLU());
        in = hidden_size;
    }
    seq->push_back(torch::nn::Linear(in, output_dim));
    layers_ = register_module("layers", seq);
}

torch::Tensor DreamMLPImpl::forward(torch::Tensor x) {
    return layers_->forward(x);
}

}  // namespace sixmax
```

- [ ] **Step 3: Commit**

```bash
git add sixmax/src/dream/nets.h sixmax/src/dream/nets.cpp
git commit -m "feat(dream): DreamMLP libtorch module"
```

---

## Task 4: Weighted reservoir buffers

**Files:**
- Create: `sixmax/src/dream/reservoir.h`
- Create: `sixmax/src/dream/reservoir.cpp`

**Interfaces:**
- Produces: `sixmax::WeightedReservoir` — `add(features, target, weight)`, `sample_batch(n, rng) -> (features, targets, weights)`, `size()`, `clear()`

- [ ] **Step 1: Create reservoir.h**

```cpp
// sixmax/src/dream/reservoir.h
#pragma once
#include <mutex>
#include <random>
#include <torch/torch.h>

namespace sixmax {

// Thread-safe reservoir with weighted replacement sampling.
// Capacity-bounded: when full, each new entry replaces a random existing
// entry with probability capacity/n_seen (standard reservoir algorithm).
class WeightedReservoir {
public:
    explicit WeightedReservoir(size_t capacity) : capacity_(capacity) {}

    // Add one (features, target, weight) triple. Thread-safe.
    void add(torch::Tensor features, torch::Tensor target, float weight,
             std::mt19937_64& rng);

    // Sample batch_size entries uniformly. Returns (features, targets, weights)
    // as stacked tensors. Caller must hold no lock.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_batch(size_t batch_size, std::mt19937_64& rng) const;

    size_t size() const;
    void clear();

private:
    struct Entry { torch::Tensor features, target; float weight; };
    size_t capacity_;
    size_t n_seen_ = 0;
    std::vector<Entry> entries_;
    mutable std::mutex mu_;
};

}  // namespace sixmax
```

- [ ] **Step 2: Create reservoir.cpp**

```cpp
// sixmax/src/dream/reservoir.cpp
#include "dream/reservoir.h"
#include <stdexcept>

namespace sixmax {

void WeightedReservoir::add(torch::Tensor features, torch::Tensor target,
                            float weight, std::mt19937_64& rng) {
    std::lock_guard<std::mutex> lock(mu_);
    ++n_seen_;
    Entry e{features.clone(), target.clone(), weight};
    if (entries_.size() < capacity_) {
        entries_.push_back(std::move(e));
    } else {
        // Replace random existing entry with probability capacity/n_seen
        size_t idx = std::uniform_int_distribution<size_t>(0, n_seen_ - 1)(rng);
        if (idx < capacity_) entries_[idx] = std::move(e);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
WeightedReservoir::sample_batch(size_t batch_size, std::mt19937_64& rng) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.empty())
        throw std::runtime_error("WeightedReservoir::sample_batch: empty reservoir");
    size_t n = entries_.size();
    std::vector<torch::Tensor> fs, ts;
    std::vector<float> ws;
    fs.reserve(batch_size); ts.reserve(batch_size); ws.reserve(batch_size);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        const Entry& e = entries_[dist(rng)];
        fs.push_back(e.features);
        ts.push_back(e.target);
        ws.push_back(e.weight);
    }
    return {torch::stack(fs), torch::stack(ts),
            torch::tensor(ws, torch::kFloat32)};
}

size_t WeightedReservoir::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

void WeightedReservoir::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
    n_seen_ = 0;
}

}  // namespace sixmax
```

- [ ] **Step 3: Commit**

```bash
git add sixmax/src/dream/reservoir.h sixmax/src/dream/reservoir.cpp
git commit -m "feat(dream): thread-safe weighted reservoir buffer"
```

---

## Task 5: DreamTrainer

**Files:**
- Create: `sixmax/src/dream/trainer.h`
- Create: `sixmax/src/dream/trainer.cpp`

**Interfaces:**
- Consumes: `EngineGameState` accessors (Task 1), `encode_state` (Task 2), `DreamMLP` (Task 3), `WeightedReservoir` (Task 4)
- Produces: `sixmax::DreamTrainer` — `train(uint64_t iterations)`, `save_nets(path)`

- [ ] **Step 1: Create trainer.h**

```cpp
// sixmax/src/dream/trainer.h
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include "blueprint/engine_game.h"
#include "dream/nets.h"
#include "dream/reservoir.h"

namespace sixmax {

struct DreamConfig {
    int    hidden_size    = 256;
    int    hidden_layers  = 3;
    float  lr             = 1e-3f;
    int    batch_size     = 4096;
    size_t reservoir_size = 2'000'000;
    int    train_interval = 10'000;   // traversals between retraining
    int    sgd_steps      = 2'000;
    float  epsilon        = 0.06f;    // ε-greedy exploration
    int    num_threads    = 1;        // <=0 = hardware_concurrency
    uint64_t seed         = 42;
    // Stack randomization: LogNormal(ln(100), 0.5) clipped to [stack_min, stack_max]
    float  stack_min      = 20.0f;
    float  stack_max      = 250.0f;
    float  stack_log_mean = 4.605f;   // ln(100)
    float  stack_log_std  = 0.5f;
    // Players per hand: uniform [players_min, players_max]
    int    players_min    = 2;
    int    players_max    = 6;
};

using DreamGameFactory = std::function<std::unique_ptr<EngineGame>()>;

class DreamTrainer {
public:
    DreamTrainer(int n_actions, const ActionVocab* vocab,
                 const Abstraction* abstraction,
                 DreamGameFactory factory, DreamConfig cfg,
                 torch::Device device);

    void train(uint64_t iterations);
    uint64_t total_iterations() const { return iter_.load(std::memory_order_relaxed); }

    // Expose nets for checkpoint saving
    DreamMLP adv_net() const { return adv_net_; }
    DreamMLP strat_net() const { return strat_net_; }

private:
    struct TrajectoryNode {
        torch::Tensor features;
        std::vector<float> sigma;  // strategy at this node
        int a_star;                // sampled action
        int player;
    };

    void worker(uint64_t n_iterations, uint64_t seed_offset);
    void traverse(EngineGame& engine, std::mt19937_64& rng,
                  std::vector<TrajectoryNode>& traj,
                  std::vector<float>& utilities);
    std::vector<float> eps_greedy_strategy(const torch::Tensor& advantages,
                                           const std::vector<uint8_t>& mask) const;
    void retrain_adv();
    void retrain_strat();

    int n_actions_;
    const ActionVocab* vocab_;
    const Abstraction* abstraction_;
    DreamGameFactory factory_;
    DreamConfig cfg_;
    torch::Device device_;

    DreamMLP adv_net_{nullptr};
    DreamMLP strat_net_{nullptr};
    std::mutex adv_net_mu_;     // guards adv_net_ during retraining
    std::mutex strat_net_mu_;

    WeightedReservoir M_v_;   // advantage samples
    WeightedReservoir M_pi_;  // strategy samples

    std::atomic<uint64_t> iter_{0};
};

}  // namespace sixmax
```

- [ ] **Step 2: Create trainer.cpp (traversal)**

```cpp
// sixmax/src/dream/trainer.cpp
#include "dream/trainer.h"
#include "dream/features.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

namespace sixmax {

DreamTrainer::DreamTrainer(int n_actions, const ActionVocab* vocab,
                           const Abstraction* abstraction,
                           DreamGameFactory factory, DreamConfig cfg,
                           torch::Device device)
    : n_actions_(n_actions), vocab_(vocab), abstraction_(abstraction),
      factory_(std::move(factory)), cfg_(cfg), device_(device),
      adv_net_(DreamMLP(FEATURE_DIM, cfg.hidden_size, cfg.hidden_layers, n_actions)),
      strat_net_(DreamMLP(FEATURE_DIM, cfg.hidden_size, cfg.hidden_layers, n_actions)),
      M_v_(cfg.reservoir_size), M_pi_(cfg.reservoir_size) {
    adv_net_->to(device_);
    strat_net_->to(device_);
}

std::vector<float> DreamTrainer::eps_greedy_strategy(
    const torch::Tensor& adv, const std::vector<uint8_t>& mask) const {
    // Regret matching on advantages, then mix with uniform(legal) at rate epsilon
    int n = n_actions_;
    std::vector<float> sigma(n, 0.0f);
    // Positive regret matching over legal actions
    float pos_sum = 0.0f;
    auto adv_a = adv.accessor<float, 1>();
    for (int i = 0; i < n; ++i) {
        if (mask[i]) { sigma[i] = std::max(0.0f, adv_a[i]); pos_sum += sigma[i]; }
    }
    int n_legal = (int)std::count(mask.begin(), mask.end(), uint8_t(1));
    if (pos_sum < 1e-9f) {
        // All non-positive: uniform over legal
        float u = 1.0f / n_legal;
        for (int i = 0; i < n; ++i) sigma[i] = mask[i] ? u : 0.0f;
    } else {
        for (int i = 0; i < n; ++i) sigma[i] /= pos_sum;
    }
    // ε-greedy mix: σ' = (1-ε)*σ + ε*uniform(legal)
    float eps = cfg_.epsilon;
    float u = eps / n_legal;
    for (int i = 0; i < n; ++i) {
        if (mask[i]) sigma[i] = (1.0f - eps) * sigma[i] + u;
    }
    return sigma;
}

void DreamTrainer::traverse(EngineGame& engine, std::mt19937_64& rng,
                            std::vector<TrajectoryNode>& traj,
                            std::vector<float>& utilities) {
    auto state_ptr = engine.new_hand(rng);
    auto* state = static_cast<EngineGameState*>(state_ptr.get());

    traj.clear();
    // Walk forward: at each node, sample one action per ε-greedy strategy
    while (!state->is_terminal()) {
        int p = state->current_player();
        torch::Tensor feat;
        {
            torch::NoGradGuard no_grad;
            feat = encode_state(*state).to(device_);
        }

        std::vector<uint8_t> mask;
        state->legal_mask(mask);

        torch::Tensor adv_out;
        {
            std::lock_guard<std::mutex> lock(adv_net_mu_);
            torch::NoGradGuard no_grad;
            adv_out = adv_net_->forward(feat.unsqueeze(0)).squeeze(0).cpu();
        }

        auto sigma = eps_greedy_strategy(adv_out, mask);

        // Sample one action
        std::discrete_distribution<int> dist(sigma.begin(), sigma.end());
        int a_star = dist(rng);

        traj.push_back({feat.cpu(), sigma, a_star, p});
        state->apply(a_star);
    }

    // Collect utilities for all players
    int n = state->num_players();
    utilities.resize(n);
    for (int i = 0; i < n; ++i)
        utilities[i] = (float)state->utility(i);
}

void DreamTrainer::retrain_adv() {
    // Reinitialize advantage net from scratch (REINIT_ADV)
    {
        std::lock_guard<std::mutex> lock(adv_net_mu_);
        adv_net_ = DreamMLP(FEATURE_DIM, cfg_.hidden_size, cfg_.hidden_layers, n_actions_);
        adv_net_->to(device_);
    }
    torch::optim::Adam opt(adv_net_->parameters(),
                           torch::optim::AdamOptions(cfg_.lr));
    std::mt19937_64 rng(iter_.load());
    for (int step = 0; step < cfg_.sgd_steps; ++step) {
        auto [feat, target, weights] = M_v_.sample_batch(cfg_.batch_size, rng);
        feat = feat.to(device_);
        target = target.to(device_);
        weights = weights.to(device_);
        opt.zero_grad();
        auto pred = adv_net_->forward(feat);
        // Weighted MSE over all action dims
        auto loss = ((pred - target).pow(2) * weights.unsqueeze(1)).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(adv_net_->parameters(), 1.0);
        opt.step();
    }
}

void DreamTrainer::retrain_strat() {
    torch::optim::Adam opt(strat_net_->parameters(),
                           torch::optim::AdamOptions(cfg_.lr));
    std::mt19937_64 rng(iter_.load() + 1);
    for (int step = 0; step < cfg_.sgd_steps; ++step) {
        auto [feat, target, weights] = M_pi_.sample_batch(cfg_.batch_size, rng);
        feat = feat.to(device_);
        target = target.to(device_);
        weights = weights.to(device_);
        opt.zero_grad();
        auto pred = torch::log_softmax(strat_net_->forward(feat), /*dim=*/1);
        // Weighted cross-entropy: -Σ target * log(pred), per-sample weighted
        auto loss = -(target * pred * weights.unsqueeze(1)).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(strat_net_->parameters(), 1.0);
        opt.step();
    }
}

void DreamTrainer::worker(uint64_t n_iterations, uint64_t seed_offset) {
    std::mt19937_64 rng(cfg_.seed + seed_offset);
    std::lognormal_distribution<float> stack_dist(cfg_.stack_log_mean,
                                                   cfg_.stack_log_std);
    std::uniform_int_distribution<int> n_players_dist(cfg_.players_min,
                                                       cfg_.players_max);
    auto engine_factory = factory_();
    std::vector<TrajectoryNode> traj;
    std::vector<float> utilities;

    for (uint64_t it = 0; it < n_iterations; ++it) {
        // Randomize game config per hand
        int n_players = n_players_dist(rng);
        std::vector<double> stacks(n_players);
        for (auto& s : stacks)
            s = (double)std::clamp(stack_dist(rng), cfg_.stack_min, cfg_.stack_max);
        engine_factory->set_config(EngineConfig{n_players, 100.0});
        engine_factory->set_stacks(stacks);

        traverse(*engine_factory, rng, traj, utilities);

        uint64_t t = iter_.fetch_add(1, std::memory_order_relaxed) + 1;
        float weight = (float)t;

        // For each node in trajectory: store advantage + strategy samples
        for (const auto& node : traj) {
            int p = node.player;
            float u_p = p < (int)utilities.size() ? utilities[p] : 0.0f;
            float prob_star = node.sigma[node.a_star];
            if (prob_star < 1e-9f) continue;  // safety: skip near-zero IS weight

            // IS-corrected sparse advantage vector
            std::vector<float> adv_target(n_actions_, 0.0f);
            adv_target[node.a_star] = u_p / prob_star;
            auto adv_t = torch::tensor(adv_target, torch::kFloat32);
            M_v_.add(node.features, adv_t, weight, rng);

            // Strategy sample
            auto strat_t = torch::tensor(node.sigma, torch::kFloat32);
            M_pi_.add(node.features, strat_t, weight, rng);
        }

        // Retrain every train_interval traversals
        if (t % cfg_.train_interval == 0 && M_v_.size() >= (size_t)cfg_.batch_size) {
            retrain_adv();
            retrain_strat();
        }
    }
}

void DreamTrainer::train(uint64_t iterations) {
    int n_threads = cfg_.num_threads <= 0
        ? (int)std::thread::hardware_concurrency()
        : cfg_.num_threads;
    if (n_threads == 1) {
        worker(iterations, 0);
        return;
    }
    uint64_t per_thread = iterations / n_threads;
    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; ++i)
        threads.emplace_back(&DreamTrainer::worker, this, per_thread, (uint64_t)i);
    for (auto& t : threads) t.join();
}

}  // namespace sixmax
```

**Implementation note:** `EngineGame` in the current codebase does not have `set_config()` / `set_stacks()` methods — `EngineConfig` and stack overrides are passed at construction time. The factory lambda captures these by reference and reconstructs the engine per-hand:

Replace the `engine_factory->set_config(...)` / `set_stacks(...)` calls with a factory that constructs a fresh `EngineGame` per-hand config. Modify the `DreamGameFactory` to accept the randomized parameters: simplest approach is to not use a GameFactory at all, and instead have the worker directly create `HandState` with randomized stacks:

```cpp
// Instead of engine_factory->set_config/set_stacks, do:
EngineConfig cfg{n_players, 100.0};
// Create a fresh HandState directly with randomized stacks:
auto state = HandState::deal(cfg, /*button=*/0, rng);
// But we need the vocab and abstraction for EngineGameState...
// Wrap as: EngineGameState(std::move(state), vocab_, abstraction_)
```

Remove `DreamGameFactory` from the header and store `vocab_` and `abstraction_` for use inside `worker()`. Replace the `traverse()` first line with:
```cpp
EngineGameState state(HandState::deal(cfg, /*button=*/rng() % n_players, rng),
                      vocab_, abstraction_);
```
Adjust the traverse signature accordingly.

- [ ] **Step 3: Commit**

```bash
git add sixmax/src/dream/trainer.h sixmax/src/dream/trainer.cpp
git commit -m "feat(dream): DreamTrainer — outcome-sampling MCCFR with IS-weighted advantages"
```

---

## Task 6: Checkpoint + DreamStrategy

**Files:**
- Create: `sixmax/src/dream/checkpoint.h`
- Create: `sixmax/src/dream/checkpoint.cpp`

**Interfaces:**
- Consumes: `DreamMLP` (Task 3), `ActionVocab`
- Produces: `DreamStrategy::load(path, device)`, `DreamStrategy::get_probs(state) -> vector<double>`

- [ ] **Step 1: Create checkpoint.h**

```cpp
// sixmax/src/dream/checkpoint.h
#pragma once
#include <string>
#include <torch/torch.h>
#include "blueprint/engine_game.h"
#include "dream/nets.h"
#include "vocab/vocab.h"

namespace sixmax {

// Save SIXDM001 checkpoint. Atomically writes to path (tmp+rename).
// Embeds vocab_hash and iteration count in "meta" sub-archive.
void save_dream_checkpoint(const std::string& path,
                           const DreamMLP& adv_net,
                           const DreamMLP& strat_net,
                           uint64_t iterations,
                           uint64_t vocab_hash);

// Inference-only strategy loaded from a SIXDM001 checkpoint.
// Loads only the "strat" sub-archive.
class DreamStrategy {
public:
    static DreamStrategy load(const std::string& path, torch::Device device,
                              const ActionVocab& vocab);

    // Returns softmax probabilities over all vocab actions.
    // Illegal actions (per legal_mask) are zeroed and the result renormalized.
    std::vector<double> get_probs(const EngineGameState& state) const;

    uint64_t iterations() const { return iterations_; }

private:
    DreamStrategy() = default;
    DreamMLP strat_net_{nullptr};
    torch::Device device_{torch::kCPU};
    uint64_t iterations_ = 0;
    int n_actions_ = 0;
    const ActionVocab* vocab_ = nullptr;
};

}  // namespace sixmax
```

- [ ] **Step 2: Create checkpoint.cpp**

```cpp
// sixmax/src/dream/checkpoint.cpp
#include "dream/checkpoint.h"
#include "dream/features.h"
#include <filesystem>
#include <stdexcept>

namespace sixmax {

static constexpr uint32_t kMagic = 0x5349584D;  // 'SIXM' (SIXDM001)
static const std::string kMagicStr = "SIXDM001";

void save_dream_checkpoint(const std::string& path,
                           const DreamMLP& adv_net,
                           const DreamMLP& strat_net,
                           uint64_t iterations,
                           uint64_t vocab_hash) {
    std::string tmp = path + ".tmp";
    torch::serialize::OutputArchive root;

    // Save nets
    torch::serialize::OutputArchive adv_ar, strat_ar;
    adv_net->save(adv_ar);
    strat_net->save(strat_ar);
    root.write("adv", adv_ar);
    root.write("strat", strat_ar);

    // Save meta
    torch::serialize::OutputArchive meta;
    meta.write("magic",      torch::tensor((int64_t)kMagic));
    meta.write("iterations", torch::tensor((int64_t)iterations));
    meta.write("vocab_hash", torch::tensor((int64_t)vocab_hash));
    root.write("meta", meta);

    root.save_to(tmp);
    std::filesystem::rename(tmp, path);
}

DreamStrategy DreamStrategy::load(const std::string& path,
                                  torch::Device device,
                                  const ActionVocab& vocab) {
    torch::serialize::InputArchive root;
    root.load_from(path);

    // Verify magic and vocab hash
    torch::serialize::InputArchive meta;
    root.read("meta", meta);
    torch::Tensor vh_t;
    meta.read("vocab_hash", vh_t);
    uint64_t saved_hash = (uint64_t)vh_t.item<int64_t>();
    if (saved_hash != vocab.hash())
        throw std::runtime_error("DreamStrategy::load: vocab hash mismatch — "
                                 "checkpoint was trained with a different action set");

    torch::Tensor iter_t;
    meta.read("iterations", iter_t);

    DreamStrategy s;
    s.device_ = device;
    s.vocab_ = &vocab;
    s.n_actions_ = (int)vocab.size();
    s.iterations_ = (uint64_t)iter_t.item<int64_t>();

    // Load strategy net only
    s.strat_net_ = DreamMLP(FEATURE_DIM, 256, 3, s.n_actions_);
    torch::serialize::InputArchive strat_ar;
    root.read("strat", strat_ar);
    s.strat_net_->load(strat_ar);
    s.strat_net_->to(device);
    s.strat_net_->eval();

    return s;
}

std::vector<double> DreamStrategy::get_probs(const EngineGameState& state) const {
    torch::NoGradGuard no_grad;
    auto feat = encode_state(state).to(device_).unsqueeze(0);
    auto logits = strat_net_->forward(feat).squeeze(0).cpu();

    std::vector<uint8_t> mask;
    state.legal_mask(const_cast<std::vector<uint8_t>&>(mask));

    // Zero illegal logits to -inf before softmax
    auto logits_a = logits.accessor<float, 1>();
    for (int i = 0; i < n_actions_; ++i)
        if (!mask[i]) logits_a[i] = -1e9f;

    auto probs = torch::softmax(logits, 0);
    auto probs_a = probs.accessor<float, 1>();

    std::vector<double> result(n_actions_);
    for (int i = 0; i < n_actions_; ++i) result[i] = (double)probs_a[i];
    return result;
}

}  // namespace sixmax
```

**Implementation note:** `legal_mask()` takes a non-const `std::vector<uint8_t>&`. Since `get_probs()` is const, the cast above is safe because legal_mask only writes to its argument. Alternatively, make the local mask a mutable local variable and pass that.

- [ ] **Step 3: Commit**

```bash
git add sixmax/src/dream/checkpoint.h sixmax/src/dream/checkpoint.cpp
git commit -m "feat(dream): SIXDM001 checkpoint format + DreamStrategy inference"
```

---

## Task 7: BUCK update + configs + bindings

**Files:**
- Modify: `sixmax/BUCK`
- Modify: `sixmax/src/bindings/bindings.cpp`
- Modify: `sixmax/configs/default.toml`

- [ ] **Step 1: Update sixmax/BUCK**

Read the current `sixmax/BUCK` and replace with a two-target layout (mirrors `neural_cfr/BUCK`):

```python
_python_include = read_config("python", "include_path", "")
_cpu_tune = read_config("build", "cpu_tune", "native")

# Core: all implementation including dream/, no pybind11
cxx_library(
    name = "core",
    srcs = [
        "src/vocab/vocab.cpp",
        "src/blueprint/kuhn.cpp",
        "src/blueprint/mccfr.cpp",
        "src/blueprint/trainer.cpp",
        "src/engine/engine.cpp",
        "src/blueprint/engine_game.cpp",
        "src/abstraction/abstraction.cpp",
        "src/abstraction/abstract_key.cpp",
        "src/blueprint/checkpoint.cpp",
        "src/dream/features.cpp",
        "src/dream/nets.cpp",
        "src/dream/reservoir.cpp",
        "src/dream/trainer.cpp",
        "src/dream/checkpoint.cpp",
    ],
    headers = glob(["src/**/*.h"]),
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"],
    exported_preprocessor_flags = ["-Isixmax/src"],
    deps = ["//common:evaluator", "//third_party:libtorch"],
    exported_deps = ["//common:evaluator"],
    preferred_linkage = "static",
    visibility = ["PUBLIC"],
)

# Python extension: bindings only
cxx_library(
    name = "sixmax",
    srcs = ["src/bindings/bindings.cpp"],
    headers = [],
    header_namespace = "",
    compiler_flags = ["-std=c++17", "-O3", "-mcpu=" + _cpu_tune, "-DNDEBUG"] +
                     (["-I" + _python_include] if _python_include else []),
    deps = [":core", "//third_party:pybind11", "//third_party:libtorch"],
    preferred_linkage = "shared",
    soname = "sixmax.so",
    linker_flags = ["-undefined", "dynamic_lookup"] if host_info().os.is_macos else [],
    visibility = ["PUBLIC"],
)
```

- [ ] **Step 2: Add DreamStrategy to bindings.cpp**

Read `sixmax/src/bindings/bindings.cpp`. Find where `BlueprintStrategy` is exposed to Python. Add `DreamStrategy` immediately after:

```cpp
#include "dream/checkpoint.h"  // add this include at the top

// In the PYBIND11_MODULE block, after BlueprintStrategy binding:
py::class_<DreamStrategy>(m, "DreamStrategy")
    .def_static("load",
        [](const std::string& path, const std::string& device_str,
           const ActionVocab& vocab) {
            auto device = torch::Device(device_str);
            return DreamStrategy::load(path, device, vocab);
        },
        py::arg("path"), py::arg("device") = "cpu", py::arg("vocab"))
    .def("get_probs",
        [](const DreamStrategy& s, const EngineGameState& state) {
            return s.get_probs(state);
        })
    .def_property_readonly("iterations", &DreamStrategy::iterations);
```

Also expose `DreamTrainer` for the training script:

```cpp
#include "dream/trainer.h"

py::class_<DreamTrainer>(m, "DreamTrainer")
    .def(py::init([](int n_actions, const ActionVocab* vocab,
                     const Abstraction* abstraction,
                     DreamConfig cfg, const std::string& device_str) {
        return std::make_unique<DreamTrainer>(
            n_actions, vocab, abstraction, cfg,
            torch::Device(device_str));
    }))
    .def("train", &DreamTrainer::train)
    .def("total_iterations", &DreamTrainer::total_iterations)
    .def("save", [](const DreamTrainer& t, const std::string& path,
                    uint64_t vocab_hash) {
        save_dream_checkpoint(path, t.adv_net(), t.strat_net(),
                              t.total_iterations(), vocab_hash);
    });

py::class_<DreamConfig>(m, "DreamConfig")
    .def(py::init<>())
    .def_readwrite("hidden_size", &DreamConfig::hidden_size)
    .def_readwrite("hidden_layers", &DreamConfig::hidden_layers)
    .def_readwrite("lr", &DreamConfig::lr)
    .def_readwrite("batch_size", &DreamConfig::batch_size)
    .def_readwrite("reservoir_size", &DreamConfig::reservoir_size)
    .def_readwrite("train_interval", &DreamConfig::train_interval)
    .def_readwrite("sgd_steps", &DreamConfig::sgd_steps)
    .def_readwrite("epsilon", &DreamConfig::epsilon)
    .def_readwrite("num_threads", &DreamConfig::num_threads)
    .def_readwrite("seed", &DreamConfig::seed)
    .def_readwrite("stack_min", &DreamConfig::stack_min)
    .def_readwrite("stack_max", &DreamConfig::stack_max)
    .def_readwrite("players_min", &DreamConfig::players_min)
    .def_readwrite("players_max", &DreamConfig::players_max);
```

- [ ] **Step 3: Add [train.dream] to default.toml**

Append to `sixmax/configs/default.toml`:

```toml
[train.dream]
hidden_size    = 256
hidden_layers  = 3
lr             = 1e-3
batch_size     = 4096
reservoir_size = 2_000_000
train_interval = 10_000
sgd_steps      = 2_000
epsilon        = 0.06
num_threads    = 0        # 0 = hardware_concurrency
seed           = 42
stack_min      = 20.0
stack_max      = 250.0
stack_log_mean = 4.605    # ln(100)
stack_log_std  = 0.5
players_min    = 2
players_max    = 6
device         = "cpu"    # "mps" for MacBook, "cuda" for cloud
checkpoint_interval = 0   # 0 = save at end only
checkpoint     = "sixmax/checkpoints/dream.pt"
```

- [ ] **Step 4: Build**

```bash
~/bin/buck2 build //sixmax:sixmax 2>&1 | tail -30
```
Expected: `BUILD SUCCEEDED`. Fix any compile errors before proceeding.

- [ ] **Step 5: Smoke import**

```bash
uv run python -c "import sys; sys.path.insert(0, '.'); import sixmax; print(dir(sixmax))"
```
Expected: output includes `DreamStrategy`, `DreamTrainer`, `DreamConfig`.

- [ ] **Step 6: Commit**

```bash
git add sixmax/BUCK sixmax/src/bindings/bindings.cpp sixmax/configs/default.toml
git commit -m "feat(dream): wire BUCK build, expose DreamStrategy+DreamTrainer to Python, add [train.dream] config"
```

---

## Task 8: Python unit tests (Tier 1)

**Files:**
- Create: `tests/sixmax/test_dream_features.py`
- Create: `tests/sixmax/test_dream_nets.py`
- Create: `tests/sixmax/test_dream_reservoir.py`
- Create: `tests/sixmax/test_dream_checkpoint.py`

The existing `tests/sixmax/conftest.py` force-loads `sixmax.so` — all tests below rely on that fixture. Read that file before writing tests to understand the fixture pattern.

- [ ] **Step 1: Write test_dream_features.py**

```python
# tests/sixmax/test_dream_features.py
import torch, sixmax

def _make_state(n_players=6, n_board=0):
    """Helper: deal a hand with the default vocab, return EngineGameState."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "vocab_config", pathlib.Path("sixmax/vocab_config.py"))
    vc = importlib.util.module_from_spec(spec); spec.loader.exec_module(vc)
    vocab = vc.load_vocab("sixmax/configs/default.toml", "blueprint")
    cfg = sixmax.EngineConfig(n_players, 100.0)
    # Use a fixed seed game state — pull from fixtures if available
    # For now, just check that blueprint strategy produces a state we can encode
    state = sixmax.BlueprintStrategy  # not what we want — use engine directly
    # NOTE: If EngineGameState cannot be constructed directly from Python,
    # use a running BlueprintStrategy to get a real state, or call
    # sixmax.HandState.deal() if exposed. Check conftest.py for the pattern
    # used in existing tests to get an EngineGameState.

def test_feature_dim(blueprint_hu_ckpt, default_vocab):
    """Encoding produces a 154-dim tensor."""
    # Use blueprint_hu_ckpt fixture to get a real game state
    # (see existing tests for how to iterate the engine to a mid-hand state)
    # Placeholder: check constant is correct
    assert sixmax.FEATURE_DIM == 154

def test_chip_norm():
    assert sixmax.CHIP_NORM == 100.0

def test_raise_norm():
    assert sixmax.RAISE_NORM == 5.0
```

**Implementation note:** Check `tests/sixmax/conftest.py` and existing test files (e.g., `test_blueprint_range_chart.py`) to find the fixture that provides a live `EngineGameState` mid-hand. The `blueprint_hu_ckpt` fixture or a simple traversal loop is likely the right entry point. Expand the tests above using that fixture to verify:
- Output shape is `(154,)`
- Hole card dims 0–33 sum to 2.0 (each card: exactly 2 ones)
- Board dims 34–118 sum to 3*2=6 on the flop, 0 preflop
- Street one-hot sums to 1.0
- Acting player one-hot sums to 1.0
- All values finite (no NaN/inf)

- [ ] **Step 2: Write test_dream_nets.py**

```python
# tests/sixmax/test_dream_nets.py
import torch, sixmax

def test_output_shape_6_actions():
    cfg = sixmax.DreamConfig()
    net = sixmax.DreamMLP(154, 256, 3, 6)
    x = torch.randn(4, 154)
    out = net(x)
    assert out.shape == (4, 6)

def test_output_shape_10_actions():
    net = sixmax.DreamMLP(154, 256, 3, 10)
    x = torch.randn(1, 154)
    out = net(x)
    assert out.shape == (1, 10)

def test_output_shape_matches_vocab(default_vocab):
    n = default_vocab.size()
    net = sixmax.DreamMLP(154, 64, 2, n)
    x = torch.randn(2, 154)
    assert net(x).shape == (2, n)
```

**Note:** `default_vocab` fixture must be defined in conftest.py or inline. Check existing conftest.py; if not present, add:
```python
@pytest.fixture
def default_vocab():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "vocab_config", pathlib.Path("sixmax/vocab_config.py"))
    vc = importlib.util.module_from_spec(spec); spec.loader.exec_module(vc)
    return vc.load_vocab("sixmax/configs/default.toml", "blueprint")
```

- [ ] **Step 3: Write test_dream_reservoir.py**

```python
# tests/sixmax/test_dream_reservoir.py
import torch, sixmax
import random

def _rng():
    return random.Random(42)  # not used directly; reservoir uses C++ rng

def test_size_cap():
    r = sixmax.WeightedReservoir(capacity=10)
    for i in range(50):
        feat = torch.zeros(154)
        tgt  = torch.zeros(6)
        r.add(feat, tgt, float(i))
    assert r.size() == 10

def test_add_then_sample():
    r = sixmax.WeightedReservoir(capacity=1000)
    for i in range(100):
        r.add(torch.randn(154), torch.randn(6), 1.0)
    feat, tgt, weights = r.sample_batch(32)
    assert feat.shape   == (32, 154)
    assert tgt.shape    == (32, 6)
    assert weights.shape == (32,)
    assert (weights > 0).all()

def test_clear():
    r = sixmax.WeightedReservoir(capacity=100)
    r.add(torch.zeros(154), torch.zeros(6), 1.0)
    r.clear()
    assert r.size() == 0
```

**Note:** `WeightedReservoir` needs a Python seed or a default internal RNG. Update the pybind11 binding for `add()` and `sample_batch()` to use an internal seeded RNG if not accepting one from Python. Simplest: the binding stores an `mt19937_64` seeded at construction and uses it internally for Python calls.

- [ ] **Step 4: Write test_dream_checkpoint.py**

```python
# tests/sixmax/test_dream_checkpoint.py
import torch, sixmax, pathlib, tempfile

def test_checkpoint_roundtrip(default_vocab, tmp_path):
    path = str(tmp_path / "dream_test.pt")
    n = default_vocab.size()

    # Create and save a net with known output on a fixed input
    cfg = sixmax.DreamConfig()
    cfg.hidden_size = 64
    cfg.hidden_layers = 2
    adv  = sixmax.DreamMLP(154, 64, 2, n)
    strat = sixmax.DreamMLP(154, 64, 2, n)
    sixmax.save_dream_checkpoint(path, adv, strat, 999, default_vocab.hash())

    # Load DreamStrategy and verify it produces valid probs
    # (need a real EngineGameState — use fixture or skip full inference test here)
    # At minimum: verify load doesn't throw and iterations() is correct
    ds = sixmax.DreamStrategy.load(path, "cpu", default_vocab)
    assert ds.iterations() == 999

def test_vocab_hash_mismatch(default_vocab, tmp_path):
    path = str(tmp_path / "bad.pt")
    n = default_vocab.size()
    adv  = sixmax.DreamMLP(154, 64, 2, n)
    strat = sixmax.DreamMLP(154, 64, 2, n)
    sixmax.save_dream_checkpoint(path, adv, strat, 1, 0xDEADBEEF)  # wrong hash
    import pytest
    with pytest.raises(RuntimeError, match="vocab hash mismatch"):
        sixmax.DreamStrategy.load(path, "cpu", default_vocab)
```

- [ ] **Step 5: Run all Tier 1 tests**

```bash
uv run pytest tests/sixmax/test_dream_features.py \
              tests/sixmax/test_dream_nets.py \
              tests/sixmax/test_dream_reservoir.py \
              tests/sixmax/test_dream_checkpoint.py -v
```
Expected: all pass. Fix any failures before proceeding to Task 9.

- [ ] **Step 6: Commit**

```bash
git add tests/sixmax/test_dream_features.py tests/sixmax/test_dream_nets.py \
        tests/sixmax/test_dream_reservoir.py tests/sixmax/test_dream_checkpoint.py
git commit -m "test(dream): Tier 1 unit tests — features, nets, reservoir, checkpoint"
```

---

## Task 9: Kuhn convergence gate (Tier 2)

**Files:**
- Modify: `sixmax/src/blueprint/kuhn.h` — add `card(int p)` and `history_code()` to `KuhnState`
- Create: `tests/sixmax/test_dream_kuhn.py`

This test validates the IS weight computation, reservoir, and retraining loop before committing to a full multi-day training run. It uses a minimal 9-dim Kuhn feature encoder instead of the full 154-dim EngineGameState encoder.

- [ ] **Step 1: Add accessors to KuhnState**

In `sixmax/src/blueprint/kuhn.h`, add to the `KuhnState` public section:

```cpp
int card(int player) const { return cards_[player]; }
// History code: 0=empty 1=check 2=bet 3=check+bet (matches key_for convention)
int history_code() const {
    if (history_.empty()) return 0;
    if (history_.size() == 1) return history_[0] == 0 ? 1 : 2;
    return 3;  // check then bet
}
int acting_player() const { return (int)history_.size() % 2; }
```

Expose these in bindings.cpp:
```cpp
py::class_<KuhnState>(m, "KuhnState")
    .def("card", &KuhnState::card)
    .def("history_code", &KuhnState::history_code)
    .def("acting_player", &KuhnState::acting_player);
```

Rebuild: `~/bin/buck2 build //sixmax:sixmax`

- [ ] **Step 2: Write test_dream_kuhn.py**

The test runs a self-contained outcome-sampling MCCFR with tiny nets on Kuhn poker. It reuses `DreamMLP` and `WeightedReservoir` from the bindings, but implements its own Python-level traversal (avoiding the C++ DreamTrainer's EngineGameState dependency).

```python
# tests/sixmax/test_dream_kuhn.py
"""
Kuhn poker convergence gate for DREAM IS weights + reservoir + retraining.
Uses 9-dim feature encoding (card one-hot + history one-hot + player one-hot).
Trains for 50K traversals. Expected strategy-net value for P0 ≈ -1/18 ≈ -0.0556.
"""
import torch
import sixmax
import random
import math

KUHN_FEATURE_DIM = 9

def encode_kuhn(card: int, history_code: int, player: int) -> torch.Tensor:
    """9-dim: 3 card one-hot + 4 history one-hot + 2 player one-hot."""
    t = torch.zeros(9)
    t[card] = 1.0           # dims 0-2: J/Q/K
    t[3 + history_code] = 1.0  # dims 3-6: empty/check/bet/check-bet
    t[7 + player] = 1.0     # dims 7-8: player 0/1
    return t

def regret_match(advantages: list, mask: list) -> list:
    pos = [max(0.0, advantages[i]) if mask[i] else 0.0 for i in range(len(mask))]
    total = sum(pos)
    n_legal = sum(mask)
    if total < 1e-9:
        return [1.0 / n_legal if m else 0.0 for m in mask]
    return [p / total for p in pos]

def eps_greedy(sigma: list, eps: float = 0.06) -> list:
    n_legal = sum(1 for s in sigma if s > 0)
    if n_legal == 0: return sigma
    u = eps / n_legal
    return [(1 - eps) * s + (u if s > 0 else 0.0) for s in sigma]

def kuhn_utility(cards: list, history: list) -> list:
    """Returns [u0, u1] for Kuhn terminal state."""
    # Standard Kuhn payoffs: ante=1, bet=1
    h = tuple(history)
    c0, c1 = cards
    winner = 0 if c0 > c1 else 1  # higher card wins showdown
    if h == (0,):       return [-1, 1]      # p0 pass, p1 pass: lower loses ante
    if h == (1,):       return [-1, 1]      # p0 bet, p1 fold
    if h == (0, 0):     return [-1, 1] if winner == 1 else [1, -1]
    if h == (0, 1):     return [-2, 2] if winner == 1 else [2, -2]
    if h == (1, 0):     return [1, -1]
    if h == (1, 1):     return [-2, 2] if winner == 1 else [2, -2]
    raise ValueError(f"unexpected history {h}")

def kuhn_is_terminal(history: list) -> bool:
    h = tuple(history)
    return h in ((0,), (1, 0), (0, 0), (0, 1), (1, 1))

def test_dream_kuhn_convergence():
    torch.manual_seed(0)
    rng = random.Random(0)

    adv_net = sixmax.DreamMLP(KUHN_FEATURE_DIM, 32, 2, 2)
    strat_net = sixmax.DreamMLP(KUHN_FEATURE_DIM, 32, 2, 2)
    M_v  = sixmax.WeightedReservoir(200_000)
    M_pi = sixmax.WeightedReservoir(200_000)

    TRAIN_INTERVAL = 1000
    BATCH = 256
    SGD_STEPS = 200
    LR = 1e-3
    EPS = 0.06
    N_ITER = 50_000

    def get_strategy(card, history_code, player):
        feat = encode_kuhn(card, history_code, player)
        with torch.no_grad():
            adv = adv_net(feat.unsqueeze(0)).squeeze(0).tolist()
        sigma = regret_match(adv, [1, 1])
        return eps_greedy(sigma, EPS)

    for t in range(1, N_ITER + 1):
        # Deal
        cards = random.sample([0, 1, 2], 2)
        history = []
        trajectory = []

        # Forward pass: sample one action at each node
        while not kuhn_is_terminal(history):
            player = len(history) % 2
            hcode = (0 if not history else
                     1 if history == [0] else
                     2 if history == [1] else 3)
            sigma = get_strategy(cards[player], hcode, player)
            a_star = rng.choices([0, 1], weights=sigma)[0]
            trajectory.append((cards[player], hcode, player, sigma, a_star))
            history.append(a_star)

        utils = kuhn_utility(cards, history)
        weight = float(t)

        # Backward pass: IS-corrected advantage + strategy samples
        for (card, hcode, player, sigma, a_star) in trajectory:
            feat = encode_kuhn(card, hcode, player)
            u_p = utils[player]
            prob_star = sigma[a_star]
            if prob_star < 1e-9: continue

            adv_target = torch.zeros(2)
            adv_target[a_star] = u_p / prob_star
            M_v.add(feat, adv_target, weight)

            strat_target = torch.tensor(sigma)
            M_pi.add(feat, strat_target, weight)

        if t % TRAIN_INTERVAL == 0 and M_v.size() >= BATCH:
            # Retrain adv net from scratch
            adv_net = sixmax.DreamMLP(KUHN_FEATURE_DIM, 32, 2, 2)
            opt = torch.optim.Adam(adv_net.parameters(), lr=LR)
            for _ in range(SGD_STEPS):
                feat_b, tgt_b, w_b = M_v.sample_batch(BATCH)
                opt.zero_grad()
                pred = adv_net(feat_b)
                loss = ((pred - tgt_b).pow(2) * w_b.unsqueeze(1)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adv_net.parameters(), 1.0)
                opt.step()

            # Retrain strategy net
            opt2 = torch.optim.Adam(strat_net.parameters(), lr=LR)
            for _ in range(SGD_STEPS):
                feat_b, tgt_b, w_b = M_pi.sample_batch(BATCH)
                opt2.zero_grad()
                log_probs = torch.log_softmax(strat_net(feat_b), dim=1)
                loss = -(tgt_b * log_probs * w_b.unsqueeze(1)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(strat_net.parameters(), 1.0)
                opt2.step()

    # Evaluate: compute P0 EV under strat_net's strategy
    def strategy_net_probs(card, history_code, player):
        feat = encode_kuhn(card, history_code, player)
        with torch.no_grad():
            logits = strat_net(feat.unsqueeze(0)).squeeze(0)
            return torch.softmax(logits, 0).tolist()

    # Enumerate all 6 Kuhn deals and compute exact EV
    total_ev = 0.0
    for c0 in range(3):
        for c1 in range(3):
            if c0 == c1: continue
            cards = [c0, c1]
            # P0 acts first (history empty)
            s0 = strategy_net_probs(c0, 0, 0)
            ev = 0.0
            for a0 in range(2):
                h1 = [a0]
                if kuhn_is_terminal(h1):
                    u = kuhn_utility(cards, h1)
                    ev += s0[a0] * u[0]
                else:
                    hcode1 = 1 if a0 == 0 else 2
                    s1 = strategy_net_probs(c1, hcode1, 1)
                    for a1 in range(2):
                        h2 = [a0, a1]
                        u = kuhn_utility(cards, h2)
                        ev += s0[a0] * s1[a1] * u[0]
            total_ev += ev
    ev_p0 = total_ev / 6.0  # average over 6 deals

    # Nash value for P0 is -1/18 ≈ -0.0556
    assert abs(ev_p0 - (-1.0 / 18.0)) < 0.02, \
        f"Kuhn EV for P0 = {ev_p0:.4f}, expected ≈ -0.0556 (±0.02)"
```

- [ ] **Step 3: Run the convergence gate**

```bash
uv run pytest tests/sixmax/test_dream_kuhn.py -v -s
```
Expected: PASS in < 2 minutes on MacBook. If it fails, check:
- IS weight formula: `adv_target[a_star] = u_p / prob_star` (not multiplied by reach prob)
- Advantage net is reinitialized each cycle (`adv_net = DreamMLP(...)`)
- Strategy net is NOT reinitialized — it accumulates

- [ ] **Step 4: Commit**

```bash
git add sixmax/src/blueprint/kuhn.h tests/sixmax/test_dream_kuhn.py \
        sixmax/src/bindings/bindings.cpp
git commit -m "test(dream): Kuhn poker convergence gate — IS weights + reservoir + retraining"
```

---

## Task 10: train_dream.py + openpoker_bot.py update

**Files:**
- Create: `scripts/train_dream.py`
- Modify: `scripts/openpoker_bot.py`

**Interfaces:**
- Consumes: `sixmax.DreamTrainer`, `sixmax.DreamConfig`, `sixmax.save_dream_checkpoint`
- Produces: `dream_*.pt` checkpoint

- [ ] **Step 1: Create scripts/train_dream.py**

Read `scripts/train_neural.py` as a reference for the CLI pattern. Then write:

```python
#!/usr/bin/env python3
"""Train DREAM neural blueprint for sixmax/."""
import argparse, sys, pathlib, importlib.util, time

def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(args.config, "rb") as f:
        cfg_data = tomllib.load(f)

    # Force-load sixmax extension (same pattern as conftest.py)
    import importlib
    so = next(pathlib.Path("sixmax").glob("*.so"), None)
    if so is None:
        sys.exit("sixmax.so not found — run: ~/bin/buck2 build //sixmax:sixmax")
    spec = importlib.util.spec_from_file_location("sixmax", so)
    sixmax = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = sixmax
    spec.loader.exec_module(sixmax)

    vc = load_module("sixmax/vocab_config.py", "vocab_config")
    vocab = vc.load_vocab(args.config, "blueprint")

    dc = cfg_data.get("train", {}).get("dream", {})
    dream_cfg = sixmax.DreamConfig()
    dream_cfg.hidden_size    = dc.get("hidden_size",    256)
    dream_cfg.hidden_layers  = dc.get("hidden_layers",  3)
    dream_cfg.lr             = dc.get("lr",             1e-3)
    dream_cfg.batch_size     = dc.get("batch_size",     4096)
    dream_cfg.reservoir_size = dc.get("reservoir_size", 2_000_000)
    dream_cfg.train_interval = dc.get("train_interval", 10_000)
    dream_cfg.sgd_steps      = dc.get("sgd_steps",      2_000)
    dream_cfg.epsilon        = dc.get("epsilon",         0.06)
    dream_cfg.num_threads    = dc.get("num_threads",     0)
    dream_cfg.seed           = dc.get("seed",            42)
    dream_cfg.stack_min      = dc.get("stack_min",       20.0)
    dream_cfg.stack_max      = dc.get("stack_max",       250.0)
    dream_cfg.players_min    = dc.get("players_min",     2)
    dream_cfg.players_max    = dc.get("players_max",     6)
    device_str = args.device or dc.get("device", "cpu")

    total_iters = args.iterations or dc.get("iterations", 1_000_000)
    ckpt_interval = dc.get("checkpoint_interval", 0)
    ckpt_path = args.checkpoint

    # Load abstraction
    abs_cfg = cfg_data.get("abstraction", {})
    abstraction = sixmax.Abstraction(
        abs_cfg.get("flop_buckets", 50),
        abs_cfg.get("turn_buckets", 50),
        abs_cfg.get("river_buckets", 20),
        abs_cfg.get("equity_rollouts", 100),
        abs_cfg.get("quantile_samples", 10000),
        abs_cfg.get("seed", 20260719),
    )

    trainer = sixmax.DreamTrainer(vocab.size(), vocab, abstraction,
                                  dream_cfg, device_str)

    if args.resume:
        print(f"Resume not yet supported for DreamTrainer (nets only). "
              f"Starting fresh.")

    print(f"Training DREAM blueprint: {total_iters} iterations, device={device_str}")
    start = time.time()

    if ckpt_interval > 0:
        done = 0
        while done < total_iters:
            chunk = min(ckpt_interval, total_iters - done)
            trainer.train(chunk)
            done += chunk
            elapsed = time.time() - start
            print(f"  {done}/{total_iters} iters | {elapsed:.0f}s elapsed")
            trainer.save(ckpt_path, vocab.hash())
            print(f"  Saved checkpoint → {ckpt_path}")
    else:
        trainer.train(total_iters)
        trainer.save(ckpt_path, vocab.hash())
        print(f"Saved → {ckpt_path} ({time.time()-start:.0f}s)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update openpoker_bot.py auto-detect**

Read `scripts/openpoker_bot.py` and find the section that auto-detects checkpoint type by file extension. It currently maps `.pt` → `neural_cfr`. Add a branch for `dream_*.pt` before the generic `.pt` branch:

```python
# Find the section with something like:
#   if path.endswith(".pt"):
#       ...neural_cfr...
# And change it to:

import pathlib
p = pathlib.Path(path)
if p.suffix == ".pt" and p.stem.startswith("dream_"):
    # Load DreamStrategy
    strategy = sixmax.DreamStrategy.load(str(path), device="cpu", vocab=vocab)
    # ... wire into the agent
elif p.suffix == ".pt":
    # Existing neural_cfr path
    ...
elif p.suffix == ".bin":
    # Existing tabular blueprint path
    ...
```

Read the actual bot code to find the exact variable names and class used. The wiring should follow the same pattern as `BlueprintStrategy` since `DreamStrategy.get_probs()` has the identical signature.

- [ ] **Step 3: Smoke test the training script (1K iterations)**

```bash
uv run python scripts/train_dream.py \
    --config sixmax/configs/default.toml \
    --checkpoint /tmp/dream_smoke.pt \
    --iterations 1000
```
Expected: runs without error, prints progress, saves `/tmp/dream_smoke.pt`.

- [ ] **Step 4: Verify checkpoint loads**

```python
uv run python -c "
import sys, importlib, pathlib
so = next(pathlib.Path('sixmax').glob('*.so'))
spec = importlib.util.spec_from_file_location('sixmax', so)
sixmax = importlib.util.module_from_spec(spec); sys.modules['sixmax'] = sixmax
spec.loader.exec_module(sixmax)
import importlib.util as iu
spec2 = iu.spec_from_file_location('vc', 'sixmax/vocab_config.py')
vc = iu.module_from_spec(spec2); spec2.loader.exec_module(vc)
vocab = vc.load_vocab('sixmax/configs/default.toml', 'blueprint')
ds = sixmax.DreamStrategy.load('/tmp/dream_smoke.pt', 'cpu', vocab)
print('iterations:', ds.iterations())
"
```
Expected: prints `iterations: 1000`.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_dream.py scripts/openpoker_bot.py
git commit -m "feat(dream): train_dream.py script + openpoker_bot dream_*.pt auto-detect"
```

---

## Self-Review Checklist

- [x] Engine raise cap: removed in apply(), capped in abstract_key() at 3 — tabular blueprint unaffected
- [x] 154-dim feature vector: dims verified to sum to 154 (34+85+4+1+6+6+6+6+1+1+4)
- [x] IS weight formula specified: `adv[a*] = u_p / σ(a*)`, zeros elsewhere
- [x] Advantage net reinitialized each cycle; strategy net continuous
- [x] Kuhn convergence gate: validates IS weights in isolation from EngineGameState
- [x] SIXDM001 vocab hash mismatch throws on load
- [x] `device` not stored in checkpoint — passed at load time
- [x] `openpoker_bot.py` auto-detect updated for `dream_*.pt`
- [x] Task 5 DreamTrainer note: `set_config`/`set_stacks` don't exist — implementer must construct `HandState::deal()` directly with per-hand config
- [x] Task 8 WeightedReservoir Python RNG: binding needs internal RNG for Python-facing `add()`/`sample_batch()`
- [x] Task 8 `legal_mask()` const correctness: local mutable variable needed in `get_probs()`
