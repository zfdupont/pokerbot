# Neural CFR Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three correctness issues in the `neural_cfr/` Deep CFR implementation: train/inference feature mismatch for street bets, drastically undertrained networks vs the paper's regime, and a uniform (instead of argmax) regret-matching fallback.

**Architecture:** Three parallel tracks in separate git worktrees — Track A (traversal/regret matching), Track B (trainer regime + TOML config), Track C (inference feature parity, depends on Track A) — followed by a sequential integration track that merges, updates docs, smoke-tests, and runs acceptance evals. All C++ changes live in the `neural_cfr/` Buck2 extension; Python changes live in `scripts/`.

**Tech Stack:** C++17, libtorch, pybind11, Buck2 (`~/bin/buck2`), GoogleTest, Python 3.11+ (`uv`), stdlib `tomllib`.

**Spec:** `docs/superpowers/specs/2026-07-07-neural-cfr-fixes-design.md`

## Global Constraints

- Build: `~/bin/buck2 build //neural_cfr:neural_cfr` (extension), `~/bin/buck2 test //neural_cfr/tests:<name>` (C++ tests). Buck2 is at `~/bin/buck2`, NOT on PATH.
- Python: always `uv run python …` / `uv run pytest …` from repo root.
- Action vocabulary (index order is load-bearing): `0=fold 1=check 2=call 3=b0.5 4=b1.0 5=allin`.
- Feature layout (134 dims): dims 129/130 = acting player's / opponent's current-street bets, `CHIP_NORM=200`, `RAISE_NORM=2`; betting history capped at 2 per street.
- Training scale: `STARTING_STACK=100`, `DEFAULT_BIG_BLIND=1` (constants in `neural_cfr/src/game/abstract_state.h`).
- Checkpoint format: named sub-archives `adv0`/`adv1`/`strat` — do not change; `neural_cfr.Strategy` loads only `strat`.
- New trainer defaults (from spec, exact values): `train_interval=10_000`, `sgd_steps=2_000`, `reinit_adv=true`, `lr=1e-3`, grad-norm clip `1.0`.
- Do NOT commit `.vscode/` or any `.env` files.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## Worktree / Parallelism Map

```
Task 0 (baseline commit, on feature/neural-cfr)  ← sequential, do first
├── Track A  (worktree ncfr-regret)   — Task A1                    ┐
├── Track B  (worktree ncfr-trainer)  — Tasks B1→B2→B3→B4          ├ parallel
└── Track C  (worktree ncfr-parity)   — Tasks C1→C2→C3  ← branch   ┘
                                         from Track A's branch (needs
                                         regret_match from A1)
Integration (back on feature/neural-cfr): I1 merge → I2 docs → I3 smoke → I4 acceptance
```

Merge order: **A → B → C**. A and B start immediately from the Task 0 commit. C branches from A's completed branch (or from `feature/neural-cfr` after A merges). File-conflict analysis: A touches `traversal.{h,cpp}` + `test_traversal.cpp`; B touches `mlp.{h,cpp}`, `trainer.{h,cpp}`, `test_trainer.cpp` (new), `tests/BUCK`, `train_neural.py`, `configs/` (new), and the `PYBIND11_MODULE` block at the bottom of `bindings.cpp`; C touches the `Strategy`/`AdvantageProbe` classes at the top of `bindings.cpp`, new `inference.{h,cpp}`, `test_features.cpp`, and the two Python callers. The only shared file is `bindings.cpp`, in disjoint regions (B: module registration block; C: class bodies) — git merges these hunks cleanly.

Use the `superpowers:using-git-worktrees` skill to create each worktree. Each worktree needs `.buckconfig.local` (gitignored, machine-specific Python include path) — copy it from the main checkout: `cp /Users/zfdupont/pokerbot/.buckconfig.local <worktree>/` before building.

---

### Task 0: Commit baseline WIP (sequential, before any worktree)

The working tree has uncommitted changes that all tracks build on: `neural_cfr/src/bindings/bindings.cpp` (adds the `AdvantageProbe` class), `scripts/eval_openspiel.py`, and untracked `scripts/eval_neural_vs_tabular.py`.

**Files:**
- Commit: `neural_cfr/src/bindings/bindings.cpp`, `scripts/eval_openspiel.py`, `scripts/eval_neural_vs_tabular.py`
- Do NOT commit: `.vscode/`

- [ ] **Step 1: Review the pending diff**

Run: `git diff neural_cfr/src/bindings/bindings.cpp scripts/eval_openspiel.py | head -100` and `git status`
Expected: the bindings diff adds `AdvantageProbe`; no secrets anywhere.

- [ ] **Step 2: Verify the extension still builds**

Run: `~/bin/buck2 build //neural_cfr:neural_cfr`
Expected: `BUILD SUCCEEDED`

- [ ] **Step 3: Commit**

```bash
git add neural_cfr/src/bindings/bindings.cpp scripts/eval_openspiel.py scripts/eval_neural_vs_tabular.py
git commit -m "feat: AdvantageProbe diagnostics + neural-vs-tabular eval script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Track A — worktree `ncfr-regret` (branch `ncfr/regret-match`)

### Task A1: Argmax fallback in regret matching

**Files:**
- Modify: `neural_cfr/src/cfr/traversal.h` (add `regret_match` declaration)
- Modify: `neural_cfr/src/cfr/traversal.cpp:13-28` (de-static, argmax fallback)
- Test: `neural_cfr/tests/test_traversal.cpp` (new tests; also fix stale `external_sample` call sites that predate the `rng` parameter)

**Interfaces:**
- Consumes: existing `external_sample(...)` signature from `traversal.h` (requires `std::mt19937& rng` — the current test file omits it and does not compile).
- Produces: `std::vector<float> regret_match(const std::array<float, 6>& advantages, const std::vector<int>& legal_indices);` declared in `traversal.h` — **Track C's Task C2 calls this exact signature.** Returned vector is index-aligned with `legal_indices`.

- [ ] **Step 1: Write the failing tests**

Append to `neural_cfr/tests/test_traversal.cpp`, and fix the three existing tests to pass an RNG (they currently call the pre-`rng` signature). The full updated file:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "cfr/traversal.h"
#include "cfr/reservoir_buffer.h"
#include "net/mlp.h"
#include "game/abstract_state.h"

TEST(Traversal, BuffersPopulatedAfterTraversal) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(10000), mv1(10000), mpi(10000);
    std::mt19937 rng{42};

    // Run 20 traversals for each player
    for (int t = 1; t <= 20; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t, rng);
        s = deal_heads_up();
        external_sample(s, 1, adv1, adv0, strat, mv1, mpi, t, rng);
    }

    EXPECT_GT(mv0.size(), 0u);
    EXPECT_GT(mv1.size(), 0u);
    EXPECT_GT(mpi.size(), 0u);
}

TEST(Traversal, ReturnedEVIsFinite) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    std::mt19937 rng{42};
    auto s = deal_heads_up();
    float ev = external_sample(s, 0, adv0, adv1, strat, mv0, mpi, 1, rng);
    EXPECT_TRUE(std::isfinite(ev));
}

TEST(Traversal, FeatureDimInBuffer) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    std::mt19937 rng{42};
    for (int t = 1; t <= 5; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t, rng);
    }
    ASSERT_GT(mv0.size(), 0u);
    EXPECT_EQ(mv0.data()[0].features.size(), 134u);
    EXPECT_EQ(mv0.data()[0].targets.size(), 6u);
}

TEST(RegretMatch, ProportionalWhenPositiveAdvantagesExist) {
    std::array<float, 6> adv{};
    adv[1] = 1.0f;   // check
    adv[3] = 3.0f;   // b0.5
    std::vector<int> legal{1, 3, 5};
    auto p = regret_match(adv, legal);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_NEAR(p[0], 0.25f, 1e-6);
    EXPECT_NEAR(p[1], 0.75f, 1e-6);
    EXPECT_NEAR(p[2], 0.0f,  1e-6);
}

TEST(RegretMatch, ArgmaxWhenAllNonPositive) {
    // Brown et al. 2019: when no advantage is positive, play the
    // highest-advantage action as a pure strategy — NOT uniform.
    std::array<float, 6> adv{};
    adv[0] = -0.5f;
    adv[2] = -0.1f;  // best of the legal set
    adv[5] = -2.0f;
    std::vector<int> legal{0, 2, 5};
    auto p = regret_match(adv, legal);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 1.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
}

TEST(RegretMatch, AllExactlyZeroPicksFirstArgmax) {
    // Ties broken by first index — deterministic, any pure choice is valid.
    std::array<float, 6> adv{};
    std::vector<int> legal{0, 1};
    auto p = regret_match(adv, legal);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/bin/buck2 test //neural_cfr/tests:test_traversal`
Expected: compile FAILURE — `regret_match` is not declared in `traversal.h` (it is currently `static` in the .cpp).

- [ ] **Step 3: Implement**

In `neural_cfr/src/cfr/traversal.h`, after the `external_sample` declaration add:

```cpp
// Regret matching over legal action indices: ReLU(advantages) normalized.
// When no advantage is positive, returns a one-hot on the argmax advantage
// (Brown et al. 2019) rather than uniform. Result is index-aligned with
// legal_indices.
std::vector<float> regret_match(
    const std::array<float, 6>& advantages,
    const std::vector<int>& legal_indices);
```

In `neural_cfr/src/cfr/traversal.cpp`, replace the whole `static std::vector<float> regret_match(...)` function (lines 11–28) with:

```cpp
// Regret matching: given raw advantage logits and legal action indices,
// return a probability distribution via ReLU + normalize. When all
// advantages are non-positive, play the argmax advantage as a pure
// strategy (Brown et al. 2019) — uniform here would both add noise at
// traverser nodes and train the strategy net toward uniform via M_π.
std::vector<float> regret_match(
    const std::array<float, 6>& advantages,
    const std::vector<int>& legal_indices)
{
    std::vector<float> pos(legal_indices.size());
    float total = 0.0f;
    for (size_t i = 0; i < legal_indices.size(); ++i) {
        pos[i] = std::max(0.0f, advantages[legal_indices[i]]);
        total += pos[i];
    }
    if (total > 0.0f) {
        for (auto& p : pos) p /= total;
    } else {
        size_t best = 0;
        for (size_t i = 1; i < legal_indices.size(); ++i)
            if (advantages[legal_indices[i]] > advantages[legal_indices[best]])
                best = i;
        std::fill(pos.begin(), pos.end(), 0.0f);
        pos[best] = 1.0f;
    }
    return pos;
}
```

Note: the existing definition uses `std::array<float, NUM_ACTIONS>`; `NUM_ACTIONS` is 6 (from `net/features.h`, already included by traversal.cpp). Write the definition with a literal `6` to match the header, which deliberately avoids including the torch-heavy `features.h`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/bin/buck2 test //neural_cfr/tests:test_traversal`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run the other C++ test targets (regression)**

Run: `~/bin/buck2 test //neural_cfr/tests:test_abstract_state //neural_cfr/tests:test_features //neural_cfr/tests:test_reservoir_buffer`
Expected: PASS.

- [ ] **Step 6: Document the `payoff()` stack invariant (spec "out of scope" note)**

In `neural_cfr/src/game/abstract_state.cpp`, at the top of `AbstractState::payoff` (line 46), add above the `invest` line:

```cpp
    // Assumes the hand was dealt with starting_stack == STARTING_STACK.
    // deal_heads_up() accepts other stack sizes, but payoff() would then be
    // silently wrong — Trainer always uses the constant.
```

- [ ] **Step 7: Commit**

```bash
git add neural_cfr/src/cfr/traversal.h neural_cfr/src/cfr/traversal.cpp neural_cfr/tests/test_traversal.cpp neural_cfr/src/game/abstract_state.cpp
git commit -m "fix: argmax fallback in regret matching per Brown et al. 2019

When all predicted advantages are non-positive, play the argmax advantage
as a pure strategy instead of uniform. Uniform was polluting M_pi at
opponent nodes, training the strategy net toward uniform play in
pessimistic states. Also expose regret_match in traversal.h for reuse.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Track B — worktree `ncfr-trainer` (branch `ncfr/trainer-regime`)

### Task B1: `MLP::reset_parameters()`

**Files:**
- Modify: `neural_cfr/src/net/mlp.h`, `neural_cfr/src/net/mlp.cpp`
- Create: `neural_cfr/tests/test_trainer.cpp` (starts with the MLP test; Task B2 adds trainer tests)
- Modify: `neural_cfr/tests/BUCK` (new test target)

**Interfaces:**
- Produces: `void MLP::reset_parameters();` — reinitializes all four Linear layers in place (same parameter tensors, new values; safe to call on a net whose parameters an optimizer references, but Task B2 always pairs it with a fresh optimizer).

- [ ] **Step 1: Write the failing test**

Create `neural_cfr/tests/test_trainer.cpp`:

```cpp
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "net/mlp.h"
#include "cfr/trainer.h"

TEST(MLP, ResetParametersChangesWeights) {
    MLP net;
    auto before = net.fc1->weight.clone();
    net.reset_parameters();
    EXPECT_FALSE(torch::allclose(before, net.fc1->weight));
    // Shape must be preserved
    EXPECT_TRUE(before.sizes() == net.fc1->weight.sizes());
}
```

Add to `neural_cfr/tests/BUCK`:

```python
cxx_test(
    name = "test_trainer",
    srcs = ["test_trainer.cpp"],
    deps = ["//neural_cfr:core", "//third_party:gtest", "//third_party:libtorch"],
    compiler_flags = ["-std=c++17"],
    env = _libtorch_env,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: compile FAILURE — `reset_parameters` is not a member of `MLP`.

- [ ] **Step 3: Implement**

In `neural_cfr/src/net/mlp.h`, add to the struct after `forward`:

```cpp
    // Reinitialize all layers in place (Deep CFR retrains nets from scratch
    // each CFR iteration). Pair with a fresh optimizer.
    void reset_parameters();
```

In `neural_cfr/src/net/mlp.cpp`, append:

```cpp
void MLP::reset_parameters() {
    for (auto* fc : {&fc1, &fc2, &fc3, &fc4})
        (*fc)->reset_parameters();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neural_cfr/src/net/mlp.h neural_cfr/src/net/mlp.cpp neural_cfr/tests/test_trainer.cpp neural_cfr/tests/BUCK
git commit -m "feat: MLP::reset_parameters for from-scratch retraining

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task B2: Trainer restructure — CFR iterations, training events, `train_strategy`

**Files:**
- Modify: `neural_cfr/src/cfr/trainer.h`, `neural_cfr/src/cfr/trainer.cpp`
- Test: `neural_cfr/tests/test_trainer.cpp` (append)

**Interfaces:**
- Consumes: `MLP::reset_parameters()` from Task B1.
- Produces (Task B3 binds these; Task I3 exercises them from Python):
  - `Trainer(size_t reservoir_size, size_t batch_size, float lr, int train_interval, int num_threads, float epsilon, int sgd_steps, bool reinit_adv)` — two new trailing params.
  - `void Trainer::train_strategy(int sgd_steps = -1);` — retrains the strategy net on M_π (`-1` → use constructor `sgd_steps`). Skips with a stderr warning if `mpi_.size() < batch_size`.
  - `checkpoint(path)` now calls `train_strategy()` before saving.
  - New constants in `trainer.h`: `DEFAULT_TRAIN_INTERVAL=10'000`, `DEFAULT_SGD_STEPS=2'000`, `DEFAULT_REINIT_ADV=true`, `DEFAULT_GRAD_CLIP=1.0f`, `DEFAULT_LR=1e-3f`.

- [ ] **Step 1: Write the failing tests**

Append to `neural_cfr/tests/test_trainer.cpp`:

```cpp
#include <cstdio>
#include <fstream>

// Tiny end-to-end run: 2 CFR iterations of 50 traversal-pairs, 5 SGD steps
// per training event, reinit on. Verifies the restructured loop runs,
// checkpoint() triggers strategy training, and the file round-trips.
TEST(Trainer, TinyRunTrainsAndCheckpoints) {
    Trainer t(/*reservoir_size=*/10000, /*batch_size=*/64, /*lr=*/1e-3f,
              /*train_interval=*/50, /*num_threads=*/2, /*epsilon=*/0.06f,
              /*sgd_steps=*/5, /*reinit_adv=*/true);
    t.run(100);

    const std::string path = "/tmp/neural_cfr_test_ckpt.pt";
    t.checkpoint(path);
    std::ifstream f(path);
    ASSERT_TRUE(f.good());
    f.close();

    Trainer t2(10000, 64, 1e-3f, 50, 2, 0.06f, 5, true);
    t2.load(path);  // throws on failure
    std::remove(path.c_str());
}

// With an empty buffer (no run), checkpoint must still produce a loadable
// file and train_strategy must skip rather than reinit-then-not-train.
TEST(Trainer, CheckpointWithEmptyBuffersStillSaves) {
    Trainer t(1000, 64, 1e-3f, 50, 1, 0.06f, 5, true);
    const std::string path = "/tmp/neural_cfr_test_ckpt_empty.pt";
    t.checkpoint(path);
    Trainer t2(1000, 64, 1e-3f, 50, 1, 0.06f, 5, true);
    t2.load(path);
    std::remove(path.c_str());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: compile FAILURE — Trainer has no 8-arg constructor.

- [ ] **Step 3: Rewrite `neural_cfr/src/cfr/trainer.h`**

```cpp
#pragma once
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"
#include <string>
#include <memory>
#include <torch/optim.h>

constexpr int   DEFAULT_BATCH_SIZE       = 4096;
// From-scratch training rate (Brown et al. 2019). The old 1e-4 was a
// fine-tuning rate for the pre-2026-07 continual regime.
constexpr float DEFAULT_LR               = 1e-3f;
constexpr int   DEFAULT_RESERVOIR_SIZE   = 2'000'000;
// Traversal-pairs per CFR iteration (paper: 10k traversals/iteration).
constexpr int   DEFAULT_TRAIN_INTERVAL   = 10'000;
// SGD mini-batches per training event (paper: 4000 at batch 10k).
constexpr int   DEFAULT_SGD_STEPS        = 2'000;
// Reinitialize advantage nets before each training event (paper ablation:
// from-scratch retraining beats continual fine-tuning).
constexpr bool  DEFAULT_REINIT_ADV       = true;
constexpr float DEFAULT_GRAD_CLIP        = 1.0f;
// 0 = use std::thread::hardware_concurrency()
constexpr int   DEFAULT_NUM_THREADS      = 0;
// ε-greedy exploration at opponent nodes
constexpr float DEFAULT_EPSILON          = 0.06f;

class Trainer {
public:
    explicit Trainer(
        size_t reservoir_size  = DEFAULT_RESERVOIR_SIZE,
        size_t batch_size      = DEFAULT_BATCH_SIZE,
        float  lr              = DEFAULT_LR,
        int    train_interval  = DEFAULT_TRAIN_INTERVAL,
        int    num_threads     = DEFAULT_NUM_THREADS,
        float  epsilon         = DEFAULT_EPSILON,
        int    sgd_steps       = DEFAULT_SGD_STEPS,
        bool   reinit_adv      = DEFAULT_REINIT_ADV);

    void run(int iterations);
    // Retrain the strategy net on M_π (from scratch when reinit_adv).
    // sgd_steps = -1 → use the constructor value. Called automatically by
    // checkpoint(); exposed for manual use from Python.
    void train_strategy(int sgd_steps = -1);
    void checkpoint(const std::string& path);
    void load(const std::string& path);

private:
    MLP adv0_, adv1_, strat_;
    ReservoirBuffer<BufferEntry> mv0_, mv1_, mpi_;
    std::unique_ptr<torch::optim::Adam> opt_adv0_, opt_adv1_, opt_strat_;
    size_t batch_size_;
    float  lr_;
    int    train_interval_;
    int    num_threads_;
    float  epsilon_;
    int    sgd_steps_;
    bool   reinit_adv_;

    // One SGD mini-batch. mode: "advantage" = weighted MSE,
    // "strategy" = weighted cross-entropy.
    void train_step(MLP& net, torch::optim::Adam& opt,
                    ReservoirBuffer<BufferEntry>& buffer,
                    const std::string& mode);
    // One training event: optional reinit (net + fresh optimizer), then
    // `steps` mini-batches. No-op (keeps current net) if the buffer holds
    // fewer than batch_size_ samples.
    void train_event(MLP& net, std::unique_ptr<torch::optim::Adam>& opt,
                     ReservoirBuffer<BufferEntry>& buffer,
                     const std::string& mode, int steps, bool reinit);
};
```

- [ ] **Step 4: Update `neural_cfr/src/cfr/trainer.cpp`**

Constructor (replace lines 21–31):

```cpp
Trainer::Trainer(size_t reservoir_size, size_t batch_size, float lr, int train_interval,
                 int num_threads, float epsilon, int sgd_steps, bool reinit_adv)
    : mv0_(reservoir_size), mv1_(reservoir_size), mpi_(reservoir_size),
      opt_adv0_(std::make_unique<torch::optim::Adam>(adv0_.parameters(), torch::optim::AdamOptions(lr))),
      opt_adv1_(std::make_unique<torch::optim::Adam>(adv1_.parameters(), torch::optim::AdamOptions(lr))),
      opt_strat_(std::make_unique<torch::optim::Adam>(strat_.parameters(), torch::optim::AdamOptions(lr))),
      batch_size_(batch_size),
      lr_(lr),
      train_interval_(train_interval),
      num_threads_(num_threads > 0 ? num_threads : (int)std::thread::hardware_concurrency()),
      epsilon_(epsilon),
      sgd_steps_(sgd_steps),
      reinit_adv_(reinit_adv)
{}
```

In `train_step`, add gradient clipping between `loss.backward()` and `opt.step()`:

```cpp
    loss.backward();
    torch::nn::utils::clip_grad_norm_(net.parameters(), DEFAULT_GRAD_CLIP);
    opt.step();
```

Add `train_event` and `train_strategy` after `train_step`:

```cpp
void Trainer::train_event(MLP& net, std::unique_ptr<torch::optim::Adam>& opt,
                          ReservoirBuffer<BufferEntry>& buffer,
                          const std::string& mode, int steps, bool reinit)
{
    if (buffer.size() < batch_size_) return;  // keep current net — never
                                              // reinit without retraining
    if (reinit) {
        net.reset_parameters();
        opt = std::make_unique<torch::optim::Adam>(
            net.parameters(), torch::optim::AdamOptions(lr_));
    }
    for (int i = 0; i < steps; ++i)
        train_step(net, *opt, buffer, mode);
}

void Trainer::train_strategy(int sgd_steps) {
    if (sgd_steps < 0) sgd_steps = sgd_steps_;
    if (mpi_.size() < batch_size_) {
        std::cerr << "train_strategy: skipped — " << mpi_.size()
                  << " samples < batch size " << batch_size_ << "\n";
        return;
    }
    train_event(strat_, opt_strat_, mpi_, "strategy", sgd_steps, reinit_adv_);
}
```

In `run()`, replace the post-traversal training block (currently lines 128–132):

```cpp
        if (!g_interrupted) {
            // Training event per CFR iteration: from-scratch retrain of the
            // advantage nets. The strategy net is trained only at
            // checkpoint time (it is never queried during traversal).
            train_event(adv0_, opt_adv0_, mv0_, "advantage", sgd_steps_, reinit_adv_);
            train_event(adv1_, opt_adv1_, mv1_, "advantage", sgd_steps_, reinit_adv_);
        }
```

In `checkpoint()`, add as the first line of the function body:

```cpp
    train_strategy();
```

`train_step` keeps its `torch::optim::Adam&` parameter — callers now pass `*opt_adv0_` etc. only via `train_event`, which dereferences the unique_ptr.

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/bin/buck2 test //neural_cfr/tests:test_trainer`
Expected: 3 tests PASS (the tiny run takes ~1–2 min on CPU: 200 traversals + 4 training events of 5 steps + strategy training at checkpoint).

- [ ] **Step 6: Commit**

```bash
git add neural_cfr/src/cfr/trainer.h neural_cfr/src/cfr/trainer.cpp neural_cfr/tests/test_trainer.cpp
git commit -m "feat: paper-faithful training regime — from-scratch retrains per CFR iteration

Restructure Trainer::run into discrete CFR iterations (train_interval
traversal-pairs) followed by a training event: reinit advantage nets and
train sgd_steps mini-batches (Brown et al. 2019). Strategy net now trains
only at checkpoint time via train_strategy(). Adds grad-norm clipping
(1.0), lr default 1e-3, train_interval default 10k, sgd_steps 2k.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task B3: Bind new Trainer parameters and `train_strategy`

**Files:**
- Modify: `neural_cfr/src/bindings/bindings.cpp` — ONLY the `py::class_<Trainer>` block inside `PYBIND11_MODULE` (bottom of file). Do not touch `Strategy`/`AdvantageProbe` (Track C owns those).

**Interfaces:**
- Consumes: Task B2's 8-arg constructor and `train_strategy(int)`.
- Produces: Python API `neural_cfr.Trainer(..., sgd_steps=2000, reinit_adv=True)` and `trainer.train_strategy(sgd_steps=-1)` — Task B4's script and Task I3 use these exact keyword names.

- [ ] **Step 1: Update the binding**

Replace the `py::class_<Trainer>` block with:

```cpp
    py::class_<Trainer>(m, "Trainer")
        .def(py::init<size_t, size_t, float, int, int, float, int, bool>(),
             py::arg("reservoir_size")  = DEFAULT_RESERVOIR_SIZE,
             py::arg("batch_size")      = DEFAULT_BATCH_SIZE,
             py::arg("lr")              = DEFAULT_LR,
             py::arg("train_interval")  = DEFAULT_TRAIN_INTERVAL,
             py::arg("num_threads")     = DEFAULT_NUM_THREADS,
             py::arg("epsilon")         = DEFAULT_EPSILON,
             py::arg("sgd_steps")       = DEFAULT_SGD_STEPS,
             py::arg("reinit_adv")      = DEFAULT_REINIT_ADV)
        .def("run",            &Trainer::run,            py::arg("iterations"))
        .def("train_strategy", &Trainer::train_strategy, py::arg("sgd_steps") = -1,
             "Retrain the strategy net on M_pi (called automatically by checkpoint)")
        .def("checkpoint",     &Trainer::checkpoint,     py::arg("path"))
        .def("load",           &Trainer::load,           py::arg("path"));
```

- [ ] **Step 2: Build and verify from Python**

```bash
~/bin/buck2 build //neural_cfr:neural_cfr --show-output
```

Then (substitute the .so dir printed above):

```bash
uv run python -c "
import ctypes, sys
for lib in ['libc10.dylib','libtorch_cpu.dylib','libtorch.dylib']:
    ctypes.CDLL('third_party/libtorch/lib/'+lib)
sys.path.insert(0, '<SO_DIR_FROM_BUCK_OUTPUT>')
import neural_cfr
t = neural_cfr.Trainer(reservoir_size=1000, batch_size=64, sgd_steps=5, reinit_adv=True, train_interval=50, num_threads=2)
t.run(50)
t.train_strategy(5)
print('OK')
"
```

Expected: `OK` (plus a possible `train_strategy: skipped` warning if the tiny buffer is under 64 — both outcomes prove the binding works).

- [ ] **Step 3: Commit**

```bash
git add neural_cfr/src/bindings/bindings.cpp
git commit -m "feat: bind sgd_steps, reinit_adv, train_strategy on Trainer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task B4: TOML config for `train_neural.py`

**Files:**
- Create: `neural_cfr/configs/default.toml`, `neural_cfr/configs/smoke.toml`
- Modify: `scripts/train_neural.py`
- Test: `tests/scripts/test_train_neural_config.py` (new; `tests/scripts/` is a new directory)

**Interfaces:**
- Consumes: Task B3's `Trainer(..., sgd_steps=, reinit_adv=)` kwargs.
- Produces:
  - `resolve_config(args, repo_root) -> dict` in `scripts/train_neural.py` (module level, importable without heavy deps) — precedence CLI > config file > `BUILTIN_DEFAULTS`.
  - `write_config_snapshot(cfg: dict, checkpoint_path: str) -> None` — writes `<checkpoint_path>.config.toml`.
  - Config files with `[training]` and `[trainer]` sections; keys named exactly like the CLI flags with underscores.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_train_neural_config.py`:

```python
"""Config resolution for scripts/train_neural.py (CLI > TOML > builtin)."""
import argparse
import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train_neural.py")


@pytest.fixture()
def train_neural():
    spec = importlib.util.spec_from_file_location("train_neural", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level imports are stdlib-only
    return mod


def _args(**overrides):
    """Namespace with every config key unset (None) unless overridden."""
    from_keys = [
        "iterations", "checkpoint_interval", "checkpoint", "reservoir_size",
        "batch_size", "lr", "train_interval", "sgd_steps", "reinit_adv",
        "num_threads", "epsilon", "eval_interval", "eval_hands", "config",
    ]
    ns = argparse.Namespace(**{k: None for k in from_keys})
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_builtin_defaults_when_no_config(train_neural, tmp_path):
    cfg = train_neural.resolve_config(_args(), str(tmp_path))  # no configs/ dir
    assert cfg["train_interval"] == 10_000
    assert cfg["sgd_steps"] == 2_000
    assert cfg["reinit_adv"] is True
    assert cfg["lr"] == pytest.approx(1e-3)


def test_config_file_overrides_builtin(train_neural, tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text("[trainer]\ntrain_interval = 10\nsgd_steps = 1\nreinit_adv = false\n")
    cfg = train_neural.resolve_config(_args(config=str(toml)), str(tmp_path))
    assert cfg["train_interval"] == 10
    assert cfg["sgd_steps"] == 1
    assert cfg["reinit_adv"] is False
    assert cfg["batch_size"] == 4096  # untouched builtin


def test_cli_overrides_config_file(train_neural, tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text("[trainer]\ntrain_interval = 10\n")
    cfg = train_neural.resolve_config(
        _args(config=str(toml), train_interval=77), str(tmp_path))
    assert cfg["train_interval"] == 77


def test_default_config_file_autoloaded(train_neural, tmp_path):
    configs = tmp_path / "neural_cfr" / "configs"
    configs.mkdir(parents=True)
    (configs / "default.toml").write_text("[training]\niterations = 123\n")
    cfg = train_neural.resolve_config(_args(), str(tmp_path))
    assert cfg["iterations"] == 123


def test_snapshot_roundtrips(train_neural, tmp_path):
    import tomllib
    cfg = {"iterations": 5, "lr": 1e-3, "reinit_adv": True,
           "checkpoint": "a/b.pt", "eval_interval": None}
    ckpt = tmp_path / "ckpt.pt"
    train_neural.write_config_snapshot(cfg, str(ckpt))
    with open(str(ckpt) + ".config.toml", "rb") as f:
        snap = tomllib.load(f)["resolved"]
    assert snap["iterations"] == 5
    assert snap["reinit_adv"] is True
    assert snap["checkpoint"] == "a/b.pt"
    assert "eval_interval" not in snap  # None keys omitted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scripts/test_train_neural_config.py -v`
Expected: FAIL — `module 'train_neural' has no attribute 'resolve_config'`.

- [ ] **Step 3: Implement in `scripts/train_neural.py`**

Add `import json` and `import tomllib` to the imports. Add at module level (after `_get_repo_root`):

```python
# Built-in defaults — lowest precedence. Keys match CLI flag names
# (underscored) and TOML keys in [training]/[trainer] sections.
BUILTIN_DEFAULTS = {
    "iterations":          100_000,
    "checkpoint_interval": None,
    "checkpoint":          "neural_cfr/checkpoints/checkpoint.pt",
    "reservoir_size":      2_000_000,
    "batch_size":          4096,
    "lr":                  1e-3,
    "train_interval":      10_000,
    "sgd_steps":           2_000,
    "reinit_adv":          True,
    "num_threads":         0,
    "epsilon":             0.06,
    "eval_interval":       None,
    "eval_hands":          500,
}


def resolve_config(args: "argparse.Namespace", repo_root: str) -> dict:
    """Merge config sources. Precedence: CLI flag > TOML file > builtin."""
    cfg = dict(BUILTIN_DEFAULTS)
    path = args.config or os.path.join(repo_root, "neural_cfr", "configs", "default.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for section in ("training", "trainer"):
            for key, value in data.get(section, {}).items():
                if key not in BUILTIN_DEFAULTS:
                    raise KeyError(f"Unknown config key [{section}].{key} in {path}")
                cfg[key] = value
    for key in BUILTIN_DEFAULTS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            cfg[key] = cli_value
    return cfg


def write_config_snapshot(cfg: dict, checkpoint_path: str) -> None:
    """Write the effective config next to the checkpoint for reproducibility."""
    lines = ["# Effective config — written by train_neural.py", "[resolved]"]
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
```

In `main()`:
1. Change every existing config-key flag's `default=` to `default=None` (`--iterations`, `--checkpoint-interval`, `--reservoir-size`, `--batch-size`, `--lr`, `--checkpoint`, `--train-interval`, `--num-threads`, `--epsilon`, `--eval-interval`, `--eval-hands`). `--resume` stays as is (CLI-only).
2. Add three flags:

```python
    parser.add_argument("--config",     type=str, default=None,
                        help="TOML config file (default: neural_cfr/configs/default.toml if present)")
    parser.add_argument("--sgd-steps",  type=int, default=None,
                        help="SGD mini-batches per training event (default: 2000)")
    parser.add_argument("--reinit-adv", action=argparse.BooleanOptionalAction, default=None,
                        help="Reinitialize advantage nets each training event (default: on)")
```

3. After `args = parser.parse_args()`:

```python
    cfg = resolve_config(args, repo_root)
    print("Effective config: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))
```

4. Replace all `args.X` uses below with `cfg["X"]` (except `args.resume`), pass the new kwargs to the Trainer:

```python
    trainer = neural_cfr.Trainer(
        reservoir_size=cfg["reservoir_size"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        train_interval=cfg["train_interval"],
        num_threads=cfg["num_threads"],
        epsilon=cfg["epsilon"],
        sgd_steps=cfg["sgd_steps"],
        reinit_adv=cfg["reinit_adv"],
    )
```

5. After each `trainer.checkpoint(cfg["checkpoint"])` call add:

```python
        write_config_snapshot(cfg, cfg["checkpoint"])
```

- [ ] **Step 4: Create the config files**

`neural_cfr/configs/default.toml`:

```toml
# Paper-faithful Deep CFR regime (Brown et al. 2019).
# Precedence: CLI flag > this file > built-in defaults.

[training]
iterations          = 5_000_000
checkpoint_interval = 500_000
checkpoint          = "neural_cfr/checkpoints/checkpoint.pt"

[trainer]
reservoir_size = 2_000_000
batch_size     = 4096
lr             = 1e-3
train_interval = 10_000   # traversal-pairs per CFR iteration
sgd_steps      = 2_000    # mini-batches per training event
reinit_adv     = true     # from-scratch retraining (key paper ablation)
num_threads    = 0        # 0 = hardware concurrency
epsilon        = 0.06
```

`neural_cfr/configs/smoke.toml`:

```toml
# Legacy cheap regime — smoke tests only. Reproduces the pre-2026-07
# continual-training behavior: 1 SGD step every 10 traversal-pairs.

[training]
iterations = 10_000
checkpoint = "neural_cfr/checkpoints/smoke.pt"

[trainer]
lr             = 1e-4
train_interval = 10
sgd_steps      = 1
reinit_adv     = false
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_train_neural_config.py -v`
Expected: 5 tests PASS.

- [ ] **Step 6: End-to-end sanity run**

Run: `uv run python scripts/train_neural.py --config neural_cfr/configs/smoke.toml --iterations 200 --checkpoint /tmp/ncfr_smoke.pt`
Expected: prints `Effective config: …`, runs 200 iterations, saves `/tmp/ncfr_smoke.pt` and `/tmp/ncfr_smoke.pt.config.toml` (verify both exist; may print a `train_strategy: skipped` warning — fine at this scale). Note: this requires Tasks B2+B3 built.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_neural.py neural_cfr/configs/default.toml neural_cfr/configs/smoke.toml tests/scripts/test_train_neural_config.py
git commit -m "feat: TOML config for neural training (CLI > file > builtin) + checkpoint config snapshots

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Track C — worktree `ncfr-parity` (branch `ncfr/inference-parity`, based on `ncfr/regret-match`)

### Task C1: `make_inference_state` in core + street-bet args + parity tests

**Files:**
- Create: `neural_cfr/src/net/inference.h`, `neural_cfr/src/net/inference.cpp` (in `:core` so tests can link it; the BUCK glob picks it up automatically)
- Modify: `neural_cfr/src/bindings/bindings.cpp` — `Strategy::get_action_probs` and the `AdvantageProbe` private helpers/public methods (delete `_make_state`/`_build_legal` duplication); the pybind `.def` argument lists for all three methods
- Test: `neural_cfr/tests/test_features.cpp` (append)

**Interfaces:**
- Consumes: `encode_features(state, player)` from `net/features.h`; `AbstractState` from `game/abstract_state.h`.
- Produces (declared in `neural_cfr/src/net/inference.h`):

```cpp
AbstractState make_inference_state(
    const std::vector<int>& hole_cards,
    const std::vector<int>& board_cards,
    int street, float pot, float stack, float to_call,
    const std::vector<int>& raises_per_street,
    int position,
    float my_street_bet  = -1.0f,
    float opp_street_bet = -1.0f);
```

  Python API change (Tasks C2/C3 use these exact kwarg names): `get_action_probs(..., my_street_bet=-1.0, opp_street_bet=-1.0)` and the same two trailing kwargs on `AdvantageProbe.get_advantage_probs` / `get_raw_advantages`. Sentinel `-1` (either one) = "not provided" → fallback `player_bets = {0, to_call}`. **`pot` must include all street bets** in both modes.

- [ ] **Step 1: Write the failing tests**

Append to `neural_cfr/tests/test_features.cpp` (it already includes gtest and features.h; add `#include "net/inference.h"` and `#include "game/abstract_state.h"` if missing):

```cpp
// --- Train/inference feature parity -------------------------------------
// The features encoded during traversal and the features built by
// make_inference_state for the same decision point must be identical.

static std::vector<int> _history_vec(const AbstractState& s) {
    return {s.betting_history[0], s.betting_history[1],
            s.betting_history[2], s.betting_history[3]};
}

TEST(InferenceParity, PreflopFacingRaise) {
    auto s = deal_heads_up();      // SB=p0 acts first
    s = s.apply_action("b1.0");    // SB raises; BB (p1) faces it
    int p = s.acting_player();
    ASSERT_EQ(p, 1);
    float to_call = s.current_bet - s.player_bets[p];

    auto inf = make_inference_state(
        {s.hole_cards[p][0], s.hole_cards[p][1]}, /*board=*/{},
        s.street, s.pot, s.stacks[p], to_call, _history_vec(s), p,
        /*my_street_bet=*/s.player_bets[p],
        /*opp_street_bet=*/s.player_bets[1 - p]);

    EXPECT_TRUE(torch::allclose(encode_features(s, p), encode_features(inf, p)));
    EXPECT_EQ(s.legal_actions(), inf.legal_actions());
}

TEST(InferenceParity, PostflopBetAfterRaisedPreflop) {
    auto s = deal_heads_up();
    s = s.apply_action("b0.5");    // SB raises
    s = s.apply_action("call");    // BB calls → flop
    s = s.advance_street();
    s = s.apply_action("b0.5");    // BB leads flop; SB (p0) faces the bet
    int p = s.acting_player();
    ASSERT_EQ(p, 0);
    float to_call = s.current_bet - s.player_bets[p];

    std::vector<int> board(s.board.begin(), s.board.end());
    auto inf = make_inference_state(
        {s.hole_cards[p][0], s.hole_cards[p][1]}, board,
        s.street, s.pot, s.stacks[p], to_call, _history_vec(s), p,
        s.player_bets[p], s.player_bets[1 - p]);

    EXPECT_TRUE(torch::allclose(encode_features(s, p), encode_features(inf, p)));
    EXPECT_EQ(s.legal_actions(), inf.legal_actions());
}

TEST(InferenceParity, FallbackWithoutStreetBetsEncodesToCall) {
    // Sentinel mode: opponent street bet approximated by to_call — strictly
    // better than the old all-zeros encoding.
    auto inf = make_inference_state({51, 50}, {}, /*street=*/0, /*pot=*/4.0f,
                                    /*stack=*/97.0f, /*to_call=*/2.0f,
                                    {1, 0, 0, 0}, /*position=*/1);
    auto t = encode_features(inf, 1);
    auto d = t.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[129], 0.0f);            // my street bet / 200
    EXPECT_FLOAT_EQ(d[130], 2.0f / 200.0f);   // opp street bet / 200
}

TEST(InferenceParity, RaisesPerStreetClamped) {
    auto inf = make_inference_state({51, 50}, {}, 0, 4.0f, 97.0f, 2.0f,
                                    {7, -3, 0, 0}, 1);
    auto t = encode_features(inf, 1);  // keep tensor alive — data_ptr on a
    auto d = t.data_ptr<float>();      // temporary would dangle
    EXPECT_FLOAT_EQ(d[125], 2.0f / 2.0f);  // clamped 7 → 2, / RAISE_NORM
    EXPECT_FLOAT_EQ(d[126], 0.0f);         // clamped -3 → 0
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/bin/buck2 test //neural_cfr/tests:test_features`
Expected: compile FAILURE — `net/inference.h` does not exist.

- [ ] **Step 3: Implement the core function**

Create `neural_cfr/src/net/inference.h`:

```cpp
#pragma once
#include <vector>
#include "game/abstract_state.h"

// Build a synthetic AbstractState for inference-time feature encoding and
// legal-action derivation, matching training-time feature semantics.
//
// pot MUST include all street bets (same as during traversal).
// my_street_bet / opp_street_bet: chips committed this street by the acting
// player / opponent. Pass -1 (either) if unknown: falls back to
// player_bets = {0, to_call}, which loses the acting player's own prior
// street commitment but preserves the bet being faced.
// raises_per_street values are clamped to [0, 2] (training caps them at 2).
AbstractState make_inference_state(
    const std::vector<int>& hole_cards,
    const std::vector<int>& board_cards,
    int street, float pot, float stack, float to_call,
    const std::vector<int>& raises_per_street,
    int position,
    float my_street_bet  = -1.0f,
    float opp_street_bet = -1.0f);
```

Create `neural_cfr/src/net/inference.cpp`:

```cpp
#include "net/inference.h"
#include <algorithm>

AbstractState make_inference_state(
    const std::vector<int>& hole_cards,
    const std::vector<int>& board_cards,
    int street, float pot, float stack, float to_call,
    const std::vector<int>& raises_per_street,
    int position,
    float my_street_bet,
    float opp_street_bet)
{
    AbstractState s{};
    s.hole_cards[position][0] = hole_cards[0];
    s.hole_cards[position][1] = hole_cards[1];
    s.board = std::vector<Card>(board_cards.begin(), board_cards.end());
    s.street = street;
    s.pot = pot;
    s.stacks[position] = stack;
    s.stacks[1 - position] = stack;  // opponent stack unknown; not a feature

    if (my_street_bet >= 0.0f && opp_street_bet >= 0.0f) {
        s.player_bets[position]     = my_street_bet;
        s.player_bets[1 - position] = opp_street_bet;
        // Derive current_bet from street bets — the more precise signal —
        // rather than trusting the redundant to_call argument.
        s.current_bet = std::max(my_street_bet, opp_street_bet);
    } else {
        s.player_bets[position]     = 0.0f;
        s.player_bets[1 - position] = to_call;
        s.current_bet = to_call;
    }

    s.betting_history = {0, 0, 0, 0};
    for (int i = 0; i < 4 && i < (int)raises_per_street.size(); ++i)
        s.betting_history[i] = std::clamp(raises_per_street[i], 0, 2);
    s.folded = {false, false};
    s.to_act = {position};
    return s;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/bin/buck2 test //neural_cfr/tests:test_features`
Expected: all tests PASS (including the pre-existing ones).

- [ ] **Step 5: Rewire `bindings.cpp` to use it**

In `neural_cfr/src/bindings/bindings.cpp`:

1. Add `#include "net/inference.h"`.
2. `Strategy::get_action_probs`: add parameters `float my_street_bet = -1.0f, float opp_street_bet = -1.0f` after `position`; replace the inline `AbstractState s{}; …` construction (everything up to `auto feat = encode_features(s, position);`) with:

```cpp
        auto s = make_inference_state(hole_cards, board_cards, street, pot,
                                      stack, to_call, raises_per_street,
                                      position, my_street_bet, opp_street_bet);
        auto feat = encode_features(s, position);
```

3. `AdvantageProbe`: delete `_make_state` and `_build_legal`; add the same two trailing parameters to `get_advantage_probs`, `get_raw_advantages`, and `_forward`; replace their state construction with `make_inference_state(...)` calls (in `get_*`: `auto legal = make_inference_state(...).legal_actions();`).
4. Update the pybind `.def` lists — for all three methods append:

```cpp
             py::arg("my_street_bet")  = -1.0f,
             py::arg("opp_street_bet") = -1.0f
```

5. Update the `get_action_probs` doc comment: `pot` must include all street bets; describe the two new kwargs and the sentinel.

- [ ] **Step 6: Build and verify the Python surface**

Run: `~/bin/buck2 build //neural_cfr:neural_cfr --show-output`
Expected: `BUILD SUCCEEDED`.

Then with the printed .so dir (needs an existing checkpoint at `neural_cfr/checkpoints/checkpoint.pt`; if none exists, skip this step — the parity tests in Step 4 already cover the logic):

```bash
uv run python -c "
import ctypes, sys
for lib in ['libc10.dylib','libtorch_cpu.dylib','libtorch.dylib']:
    ctypes.CDLL('third_party/libtorch/lib/'+lib)
sys.path.insert(0, '<SO_DIR_FROM_BUCK_OUTPUT>')
import neural_cfr
s = neural_cfr.Strategy('neural_cfr/checkpoints/checkpoint.pt')
old = s.get_action_probs([51,50],[],0,3.0,99.0,1.0,[1,0,0,0],1)
new = s.get_action_probs([51,50],[],0,3.0,99.0,1.0,[1,0,0,0],1,
                         my_street_bet=1.0, opp_street_bet=2.0)
print('backcompat:', dict(old)); print('street-bets:', dict(new))
"
```

Expected: two probability dicts over the same legal actions, no exceptions.

- [ ] **Step 7: Commit**

```bash
git add neural_cfr/src/net/inference.h neural_cfr/src/net/inference.cpp neural_cfr/src/bindings/bindings.cpp neural_cfr/tests/test_features.cpp
git commit -m "fix: train/inference feature parity — street bets reach the net

make_inference_state (in :core, unit-testable) now encodes street bets
into feature dims 129/130. Callers may pass my_street_bet/opp_street_bet
for exact parity; without them, the opponent bet is approximated by
to_call (previously both dims were always zero at inference, so the net
could not see the bet it was facing). Also clamps raises_per_street to
[0,2].

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task C2: `AdvantageProbe` uses shared `regret_match` (argmax fallback mirror)

**Files:**
- Modify: `neural_cfr/src/bindings/bindings.cpp` — `AdvantageProbe::get_advantage_probs` body only

**Interfaces:**
- Consumes: `regret_match(const std::array<float, 6>&, const std::vector<int>&)` from `cfr/traversal.h` (Task A1 — this worktree branches from that branch).

- [ ] **Step 1: Replace the manual ReLU/normalize block**

Add `#include "cfr/traversal.h"` to bindings.cpp. In `get_advantage_probs`, replace the block from `std::vector<float> probs;` through the `else … 1.0f / probs.size();` loop with:

```cpp
        std::array<float, 6> adv{};
        for (int i = 0; i < 6; ++i) adv[i] = logits[i].item<float>();
        std::vector<int> legal_idx;
        for (auto& a : legal) legal_idx.push_back(action_idx(a));
        // Same regret matching (incl. argmax fallback) as traversal.
        auto probs = regret_match(adv, legal_idx);
```

The `py::dict` assembly below it stays unchanged.

- [ ] **Step 2: Build**

Run: `~/bin/buck2 build //neural_cfr:neural_cfr`
Expected: `BUILD SUCCEEDED`.

- [ ] **Step 3: Commit**

```bash
git add neural_cfr/src/bindings/bindings.cpp
git commit -m "refactor: AdvantageProbe reuses traversal regret_match (argmax fallback)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task C3: Python callers pass real street bets

**Files:**
- Modify: `scripts/openpoker_bot.py:214-223` (`HandTracker._decide_neural`)
- Modify: `scripts/eval_openspiel_neural.py` (new helper + `NeuralCFRPolicy.action_probabilities`)

**Interfaces:**
- Consumes: `get_action_probs(..., my_street_bet=, opp_street_bet=)` from Task C1.
- Notes for the implementer:
  - In `openpoker_bot.py`, `HandTracker.my_committed` is already the bot's **current-street** commitment: set to its blind in `on_hand_start`, reset to 0 in `on_community_cards`, updated in `_update_committed` (raise-to convention). In heads-up, the opponent's street bet is exactly `my_committed + to_call_chips`.
  - In `eval_openspiel_neural.py`, universal_poker `r<N>` amounts in the `Sequences` string are **cumulative for the hand** (raise-to total committed). Remaining stacks come from `[Money: m0 m1]`, starting stack is `OPENSPIEL_STARTING_STACK` (1000), big blind is `BIG_BLIND` (10).

- [ ] **Step 1: `openpoker_bot.py` — pass street bets**

In `_decide_neural`, replace the `get_action_probs` call with:

```python
        probs_dict = strategy.get_action_probs(
            hole_ints,
            board_ints,
            self.street,
            pot          / scale,
            self.my_stack / scale,
            to_call_chips / scale,
            self.raises_per_street,
            self.my_position,
            my_street_bet  = self.my_committed / scale,
            opp_street_bet = (self.my_committed + to_call_chips) / scale,
        )
```

- [ ] **Step 2: `eval_openspiel_neural.py` — derive street bets from the info state**

Add after `_count_raises`:

```python
def _street_start_committed(sequences: str, street: int, big_blind: float) -> float:
    """Per-player chips committed at the start of `street`.

    universal_poker r<N> amounts are cumulative for the hand, so the last
    raise-to amount on any completed street is both players' total at that
    street's close (the street only ends once the raise is called). With no
    raises yet, both players have matched the big blind (or nothing preflop).
    """
    committed = 0.0 if street == 0 else float(big_blind)
    for part in sequences.split("|")[:street]:
        raises = re.findall(r"r(\d+)", part)
        if raises:
            committed = float(raises[-1])
    return committed
```

In `action_probabilities`, after `to_call` is computed, add:

```python
        # Exact street bets for feature parity with training (dims 129/130).
        my_total  = OPENSPIEL_STARTING_STACK - stack_my
        opp_total = OPENSPIEL_STARTING_STACK - stack_op
        c0 = _street_start_committed(sequences, street, BIG_BLIND)
        my_sb, opp_sb = my_total - c0, opp_total - c0
        street_bet_kwargs = {}
        if my_sb >= 0.0 and opp_sb >= 0.0:
            street_bet_kwargs = {
                "my_street_bet":  my_sb  / _CHIP_SCALE,
                "opp_street_bet": opp_sb / _CHIP_SCALE,
            }
        # Negative values mean the sequence parse disagreed with the stacks —
        # omit the kwargs and let the C++ fallback (opp bet = to_call) apply.
```

and append `**street_bet_kwargs` to the `self.strategy.get_action_probs(...)` call.

- [ ] **Step 3: Verify end-to-end against OpenSpiel**

Run (requires a checkpoint; use any existing one, or the tiny one produced by Track B's smoke run after integration):

```bash
uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 50 --baseline random
```

Expected: completes without exceptions and prints a BB/100 result. (If no checkpoint exists yet in this worktree, defer this step to integration Task I1 — note it in the commit message.)

- [ ] **Step 4: Regression: tabular eval path untouched**

Run: `uv run pytest tests/ -x -q`
Expected: full suite passes (~82 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/openpoker_bot.py scripts/eval_openspiel_neural.py
git commit -m "feat: callers pass exact street bets to neural strategy

openpoker_bot derives them from my_committed (already street-scoped);
the OpenSpiel eval reconstructs them from cumulative raise-to amounts
in the Sequences string, falling back to the C++ default on mismatch.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Integration (sequential, back on `feature/neural-cfr`)

### Task I1: Merge tracks, full build + test

- [ ] **Step 1: Merge in order A → B → C**

```bash
git checkout feature/neural-cfr
git merge --no-ff ncfr/regret-match
git merge --no-ff ncfr/trainer-regime
git merge --no-ff ncfr/inference-parity
```

Expected: A and B merge cleanly. C may report a trivial `bindings.cpp` conflict with B (disjoint regions: B edited the `PYBIND11_MODULE` Trainer block, C edited the class bodies) — if so, resolve by keeping both changes.

- [ ] **Step 2: Full C++ build + tests**

```bash
~/bin/buck2 build //neural_cfr:neural_cfr
~/bin/buck2 test //neural_cfr/tests:test_abstract_state //neural_cfr/tests:test_features //neural_cfr/tests:test_reservoir_buffer //neural_cfr/tests:test_traversal //neural_cfr/tests:test_trainer
```

Expected: all PASS.

- [ ] **Step 3: Full Python suite**

Run: `uv run pytest tests/ -q`
Expected: all pass (existing ~82 + 5 new config tests).

- [ ] **Step 4: Clean up worktrees**

Remove the three worktrees and delete merged branches (`git worktree remove <path>`, `git branch -d ncfr/regret-match ncfr/trainer-regime ncfr/inference-parity`).

### Task I2: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — the `## Neural CFR (neural_cfr/)` section

- [ ] **Step 1: Update the stale content**

Apply these edits (keep surrounding text):
1. In the constants line, replace `DEFAULT_LR=1e-4, … DEFAULT_TRAIN_INTERVAL=10` with `DEFAULT_LR=1e-3, DEFAULT_RESERVOIR_SIZE=2_000_000, DEFAULT_TRAIN_INTERVAL=10_000, DEFAULT_SGD_STEPS=2_000, DEFAULT_REINIT_ADV=true (grad-norm clip 1.0)`.
2. In "Training algorithm", append:

```markdown
- Training regime (paper-faithful, 2026-07): every `train_interval` traversal-pairs = one CFR iteration → advantage nets reinitialized from scratch and trained `sgd_steps` mini-batches. Strategy net is trained from scratch on M_π only at checkpoint time (`Trainer.train_strategy`, called automatically by `checkpoint()`). Legacy continual regime: `neural_cfr/configs/smoke.toml`.
- If all predicted advantages are ≤ 0, regret matching plays argmax(advantage) as a pure strategy (not uniform).
```

3. In "Commands", note config usage:

```markdown
# Config: neural_cfr/configs/default.toml (CLI flags override; effective
# config is snapshotted to <checkpoint>.config.toml on every save)
uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml
```

4. In the feature-encoding section, after the dims table add: `At inference, pass my_street_bet/opp_street_bet to get_action_probs for exact dim-129/130 parity; pot must include all street bets.`
5. In "Checkpoint format", append: `checkpoint() retrains the strategy net on M_π before saving; buffers are still not serialized, so prefer single uninterrupted runs (a resumed run's strategy net only sees post-resume M_π).`

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new neural CFR training regime

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task I3: Smoke gate — short train + probe sanity

- [ ] **Step 1: 100k-iteration smoke train with the new regime (scaled-down SGD)**

```bash
mkdir -p neural_cfr/checkpoints
uv run python scripts/train_neural.py \
    --iterations 100000 --train-interval 10000 --sgd-steps 500 \
    --checkpoint-interval 100000 \
    --checkpoint neural_cfr/checkpoints/smoke_fixes.pt
```

Expected: 10 CFR iterations, each followed by a visible training pause; checkpoint + `.config.toml` snapshot written. Wall clock ~30–60 min.

- [ ] **Step 2: Probe spot checks**

```bash
uv run python - <<'EOF'
import ctypes, os, subprocess, sys
root = os.getcwd()
for lib in ["libc10.dylib", "libtorch_cpu.dylib", "libtorch.dylib"]:
    ctypes.CDLL(os.path.join(root, "third_party", "libtorch", "lib", lib))
out = subprocess.run([os.path.expanduser("~/bin/buck2"), "build",
                      "//neural_cfr:neural_cfr", "--show-output"],
                     capture_output=True, text=True, cwd=root)
so_dir = next(os.path.join(root, os.path.dirname(l.split()[-1]))
              for l in out.stdout.splitlines() if "neural_cfr.so" in l)
sys.path.insert(0, so_dir)
import neural_cfr

s = neural_cfr.Strategy("neural_cfr/checkpoints/smoke_fixes.pt")
# AA (As=51, Ah=50) as SB preflop, unopened: expect aggression (b*/allin+call
# mass well above fold).
aa = s.get_action_probs([51, 50], [], 0, 1.5, 99.5, 0.5, [0, 0, 0, 0], 0,
                        my_street_bet=0.5, opp_street_bet=1.0)
# 72o (7c=20, 2d=1) as BB facing an SB shove: expect fold-dominant.
trash = s.get_action_probs([20, 1], [], 0, 101.0, 99.0, 99.0, [2, 0, 0, 0], 1,
                           my_street_bet=1.0, opp_street_bet=100.0)
print("AA  SB preflop:", {k: round(v, 3) for k, v in aa.items()})
print("72o vs shove :", {k: round(v, 3) for k, v in trash.items()})
assert aa.get("fold", 0.0) < 0.3, "AA folding too much"
assert trash.get("fold", 0.0) > 0.5, "72o not folding to a shove"
print("SMOKE GATE PASSED")
EOF
```

Expected: `SMOKE GATE PASSED`. A 100k-iteration net is rough — the thresholds are deliberately loose. If it fails, debug before any long run (use `superpowers:systematic-debugging`).

- [ ] **Step 3: Commit nothing** — this task produces no source changes (the smoke checkpoint stays gitignored under `neural_cfr/checkpoints/`).

### Task I4: Acceptance — full retrain + head-to-head evals

Reference targets (spec): sanity floor = beat the current pre-fix checkpoint on both evals; primary = ≥ 0 BB/100 vs the tabular bot; secondary = at least match the tabular bot's win rate vs random. Evals at ≥ 20,000 hands (SE ≈ ±7 BB/100).

- [ ] **Step 1: Preserve the pre-fix baseline checkpoint**

```bash
cp neural_cfr/checkpoints/checkpoint.pt neural_cfr/checkpoints/prefix_baseline.pt 2>/dev/null || echo "no existing checkpoint — skip baseline comparison"
```

- [ ] **Step 2: Baseline numbers (pre-fix checkpoint, if it exists)**

```bash
uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/prefix_baseline.pt --hands 20000 --baseline random
uv run python scripts/eval_neural_vs_tabular.py --neural-checkpoint neural_cfr/checkpoints/prefix_baseline.pt --hands 20000
```

Record both BB/100 numbers.

- [ ] **Step 3: Full retrain (long-running — run in background/overnight)**

```bash
uv run python scripts/train_neural.py --config neural_cfr/configs/default.toml
```

Expected: 5M iterations = 500 CFR iterations; checkpoints every 500k with config snapshots. Total wall clock: traversal time + ~4–8 h SGD.

- [ ] **Step 4: Acceptance evals**

```bash
uv run python scripts/eval_openspiel_neural.py --checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 20000 --baseline random
uv run python scripts/eval_neural_vs_tabular.py --neural-checkpoint neural_cfr/checkpoints/checkpoint.pt --hands 20000
```

Compare against Step 2 and the targets table. Report all numbers (pass or fail) — do not cherry-pick.

- [ ] **Step 5: Record results**

Append an `## Acceptance Results (YYYY-MM-DD)` section with the numbers to `docs/superpowers/specs/2026-07-07-neural-cfr-fixes-design.md`, commit:

```bash
git add docs/superpowers/specs/2026-07-07-neural-cfr-fixes-design.md
git commit -m "docs: acceptance results for neural CFR fixes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
