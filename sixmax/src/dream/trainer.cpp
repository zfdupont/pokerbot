// sixmax/src/dream/trainer.cpp
#include "dream/trainer.h"
#include "dream/features.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>

namespace sixmax {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DreamTrainer::DreamTrainer(int n_actions, const ActionVocab* vocab,
                           const Abstraction* abstraction,
                           DreamConfig cfg, torch::Device device)
    : n_actions_(n_actions),
      vocab_(vocab),
      abstraction_(abstraction),
      cfg_(cfg),
      device_(device),
      adv_net_(DreamMLP(FEATURE_DIM, cfg.hidden_size, cfg.hidden_layers, n_actions)),
      strat_net_(DreamMLP(FEATURE_DIM, cfg.hidden_size, cfg.hidden_layers, n_actions)),
      M_v_(cfg.reservoir_size),
      M_pi_(cfg.reservoir_size) {
    adv_net_->to(device_);
    strat_net_->to(device_);
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

std::vector<float> DreamTrainer::eps_greedy_strategy(
        const torch::Tensor& adv, const std::vector<uint8_t>& mask) const {
    int n = n_actions_;
    std::vector<float> sigma(n, 0.0f);

    // Positive regret matching over legal actions
    float pos_sum = 0.0f;
    auto adv_a = adv.accessor<float, 1>();
    for (int i = 0; i < n; ++i) {
        if (mask[i]) {
            sigma[i] = std::max(0.0f, adv_a[i]);
            pos_sum += sigma[i];
        }
    }
    int n_legal = (int)std::count(mask.begin(), mask.end(), uint8_t(1));
    if (pos_sum < 1e-9f) {
        // All non-positive: fall back to uniform over legal actions
        float u = 1.0f / (float)n_legal;
        for (int i = 0; i < n; ++i) sigma[i] = mask[i] ? u : 0.0f;
    } else {
        for (int i = 0; i < n; ++i) sigma[i] /= pos_sum;
    }

    // ε-greedy mix: σ' = (1-ε)*σ + ε*uniform(legal)
    float eps = cfg_.epsilon;
    float u   = eps / (float)n_legal;
    for (int i = 0; i < n; ++i) {
        if (mask[i]) sigma[i] = (1.0f - eps) * sigma[i] + u;
    }
    return sigma;
}

// ---------------------------------------------------------------------------
// Stochastic external-sampling MCCFR traversal
// ---------------------------------------------------------------------------
//
// Algorithm (per DREAM / DeepCFR external-sampling):
//   Terminal node  → return utility for updating_player.
//   Updating player's node →
//     1. For each legal action a: clone state, apply a, recurse → v(a).
//     2. Compute sigma = eps_greedy_strategy(adv_net output, mask).
//     3. E_v = Σ sigma[a] * v(a).
//     4. adv_target[a] = v(a) - E_v  (instantaneous regret, unbiased estimator
//        of counterfactual regret under external sampling).
//     5. Store (features, adv_target, weight=t) in M_v_.
//     6. Store (features, sigma,      weight=t) in M_pi_.
//     7. Return E_v.
//   Opponent node →
//     1. Compute sigma = eps_greedy_strategy.
//     2. Sample one action a_opp from sigma.
//     3. Store (features, sigma, weight=t) in M_pi_.
//     4. Apply a_opp, recurse, return result.
//
// This keeps branching factor = 1 for all opponent nodes (tractable), while
// giving unbiased advantage estimates for the updating player.  Two traversals
// per hand (one per player) are standard in alternating-update CFR.

float DreamTrainer::traverse(EngineGameState& state, int updating_player,
                              uint64_t t, std::mt19937_64& rng) {
    if (state.is_terminal()) {
        return (float)state.utility(updating_player);
    }

    int p = state.current_player();

    torch::Tensor feat;
    {
        torch::NoGradGuard no_grad;
        feat = encode_state(state).to(device_);
    }

    std::vector<uint8_t> mask;
    state.legal_mask(mask);

    torch::Tensor adv_out;
    {
        std::lock_guard<std::mutex> lock(adv_net_mu_);
        torch::NoGradGuard no_grad;
        adv_out = adv_net_->forward(feat.unsqueeze(0)).squeeze(0).cpu();
    }

    auto sigma = eps_greedy_strategy(adv_out, mask);
    float weight = (float)t;

    if (p == updating_player) {
        // ---- Updating player's node: enumerate ALL legal actions ----

        // Collect legal action indices
        std::vector<int> legal_actions;
        legal_actions.reserve(n_actions_);
        for (int a = 0; a < n_actions_; ++a) {
            if (mask[a]) legal_actions.push_back(a);
        }

        // Recurse into each legal action by cloning the state
        std::vector<float> action_values(n_actions_, 0.0f);
        for (int a : legal_actions) {
            auto child = std::unique_ptr<EngineGameState>(
                static_cast<EngineGameState*>(state.clone().release()));
            child->apply(a);
            action_values[a] = traverse(*child, updating_player, t, rng);
        }

        // Compute expected value under current strategy
        float expected_v = 0.0f;
        for (int a : legal_actions) {
            expected_v += sigma[a] * action_values[a];
        }

        // Instantaneous regrets = v(a) - E[v] for each legal action
        std::vector<float> adv_target(n_actions_, 0.0f);
        for (int a : legal_actions) {
            adv_target[a] = action_values[a] - expected_v;
        }

        auto adv_t  = torch::tensor(adv_target, torch::kFloat32);
        auto strat_t = torch::tensor(sigma,     torch::kFloat32);
        M_v_.add(feat.cpu(), adv_t,   weight, rng);
        M_pi_.add(feat.cpu(), strat_t, weight, rng);

        return expected_v;

    } else {
        // ---- Opponent node: sample one action (outcome-sampling style) ----

        std::discrete_distribution<int> dist(sigma.begin(), sigma.end());
        int a_opp = dist(rng);

        auto strat_t = torch::tensor(sigma, torch::kFloat32);
        M_pi_.add(feat.cpu(), strat_t, weight, rng);

        state.apply(a_opp);
        return traverse(state, updating_player, t, rng);
    }
}

// ---------------------------------------------------------------------------
// Retraining
// ---------------------------------------------------------------------------

void DreamTrainer::retrain_adv() {
    // REINIT: advantage net is reset from scratch each cycle
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
        feat    = feat.to(device_);
        target  = target.to(device_);
        weights = weights.to(device_);

        std::lock_guard<std::mutex> lock(adv_net_mu_);   // hold for full update
        opt.zero_grad();
        auto pred = adv_net_->forward(feat);
        // Weighted MSE over all action dimensions
        auto loss = ((pred - target).pow(2) * weights.unsqueeze(1)).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(adv_net_->parameters(), 1.0);
        opt.step();
    }
}

void DreamTrainer::retrain_strat() {
    // Strategy net is NOT reinitialized — continues from previous weights
    torch::optim::Adam opt(strat_net_->parameters(),
                           torch::optim::AdamOptions(cfg_.lr));
    std::mt19937_64 rng(iter_.load() + 1);

    for (int step = 0; step < cfg_.sgd_steps; ++step) {
        auto [feat, target, weights] = M_pi_.sample_batch(cfg_.batch_size, rng);
        feat    = feat.to(device_);
        target  = target.to(device_);
        weights = weights.to(device_);

        std::lock_guard<std::mutex> lock(strat_net_mu_);   // hold for full update
        opt.zero_grad();
        auto pred = torch::log_softmax(strat_net_->forward(feat), /*dim=*/1);
        // Weighted cross-entropy: -Σ target * log(pred), per-sample weighted
        auto loss = -(target * pred * weights.unsqueeze(1)).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(strat_net_->parameters(), 1.0);
        opt.step();
    }
}

// ---------------------------------------------------------------------------
// Worker: one thread's traversal loop
// ---------------------------------------------------------------------------

void DreamTrainer::worker(uint64_t n_iterations, uint64_t seed_offset) {
    std::mt19937_64 rng(cfg_.seed + seed_offset);
    std::lognormal_distribution<float> stack_dist(cfg_.stack_log_mean,
                                                   cfg_.stack_log_std);
    std::uniform_int_distribution<int> n_players_dist(cfg_.players_min,
                                                       cfg_.players_max);

    // Timing accumulator for avg_traverse_ns_
    double total_traverse_ns = 0.0;
    uint64_t traverse_count  = 0;

    for (uint64_t it = 0; it < n_iterations; ++it) {
        // Randomize game config per hand
        int n_players = n_players_dist(rng);
        std::vector<double> stacks(n_players);
        for (auto& s : stacks) {
            float raw = stack_dist(rng);
            s = (double)std::clamp(raw, cfg_.stack_min, cfg_.stack_max);
        }

        // Shuffle deck and construct HandState directly (supports per-seat stacks)
        std::vector<int> deck(52);
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);

        EngineConfig eng_cfg{n_players, 100.0};
        int button = (int)(rng() % (uint64_t)n_players);

        // Stochastic external-sampling: two traversals per hand, one per player.
        // t_base is a globally-unique even weight for the reservoir; pass t_base
        // and t_base+1 for the two player traversals so weights are distinct.
        uint64_t t_base = iter_.fetch_add(2, std::memory_order_relaxed) + 1;
        // hand_count is the 1-based index of this hand (local+global), used for
        // the retrain interval gate so it fires regardless of thread count.
        uint64_t hand_count = t_base / 2 + 1;   // t_base is odd; (t_base-1)/2+1

        auto t0 = std::chrono::steady_clock::now();

        for (int updating_player = 0; updating_player < 2; ++updating_player) {
            // Re-shuffle deck for each player traversal to get independent samples
            std::shuffle(deck.begin(), deck.end(), rng);

            EngineGameState state(
                HandState(eng_cfg, button, deck, stacks),
                vocab_, abstraction_);

            traverse(state, updating_player, t_base + (uint64_t)updating_player, rng);
        }

        auto t1 = std::chrono::steady_clock::now();
        total_traverse_ns +=
            (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        traverse_count += 2;

        // Retrain every train_interval HANDS (when reservoir is large enough).
        // Using hand_count (not t_base) ensures the check fires correctly
        // regardless of whether we fetch_add by 1 or 2 per loop iteration.
        if (hand_count % (uint64_t)cfg_.train_interval == 0 &&
            M_v_.size() >= (size_t)cfg_.batch_size) {
            // Update timing stat before retraining
            if (traverse_count > 0) {
                avg_traverse_ns_.store(total_traverse_ns / (double)traverse_count,
                                       std::memory_order_relaxed);
            }
            retrain_adv();
            retrain_strat();
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void DreamTrainer::train(uint64_t iterations) {
    int n_threads = cfg_.num_threads <= 0
        ? (int)std::thread::hardware_concurrency()
        : cfg_.num_threads;

    if (n_threads == 1) {
        worker(iterations, 0);
        return;
    }

    uint64_t per_thread = iterations / (uint64_t)n_threads;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        threads.emplace_back(&DreamTrainer::worker, this,
                             per_thread, (uint64_t)i);
    for (auto& t : threads) t.join();
}

}  // namespace sixmax
