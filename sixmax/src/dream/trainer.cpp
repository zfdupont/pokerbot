// sixmax/src/dream/trainer.cpp
#include "dream/trainer.h"
#include "dream/features.h"
#include <algorithm>
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
// Traversal
// ---------------------------------------------------------------------------

void DreamTrainer::traverse(EngineGameState& state, std::mt19937_64& rng,
                             std::vector<TrajectoryNode>& traj,
                             std::vector<float>& utilities) {
    traj.clear();

    // Walk forward: at each decision node sample one action per ε-greedy strategy
    while (!state.is_terminal()) {
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

        // Sample one action
        std::discrete_distribution<int> dist(sigma.begin(), sigma.end());
        int a_star = dist(rng);

        traj.push_back({feat.cpu(), sigma, a_star, p});
        state.apply(a_star);
    }

    // Collect utilities for all players
    int n = state.num_players();
    utilities.resize(n);
    for (int i = 0; i < n; ++i)
        utilities[i] = (float)state.utility(i);
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

    std::vector<TrajectoryNode> traj;
    std::vector<float>          utilities;

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

        EngineGameState state(
            HandState(eng_cfg, button, std::move(deck), stacks),
            vocab_, abstraction_);

        traverse(state, rng, traj, utilities);

        uint64_t t = iter_.fetch_add(1, std::memory_order_relaxed) + 1;
        float weight = (float)t;

        // For each decision node: store IS-weighted advantage + strategy samples
        for (const auto& node : traj) {
            int   p         = node.player;
            float u_p       = p < (int)utilities.size() ? utilities[p] : 0.0f;
            float prob_star = node.sigma[node.a_star];
            if (prob_star < 1e-9f) continue;  // skip near-zero IS weights

            // IS-corrected sparse advantage vector: only sampled action is non-zero
            std::vector<float> adv_target(n_actions_, 0.0f);
            adv_target[node.a_star] = u_p / prob_star;
            auto adv_t = torch::tensor(adv_target, torch::kFloat32);
            M_v_.add(node.features, adv_t, weight, rng);

            // Strategy sample
            auto strat_t = torch::tensor(node.sigma, torch::kFloat32);
            M_pi_.add(node.features, strat_t, weight, rng);
        }

        // Retrain every train_interval traversals (when reservoir is large enough)
        if (t % (uint64_t)cfg_.train_interval == 0 &&
            M_v_.size() >= (size_t)cfg_.batch_size) {
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
