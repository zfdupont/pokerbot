// sixmax/src/dream/trainer.h
#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include "abstraction/abstraction.h"
#include "blueprint/engine_game.h"
#include "dream/nets.h"
#include "dream/reservoir.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

struct DreamConfig {
    int      hidden_size    = 256;
    int      hidden_layers  = 3;
    float    lr             = 1e-3f;
    int      batch_size     = 4096;
    size_t   reservoir_size = 2'000'000;
    int      train_interval = 10'000;   // traversals between retraining
    int      sgd_steps      = 2'000;
    float    epsilon        = 0.06f;    // ε-greedy exploration
    int      num_threads    = 1;        // <=0 = hardware_concurrency
    uint64_t seed           = 42;
    // Stack randomization: LogNormal(ln(100), 0.5) clipped to [stack_min, stack_max]
    float    stack_min      = 20.0f;
    float    stack_max      = 250.0f;
    float    stack_log_mean = 4.605f;   // ln(100)
    float    stack_log_std  = 0.5f;
    // Players per hand: uniform [players_min, players_max]
    int      players_min    = 2;
    int      players_max    = 6;
};

class DreamTrainer {
public:
    DreamTrainer(int n_actions, const ActionVocab* vocab,
                 const Abstraction* abstraction,
                 DreamConfig cfg, torch::Device device);

    void     train(uint64_t iterations);
    uint64_t total_iterations() const {
        return iter_.load(std::memory_order_relaxed);
    }

    // Expose nets for checkpoint saving
    DreamMLP adv_net()   const { return adv_net_; }
    DreamMLP strat_net() const { return strat_net_; }

private:
    struct TrajectoryNode {
        torch::Tensor      features;
        std::vector<float> sigma;    // strategy at this node
        int                a_star;   // sampled action
        int                player;
    };

    void worker(uint64_t n_iterations, uint64_t seed_offset);

    // Traverse a single hand: fills traj and utilities in-place.
    void traverse(EngineGameState& state, std::mt19937_64& rng,
                  std::vector<TrajectoryNode>& traj,
                  std::vector<float>& utilities);

    std::vector<float> eps_greedy_strategy(const torch::Tensor& advantages,
                                           const std::vector<uint8_t>& mask) const;

    void retrain_adv();
    void retrain_strat();

    // --- persistent state ---
    int              n_actions_;
    const ActionVocab*   vocab_;
    const Abstraction*   abstraction_;
    DreamConfig      cfg_;
    torch::Device    device_;

    DreamMLP adv_net_{nullptr};
    DreamMLP strat_net_{nullptr};
    std::mutex adv_net_mu_;    // guards adv_net_ during retraining
    std::mutex strat_net_mu_;  // guards strat_net_ during retraining

    WeightedReservoir M_v_;    // advantage samples
    WeightedReservoir M_pi_;   // strategy samples

    std::atomic<uint64_t> iter_{0};
};

}  // namespace sixmax
