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
