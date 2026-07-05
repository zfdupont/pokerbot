#pragma once
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"
#include <string>
#include <memory>
#include <torch/optim.h>

constexpr int   DEFAULT_BATCH_SIZE       = 4096;
constexpr float DEFAULT_LR               = 1e-4f;
constexpr int   DEFAULT_RESERVOIR_SIZE   = 2'000'000;
// Train networks every N CFR iterations (rounds, not per-thread).
constexpr int   DEFAULT_TRAIN_INTERVAL   = 10;
// 0 = use std::thread::hardware_concurrency()
constexpr int   DEFAULT_NUM_THREADS      = 0;
// ε-greedy exploration at opponent nodes (Brown et al. 2019 uses 0.06)
constexpr float DEFAULT_EPSILON          = 0.06f;

class Trainer {
public:
    explicit Trainer(
        size_t reservoir_size  = DEFAULT_RESERVOIR_SIZE,
        size_t batch_size      = DEFAULT_BATCH_SIZE,
        float  lr              = DEFAULT_LR,
        int    train_interval  = DEFAULT_TRAIN_INTERVAL,
        int    num_threads     = DEFAULT_NUM_THREADS,
        float  epsilon         = DEFAULT_EPSILON);

    void run(int iterations);
    void checkpoint(const std::string& path);
    void load(const std::string& path);

private:
    MLP adv0_, adv1_, strat_;
    ReservoirBuffer<BufferEntry> mv0_, mv1_, mpi_;
    torch::optim::Adam opt_adv0_, opt_adv1_, opt_strat_;
    size_t batch_size_;
    int    train_interval_;
    int    num_threads_;
    float  epsilon_;

    // Train a network on its buffer using weighted loss.
    // mode: "advantage" uses weighted MSE; "strategy" uses weighted cross-entropy.
    void train_step(MLP& net, torch::optim::Adam& opt,
                    ReservoirBuffer<BufferEntry>& buffer,
                    const std::string& mode);
};
