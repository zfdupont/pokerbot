#pragma once
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"
#include <string>
#include <memory>
#include <torch/optim.h>

constexpr int   DEFAULT_BATCH_SIZE       = 4096;
constexpr float DEFAULT_LR               = 1e-4f;
constexpr int   DEFAULT_RESERVOIR_SIZE   = 2'000'000;
// Train networks every N CFR iterations. Higher = more traversals per SGD step,
// faster wall-clock but slightly noisier gradient signal.
constexpr int   DEFAULT_TRAIN_INTERVAL   = 10;

class Trainer {
public:
    explicit Trainer(
        size_t reservoir_size  = DEFAULT_RESERVOIR_SIZE,
        size_t batch_size      = DEFAULT_BATCH_SIZE,
        float  lr              = DEFAULT_LR,
        int    train_interval  = DEFAULT_TRAIN_INTERVAL);

    void run(int iterations);
    void checkpoint(const std::string& path);
    void load(const std::string& path);

private:
    MLP adv0_, adv1_, strat_;
    ReservoirBuffer<BufferEntry> mv0_, mv1_, mpi_;
    torch::optim::Adam opt_adv0_, opt_adv1_, opt_strat_;
    size_t batch_size_;
    int    train_interval_;

    // Train a network on its buffer using weighted loss.
    // mode: "advantage" uses weighted MSE; "strategy" uses weighted cross-entropy.
    void train_step(MLP& net, torch::optim::Adam& opt,
                    ReservoirBuffer<BufferEntry>& buffer,
                    const std::string& mode);
};
