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
