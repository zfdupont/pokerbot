#include "cfr/trainer.h"
#include "cfr/traversal.h"
#include "game/abstract_state.h"
#include "net/features.h"
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
    torch::serialize::OutputArchive root;
    torch::serialize::OutputArchive a0, a1, s;
    adv0_.save(a0); adv1_.save(a1); strat_.save(s);
    root.write("adv0", a0);
    root.write("adv1", a1);
    root.write("strat", s);
    root.save_to(path);
    std::cout << "Checkpoint saved to " << path << "\n";
}

void Trainer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("Checkpoint not found: " + path);
    f.close();
    torch::serialize::InputArchive root;
    root.load_from(path);
    torch::serialize::InputArchive a0, a1, s;
    root.read("adv0", a0); root.read("adv1", a1); root.read("strat", s);
    adv0_.load(a0); adv1_.load(a1); strat_.load(s);
    std::cout << "Checkpoint loaded from " << path << "\n";
}
