#include "cfr/trainer.h"
#include "cfr/traversal.h"
#include "game/abstract_state.h"
#include "net/features.h"
#include <torch/torch.h>
#include <indicators/indicators.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <numeric>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_interrupted{false};
static void sigint_handler(int) { g_interrupted = true; }

Trainer::Trainer(size_t reservoir_size, size_t batch_size, float lr, int train_interval)
    : mv0_(reservoir_size), mv1_(reservoir_size), mpi_(reservoir_size),
      opt_adv0_(adv0_.parameters(), torch::optim::AdamOptions(lr)),
      opt_adv1_(adv1_.parameters(), torch::optim::AdamOptions(lr)),
      opt_strat_(strat_.parameters(), torch::optim::AdamOptions(lr)),
      batch_size_(batch_size),
      train_interval_(train_interval)
{}

void Trainer::train_step(MLP& net, torch::optim::Adam& opt,
                         ReservoirBuffer<BufferEntry>& buffer,
                         const std::string& mode)
{
    if (buffer.size() < batch_size_) return;  // not enough data yet

    // Sample batch_size random indices — O(batch) not O(N)
    const auto& data = buffer.data();
    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::vector<size_t> indices(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) indices[i] = dist(rng);

    // Stack features, targets, weights into tensors
    auto feat_t   = torch::zeros({(int64_t)batch_size_, FEATURE_DIM});
    auto target_t = torch::zeros({(int64_t)batch_size_, NUM_ACTIONS});
    auto weight_t = torch::zeros({(int64_t)batch_size_});

    auto fa = feat_t.accessor<float, 2>();
    auto ta = target_t.accessor<float, 2>();
    auto wa = weight_t.accessor<float, 1>();

    for (size_t i = 0; i < batch_size_; ++i) {
        const auto& e = data[indices[i]];
        for (int j = 0; j < FEATURE_DIM; ++j)   fa[i][j] = e.features[j];
        for (int j = 0; j < NUM_ACTIONS; ++j)   ta[i][j] = e.targets[j];
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
    g_interrupted = false;
    auto prev_handler = std::signal(SIGINT, sigint_handler);

    using namespace indicators;
    ProgressBar bar{
        option::BarWidth{40},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::ForegroundColor{Color::cyan},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::MaxProgress{iterations},
    };

    for (int t = 1; t <= iterations; ++t) {
        if (g_interrupted) {
            std::cerr << "\nInterrupted at iteration " << t << "\n";
            break;
        }

        // Player 0 traversal
        {
            auto s = deal_heads_up();
            external_sample(s, 0, adv0_, adv1_, strat_, mv0_, mpi_, t);
        }

        // Player 1 traversal
        {
            auto s = deal_heads_up();
            external_sample(s, 1, adv1_, adv0_, strat_, mv1_, mpi_, t);
        }

        if (t % train_interval_ == 0) {
            train_step(adv0_, opt_adv0_, mv0_, "advantage");
            train_step(adv1_, opt_adv1_, mv1_, "advantage");
            train_step(strat_, opt_strat_, mpi_, "strategy");
        }

        bar.set_option(option::PostfixText{
            "iter " + std::to_string(t) +
            "  mv0=" + std::to_string(mv0_.size()) +
            " mv1=" + std::to_string(mv1_.size()) +
            " mpi=" + std::to_string(mpi_.size())
        });
        bar.tick();
    }

    std::signal(SIGINT, prev_handler);
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
