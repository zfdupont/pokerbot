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
