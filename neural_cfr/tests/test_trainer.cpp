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

// --- Global iteration counter -------------------------------------------

TEST(Trainer, TotalIterationsAccumulatesAcrossRuns) {
    // Cheap continual config: no reinit, 1 SGD step, tiny batches.
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    EXPECT_EQ(t.total_iterations(), 0);
    t.run(50);
    EXPECT_EQ(t.total_iterations(), 50);
    t.run(70);   // second run() call must continue, not reset
    EXPECT_EQ(t.total_iterations(), 120);
}

TEST(Trainer, TotalIterationsRoundTripsThroughCheckpoint) {
    const std::string path = "/tmp/neural_cfr_test_meta_ckpt.pt";
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t.run(50);
    t.checkpoint(path);

    Trainer t2(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t2.load(path);
    EXPECT_EQ(t2.total_iterations(), 50);
    std::remove(path.c_str());
}

TEST(Trainer, LegacyCheckpointWithoutMetaLoadsAsZero) {
    // Craft a legacy-format checkpoint: only the three net sub-archives.
    const std::string path = "/tmp/neural_cfr_test_legacy_ckpt.pt";
    {
        MLP m0, m1, ms;
        torch::serialize::OutputArchive root, a0, a1, s;
        m0.save(a0); m1.save(a1); ms.save(s);
        root.write("adv0", a0);
        root.write("adv1", a1);
        root.write("strat", s);
        root.save_to(path);
    }
    Trainer t(1000, 64, 1e-3f, 50, 2, 0.06f, 1, false);
    t.run(50);                      // counter nonzero before load
    t.load(path);                   // must not throw
    EXPECT_EQ(t.total_iterations(), 0);  // legacy ⇒ counter resets to 0
    std::remove(path.c_str());
}
