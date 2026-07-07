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
