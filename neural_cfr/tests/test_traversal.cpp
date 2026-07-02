#include <gtest/gtest.h>
#include <cmath>
#include "cfr/traversal.h"
#include "cfr/reservoir_buffer.h"
#include "net/mlp.h"
#include "game/abstract_state.h"

TEST(Traversal, BuffersPopulatedAfterTraversal) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(10000), mv1(10000), mpi(10000);

    // Run 20 traversals for each player
    for (int t = 1; t <= 20; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t);
        s = deal_heads_up();
        external_sample(s, 1, adv1, adv0, strat, mv1, mpi, t);
    }

    EXPECT_GT(mv0.size(), 0u);
    EXPECT_GT(mv1.size(), 0u);
    EXPECT_GT(mpi.size(), 0u);
}

TEST(Traversal, ReturnedEVIsFinite) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    auto s = deal_heads_up();
    float ev = external_sample(s, 0, adv0, adv1, strat, mv0, mpi, 1);
    EXPECT_TRUE(std::isfinite(ev));
}

TEST(Traversal, FeatureDimInBuffer) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    for (int t = 1; t <= 5; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t);
    }
    ASSERT_GT(mv0.size(), 0u);
    EXPECT_EQ(mv0.data()[0].features.size(), 134u);
    EXPECT_EQ(mv0.data()[0].targets.size(), 6u);
}
