#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "cfr/traversal.h"
#include "cfr/reservoir_buffer.h"
#include "net/mlp.h"
#include "game/abstract_state.h"

TEST(Traversal, BuffersPopulatedAfterTraversal) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(10000), mv1(10000), mpi(10000);
    std::mt19937 rng{42};

    // Run 20 traversals for each player
    for (int t = 1; t <= 20; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t, rng);
        s = deal_heads_up();
        external_sample(s, 1, adv1, adv0, strat, mv1, mpi, t, rng);
    }

    EXPECT_GT(mv0.size(), 0u);
    EXPECT_GT(mv1.size(), 0u);
    EXPECT_GT(mpi.size(), 0u);
}

TEST(Traversal, ReturnedEVIsFinite) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    std::mt19937 rng{42};
    auto s = deal_heads_up();
    float ev = external_sample(s, 0, adv0, adv1, strat, mv0, mpi, 1, rng);
    EXPECT_TRUE(std::isfinite(ev));
}

TEST(Traversal, FeatureDimInBuffer) {
    MLP adv0, adv1, strat;
    ReservoirBuffer<BufferEntry> mv0(1000), mv1(1000), mpi(1000);
    std::mt19937 rng{42};
    for (int t = 1; t <= 5; ++t) {
        auto s = deal_heads_up();
        external_sample(s, 0, adv0, adv1, strat, mv0, mpi, t, rng);
    }
    ASSERT_GT(mv0.size(), 0u);
    EXPECT_EQ(mv0.data()[0].features.size(), 134u);
    EXPECT_EQ(mv0.data()[0].targets.size(), 6u);
}

TEST(RegretMatch, ProportionalWhenPositiveAdvantagesExist) {
    std::array<float, 6> adv{};
    adv[1] = 1.0f;   // check
    adv[3] = 3.0f;   // b0.5
    std::vector<int> legal{1, 3, 5};
    auto p = regret_match(adv, legal);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_NEAR(p[0], 0.25f, 1e-6);
    EXPECT_NEAR(p[1], 0.75f, 1e-6);
    EXPECT_NEAR(p[2], 0.0f,  1e-6);
}

TEST(RegretMatch, ArgmaxWhenAllNonPositive) {
    // Brown et al. 2019: when no advantage is positive, play the
    // highest-advantage action as a pure strategy — NOT uniform.
    std::array<float, 6> adv{};
    adv[0] = -0.5f;
    adv[2] = -0.1f;  // best of the legal set
    adv[5] = -2.0f;
    std::vector<int> legal{0, 2, 5};
    auto p = regret_match(adv, legal);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 1.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
}

TEST(RegretMatch, AllExactlyZeroPicksFirstArgmax) {
    // Ties broken by first index — deterministic, any pure choice is valid.
    std::array<float, 6> adv{};
    std::vector<int> legal{0, 1};
    auto p = regret_match(adv, legal);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
}
