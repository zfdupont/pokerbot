#include <gtest/gtest.h>
#include "net/features.h"
#include "game/abstract_state.h"
#include <torch/torch.h>

TEST(Features, ShapeIs134) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    EXPECT_EQ(t.size(0), 134);
    EXPECT_EQ(t.dtype(), torch::kFloat32);
}

TEST(Features, StreetOneHotPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // street=0: index 119 = 1.0, 120–122 = 0.0
    EXPECT_FLOAT_EQ(acc[119], 1.0f);
    EXPECT_FLOAT_EQ(acc[120], 0.0f);
}

TEST(Features, CardRankOneHot) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // hole card 0: exactly one of indices 0–12 should be 1.0
    float rank_sum = 0.0f;
    for (int i = 0; i < 13; ++i) rank_sum += acc[i];
    EXPECT_FLOAT_EQ(rank_sum, 1.0f);
}

TEST(Features, PositionEncoding) {
    auto state = deal_heads_up();
    auto t0 = encode_features(state, 0);
    auto t1 = encode_features(state, 1);
    auto acc0 = t0.accessor<float, 1>();
    auto acc1 = t1.accessor<float, 1>();
    EXPECT_FLOAT_EQ(acc0[133], 0.0f);
    EXPECT_FLOAT_EQ(acc1[133], 1.0f);
}

TEST(Features, BoardPaddedAtPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // Board cards [34–118] should all be zero at preflop
    for (int i = 34; i < 119; ++i)
        EXPECT_FLOAT_EQ(acc[i], 0.0f) << "index " << i;
}
