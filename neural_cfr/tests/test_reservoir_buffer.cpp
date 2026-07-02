#include <gtest/gtest.h>
#include "cfr/reservoir_buffer.h"
#include <vector>
#include <numeric>
#include <cmath>

TEST(ReservoirBuffer, SizeStaysBounded) {
    ReservoirBuffer<int> buf(100);
    for (int i = 0; i < 1000; ++i) buf.add(i);
    EXPECT_EQ(buf.size(), 100u);
}

TEST(ReservoirBuffer, AllItemsBeforeCapacity) {
    ReservoirBuffer<int> buf(50);
    for (int i = 0; i < 50; ++i) buf.add(i);
    EXPECT_EQ(buf.size(), 50u);
    for (int v : buf.data()) EXPECT_GE(v, 0);
}

TEST(ReservoirBuffer, ApproximatelyUniformSampling) {
    // Insert 10x capacity. Each of the 100 values should appear ~equally often
    // in the buffer across 2000 repeated trials (loose chi-squared check).
    // Expected: each value appears in reservoir with prob max_size/total = 1/10,
    // so over 2000 trials each value appears ~200 times.
    // NOTE: slot_counts is indexed by value (size=total), not by v%max_size.
    // The brief had a typo using v%max_size which made slot_counts[j] sum 10
    // values each appearing ~200 times → ~2000, contradicting the expected=200
    // threshold. Corrected here so the test is both passable and meaningful.
    const int max_size = 10;
    const int total = 10 * max_size;
    std::vector<int> slot_counts(total, 0);
    const int trials = 2000;
    for (int t = 0; t < trials; ++t) {
        ReservoirBuffer<int> buf(max_size);
        for (int i = 0; i < total; ++i) buf.add(i);
        for (int v : buf.data()) slot_counts[v]++;
    }
    // Each value should appear in ~trials*max_size/total = 200 of trials
    // Just check no single value dominates (> 3× expected)
    float expected = (float)(trials * max_size) / total;
    for (int c : slot_counts)
        EXPECT_LT((float)c, expected * 3.0f);
}
