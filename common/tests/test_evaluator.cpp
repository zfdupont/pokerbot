// The direct evaluator must return the original evaluator's scores bit for
// bit: neural_cfr exposes raw scores and the 12 - rank kicker inversion is a
// project invariant. The original sort-and-count evaluator is kept here as
// the oracle. Exhaustive over every 5-card hand; random 7-card hands here
// (the one-off exhaustive 133,784,560-hand check passed when it landed).
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "game/card.h"

namespace {

uint32_t reference_5card(std::array<Card, 5> cards) {
    std::array<int, 5> ranks, suits;
    for (int i = 0; i < 5; ++i) {
        ranks[i] = card_rank(cards[i]);
        suits[i] = card_suit(cards[i]);
    }
    std::sort(ranks.begin(), ranks.end(), std::greater<int>());  // descending

    bool flush = (suits[0]==suits[1] && suits[1]==suits[2] &&
                  suits[2]==suits[3] && suits[3]==suits[4]);

    // Straight detection (including A-2-3-4-5 wheel)
    bool straight = false;
    int straight_high = ranks[0];
    if (ranks[0]-ranks[4] == 4 &&
        std::set<int>(ranks.begin(), ranks.end()).size() == 5) {
        straight = true;
    } else if (ranks[0]==12 && ranks[1]==3 && ranks[2]==2 &&
               ranks[3]==1 && ranks[4]==0) {
        straight = true; straight_high = 3; // wheel: A-2-3-4-5, high=5
    }

    // Count rank frequencies
    std::array<int, 13> freq{};
    for (int r : ranks) freq[r]++;
    std::vector<int> counts;
    std::vector<int> rank_by_count;
    for (int r = 12; r >= 0; --r) if (freq[r]) {
        counts.push_back(freq[r]);
        rank_by_count.push_back(r);
    }
    // Sort by count desc, then rank desc
    std::vector<int> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        return counts[a] != counts[b] ? counts[a] > counts[b] : rank_by_count[a] > rank_by_count[b];
    });

    // Invert rank kickers: Ace(12)→0 (lowest bits = best within category),
    // 2(0)→12 (highest bits = worst within category). Consistent with lower=better.
    auto cat_rank = [&](uint32_t cat, std::initializer_list<int> kickers) -> uint32_t {
        uint32_t v = cat << 20;
        int shift = 16;
        for (int k : kickers) { v |= ((12 - k) << shift); shift -= 4; }
        return v;
    };

    if (flush && straight) return cat_rank(1, {straight_high});
    if (counts[order[0]] == 4) return cat_rank(2, {rank_by_count[order[0]], rank_by_count[order[1]]});
    if (counts[order[0]] == 3 && counts[order[1]] == 2)
        return cat_rank(3, {rank_by_count[order[0]], rank_by_count[order[1]]});
    if (flush) return cat_rank(4, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
    if (straight) return cat_rank(5, {straight_high});
    if (counts[order[0]] == 3) return cat_rank(6, {rank_by_count[order[0]],
        rank_by_count[order[1]], rank_by_count[order[2]]});
    if (counts[order[0]] == 2 && counts[order[1]] == 2)
        return cat_rank(7, {rank_by_count[order[0]], rank_by_count[order[1]], rank_by_count[order[2]]});
    if (counts[order[0]] == 2) return cat_rank(8, {rank_by_count[order[0]],
        rank_by_count[order[1]], rank_by_count[order[2]], rank_by_count[order[3]]});
    return cat_rank(9, {ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]});
}

uint32_t reference_7card(std::array<Card, 7> cards) {
    uint32_t best = std::numeric_limits<uint32_t>::max();
    // All C(7,5) = 21 combinations
    for (int i = 0; i < 7; ++i)
        for (int j = i+1; j < 7; ++j) {
            std::array<Card, 5> hand;
            int k = 0;
            for (int x = 0; x < 7; ++x)
                if (x != i && x != j) hand[k++] = cards[x];
            best = std::min(best, reference_5card(hand));
        }
    return best;
}

}  // namespace

TEST(Evaluator, EveryFiveCardHandMatchesReference) {
    long hands = 0;
    for (int a = 0; a < 52; ++a)
        for (int b = a + 1; b < 52; ++b)
            for (int c = b + 1; c < 52; ++c)
                for (int d = c + 1; d < 52; ++d)
                    for (int e = d + 1; e < 52; ++e) {
                        const std::array<Card, 5> h{a, b, c, d, e};
                        ASSERT_EQ(evaluate_5card(h), reference_5card(h));
                        ++hands;
                    }
    EXPECT_EQ(hands, 2598960);
}

TEST(Evaluator, RandomSevenCardHandsMatchReference) {
    std::mt19937 rng(20261022);
    std::vector<Card> deck = make_deck();
    for (int i = 0; i < 300000; ++i) {
        // Partial shuffle: the first seven cards are a uniform 7-card hand.
        for (int k = 0; k < 7; ++k) {
            std::uniform_int_distribution<int> pick(k, NUM_CARDS - 1);
            std::swap(deck[k], deck[pick(rng)]);
        }
        std::array<Card, 7> h;
        std::copy(deck.begin(), deck.begin() + 7, h.begin());
        ASSERT_EQ(evaluate_7card(h), reference_7card(h));
    }
}

TEST(Evaluator, SevenCardEdgeCasesMatchReference) {
    auto c = [](int rank, int suit) { return (rank - 2) * 4 + suit; };
    const std::vector<std::array<Card, 7>> hands = {
        // Wheel straight flush beside a higher plain straight.
        {c(14, 0), c(2, 0), c(3, 0), c(4, 0), c(5, 0), c(6, 1), c(7, 2)},
        // Two trips: the lower trips plays as the pair.
        {c(9, 0), c(9, 1), c(9, 2), c(4, 0), c(4, 1), c(4, 2), c(13, 3)},
        // Three pairs: the third pair's rank can be the kicker.
        {c(10, 0), c(10, 1), c(8, 0), c(8, 1), c(6, 0), c(6, 1), c(2, 3)},
        // Quads with a pair: the kicker is the best remaining rank.
        {c(7, 0), c(7, 1), c(7, 2), c(7, 3), c(12, 0), c(12, 1), c(3, 2)},
        // Six-card flush with a straight in other suits.
        {c(14, 2), c(11, 2), c(9, 2), c(6, 2), c(4, 2), c(2, 2), c(10, 0)},
        // Board straight on the wheel plus a six.
        {c(14, 0), c(2, 1), c(3, 2), c(4, 3), c(5, 0), c(6, 1), c(13, 2)},
    };
    for (const auto& h : hands) EXPECT_EQ(evaluate_7card(h), reference_7card(h));
}
