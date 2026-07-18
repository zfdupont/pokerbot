#include "game/card.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

// Encode 5-card hand rank as uint32_t — lower = better hand.
// Category (bits 20+): 1=str_flush, 2=quads, 3=full_house,
//   4=flush, 5=straight, 6=trips, 7=two_pair, 8=pair, 9=high_card
uint32_t evaluate_5card(std::array<Card, 5> cards) {
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

uint32_t evaluate_7card(std::array<Card, 7> cards) {
    uint32_t best = std::numeric_limits<uint32_t>::max();
    // All C(7,5) = 21 combinations
    for (int i = 0; i < 7; ++i)
        for (int j = i+1; j < 7; ++j) {
            std::array<Card, 5> hand;
            int k = 0;
            for (int x = 0; x < 7; ++x)
                if (x != i && x != j) hand[k++] = cards[x];
            best = std::min(best, evaluate_5card(hand));
        }
    return best;
}
