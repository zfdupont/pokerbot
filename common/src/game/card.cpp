#include "game/card.h"

// Encoded hand rank as uint32_t — lower = better hand.
// Category (bits 20+): 1=str_flush, 2=quads, 3=full_house,
//   4=flush, 5=straight, 6=trips, 7=two_pair, 8=pair, 9=high_card
// Kickers follow in 4-bit fields from bit 16 down, each stored as 12 - rank:
// Ace(12)→0 (lowest bits = best within category), 2(0)→12. Unused fields
// are 0. Scores are bit-identical to the original sort-and-count 5-card
// evaluator taken as the minimum over all 21 five-card subsets.
namespace {

struct Packed {
    uint32_t v;
    int shift;
};

Packed start(uint32_t cat) { return {cat << 20, 16}; }

void push(Packed& p, int rank) {
    p.v |= (uint32_t)(12 - rank) << p.shift;
    p.shift -= 4;
}

int top_rank(uint32_t mask) { return 31 - __builtin_clz(mask); }

// Pushes the `n` highest ranks of `mask` onto `p`.
void push_top(Packed& p, uint32_t mask, int n) {
    for (int i = 0; i < n; ++i) {
        const int r = top_rank(mask);
        push(p, r);
        mask &= ~(1u << r);
    }
}

// Highest straight in a 13-bit rank mask (the wheel A-2-3-4-5 is 3), or -1.
int straight_high(uint32_t mask) {
    for (int high = 12; high >= 4; --high)
        if (((mask >> (high - 4)) & 0x1Fu) == 0x1Fu) return high;
    if ((mask & 0x100Fu) == 0x100Fu) return 3;
    return -1;
}

// Best five-card hand among `n` (5..7) cards, in one pass over rank counts
// and per-suit rank masks: no allocation, no subset enumeration.
uint32_t evaluate_cards(const Card* cards, int n) {
    int count[13] = {};
    uint32_t suit_mask[4] = {};
    for (int i = 0; i < n; ++i) {
        ++count[card_rank(cards[i])];
        suit_mask[card_suit(cards[i])] |= 1u << card_rank(cards[i]);
    }
    const uint32_t ranks = suit_mask[0] | suit_mask[1] | suit_mask[2] |
                           suit_mask[3];

    int flush_suit = -1;
    for (int s = 0; s < 4; ++s)
        if (__builtin_popcount(suit_mask[s]) >= 5) flush_suit = s;
    if (flush_suit >= 0) {
        const int high = straight_high(suit_mask[flush_suit]);
        if (high >= 0) {
            Packed p = start(1);
            push(p, high);
            return p.v;
        }
    }

    // Ranks grouped by multiplicity, highest first.
    int quads = -1, trips[2], pairs[3];
    int num_trips = 0, num_pairs = 0;
    for (int r = 12; r >= 0; --r) {
        if (count[r] == 4) quads = r;
        else if (count[r] == 3) trips[num_trips++] = r;
        else if (count[r] == 2) pairs[num_pairs++] = r;
    }

    if (quads >= 0) {
        Packed p = start(2);
        push(p, quads);
        push(p, top_rank(ranks & ~(1u << quads)));
        return p.v;
    }
    if (num_trips >= 1 && (num_trips >= 2 || num_pairs >= 1)) {
        // The pair is the better of a second trips and the top pair.
        int pair = num_pairs > 0 ? pairs[0] : -1;
        if (num_trips >= 2 && trips[1] > pair) pair = trips[1];
        Packed p = start(3);
        push(p, trips[0]);
        push(p, pair);
        return p.v;
    }
    if (flush_suit >= 0) {
        Packed p = start(4);
        push_top(p, suit_mask[flush_suit], 5);
        return p.v;
    }
    const int high = straight_high(ranks);
    if (high >= 0) {
        Packed p = start(5);
        push(p, high);
        return p.v;
    }
    if (num_trips == 1) {
        Packed p = start(6);
        push(p, trips[0]);
        push_top(p, ranks & ~(1u << trips[0]), 2);
        return p.v;
    }
    if (num_pairs >= 2) {
        Packed p = start(7);
        push(p, pairs[0]);
        push(p, pairs[1]);
        push_top(p, ranks & ~((1u << pairs[0]) | (1u << pairs[1])), 1);
        return p.v;
    }
    if (num_pairs == 1) {
        Packed p = start(8);
        push(p, pairs[0]);
        push_top(p, ranks & ~(1u << pairs[0]), 3);
        return p.v;
    }
    Packed p = start(9);
    push_top(p, ranks, 5);
    return p.v;
}

}  // namespace

uint32_t evaluate_5card(std::array<Card, 5> cards) {
    return evaluate_cards(cards.data(), 5);
}

uint32_t evaluate_7card(std::array<Card, 7> cards) {
    return evaluate_cards(cards.data(), 7);
}
