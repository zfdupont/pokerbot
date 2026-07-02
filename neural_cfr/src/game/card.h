#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

// card = rank_index * 4 + suit_index
// rank_index: 0=2, 1=3, ..., 12=A
// suit_index: 0=clubs, 1=diamonds, 2=hearts, 3=spades
using Card = int;

constexpr int NUM_CARDS = 52;

inline int card_rank(Card c) { return c / 4; }   // 0–12
inline int card_suit(Card c) { return c % 4; }   // 0–3

inline std::vector<Card> make_deck() {
    std::vector<Card> deck(NUM_CARDS);
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
    return deck;
}

inline void shuffle_deck(std::vector<Card>& deck) {
    static std::mt19937 rng{std::random_device{}()};
    std::shuffle(deck.begin(), deck.end(), rng);
}

#include <limits>

// 5-card hand rank: returns encoded uint32_t — lower = better hand.
// category (bits 20+): 1=straight_flush, 2=quads, 3=full_house,
//   4=flush, 5=straight, 6=trips, 7=two_pair, 8=pair, 9=high_card
// Internal helper — use evaluate_7card for 7-card hands.
uint32_t evaluate_5card(std::array<Card, 5> cards);

// Returns best 5-card rank from 7 cards (lower = better).
uint32_t evaluate_7card(std::array<Card, 7> cards);
