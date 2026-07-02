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
