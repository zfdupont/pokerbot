#pragma once
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include "card.h"

struct AbstractState {
    std::array<std::array<Card, 2>, 2> hole_cards;  // [player][card]
    std::vector<Card> board;                          // 0–5 cards
    std::vector<Card> deck;                           // remaining runout cards
    int street;                                       // 0=preflop … 3=river
    float pot;
    std::array<float, 2> stacks;
    float current_bet;
    std::array<float, 2> player_bets;               // bets placed this street
    std::vector<int> to_act;                          // action queue
    std::array<int, 4> betting_history;              // raise counts per street, capped at 2
    std::array<bool, 2> folded;

    bool is_terminal() const;
    int acting_player() const;
    float payoff(int player) const;
    std::vector<std::string> legal_actions() const;
    AbstractState apply_action(const std::string& action) const;
    AbstractState advance_street() const;
};

std::vector<std::string> legal_abstract_actions(
    float to_call, float pot, float stack,
    float current_bet, float player_bet);

AbstractState deal_heads_up(
    float starting_stack = 100.0f,
    float big_blind = 1.0f);
