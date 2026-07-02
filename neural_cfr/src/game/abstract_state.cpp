#include "game/abstract_state.h"
#include "game/card.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>

static constexpr float STARTING_STACK = 100.0f;

// ---------- legal_abstract_actions ----------
// Port of cfr/abstraction.py:legal_abstract_actions()
std::vector<std::string> legal_abstract_actions(
    float to_call, float pot, float stack,
    float current_bet, float player_bet)
{
    std::vector<std::string> actions;
    if (to_call > 0.0f) {
        actions.push_back("fold");
        if (stack >= to_call) actions.push_back("call");
        float eff_pot = pot + to_call * 2.0f;
        for (float size : {0.5f, 1.0f}) {
            if (stack > to_call + size * eff_pot)
                actions.push_back(size == 0.5f ? "b0.5" : "b1.0");
        }
        if (stack >= to_call) actions.push_back("allin");
    } else {
        actions.push_back("check");
        for (float size : {0.5f, 1.0f}) {
            if (stack > size * pot)
                actions.push_back(size == 0.5f ? "b0.5" : "b1.0");
        }
        if (stack > 0.0f) actions.push_back("allin");
    }
    return actions;
}

// ---------- AbstractState methods ----------

bool AbstractState::is_terminal() const {
    if (folded[0] || folded[1]) return true;
    if (to_act.empty() && street >= 3) return true;
    return false;
}

int AbstractState::acting_player() const { return to_act[0]; }

float AbstractState::payoff(int player) const {
    float invest = STARTING_STACK - stacks[player];
    if (folded[0]) return player == 1 ? (pot - invest) : -invest;
    if (folded[1]) return player == 0 ? (pot - invest) : -invest;
    // Showdown
    std::array<Card, 7> p0h = {
        hole_cards[0][0], hole_cards[0][1],
        board[0], board[1], board[2], board[3], board[4]
    };
    std::array<Card, 7> p1h = {
        hole_cards[1][0], hole_cards[1][1],
        board[0], board[1], board[2], board[3], board[4]
    };
    uint32_t r0 = evaluate_7card(p0h);
    uint32_t r1 = evaluate_7card(p1h);
    if (r0 < r1)       return player == 0 ? (pot - invest) : -invest;
    if (r1 < r0)       return player == 1 ? (pot - invest) : -invest;
    return pot / 2.0f - invest;  // split
}

std::vector<std::string> AbstractState::legal_actions() const {
    int p = acting_player();
    float to_call = current_bet - player_bets[p];
    return legal_abstract_actions(to_call, pot, stacks[p], current_bet, player_bets[p]);
}

AbstractState AbstractState::apply_action(const std::string& action) const {
    AbstractState next = *this;
    int p = acting_player();

    if (action == "b0.5" || action == "b1.0" || action == "allin") {
        next.betting_history[street] = std::min(next.betting_history[street] + 1, 2);
    }

    if (action == "fold") {
        next.folded[p] = true;
        next.to_act.clear();
    } else if (action == "check") {
        next.to_act.erase(next.to_act.begin());
    } else if (action == "call") {
        float to_call = std::min(current_bet - player_bets[p], stacks[p]);
        next.stacks[p] -= to_call;
        next.player_bets[p] += to_call;
        next.pot += to_call;
        next.to_act.erase(next.to_act.begin());
    } else if (action == "b0.5" || action == "b1.0" || action == "allin") {
        float to_call = current_bet - player_bets[p];
        float additional;
        if (action == "allin") {
            additional = stacks[p];
        } else {
            float size = (action == "b0.5") ? 0.5f : 1.0f;
            float eff_pot = pot + to_call * 2.0f;
            additional = std::min(to_call + size * eff_pot, stacks[p]);
        }
        next.stacks[p] -= additional;
        next.player_bets[p] += additional;
        next.pot += additional;
        next.current_bet = std::max(next.current_bet, next.player_bets[p]);
        // Other non-folded players with chips still to act
        next.to_act.clear();
        for (int i = 0; i < 2; ++i)
            if (!next.folded[i] && next.stacks[i] > 0.0f && i != p)
                next.to_act.push_back(i);
    } else {
        throw std::invalid_argument("Unknown action: " + action);
    }
    return next;
}

AbstractState AbstractState::advance_street() const {
    assert(to_act.empty() && !std::all_of(folded.begin(), folded.end(), [](bool b){ return b; }));
    AbstractState next = *this;
    next.street = street + 1;
    int cards_to_deal = (next.street == 1) ? 3 : 1;
    for (int i = 0; i < cards_to_deal; ++i) {
        next.board.push_back(next.deck.back());
        next.deck.pop_back();
    }
    next.current_bet = 0.0f;
    next.player_bets = {0.0f, 0.0f};
    // Postflop: p1 (BB/OOP) acts first
    next.to_act.clear();
    for (int i : {1, 0})
        if (!next.folded[i] && next.stacks[i] > 0.0f)
            next.to_act.push_back(i);
    return next;
}

// ---------- deal_heads_up ----------
AbstractState deal_heads_up(float starting_stack, float big_blind) {
    auto deck = make_deck();
    shuffle_deck(deck);

    AbstractState s;
    s.hole_cards[0] = {deck[0], deck[1]};
    s.hole_cards[1] = {deck[2], deck[3]};
    // Pre-deal 5 runout cards (indices 4–8), stored reversed for pop_back
    s.deck = std::vector<Card>(deck.begin() + 4, deck.begin() + 9);
    std::reverse(s.deck.begin(), s.deck.end());

    float sb = std::min(big_blind * 0.5f, starting_stack);
    float bb = std::min(big_blind, starting_stack);

    s.board = {};
    s.street = 0;
    s.pot = sb + bb;
    s.stacks = {starting_stack - sb, starting_stack - bb};
    s.current_bet = bb;
    s.player_bets = {sb, bb};
    s.to_act = {0, 1};  // SB acts first preflop
    s.betting_history = {0, 0, 0, 0};
    s.folded = {false, false};
    return s;
}
