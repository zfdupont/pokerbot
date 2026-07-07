#include "net/inference.h"
#include <algorithm>

AbstractState make_inference_state(
    const std::vector<int>& hole_cards,
    const std::vector<int>& board_cards,
    int street, float pot, float stack, float to_call,
    const std::vector<int>& raises_per_street,
    int position,
    float my_street_bet,
    float opp_street_bet)
{
    AbstractState s{};
    s.hole_cards[position][0] = hole_cards[0];
    s.hole_cards[position][1] = hole_cards[1];
    s.board = std::vector<Card>(board_cards.begin(), board_cards.end());
    s.street = street;
    s.pot = pot;
    s.stacks[position] = stack;
    s.stacks[1 - position] = stack;  // opponent stack unknown; not a feature

    if (my_street_bet >= 0.0f && opp_street_bet >= 0.0f) {
        s.player_bets[position]     = my_street_bet;
        s.player_bets[1 - position] = opp_street_bet;
        // Derive current_bet from street bets — the more precise signal —
        // rather than trusting the redundant to_call argument.
        s.current_bet = std::max(my_street_bet, opp_street_bet);
    } else {
        s.player_bets[position]     = 0.0f;
        s.player_bets[1 - position] = to_call;
        s.current_bet = to_call;
    }

    s.betting_history = {0, 0, 0, 0};
    for (int i = 0; i < 4 && i < (int)raises_per_street.size(); ++i)
        s.betting_history[i] = std::clamp(raises_per_street[i], 0, 2);
    s.folded = {false, false};
    s.to_act = {position};
    return s;
}
