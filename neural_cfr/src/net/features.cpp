#include "net/features.h"
#include "game/card.h"
#include <cstring>

static void encode_card(float* buf, Card c) {
    // 17 floats: 13 rank one-hot + 4 suit one-hot
    std::memset(buf, 0, 17 * sizeof(float));
    if (c >= 0) {
        buf[card_rank(c)] = 1.0f;
        buf[13 + card_suit(c)] = 1.0f;
    }
    // if c < 0: padding card — stays zero
}

torch::Tensor encode_features(const AbstractState& state, int player) {
    auto t = torch::zeros({FEATURE_DIM}, torch::kFloat32);
    float* d = t.data_ptr<float>();

    // [0–33] hole cards (player's own cards first)
    encode_card(d + 0,  state.hole_cards[player][0]);
    encode_card(d + 17, state.hole_cards[player][1]);

    // [34–118] board cards, zero-padded to 5
    for (int i = 0; i < 5; ++i) {
        Card c = (i < (int)state.board.size()) ? state.board[i] : -1;
        encode_card(d + 34 + i * 17, c);
    }

    // [119–122] street one-hot
    d[119 + state.street] = 1.0f;

    // [123] pot normalized, [124] stack normalized
    d[123] = state.pot / 200.0f;
    d[124] = state.stacks[player] / 200.0f;

    // [125–128] betting history (raise counts per street, normalized by 2)
    for (int i = 0; i < 4; ++i)
        d[125 + i] = state.betting_history[i] / 2.0f;

    // [129–132] player bets per street (current street's bet, normalized)
    d[129] = state.player_bets[player] / 200.0f;
    d[130] = state.player_bets[1-player] / 200.0f;
    d[131] = 0.0f;  // reserved
    d[132] = 0.0f;

    // [133] position
    d[133] = static_cast<float>(player);

    return t;
}
