#include "dream/features.h"
#include <algorithm>

namespace sixmax {

namespace {

void encode_card(float* out, int card_code) {
    // card_code = (rank-2)*4 + suit; rank 0-12, suit 0-3
    for (int i = 0; i < 17; ++i) out[i] = 0.0f;
    out[card_code / 4] = 1.0f;      // rank one-hot (0-12)
    out[13 + card_code % 4] = 1.0f; // suit one-hot (0-3)
}

void encode_card_or_pad(float* out, int card_code) {
    if (card_code < 0) { for (int i = 0; i < 17; ++i) out[i] = 0.0f; }
    else encode_card(out, card_code);
}

}  // namespace

torch::Tensor encode_state(const EngineGameState& state) {
    auto t = torch::zeros({FEATURE_DIM}, torch::kFloat32);
    float* d = t.data_ptr<float>();

    int p = state.current_player();

    // 0–33: hole cards (2 × 17)
    auto hc = state.hole_cards(p);
    encode_card(d + 0,  hc[0]);
    encode_card(d + 17, hc[1]);

    // 34–118: board (5 × 17), zero-pad missing cards
    auto board = state.board();
    board.resize(5, -1);  // pad to 5
    for (int i = 0; i < 5; ++i)
        encode_card_or_pad(d + 34 + i * 17, board[i]);

    // 119–122: street one-hot
    d[119 + (int)state.street()] = 1.0f;

    // 123: pot
    d[123] = (float)(state.pot() / CHIP_NORM);

    // 124–141: per-seat stacks, street bets, live mask
    int n = state.num_players();
    int n_active = 0;
    for (int s = 0; s < 6; ++s) {
        if (s < n) {
            const auto& ps = state.player_state(s);
            bool live = !ps.folded && !ps.all_in;
            d[124 + s] = (float)(ps.stack / CHIP_NORM);
            d[130 + s] = (float)(ps.street_bet / CHIP_NORM);
            d[136 + s] = live ? 1.0f : 0.0f;
            if (!ps.folded) ++n_active;
        }
        // seats >= n stay 0
    }

    // 142–147: acting player one-hot
    if (p < 6) d[142 + p] = 1.0f;

    // 148: n_active / 6
    d[148] = (float)n_active / 6.0f;

    // 149: to_call
    d[149] = (float)(state.to_call() / CHIP_NORM);

    // 150–153: raises_per_street clipped at 5 / RAISE_NORM
    auto raises = state.raises_per_street();
    for (int i = 0; i < 4; ++i)
        d[150 + i] = std::min((int)raises[i], 5) / RAISE_NORM;

    return t;
}

}  // namespace sixmax
