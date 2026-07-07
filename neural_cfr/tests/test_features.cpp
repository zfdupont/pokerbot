#include <gtest/gtest.h>
#include "net/features.h"
#include "net/inference.h"
#include "game/abstract_state.h"
#include <torch/torch.h>

TEST(Features, ShapeIs134) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    EXPECT_EQ(t.size(0), 134);
    EXPECT_EQ(t.dtype(), torch::kFloat32);
}

TEST(Features, StreetOneHotPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // street=0: index 119 = 1.0, 120–122 = 0.0
    EXPECT_FLOAT_EQ(acc[119], 1.0f);
    EXPECT_FLOAT_EQ(acc[120], 0.0f);
}

TEST(Features, CardRankOneHot) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // hole card 0: exactly one of indices 0–12 should be 1.0
    float rank_sum = 0.0f;
    for (int i = 0; i < 13; ++i) rank_sum += acc[i];
    EXPECT_FLOAT_EQ(rank_sum, 1.0f);
}

TEST(Features, PositionEncoding) {
    auto state = deal_heads_up();
    auto t0 = encode_features(state, 0);
    auto t1 = encode_features(state, 1);
    auto acc0 = t0.accessor<float, 1>();
    auto acc1 = t1.accessor<float, 1>();
    EXPECT_FLOAT_EQ(acc0[133], 0.0f);
    EXPECT_FLOAT_EQ(acc1[133], 1.0f);
}

TEST(Features, BoardPaddedAtPreflop) {
    auto state = deal_heads_up();
    auto t = encode_features(state, 0);
    auto acc = t.accessor<float, 1>();
    // Board cards [34–118] should all be zero at preflop
    for (int i = 34; i < 119; ++i)
        EXPECT_FLOAT_EQ(acc[i], 0.0f) << "index " << i;
}

// --- Train/inference feature parity -------------------------------------
// The features encoded during traversal and the features built by
// make_inference_state for the same decision point must be identical.

static std::vector<int> _history_vec(const AbstractState& s) {
    return {s.betting_history[0], s.betting_history[1],
            s.betting_history[2], s.betting_history[3]};
}

TEST(InferenceParity, PreflopFacingRaise) {
    auto s = deal_heads_up();      // SB=p0 acts first
    s = s.apply_action("b1.0");    // SB raises; BB (p1) faces it
    int p = s.acting_player();
    ASSERT_EQ(p, 1);
    float to_call = s.current_bet - s.player_bets[p];

    auto inf = make_inference_state(
        {s.hole_cards[p][0], s.hole_cards[p][1]}, /*board=*/{},
        s.street, s.pot, s.stacks[p], to_call, _history_vec(s), p,
        /*my_street_bet=*/s.player_bets[p],
        /*opp_street_bet=*/s.player_bets[1 - p]);

    EXPECT_TRUE(torch::allclose(encode_features(s, p), encode_features(inf, p)));
    EXPECT_EQ(s.legal_actions(), inf.legal_actions());
}

TEST(InferenceParity, PostflopBetAfterRaisedPreflop) {
    auto s = deal_heads_up();
    s = s.apply_action("b0.5");    // SB raises
    s = s.apply_action("call");    // BB calls → flop
    s = s.advance_street();
    s = s.apply_action("b0.5");    // BB leads flop; SB (p0) faces the bet
    int p = s.acting_player();
    ASSERT_EQ(p, 0);
    float to_call = s.current_bet - s.player_bets[p];

    std::vector<int> board(s.board.begin(), s.board.end());
    auto inf = make_inference_state(
        {s.hole_cards[p][0], s.hole_cards[p][1]}, board,
        s.street, s.pot, s.stacks[p], to_call, _history_vec(s), p,
        s.player_bets[p], s.player_bets[1 - p]);

    EXPECT_TRUE(torch::allclose(encode_features(s, p), encode_features(inf, p)));
    EXPECT_EQ(s.legal_actions(), inf.legal_actions());
}

TEST(InferenceParity, FallbackWithoutStreetBetsEncodesToCall) {
    // Sentinel mode: opponent street bet approximated by to_call — strictly
    // better than the old all-zeros encoding.
    auto inf = make_inference_state({51, 50}, {}, /*street=*/0, /*pot=*/4.0f,
                                    /*stack=*/97.0f, /*to_call=*/2.0f,
                                    {1, 0, 0, 0}, /*position=*/1);
    auto t = encode_features(inf, 1);
    auto d = t.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[129], 0.0f);            // my street bet / 200
    EXPECT_FLOAT_EQ(d[130], 2.0f / 200.0f);   // opp street bet / 200
}

TEST(InferenceParity, RaisesPerStreetClamped) {
    auto inf = make_inference_state({51, 50}, {}, 0, 4.0f, 97.0f, 2.0f,
                                    {7, -3, 0, 0}, 1);
    auto t = encode_features(inf, 1);  // keep tensor alive — data_ptr on a
    auto d = t.data_ptr<float>();      // temporary would dangle
    EXPECT_FLOAT_EQ(d[125], 2.0f / 2.0f);  // clamped 7 → 2, / RAISE_NORM
    EXPECT_FLOAT_EQ(d[126], 0.0f);         // clamped -3 → 0
}
