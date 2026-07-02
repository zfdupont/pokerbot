#include <gtest/gtest.h>
#include "game/abstract_state.h"
#include <algorithm>

TEST(AbstractState, DealHeadsUpInitialState) {
    auto state = deal_heads_up();
    EXPECT_EQ(state.street, 0);
    EXPECT_FLOAT_EQ(state.pot, 1.5f);           // SB(0.5) + BB(1.0)
    EXPECT_FLOAT_EQ(state.stacks[0], 99.5f);    // SB posted
    EXPECT_FLOAT_EQ(state.stacks[1], 99.0f);    // BB posted
    EXPECT_FLOAT_EQ(state.current_bet, 1.0f);
    EXPECT_EQ(state.to_act.size(), 2u);
    EXPECT_EQ(state.to_act[0], 0);              // SB acts first preflop
    EXPECT_FALSE(state.is_terminal());
}

TEST(AbstractState, FoldIsTerminal) {
    auto state = deal_heads_up();
    auto next = state.apply_action("fold");
    EXPECT_TRUE(next.is_terminal());
    // p0 folded: p1 wins pot minus p1's investment
    float p1_invest = 100.0f - next.stacks[1];
    EXPECT_FLOAT_EQ(next.payoff(1), next.pot - p1_invest);
    EXPECT_FLOAT_EQ(next.payoff(0), -(100.0f - next.stacks[0]));
}

TEST(AbstractState, CallAdvancesStreet) {
    auto state = deal_heads_up();
    // p0 calls, p1 checks → to_act empty → advance_street
    auto s1 = state.apply_action("call");
    EXPECT_EQ(s1.to_act.size(), 1u);  // p1 still to act
    auto s2 = s1.apply_action("check");
    EXPECT_EQ(s2.to_act.size(), 0u);  // street over
    auto s3 = s2.advance_street();
    EXPECT_EQ(s3.street, 1);           // flop
    EXPECT_EQ(s3.board.size(), 3u);
    EXPECT_EQ(s3.to_act[0], 1);        // BB acts first postflop
}

TEST(AbstractState, BettingHistoryTracksRaises) {
    auto state = deal_heads_up();
    auto s1 = state.apply_action("b0.5");
    EXPECT_EQ(s1.betting_history[0], 1);  // one raise preflop
    auto s2 = s1.apply_action("b1.0");
    EXPECT_EQ(s2.betting_history[0], 2);  // two raises (cap)
    auto s3 = s2.apply_action("b0.5");
    EXPECT_EQ(s3.betting_history[0], 2);  // still capped at 2
}

TEST(AbstractState, LegalActionsPreflop) {
    auto state = deal_heads_up();
    auto actions = state.legal_actions();
    // Facing BB=1.0, p0 can fold/call/raise/allin
    EXPECT_TRUE(std::find(actions.begin(), actions.end(), "fold") != actions.end());
    EXPECT_TRUE(std::find(actions.begin(), actions.end(), "call") != actions.end());
}
