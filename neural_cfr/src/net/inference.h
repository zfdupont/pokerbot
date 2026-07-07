#pragma once
#include <vector>
#include "game/abstract_state.h"

// Build a synthetic AbstractState for inference-time feature encoding and
// legal-action derivation, matching training-time feature semantics.
//
// pot MUST include all street bets (same as during traversal).
// my_street_bet / opp_street_bet: chips committed this street by the acting
// player / opponent. Pass -1 (either) if unknown: falls back to
// player_bets = {0, to_call}, which loses the acting player's own prior
// street commitment but preserves the bet being faced.
// raises_per_street values are clamped to [0, 2] (training caps them at 2).
AbstractState make_inference_state(
    const std::vector<int>& hole_cards,
    const std::vector<int>& board_cards,
    int street, float pot, float stack, float to_call,
    const std::vector<int>& raises_per_street,
    int position,
    float my_street_bet  = -1.0f,
    float opp_street_bet = -1.0f);
