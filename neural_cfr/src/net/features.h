#pragma once
#include <torch/torch.h>
#include "game/abstract_state.h"

constexpr int   FEATURE_DIM = 134;
constexpr int   NUM_ACTIONS = 6;
constexpr float CHIP_NORM   = 200.0f;  // normalizes pot, stack, and per-player bets
constexpr float RAISE_NORM  = 2.0f;    // normalizes raise counts per street (capped at 2)

// Encode game state from perspective of `player` into a [134] float tensor.
torch::Tensor encode_features(const AbstractState& state, int player);
