#pragma once
#include <torch/torch.h>
#include "game/abstract_state.h"

constexpr int FEATURE_DIM = 134;
constexpr int NUM_ACTIONS = 6;

// Encode game state from perspective of `player` into a [134] float tensor.
torch::Tensor encode_features(const AbstractState& state, int player);
