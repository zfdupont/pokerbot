#pragma once
#include <torch/torch.h>
#include "blueprint/engine_game.h"

namespace sixmax {

inline constexpr int   FEATURE_DIM = 154;
inline constexpr float CHIP_NORM   = 100.0f;
inline constexpr float RAISE_NORM  = 5.0f;

// Encode the current EngineGameState into a 154-dim float tensor (CPU).
// Layout (see spec for full table):
//   0–33:   hole cards × 17 (13 rank + 4 suit one-hot)
//   34–118: board cards × 17, zero-padded
//   119–122: street one-hot
//   123:    pot / CHIP_NORM
//   124–129: stack[seat 0-5] / CHIP_NORM (0 if absent or all-in)
//   130–135: street_bet[seat 0-5] / CHIP_NORM
//   136–141: live_mask[seat 0-5] in {0,1}
//   142–147: acting player one-hot
//   148:    n_active / 6.0
//   149:    to_call / CHIP_NORM
//   150–153: min(raises_per_street[0-3], 5) / RAISE_NORM
torch::Tensor encode_state(const EngineGameState& state);

}  // namespace sixmax
