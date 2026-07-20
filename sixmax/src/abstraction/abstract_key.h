#pragma once
#include <array>
#include <cstdint>

namespace sixmax {

// Pot-size bucket used by the abstraction infoset key. Thresholds are part of
// the artifact contract; changing them invalidates checkpoints.
int pot_bucket(double pot_bb);

// The single source of truth for the abstraction infoset-key bit layout.
// Both EngineGameState::abstract_key (trainer) and the deployment bridge
// (Python, via bindings) route through this so they can never diverge.
//   card   bits 0-7   street bits 8-9   raises[st] bits 10-17 (2 bits each)
//   pot    bits 18-19 live   bits 20-22 after      bits 23-25
uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after);

}  // namespace sixmax
