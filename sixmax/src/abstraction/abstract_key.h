#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace sixmax {

// Pot-size bucket used by the legacy abstraction infoset key. Thresholds are
// part of the artifact contract; changing them invalidates checkpoints.
int pot_bucket(double pot_bb);

// Config-defined pot layout. BB mode edges are upper bounds on the pot in BB;
// SPR mode edges are upper bounds on effective stack behind / pot at the
// decision. The legacy layout (BB, 7/15/40) keeps the historical key bits and
// hash; every other layout moves the pot index to bits 26-29.
enum class PotMode : int32_t { BB = 0, SPR = 1 };

struct PotLayout {
    PotMode mode = PotMode::BB;
    std::vector<double> edges = {7.0, 15.0, 40.0};
    bool is_legacy() const;   // mode==BB && edges=={7,15,40}
    int num_buckets() const;  // edges.size()+1
};

// Throws std::invalid_argument containing "pot_edges" unless the edges are
// non-empty, <= 15 entries, strictly increasing, and > 0.
void validate_pot_layout(const PotLayout& layout);

int pot_bucket(const PotLayout& layout, double pot_bb, double eff_stack_bb);

// Opt-in infoset-key field: the size class of the last bet or raise on each
// street (bits 30-41, 3 bits per street). Off keeps the historical key.
enum class BetHistory : int32_t { Off = 0, LastRaise = 1 };

// The single source of truth for the abstraction infoset-key bit layout.
// Both EngineGameState::abstract_key (trainer) and the deployment bridge
// (Python, via bindings) route through this so they can never diverge.
//   card   bits 0-7   street bits 8-9   raises[st] bits 10-17 (2 bits each)
//   live   bits 20-22 after      bits 23-25
//   legacy layout: pot bits 18-19
//   other layouts: pot bits 26-29 (bits 18-19 stay zero)
//   last_raise[st] bits 30-41 (3 each; zeros unless bet history is on)
uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           int pot_idx, bool legacy_layout, int live,
                           int after,
                           const std::array<uint8_t, 4>& last_raise = {});

// Legacy wrapper: pots 7/15/40 at bits 18-19.
uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after);

}  // namespace sixmax
