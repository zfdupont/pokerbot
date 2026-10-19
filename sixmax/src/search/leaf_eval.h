#pragma once
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>
#include "blueprint/checkpoint.h"
#include "blueprint/engine_game.h"
#include "engine/engine.h"

namespace sixmax {

struct LeafEvalConfig {
    int samples = 256;     // (opponent hand, runout) draws per leaf
    uint64_t seed = 1;
    int threads = 0;       // leaf-matrix worker threads; <=0 => all cores
};

// End-of-street leaf values. A board-complete (river) leaf is an EXACT showdown
// (safe_eval, no blueprint, NOT engine-terminal); earlier leaves roll the
// blueprint to showdown over sampled runouts. Replaced wholesale by the Phase-3
// value net, so keep the interface to one matrix call.
//
// `our_hands`/`opp_hands` are canonical 1326-space combo indices for the acting
// player and the other player respectively; the returned row-major
// [our_hand][opp_hand] matrix is the payoff to `actor`.
class LeafEval {
public:
    // Exact-only mode (board-complete leaves): no blueprint required.
    explicit LeafEval(LeafEvalConfig cfg) : bp_(nullptr), cfg_(cfg) {}
    bool has_blueprint() const { return bp_ != nullptr; }
    // Bound the (expensive) sampled continuation: matrix() stops early once the
    // deadline passes, leaving the remaining entries at 0. Set before a solve;
    // defaults to no deadline so direct/test use is unaffected.
    void set_deadline(std::chrono::steady_clock::time_point t) { deadline_ = t; }
    bool expired() const { return std::chrono::steady_clock::now() >= deadline_; }
    // Full mode: adds the sampled flop/turn continuation.
    LeafEval(const BlueprintStrategy& bp, LeafEvalConfig cfg)
        : bp_(&bp), cfg_(cfg) {}
    // Internal pointer form (Subgame owns the blueprint or passes null).
    LeafEval(const BlueprintStrategy* bp, LeafEvalConfig cfg)
        : bp_(bp), cfg_(cfg) {}

    std::vector<double> matrix(const EngineGameState& leaf, int actor,
                               const std::vector<int>& our_hands,
                               const std::vector<int>& opp_hands) const;

private:
    // Blueprint continuation for a flop/turn leaf, one sampled runout.
    double rollout(const EngineGameState& leaf, int actor, int our_combo,
                   int opp_combo, std::mt19937_64& rng) const;

    const BlueprintStrategy* bp_;   // null in exact-only mode
    LeafEvalConfig cfg_;
    std::chrono::steady_clock::time_point deadline_ =
        std::chrono::steady_clock::time_point::max();
};

}  // namespace sixmax
