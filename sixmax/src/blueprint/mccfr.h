#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include "blueprint/game.h"

namespace sixmax {

struct InfosetData {
    std::vector<double> regret;
    std::vector<double> strategy_sum;
};

// External-sampling MCCFR with linear weighting: regret and strategy-sum
// updates at iteration t are multiplied by t. Single-threaded; the
// multithreaded trainer with abstraction-based keys is Phase 1b.
class MCCFRTrainer {
public:
    MCCFRTrainer(Game& game, uint64_t seed) : game_(game), rng_(seed) {}
    void train(uint64_t iterations);  // one traversal per player per iteration
    uint64_t iterations() const { return iter_; }
    size_t num_infosets() const { return table_.size(); }
    // Normalized average strategy for a visited infoset; empty if unseen
    // or never accumulated. Illegal actions hold probability 0 because
    // strategy_sum only ever accumulates on masked-legal actions.
    std::vector<double> average_strategy(uint64_t key) const;

private:
    double traverse(GameState& s, int traverser);
    std::vector<double> matched_strategy(const InfosetData& d,
                                         const std::vector<uint8_t>& mask) const;
    Game& game_;
    std::unordered_map<uint64_t, InfosetData> table_;
    std::mt19937_64 rng_;
    uint64_t iter_ = 0;
    double weight_ = 0.0;
};

// Exact EV for player 0 under the trainer's average strategy, by
// enumerating all 6 Kuhn deals (uniform strategy where unaccumulated).
// Test helper for the closed-form value -1/18.
double kuhn_exact_value(const MCCFRTrainer& t);

}  // namespace sixmax
