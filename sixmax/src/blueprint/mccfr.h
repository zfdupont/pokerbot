#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>
#include "blueprint/game.h"

namespace sixmax {

// Compile-time upper bound on the action-vocabulary width. The blueprint
// action set is fold/check/call + a handful of bet sizes + all-in (<= ~10);
// 16 is a safe ceiling. Hot-path per-node buffers (sigma/utility) are fixed
// std::array<T, kMaxActions> stack storage keyed off this bound, so the
// recursive traversal allocates nothing per node for them. Every user must
// guard n <= kMaxActions.
inline constexpr int kMaxActions = 16;

struct InfosetData {
    std::vector<double> regret;
    std::vector<double> strategy_sum;
};

// Regret matching over the masked-legal actions; uniform over legal when no
// positive regret. Shared by MCCFRTrainer and BlueprintTrainer.
std::vector<double> regret_matched(const std::vector<double>& regret,
                                   const std::vector<uint8_t>& mask);

// Out-parameter form: writes sigma[0..n) into caller-provided fixed storage
// instead of heap-allocating a std::vector. Behaviour is bit-identical to the
// return-by-value overload above; only n leading entries are read/written.
void regret_matched(const std::vector<double>& regret,
                    const std::vector<uint8_t>& mask, int n,
                    std::array<double, kMaxActions>& out);

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
    void matched_strategy(const InfosetData& d,
                          const std::vector<uint8_t>& mask, int n,
                          std::array<double, kMaxActions>& out) const;
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

// Lookup-based version: call with a function that returns the average strategy
// for a given infoset key. Used by both MCCFRTrainer and BlueprintTrainer.
double kuhn_exact_value_lookup(
    const std::function<std::vector<double>(uint64_t)>& avg);

}  // namespace sixmax
