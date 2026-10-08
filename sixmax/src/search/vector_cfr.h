#pragma once
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "search/vector_game.h"

namespace sixmax {

struct VectorCfrConfig {
    uint64_t seed = 1;
    int max_iters = 2000;         // hard iteration cap
    double budget_ms = 5000.0;    // wall-clock deadline, checked between iterations
};

struct VectorCfrResult {
    // Expected value to `query_player` under the recovered average strategy
    // (the Kuhn -1/18 gate; card-removal-normalised). Not a single hand's value.
    double root_value = 0.0;
    int iterations = 0;
    bool deadline_hit = false;
};

// CFR+ (regret-matching+ with linear averaging) over a public tree.
// Two-player zero-sum, one recursion pass:
//   * each node keeps regret and average-strategy tables indexed by
//     [hand][action];
//   * the traversal returns, per player, a counterfactual value vector: for
//     player p and hand h, sum over the OPPONENT's hands weighted by the
//     opponent's reach. Returning both vectors in one pass is valid because
//     the actor's reach is folded into each child call.
class VectorCfr {
public:
    explicit VectorCfr(VectorCfrConfig cfg) : cfg_(cfg) {}

    VectorCfrResult solve(const VectorGame& game, int query_player, int query_hand);
    // Average strategy at `node_id` for `query_player` holding `hand`.
    // Empty when the node was never visited as an actor, or all regrets <= 0
    // during the whole run (caller falls back to uniform-over-legal).
    std::vector<double> average_strategy(uint64_t node_id, int hand) const;
    // Test seam: index of the arg-max action at the root for `hand`.
    int rank_root_action(int query_player, int hand) const;
    // Node id of the root (the regret/average key for the root decision).
    uint64_t root_id() const { return root_id_; }

private:
    struct NodeData {
        std::vector<double> regret;    // [n_hands * n_actions]
        std::vector<double> strat_sum; // [n_hands * n_actions]
    };
    // v[p][h] for p in {0,1}: counterfactual value of p's hand h.
    std::array<std::vector<double>, 2> walk(
            const VectorState& s, int depth,
            const std::array<std::vector<double>, 2>& reach, double weight);
    // Expected value to `query_player` under the recovered AVERAGE strategy over
    // the joint prior, restricted to compatible pairs. This is the gate value;
    // it does not depend on the solver's internal reach bookkeeping.
    double average_game_value(const VectorGame& game, int query_player) const;
    double ev_under_avg(const VectorState& s, int query_player, int h0, int h1) const;

    VectorCfrConfig cfg_;
    const VectorGame* game_ = nullptr;   // non-owning; valid only during solve()
    int num_hands_[2] = {0, 0};
    int query_player_ = 0;
    uint64_t root_id_ = 0;
    std::unordered_map<uint64_t, NodeData> table_;
    std::unordered_map<uint64_t, int> node_actions_;
    std::unordered_map<uint64_t, int> node_actor_;
    bool deadline_hit_ = false;
};

}  // namespace sixmax
