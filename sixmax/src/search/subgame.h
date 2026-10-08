#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>
#include "blueprint/engine_game.h"
#include "search/leaf_eval.h"
#include "search/range_tracker.h"
#include "search/vector_game.h"
#include "vocab/vocab.h"

namespace sixmax {

struct SubgameConfig {
    int max_nodes = 20000;   // expansion cap; the frontier becomes leaves
};

class SubgameState;

// The current street's public betting tree, as a VectorGame. Depth-limited:
// every node past the street's last action is a LEAF whose utility matrix is
// supplied by LeafEval (Task 4). Asymmetric hand spaces: the hero seat holds one
// exact combo; the opponent seat ranges over the surviving combos of the
// RangeTracker posterior, which also supplies the opponent's prior reach.
//
// Storage is FULL-VOCAB indexed: `mask`/`children` are vocab.size() long, so
// vocab action `a` maps to `children[a]` and illegal actions are masked (never
// reordered or filtered).
class Subgame : public VectorGame {
public:
    Subgame(const EngineGameState& street_root, const ActionVocab& vocab,
            const RangeTracker& ranges, int hero, std::array<int, 2> hero_hole,
            SubgameConfig cfg, const BlueprintStrategy* bp = nullptr,
            LeafEvalConfig leaf_cfg = LeafEvalConfig{},
            std::chrono::steady_clock::time_point deadline =
                std::chrono::steady_clock::time_point::max());
    ~Subgame() override;

    int num_hands(int player) const override {
        return player == hero_seat_ ? 1 : (int)opp_hands_.size();
    }
    bool compatible(int h0, int h1) const override;        // no shared card
    std::vector<double> prior_reach(int player) const override;
    std::unique_ptr<VectorState> root() const override;
    size_t node_count() const { return nodes_.size(); }
    std::vector<uint64_t> node_ids() const;
    // False if any leaf could not be fully valued (deadline hit, or a sampled
    // leaf with no blueprint). The solve would then rest on zero-valued leaves,
    // so SearchStrategy must fall back to the blueprint rather than trust it.
    bool complete() const { return complete_; }

    // Index translation used by SubgameState::utility and LeafEval wiring.
    int combo_for(int player, int hand) const {            // compact index -> 1326-combo
        return player == hero_seat_ ? hero_combo_ : opp_hands_[hand];
    }
    // Node id reached by following a legal vocab-action prefix from the root;
    // the root id if the prefix leaves the built tree.
    uint64_t query_node_id(const std::vector<int>& actions) const;

private:
    friend class SubgameState;
    struct Node {
        uint64_t id = 0;
        int actor = -1;
        bool leaf = false;
        std::vector<uint8_t> mask;         // vocab.size()
        std::vector<int> children;         // vocab.size(), -1 where none
        std::vector<double> leaf_matrix;   // payoff to hero, [1 * num_hands(opp)]
    };
    int build(const EngineGameState& s, int depth);
    uint64_t make_node_id(const EngineGameState& s, int depth) const;

    int hero_seat_ = 0;
    int hero_combo_ = -1;                      // the hero's one exact combo (1326-space)
    std::vector<int> opp_hands_;               // opponent combos surviving card removal
    std::vector<double> opp_prior_;            // posterior over opp_hands_, renormalised
    Street street_ = Street::Preflop;
    const ActionVocab* vocab_ = nullptr;
    LeafEval leaf_eval_;
    std::vector<Node> nodes_;
    SubgameConfig cfg_;
    bool complete_ = true;   // cleared when a leaf can't be fully valued
};

class SubgameState : public VectorState {
public:
    SubgameState(const Subgame* sg, int node) : sg_(sg), node_(node) {}
    int num_players() const override { return 2; }
    int actor() const override { return sg_->nodes_[node_].actor; }
    bool is_leaf() const override { return sg_->nodes_[node_].leaf; }
    int num_actions() const override { return sg_->vocab_->size(); }
    void legal_mask(std::vector<uint8_t>& mask) const override {
        mask = sg_->nodes_[node_].mask;
    }
    std::unique_ptr<VectorState> apply(int action) const override;
    double utility(int player, int h0, int h1) const override;
    uint64_t id() const override { return sg_->nodes_[node_].id; }

private:
    const Subgame* sg_;
    int node_;
};

}  // namespace sixmax
