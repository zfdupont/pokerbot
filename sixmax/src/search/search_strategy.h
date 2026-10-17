#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "blueprint/checkpoint.h"
#include "engine/engine.h"
#include "search/leaf_eval.h"
#include "search/range_tracker.h"
#include "search/subgame.h"
#include "search/vector_cfr.h"
#include "vocab/vocab.h"

namespace sixmax {

struct SearchConfig {
    bool enabled = false;
    double budget_ms = 5000.0;
    int max_nodes = 20000;
    int max_iters = 2000;
    int leaf_samples = 256;
    uint64_t seed = 1;
};

// Orchestrates one decision. Contract: never throws; never returns a partial
// or illegal distribution; an empty return means "use the blueprint".
class SearchStrategy {
public:
    SearchStrategy(const BlueprintStrategy& bp, const ActionVocab& vocab,
                   const EngineConfig& cfg, SearchConfig sc);

    // [actions.search]-indexed distribution over the hero's legal actions.
    // EMPTY means "fall back to the blueprint". Never throws. `decision_state`
    // is the STREET ROOT (start of the current street); `observed_actions` is
    // the vocab-action prefix taken since that root.
    std::vector<double> probs(const EngineGameState& decision_state, int hero,
                              const std::vector<int>& observed_actions,
                              const std::vector<double>& opp_prior = {});

    bool last_decision_searched() const { return last_searched_; }
    int searches() const { return searches_; }
    int fallbacks() const { return fallbacks_; }

private:
    const BlueprintStrategy& bp_;
    const ActionVocab& vocab_;
    EngineConfig cfg_;
    SearchConfig sc_;
    bool last_searched_ = false;
    int searches_ = 0, fallbacks_ = 0;
};

}  // namespace sixmax
