#include "search/search_strategy.h"
#include <chrono>

namespace sixmax {

SearchStrategy::SearchStrategy(const BlueprintStrategy& bp, const ActionVocab& vocab,
                              const EngineConfig& cfg, SearchConfig sc)
    : bp_(bp), vocab_(vocab), cfg_(cfg), sc_(sc) {}

std::vector<double> SearchStrategy::probs(const EngineGameState& decision_state,
                                          int hero,
                                          const std::vector<int>& observed_actions) {
    last_searched_ = false;
    if (!sc_.enabled) { ++fallbacks_; return {}; }
    try {
        const auto start = std::chrono::steady_clock::now();
        const std::vector<int> board = decision_state.board();
        const auto hc = decision_state.hole_cards(hero);
        const std::array<int, 2> hero_hole{hc[0], hc[1]};

        // Replay the observed prefix from the street root, Bayes-updating the
        // opponent range with the blueprint at each opponent decision.
        std::unique_ptr<EngineGameState> cur = decision_state.clone_engine();
        RangeTracker tracker(cfg_, vocab_);
        tracker.reset(board, hero_hole);
        std::vector<int> prefix;
        for (int action : observed_actions) {
            if (cur->is_terminal()) break;
            const int actor = cur->current_player();
            if (actor != hero) tracker.observe(bp_, *cur, actor, action);
            cur->apply(action);
            prefix.push_back(action);
        }

        // Build the street subgame over the tracked range. Reserve part of the
        // budget for the solve: leaf evaluation is bounded by an internal
        // deadline so a pathological tree can never block the caller.
        SubgameConfig sg_cfg;
        sg_cfg.max_nodes = sc_.max_nodes;
        LeafEvalConfig leaf_cfg;
        leaf_cfg.samples = sc_.leaf_samples;
        leaf_cfg.seed = sc_.seed;
        const double leaf_ms = sc_.budget_ms * 0.6;
        const auto leaf_deadline =
            start + std::chrono::milliseconds(static_cast<long long>(leaf_ms));
        Subgame sg(decision_state, vocab_, tracker, hero, hero_hole, sg_cfg,
                   &bp_, leaf_cfg, leaf_deadline);
        const uint64_t query_id = sg.query_node_id(prefix);

        const auto now = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(now - start).count();
        double remaining_ms = sc_.budget_ms - elapsed_ms;
        if (remaining_ms < 1.0) remaining_ms = 1.0;   // at least one deadline check

        VectorCfrConfig vc;
        vc.seed = sc_.seed;
        vc.max_iters = sc_.max_iters;
        vc.budget_ms = remaining_ms;
        VectorCfr solver(vc);
        solver.solve(sg, hero, /*query_hand=*/0);
        std::vector<double> p = solver.average_strategy(query_id, /*hand=*/0);
        if (p.empty()) { ++fallbacks_; return {}; }
        double total = 0.0;
        for (double v : p) total += v;
        if (!(total > 0.0)) { ++fallbacks_; return {}; }
        for (double& v : p) v /= total;
        last_searched_ = true;
        ++searches_;
        return p;
    } catch (...) {
        ++fallbacks_;
        return {};
    }
}

}  // namespace sixmax
