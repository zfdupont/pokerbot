#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "blueprint/checkpoint.h"
#include "blueprint/engine_game.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

// The OPPONENT's range over the 1,326 two-card combos, canonically indexed by
// (min,max) card code. v1 tracks exactly one opponent: `reset`/`observe` build a
// single posterior and `range()` returns it. The hero's own hand is NOT tracked
// here — it is a known point mass supplied straight to the Subgame. The
// `int player` argument on `range()`/`combo_prob()` is vestigial (kept for the
// binding) and ignored; both always return the opponent posterior.
class RangeTracker {
public:
    RangeTracker(const EngineConfig& cfg, const ActionVocab& vocab)
        : cfg_(cfg), vocab_(vocab), range_(1326, 0.0) {}

    static int combo_index(int c0, int c1);
    static std::array<int, 2> combo_cards(int idx);

    // Uniform over combos not blocked by `board` or `our_hole`.
    void reset(const std::vector<int>& board, const std::array<int, 2>& our_hole);

    // Bayes reweight for `actor` taking `action` from `pre_action`, using the
    // blueprint's policy at that state as the opponent model. Renormalises;
    // an all-zero posterior degrades to uniform-over-the-previous-support.
    void observe(const BlueprintStrategy& blueprint, const EngineGameState& pre_action,
                 int actor, int action);

    // Zero every combo holding one of `cards` (newly revealed board cards) and
    // renormalise. If that would leave zero mass, the prior is kept.
    void remove_blocked(const std::vector<int>& cards);

    // Overwrite the posterior with `prior` (renormalised copy). A wrong-size or
    // all-zero prior is ignored, leaving the current range (e.g. a uniform
    // reset) intact — the degenerate-safe path the search relies on.
    void set_prior(const std::vector<double>& prior);

    const std::vector<double>& range(int) const { return range_; }
    double combo_prob(std::array<int, 2> hole) const {
        return range_[combo_index(hole[0], hole[1])];
    }

    // Test seams (never used in production paths).
    void set_range_for_test(int, const std::vector<double>& probs) { range_ = probs; }
    void observe_with_probs(int, const std::vector<double>& action_probs_per_combo);

private:
    EngineConfig cfg_;
    const ActionVocab& vocab_;
    std::vector<double> range_;
};

}  // namespace sixmax
