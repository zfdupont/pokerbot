#include "search/range_tracker.h"
#include <algorithm>
#include <cmath>

namespace sixmax {

int RangeTracker::combo_index(int c0, int c1) {
    if (c0 > c1) std::swap(c0, c1);
    // Number of combos whose first card is < c0, plus the offset within c0.
    return c0 * 51 - (c0 * (c0 - 1)) / 2 + (c1 - c0 - 1);
}

std::array<int, 2> RangeTracker::combo_cards(int idx) {
    for (int c0 = 0; c0 < 52; ++c0) {
        const int base = c0 * 51 - (c0 * (c0 - 1)) / 2;
        const int count = 51 - c0;
        if (idx < base + count) return {c0, c0 + 1 + (idx - base)};
    }
    return {-1, -1};
}

void RangeTracker::reset(const std::vector<int>& board,
                         const std::array<int, 2>& our_hole) {
    std::array<bool, 52> blocked{};
    for (int c : board) blocked[c] = true;
    blocked[our_hole[0]] = blocked[our_hole[1]] = true;
    range_.assign(1326, 0.0);
    double total = 0.0;
    for (int i = 0; i < 1326; ++i) {
        auto c = combo_cards(i);
        if (blocked[c[0]] || blocked[c[1]]) continue;
        range_[i] = 1.0;
        total += 1.0;
    }
    if (total > 0.0)
        for (double& v : range_) v /= total;
}

void RangeTracker::observe(const BlueprintStrategy& blueprint,
                           const EngineGameState& pre_action, int actor, int action) {
    if (action < 0) return;   // unknown/off-tree observation: no update
    std::vector<double> likelihood(range_.size(), 0.0);
    for (size_t i = 0; i < range_.size(); ++i) {
        if (range_[i] <= 0.0) continue;
        auto hole = combo_cards(static_cast<int>(i));
        if (hole[0] < 0) continue;
        EngineGameState probe = pre_action;   // value copy: set_hole_cards below
        probe.mutable_hand().set_hole_cards(actor, hole[0], hole[1]);
        std::vector<double> probs = blueprint.probs_for(probe);
        std::vector<uint8_t> mask;
        probe.legal_mask(mask);
        if (!probs.empty() && action < (int)probs.size()) {
            likelihood[i] = probs[action];
        } else {
            int legal = 0;
            for (uint8_t m : mask) legal += m ? 1 : 0;
            likelihood[i] = (legal > 0 && action < (int)mask.size() && mask[action])
                                ? 1.0 / legal
                                : 0.0;
        }
    }
    observe_with_probs(actor, likelihood);
}

void RangeTracker::observe_with_probs(int, const std::vector<double>& w) {
    // Multiply into a temporary so a degenerate observation (zero total) leaves
    // the prior untouched rather than zeroing it out before the guard.
    std::vector<double> post(range_.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < range_.size(); ++i) {
        post[i] = range_[i] * (i < w.size() ? w[i] : 0.0);
        total += post[i];
    }
    if (total <= 0.0) return;                    // keep the prior: degenerate update
    for (double& v : post) v /= total;
    range_ = std::move(post);
}

}  // namespace sixmax
