#include "search/range_tracker.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace sixmax {

int RangeTracker::combo_index(int c0, int c1) {
    if (c0 > c1) std::swap(c0, c1);
    // Number of combos whose first card is < c0, plus the offset within c0.
    return c0 * 51 - (c0 * (c0 - 1)) / 2 + (c1 - c0 - 1);
}

std::array<int, 2> RangeTracker::combo_cards(int idx) {
    // Inverse of combo_index, built once: hot loops call this per combo.
    static const std::array<std::array<int, 2>, 1326> table = [] {
        std::array<std::array<int, 2>, 1326> t{};
        for (int c0 = 0; c0 < 52; ++c0)
            for (int c1 = c0 + 1; c1 < 52; ++c1)
                t[combo_index(c0, c1)] = {c0, c1};
        return t;
    }();
    if (idx < 0 || idx >= 1326) return {-1, -1};
    return table[idx];
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
    // Legality is hole-independent, so the unseen-infoset fallback (uniform
    // over the legal mask) is one number for every combo.
    std::vector<uint8_t> mask;
    pre_action.legal_mask(mask);
    int legal = 0;
    for (uint8_t m : mask) legal += m ? 1 : 0;
    const double fallback =
        (legal > 0 && action < (int)mask.size() && mask[action]) ? 1.0 / legal
                                                                 : 0.0;
    // One probe whose actor hole cards are overwritten per combo; combos
    // differ only through the card bucket, so the likelihood is memoised by
    // infoset key.
    EngineGameState probe = pre_action;
    std::unordered_map<uint64_t, double> by_key;
    for (size_t i = 0; i < range_.size(); ++i) {
        if (range_[i] <= 0.0) continue;
        auto hole = combo_cards(static_cast<int>(i));
        if (hole[0] < 0) continue;
        probe.mutable_hand().set_hole_cards(actor, hole[0], hole[1]);
        const uint64_t key = probe.abstract_key(blueprint.abstraction());
        auto it = by_key.find(key);
        if (it == by_key.end()) {
            const std::vector<double> probs = blueprint.probs(key);
            const double v = (!probs.empty() && action < (int)probs.size())
                                 ? probs[action]
                                 : fallback;
            it = by_key.emplace(key, v).first;
        }
        likelihood[i] = it->second;
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

void RangeTracker::remove_blocked(const std::vector<int>& cards) {
    std::array<bool, 52> blocked{};
    bool any = false;
    for (int c : cards) {
        if (c < 0 || c >= 52 || blocked[c]) continue;
        blocked[c] = true;
        any = true;
    }
    if (!any) return;
    std::vector<double> post(range_.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < range_.size(); ++i) {
        const auto hole = combo_cards((int)i);
        if (hole[0] < 0 || blocked[hole[0]] || blocked[hole[1]]) continue;
        post[i] = range_[i];
        total += post[i];
    }
    if (total <= 0.0) return;  // everything blocked: keep the prior
    for (double& v : post) v /= total;
    range_ = std::move(post);
}

void RangeTracker::set_prior(const std::vector<double>& prior) {
    if (prior.size() != range_.size()) return;        // malformed: ignore
    double total = 0.0;
    for (double v : prior) total += (v > 0.0 ? v : 0.0);
    if (total <= 0.0) return;                          // degenerate: keep reset
    for (size_t i = 0; i < range_.size(); ++i)
        range_[i] = (prior[i] > 0.0 ? prior[i] : 0.0) / total;
}

}  // namespace sixmax
