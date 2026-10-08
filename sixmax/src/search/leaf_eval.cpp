#include "search/leaf_eval.h"
#include <array>
#include <stdexcept>
#include "exploit/policy.h"
#include "game/safe_eval.h"
#include "search/range_tracker.h"

namespace sixmax {

double LeafEval::rollout(const EngineGameState& leaf, int actor, int our_combo,
                         int opp_combo, std::mt19937_64& rng) const {
    // Clone the bridge state so the rollout inherits its ActionVocab (needed by
    // sample_blueprint); then overwrite the hidden cards directly.
    std::unique_ptr<EngineGameState> probe = leaf.clone_engine();
    HandState& h = probe->mutable_hand();
    const auto ours = RangeTracker::combo_cards(our_combo);
    const auto theirs = RangeTracker::combo_cards(opp_combo);
    h.set_hole_cards(actor, ours[0], ours[1]);
    h.set_hole_cards(1 - actor, theirs[0], theirs[1]);

    // Pool: every card not already visible/held. reroll_hidden then samples the
    // opponent's temporary hole cards (overwritten below) and the remaining
    // board; excluding both combos from the pool keeps the board collision-free.
    std::array<bool, 52> used{};
    for (int c : leaf.board()) used[c] = true;
    used[ours[0]] = used[ours[1]] = true;
    used[theirs[0]] = used[theirs[1]] = true;
    std::vector<int> pool;
    for (int c = 0; c < 52; ++c) if (!used[c]) pool.push_back(c);
    h.reroll_hidden(actor, std::move(pool), rng);
    // Restore the opponent's exact combo (its cards were excluded from the pool).
    h.set_hole_cards(1 - actor, theirs[0], theirs[1]);

    while (!probe->is_terminal()) {
        int a = sample_blueprint(*bp_, *probe, /*uniform=*/false, rng);
        if (a < 0) break;   // defensive: no legal action
        probe->apply(a);
    }
    return probe->utility(actor);
}

std::vector<double> LeafEval::matrix(const EngineGameState& leaf, int actor,
                                     const std::vector<int>& our_hands,
                                     const std::vector<int>& opp_hands) const {
    const size_t n = our_hands.size(), m = opp_hands.size();
    std::vector<double> out(n * m, 0.0);
    const std::vector<int> board = leaf.board();
    const bool board_complete = board.size() == 5;
    std::array<bool, 52> on_board{};
    for (int c : board) on_board[c] = true;
    // HU showdown swing: the winner nets +pot/2 (contributed half, takes all).
    // A single per-leaf constant; CFR only uses relative action values.
    const double stake = leaf.hand().pot() / 2.0;
    std::array<int, 7> hero7{}, opp7{};
    if (board_complete)
        for (int b = 0; b < 5; ++b) { hero7[2 + b] = board[b]; opp7[2 + b] = board[b]; }

    std::mt19937_64 rng(cfg_.seed ^
                        (static_cast<uint64_t>(leaf.street()) * 0x9E3779B97F4A7C15ull));

    for (size_t i = 0; i < n; ++i) {
        const auto a = RangeTracker::combo_cards(our_hands[i]);
        if (on_board[a[0]] || on_board[a[1]]) continue;        // card removal: board
        for (size_t j = 0; j < m; ++j) {
            const auto b = RangeTracker::combo_cards(opp_hands[j]);
            if (on_board[b[0]] || on_board[b[1]]) continue;    // card removal: board
            if (a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1])
                continue;                                       // shared card (incl. identical)
            if (board_complete) {
                // Exact showdown. NOTE: not engine-terminal; do NOT call payoffs().
                hero7[0] = a[0]; hero7[1] = a[1];
                opp7[0] = b[0];  opp7[1] = b[1];
                const auto hr = safe_eval::rank7(hero7);
                const auto orr = safe_eval::rank7(opp7);
                out[i * m + j] = hr.beats(orr) ? stake : (orr.beats(hr) ? -stake : 0.0);
            } else {
                if (!bp_) throw std::logic_error("sampled leaf needs a blueprint");
                if (expired()) return out;   // budget exhausted: leave the rest at 0
                double acc = 0.0;
                for (int s = 0; s < cfg_.samples; ++s) {
                    acc += rollout(leaf, actor, our_hands[i], opp_hands[j], rng);
                    if ((s & 15) == 15 && expired()) break;
                }
                out[i * m + j] = cfg_.samples > 0 ? acc / cfg_.samples : 0.0;
            }
        }
    }
    return out;
}

}  // namespace sixmax
