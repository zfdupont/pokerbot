#include "search/vector_cfr.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace sixmax {
namespace {
constexpr int kMaxActions = 16;   // search grids are small; guards stack arrays
}

VectorCfrResult VectorCfr::solve(const VectorGame& game, int query_player,
                                 int query_hand) {
    if (query_player < 0 || query_player > 1)
        throw std::invalid_argument("query_player must be 0 or 1");
    query_player_ = query_player;
    num_hands_[0] = game.num_hands(0);
    num_hands_[1] = game.num_hands(1);
    if (query_hand < 0 || query_hand >= num_hands_[query_player])
        throw std::invalid_argument("query_hand out of range");
    game_ = &game;             // non-owning; used by walk() and average_game_value()
    table_.clear();
    node_actions_.clear();
    node_actor_.clear();
    deadline_hit_ = false;

    auto root = game.root();
    root_id_ = root->id();
    node_actions_[root_id_] = root->num_actions();
    node_actor_[root_id_] = root->actor();

    // Root reaches come from the game's priors: uniform for Kuhn, the tracked
    // opponent range (and the hero point mass) for a Subgame. Card removal is
    // the game's job (conflicting pairs contribute 0 utility).
    std::array<std::vector<double>, 2> reach;
    for (int p = 0; p < 2; ++p) {
        reach[p] = game.prior_reach(p);
        if ((int)reach[p].size() != num_hands_[p])
            throw std::invalid_argument("prior_reach size != num_hands");
    }

    const auto start = std::chrono::steady_clock::now();
    int it = 0;
    for (; it < cfg_.max_iters; ++it) {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start).count();
        if (ms >= cfg_.budget_ms) { deadline_hit_ = true; break; }
        const double w = static_cast<double>(it + 1);   // linear CFR+ weighting
        walk(*root, 0, reach, w);
    }
    VectorCfrResult r;
    // The gate value is computed from the AVERAGE strategy by an exact walk over
    // compatible deals, mirroring kuhn_exact_value_lookup (mccfr.cpp) — NOT read
    // from the solver's per-hand reach-weighted values.
    r.root_value = average_game_value(game, query_player);
    r.iterations = it;
    r.deadline_hit = deadline_hit_;
    return r;
}

double VectorCfr::average_game_value(const VectorGame& game, int qp) const {
    const int nq = num_hands_[qp], no = num_hands_[1 - qp];
    const std::vector<double> pq = game.prior_reach(qp);
    const std::vector<double> po = game.prior_reach(1 - qp);
    double num = 0.0, Z = 0.0;
    for (int hq = 0; hq < nq; ++hq)
        for (int ho = 0; ho < no; ++ho) {
            const int h0 = (qp == 0) ? hq : ho, h1 = (qp == 0) ? ho : hq;
            if (!game.compatible(h0, h1)) continue;
            const double wt = pq[hq] * po[ho];   // joint prior over this pair
            if (wt == 0.0) continue;
            Z += wt;
            num += wt * ev_under_avg(*game.root(), qp, h0, h1);
        }
    return Z > 0.0 ? num / Z : 0.0;
}

double VectorCfr::ev_under_avg(const VectorState& s, int qp, int h0, int h1) const {
    if (s.is_leaf()) return s.utility(qp, h0, h1);
    const int actor = s.actor();
    const int actor_hand = (actor == 0) ? h0 : h1;
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    std::vector<double> sigma = average_strategy(s.id(), actor_hand);
    double ev = 0.0;
    int legal = 0;
    for (int a = 0; a < (int)mask.size(); ++a) legal += mask[a] ? 1 : 0;
    for (int a = 0; a < (int)mask.size(); ++a) {
        if (!mask[a]) continue;
        const double p = sigma.empty() ? 1.0 / legal : sigma[a];
        if (p <= 0.0) continue;
        ev += p * ev_under_avg(*s.apply(a), qp, h0, h1);
    }
    return ev;
}

std::array<std::vector<double>, 2> VectorCfr::walk(
        const VectorState& s, int depth,
        const std::array<std::vector<double>, 2>& reach, double w) {
    std::array<std::vector<double>, 2> out;
    if (s.is_leaf()) {
        const int n0 = num_hands_[0], n1 = num_hands_[1];
        out[0].assign(n0, 0.0);
        out[1].assign(n1, 0.0);
        for (int h0 = 0; h0 < n0; ++h0)
            for (int h1 = 0; h1 < n1; ++h1) {
                if (!game_->compatible(h0, h1)) continue;   // card removal
                // Counterfactual value: weight each player's hand by the
                // OPPONENT's reach only. Folding in a player's own reach cancels
                // in per-hand regret-matching (so the learned strategy is
                // unaffected) but corrupts the root value readout.
                const double r0 = reach[0][h0], r1 = reach[1][h1];
                if (r1 != 0.0) out[0][h0] += r1 * s.utility(0, h0, h1);
                if (r0 != 0.0) out[1][h1] += r0 * s.utility(1, h0, h1);
            }
        return out;
    }
    const int p = s.actor();
    const int q = 1 - p;
    const uint64_t id = s.id();
    const int n = s.num_actions();
    if (n <= 0 || n > kMaxActions) throw std::runtime_error("bad action count");
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    node_actions_[id] = n;
    node_actor_[id] = p;

    NodeData& d = table_[id];
    if (d.regret.empty()) {
        d.regret.assign(static_cast<size_t>(num_hands_[p]) * n, 0.0);
        d.strat_sum.assign(static_cast<size_t>(num_hands_[p]) * n, 0.0);
    }

    // Regret-matching+ with the node's per-hand reach weighting for averaging.
    std::vector<double> sigma(static_cast<size_t>(num_hands_[p]) * n, 0.0);
    for (int h = 0; h < num_hands_[p]; ++h) {
        double pos = 0.0;
        int legal = 0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            ++legal;
            const double r = d.regret[static_cast<size_t>(h) * n + a];
            if (r > 0.0) pos += r;
        }
        if (pos > 0.0) {
            for (int a = 0; a < n; ++a)
                if (mask[a]) {
                    const double r = d.regret[static_cast<size_t>(h) * n + a];
                    sigma[static_cast<size_t>(h) * n + a] = r > 0.0 ? r / pos : 0.0;
                }
        } else {
            for (int a = 0; a < n; ++a)
                if (mask[a]) sigma[static_cast<size_t>(h) * n + a] = 1.0 / legal;
        }
    }

    out[0].assign(num_hands_[0], 0.0);
    out[1].assign(num_hands_[1], 0.0);
    std::vector<std::vector<double>> vp_by_action(n);   // p's values per action
    std::vector<std::vector<double>> vq_by_action(n);   // q's values per action

    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        auto child = s.apply(a);
        auto child_reach = reach;
        for (int h = 0; h < num_hands_[p]; ++h)
            child_reach[p][h] = reach[p][h] * sigma[static_cast<size_t>(h) * n + a];
        auto v = walk(*child, depth + 1, child_reach, w);
        vp_by_action[a] = v[p];
        vq_by_action[a] = v[q];
        for (int h = 0; h < num_hands_[q]; ++h) out[q][h] += v[q][h];
    }

    // p's node value and regret update (regrets are weighted by w, as in CFR+).
    for (int h = 0; h < num_hands_[p]; ++h) {
        double node_value = 0.0;
        for (int a = 0; a < n; ++a)
            if (mask[a]) node_value += sigma[static_cast<size_t>(h) * n + a] * vp_by_action[a][h];
        out[p][h] = node_value;
        const double reach_h = reach[p][h];
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            const size_t k = static_cast<size_t>(h) * n + a;
            d.regret[k] = std::max(0.0, d.regret[k] + w * (vp_by_action[a][h] - node_value));
            // Average strategy for BOTH seats: average_game_value walks the
            // recovered average for either actor, so the acting player's
            // strategy_sum must always be accumulated (regret updates above are
            // unchanged). This mirrors the tabular trainer's average_strategy.
            d.strat_sum[k] += w * reach_h * sigma[k];
        }
    }
    return out;
}

std::vector<double> VectorCfr::average_strategy(uint64_t node_id, int hand) const {
    auto it = table_.find(node_id);
    if (it == table_.end()) return {};
    const int n = node_actions_.at(node_id);
    const auto& ss = it->second.strat_sum;
    double total = 0.0;
    for (int a = 0; a < n; ++a) total += ss[static_cast<size_t>(hand) * n + a];
    if (total <= 0.0) return {};
    std::vector<double> out(n, 0.0);
    for (int a = 0; a < n; ++a) out[a] = ss[static_cast<size_t>(hand) * n + a] / total;
    return out;
}

int VectorCfr::rank_root_action(int, int hand) const {
    auto probs = average_strategy(root_id_, hand);
    if (probs.empty()) return -1;
    int best = -1;
    double bestv = -1.0;
    for (size_t a = 0; a < probs.size(); ++a)
        if (probs[a] > bestv) { bestv = probs[a]; best = static_cast<int>(a); }
    return best;
}

}  // namespace sixmax
