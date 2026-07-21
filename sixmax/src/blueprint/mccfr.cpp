#include "blueprint/mccfr.h"
#include <array>
#include <cassert>
#include "blueprint/kuhn.h"

namespace sixmax {

std::vector<double> regret_matched(const std::vector<double>& regret,
                                   const std::vector<uint8_t>& mask) {
    int n = (int)regret.size();
    std::vector<double> sigma(n, 0.0);
    double pos = 0.0;
    for (int a = 0; a < n; ++a)
        if (mask[a] && regret[a] > 0.0) pos += regret[a];
    if (pos > 0.0) {
        for (int a = 0; a < n; ++a)
            if (mask[a] && regret[a] > 0.0) sigma[a] = regret[a] / pos;
    } else {
        int legal = 0;
        for (int a = 0; a < n; ++a) legal += mask[a] ? 1 : 0;
        for (int a = 0; a < n; ++a) if (mask[a]) sigma[a] = 1.0 / legal;
    }
    return sigma;
}

void regret_matched(const std::vector<double>& regret,
                    const std::vector<uint8_t>& mask, int n,
                    std::array<double, kMaxActions>& out) {
    for (int a = 0; a < n; ++a) out[a] = 0.0;
    double pos = 0.0;
    for (int a = 0; a < n; ++a)
        if (mask[a] && regret[a] > 0.0) pos += regret[a];
    if (pos > 0.0) {
        for (int a = 0; a < n; ++a)
            if (mask[a] && regret[a] > 0.0) out[a] = regret[a] / pos;
    } else {
        int legal = 0;
        for (int a = 0; a < n; ++a) legal += mask[a] ? 1 : 0;
        for (int a = 0; a < n; ++a) if (mask[a]) out[a] = 1.0 / legal;
    }
}

void MCCFRTrainer::matched_strategy(
        const InfosetData& d, const std::vector<uint8_t>& mask, int n,
        std::array<double, kMaxActions>& out) const {
    regret_matched(d.regret, mask, n, out);
}

void MCCFRTrainer::train(uint64_t iterations) {
    for (uint64_t i = 0; i < iterations; ++i) {
        ++iter_;
        weight_ = (double)iter_;  // linear CFR weighting
        for (int t = 0; t < game_.num_players(); ++t) {
            auto s = game_.new_hand(rng_);
            traverse(*s, t);
        }
    }
}

double MCCFRTrainer::traverse(GameState& s, int traverser) {
    if (s.is_terminal()) return s.utility(traverser);
    int n = game_.num_actions();
    assert(n <= kMaxActions);
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    InfosetData& d = table_[s.infoset_key()];
    if (d.regret.empty()) {
        d.regret.assign(n, 0.0);
        d.strategy_sum.assign(n, 0.0);
    }
    std::array<double, kMaxActions> sigma;
    matched_strategy(d, mask, n, sigma);

    if (s.current_player() == traverser) {
        std::array<double, kMaxActions> u{};
        double ev = 0.0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            auto child = s.clone();
            child->apply(a);
            u[a] = traverse(*child, traverser);
            ev += sigma[a] * u[a];
        }
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.regret[a] += weight_ * (u[a] - ev);
        return ev;
    }
    // Opponent/chance-free node: accumulate average strategy, sample one.
    for (int a = 0; a < n; ++a)
        if (mask[a]) d.strategy_sum[a] += weight_ * sigma[a];
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng_), acc = 0.0;
    int chosen = -1;
    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        acc += sigma[a];
        chosen = a;
        if (r <= acc) break;
    }
    s.apply(chosen);
    return traverse(s, traverser);
}

std::vector<double> MCCFRTrainer::average_strategy(uint64_t key) const {
    auto it = table_.find(key);
    if (it == table_.end()) return {};
    const auto& ss = it->second.strategy_sum;
    double total = 0.0;
    for (double v : ss) total += v;
    if (total <= 0.0) return {};
    std::vector<double> out(ss.size());
    for (size_t a = 0; a < ss.size(); ++a) out[a] = ss[a] / total;
    return out;
}

namespace {
double kuhn_ev_p0(const std::function<std::vector<double>(uint64_t)>& avg,
                  const KuhnState& s) {
    if (s.is_terminal()) return s.utility(0);
    std::vector<double> sigma = avg(s.infoset_key());
    if (sigma.empty()) sigma = {0.5, 0.5};
    double ev = 0.0;
    for (int a = 0; a < 2; ++a) {
        if (sigma[a] <= 0.0) continue;
        KuhnState child = s;
        child.apply(a);
        ev += sigma[a] * kuhn_ev_p0(avg, child);
    }
    return ev;
}
}  // namespace

double kuhn_exact_value_lookup(
        const std::function<std::vector<double>(uint64_t)>& avg) {
    double total = 0.0;
    for (int c0 = 0; c0 < 3; ++c0)
        for (int c1 = 0; c1 < 3; ++c1)
            if (c0 != c1) total += kuhn_ev_p0(avg, KuhnState(c0, c1));
    return total / 6.0;
}

double kuhn_exact_value(const MCCFRTrainer& t) {
    return kuhn_exact_value_lookup(
        [&](uint64_t k) { return t.average_strategy(k); });
}

}  // namespace sixmax
