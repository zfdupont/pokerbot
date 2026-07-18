#include "blueprint/mccfr.h"
#include "blueprint/kuhn.h"

namespace sixmax {

std::vector<double> MCCFRTrainer::matched_strategy(
        const InfosetData& d, const std::vector<uint8_t>& mask) const {
    int n = (int)d.regret.size();
    std::vector<double> sigma(n, 0.0);
    double pos = 0.0;
    for (int a = 0; a < n; ++a)
        if (mask[a] && d.regret[a] > 0.0) pos += d.regret[a];
    if (pos > 0.0) {
        for (int a = 0; a < n; ++a)
            if (mask[a] && d.regret[a] > 0.0) sigma[a] = d.regret[a] / pos;
    } else {
        int legal = 0;
        for (int a = 0; a < n; ++a) legal += mask[a] ? 1 : 0;
        for (int a = 0; a < n; ++a) if (mask[a]) sigma[a] = 1.0 / legal;
    }
    return sigma;
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
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    InfosetData& d = table_[s.infoset_key()];
    if (d.regret.empty()) {
        d.regret.assign(n, 0.0);
        d.strategy_sum.assign(n, 0.0);
    }
    std::vector<double> sigma = matched_strategy(d, mask);

    if (s.current_player() == traverser) {
        std::vector<double> u(n, 0.0);
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
double kuhn_ev_p0(const MCCFRTrainer& t, const KuhnState& s) {
    if (s.is_terminal()) return s.utility(0);
    std::vector<double> sigma = t.average_strategy(s.infoset_key());
    if (sigma.empty()) sigma = {0.5, 0.5};
    double ev = 0.0;
    for (int a = 0; a < 2; ++a) {
        if (sigma[a] <= 0.0) continue;
        KuhnState child = s;
        child.apply(a);
        ev += sigma[a] * kuhn_ev_p0(t, child);
    }
    return ev;
}
}  // namespace

double kuhn_exact_value(const MCCFRTrainer& t) {
    double total = 0.0;
    for (int c0 = 0; c0 < 3; ++c0)
        for (int c1 = 0; c1 < 3; ++c1)
            if (c0 != c1) total += kuhn_ev_p0(t, KuhnState(c0, c1));
    return total / 6.0;
}

}  // namespace sixmax
