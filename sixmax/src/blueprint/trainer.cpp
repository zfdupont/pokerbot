#include "blueprint/trainer.h"
#include <array>
#include <cassert>
#include <thread>

namespace sixmax {

void BlueprintTrainer::train(uint64_t iterations) {
    const uint64_t target = iter_.load() + iterations;
    int T = cfg_.num_threads > 0 ? cfg_.num_threads
                                 : (int)std::thread::hardware_concurrency();
    if (T < 1) T = 1;
    auto worker = [&](int tid) {
        auto game = factory_();
        std::mt19937_64 rng(cfg_.seed * 0x9E3779B97F4A7C15ull +
                            (uint64_t)tid + 1);
        const int n = game->num_actions();
        for (;;) {
            uint64_t t = iter_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (t > target) {
                iter_.fetch_sub(1, std::memory_order_relaxed);
                break;  // each thread over-grabs at most once
            }
            const double w = (double)t;  // linear CFR weight
            for (int p = 0; p < game->num_players(); ++p) {
                auto s = game->new_hand(rng);
                traverse(*s, p, w, rng, n);
            }
        }
    };
    if (T == 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int i = 0; i < T; ++i) threads.emplace_back(worker, i);
    for (auto& th : threads) th.join();
}

double BlueprintTrainer::traverse(GameState& s, int traverser, double w,
                                  std::mt19937_64& rng, int n) {
    if (s.is_terminal()) return s.utility(traverser);
    assert(n <= kMaxActions);
    std::vector<uint8_t> mask;
    s.legal_mask(mask);
    const uint64_t key = s.infoset_key();
    Shard& sh = shards_[shard_of(key)];
    std::array<double, kMaxActions> sigma;
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        if (d.regret.empty()) {
            d.regret.assign(n, 0.0);
            d.strategy_sum.assign(n, 0.0);
        }
        regret_matched(d.regret, mask, n, sigma);
    }
    if (s.current_player() == traverser) {
        std::array<double, kMaxActions> u{};
        double ev = 0.0;
        for (int a = 0; a < n; ++a) {
            if (!mask[a]) continue;
            auto child = s.clone();
            child->apply(a);
            u[a] = traverse(*child, traverser, w, rng, n);
            ev += sigma[a] * u[a];
        }
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.regret[a] += w * (u[a] - ev);
        return ev;
    }
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        InfosetData& d = sh.map[key];
        for (int a = 0; a < n; ++a)
            if (mask[a]) d.strategy_sum[a] += w * sigma[a];
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng), acc = 0.0;
    int chosen = -1;
    for (int a = 0; a < n; ++a) {
        if (!mask[a]) continue;
        acc += sigma[a];
        chosen = a;
        if (r <= acc) break;
    }
    s.apply(chosen);
    return traverse(s, traverser, w, rng, n);
}

size_t BlueprintTrainer::num_infosets() const {
    size_t total = 0;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        total += sh.map.size();
    }
    return total;
}

std::vector<double> BlueprintTrainer::average_strategy(uint64_t key) const {
    const Shard& sh = shards_[shard_of(key)];
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(key);
    if (it == sh.map.end()) return {};
    const auto& ss = it->second.strategy_sum;
    double total = 0.0;
    for (double v : ss) total += v;
    if (total <= 0.0) return {};
    std::vector<double> out(ss.size());
    for (size_t a = 0; a < ss.size(); ++a) out[a] = ss[a] / total;
    return out;
}

std::vector<uint64_t> BlueprintTrainer::keys() const {
    std::vector<uint64_t> out;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        for (const auto& [k, v] : sh.map) out.push_back(k);
    }
    return out;
}

std::unordered_map<uint64_t, InfosetData> BlueprintTrainer::export_table()
        const {
    std::unordered_map<uint64_t, InfosetData> out;
    for (const auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        out.insert(sh.map.begin(), sh.map.end());
    }
    return out;
}

void BlueprintTrainer::import_table(
        std::unordered_map<uint64_t, InfosetData> table, uint64_t iterations) {
    for (auto& sh : shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.map.clear();
    }
    for (auto& [k, v] : table) {
        Shard& sh = shards_[shard_of(k)];
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.map.emplace(k, std::move(v));
    }
    iter_.store(iterations, std::memory_order_relaxed);
}

}  // namespace sixmax
