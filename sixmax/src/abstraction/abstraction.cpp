#include "abstraction/abstraction.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include "game/safe_eval.h"

namespace sixmax {

int preflop_class(const std::array<int, 2>& hole) {
    int r0 = hole[0] / 4, r1 = hole[1] / 4;
    int hi = std::max(r0, r1), lo = std::min(r0, r1);
    bool suited = (hole[0] % 4) == (hole[1] % 4);
    if (hi == lo) return hi * 13 + hi;
    return suited ? hi * 13 + lo : lo * 13 + hi;
}

namespace {

uint64_t fnv_mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
    return h;
}

uint64_t equity_seed(const std::array<int, 2>& hole,
                     const std::vector<int>& board, uint64_t salt) {
    std::vector<int> bd = board;
    std::sort(bd.begin(), bd.end());
    uint64_t h = 1469598103934665603ull;
    h = fnv_mix(h, (uint64_t)std::min(hole[0], hole[1]));
    h = fnv_mix(h, (uint64_t)std::max(hole[0], hole[1]));
    // +64 namespaces board codes away from hole codes (same trick as the
    // naive keyer in engine_game.cpp).
    for (int c : bd) h = fnv_mix(h, (uint64_t)(c + 64));
    return fnv_mix(h, salt);
}

}  // namespace

double hand_equity(const std::array<int, 2>& hole,
                   const std::vector<int>& board, int rollouts,
                   uint64_t salt) {
    std::mt19937_64 rng(equity_seed(hole, board, salt));
    bool used[52] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int c : board) used[c] = true;
    std::vector<int> deck;
    deck.reserve(52);
    for (int c = 0; c < 52; ++c)
        if (!used[c]) deck.push_back(c);
    const int need = 2 + (5 - (int)board.size());  // opp hole + runout
    double score = 0.0;
    for (int r = 0; r < rollouts; ++r) {
        // Partial Fisher-Yates over the live deck; prior permutations do
        // not bias later draws.
        for (int i = 0; i < need; ++i) {
            std::uniform_int_distribution<int> d(i, (int)deck.size() - 1);
            std::swap(deck[i], deck[d(rng)]);
        }
        std::array<int, 7> mine{}, theirs{};
        mine[0] = hole[0];
        mine[1] = hole[1];
        theirs[0] = deck[0];
        theirs[1] = deck[1];
        int k = 2;
        for (int c : board) { mine[k] = theirs[k] = c; ++k; }
        for (int i = 2; i < need; ++i) { mine[k] = theirs[k] = deck[i]; ++k; }
        auto a = safe_eval::rank7(mine);
        auto b = safe_eval::rank7(theirs);
        if (a.beats(b)) score += 1.0;
        else if (a.ties(b)) score += 0.5;
    }
    return score / rollouts;
}

Abstraction::Abstraction(const AbstractionConfig& cfg) : cfg_(cfg) {
    std::mt19937_64 rng(cfg.seed);
    const int nbuckets[3] = {cfg.flop_buckets, cfg.turn_buckets,
                             cfg.river_buckets};
    for (int s = 0; s < 3; ++s) {
        const int board_n = 3 + s;
        std::vector<double> eqs;
        eqs.reserve(cfg.quantile_samples);
        for (int i = 0; i < cfg.quantile_samples; ++i) {
            std::array<int, 52> deck;
            for (int c = 0; c < 52; ++c) deck[c] = c;
            for (int j = 0; j < 2 + board_n; ++j) {
                std::uniform_int_distribution<int> d(j, 51);
                std::swap(deck[j], deck[d(rng)]);
            }
            std::array<int, 2> hole{deck[0], deck[1]};
            std::vector<int> board(deck.begin() + 2,
                                   deck.begin() + 2 + board_n);
            eqs.push_back(
                hand_equity(hole, board, cfg.equity_rollouts, cfg.seed));
        }
        std::sort(eqs.begin(), eqs.end());
        auto& e = edges_[s];
        for (int k = 1; k < nbuckets[s]; ++k)
            e.push_back(eqs[(size_t)((double)k * eqs.size() / nbuckets[s])]);
    }
}

Abstraction::Abstraction(const AbstractionConfig& cfg,
                         std::array<std::vector<double>, 3> edges)
    : cfg_(cfg), edges_(std::move(edges)) {}

int Abstraction::bucket(const std::array<int, 2>& hole,
                        const std::vector<int>& board) const {
    if (board.size() < 3 || board.size() > 5)
        throw std::invalid_argument(
            "Abstraction::bucket: board must have 3, 4, or 5 cards");
    const int s = (int)board.size() - 3;  // 0=flop 1=turn 2=river
    // Salt MUST be cfg_.seed — the same salt used when sampling the quantile
    // edges in the constructor. Lookup-time equity estimates and the edge
    // distribution must come from the same estimator, or percentile buckets
    // would be calibrated against a different distribution than the lookups.
    double eq = hand_equity(hole, board, cfg_.equity_rollouts, cfg_.seed);
    const auto& e = edges_[s];
    return (int)(std::upper_bound(e.begin(), e.end(), eq) - e.begin());
}

int Abstraction::num_buckets(int street) const {
    switch (street) {
        case 1: return cfg_.flop_buckets;
        case 2: return cfg_.turn_buckets;
        default: return cfg_.river_buckets;
    }
}

uint64_t Abstraction::hash() const {
    uint64_t h = 1469598103934665603ull;
    auto mix_u = [&](uint64_t v) { h = fnv_mix(h, v); };
    auto mix_d = [&](double v) {
        uint64_t b;
        __builtin_memcpy(&b, &v, sizeof b);
        h = fnv_mix(h, b);
    };
    mix_u((uint64_t)cfg_.flop_buckets);
    mix_u((uint64_t)cfg_.turn_buckets);
    mix_u((uint64_t)cfg_.river_buckets);
    mix_u((uint64_t)cfg_.equity_rollouts);
    mix_u((uint64_t)cfg_.quantile_samples);
    mix_u(cfg_.seed);
    for (const auto& e : edges_) {
        mix_u((uint64_t)e.size());
        for (double v : e) mix_d(v);
    }
    return h;
}

}  // namespace sixmax
