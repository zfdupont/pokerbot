#include "vocab/vocab.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sixmax {

// Marker for a Bet index whose size unit has more than 3 entries.
constexpr uint8_t kRaiseClassUnavailable = 0xFF;

static uint64_t fnv1a(const std::vector<AbstractAction>& as) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    for (const auto& a : as) {
        mix((uint64_t)a.type);
        mix((uint64_t)a.unit);
        uint64_t bits; double s = a.size;
        static_assert(sizeof(bits) == sizeof(s));
        __builtin_memcpy(&bits, &s, sizeof(bits));
        mix(bits);
    }
    return h;
}

ActionVocab::ActionVocab(std::vector<AbstractAction> actions)
    : actions_(std::move(actions)),
      raise_classes_(actions_.size(), 0),
      hash_(fnv1a(actions_)) {
    // Precompute the per-index class table. A unit with more than 3 Bet sizes
    // cannot be ranked in 3 bits: mark those entries unavailable but keep the
    // vocab constructible, because off mode never asks for a class.
    int bet_count[2] = {0, 0};
    for (const auto& a : actions_)
        if (a.type == ActionType::Bet) ++bet_count[(int)a.unit];
    for (int u = 0; u < 2; ++u)
        if (bet_count[u] > 3) supports_raise_classes_ = false;

    int rank[2] = {0, 0};
    for (size_t i = 0; i < actions_.size(); ++i) {
        const auto& a = actions_[i];
        switch (a.type) {
            case ActionType::Fold:
            case ActionType::Check:
            case ActionType::Call:
                raise_classes_[i] = 0;
                break;
            case ActionType::AllIn:
                raise_classes_[i] = 4;
                break;
            case ActionType::Bet: {
                const int u = (int)a.unit;
                raise_classes_[i] = bet_count[u] > 3
                    ? kRaiseClassUnavailable
                    : (uint8_t)(1 + rank[u]);
                ++rank[u];
                break;
            }
        }
    }
}

double ActionVocab::target_bb(int idx, const BetContext& ctx) const {
    const auto& a = actions_[idx];
    double target;
    switch (a.type) {
        case ActionType::AllIn: return ctx.stack;
        case ActionType::Bet:
            target = (a.unit == SizeUnit::BB)
                ? a.size
                : ctx.current_bet + a.size * (ctx.pot + 2.0 * ctx.to_call);
            return std::min(target, ctx.stack);
        default: return 0.0;  // fold/check/call carry no size
    }
}

int ActionVocab::raise_class(int idx) const {
    const uint8_t cls = raise_classes_[idx];
    if (cls == kRaiseClassUnavailable)
        throw std::invalid_argument(
            "raise_class: at most 3 Bet sizes per unit are supported");
    return (int)cls;
}

int ActionVocab::nearest(double bet_to_bb, const BetContext& ctx, double u) const {
    // Candidates: Pot-sized bets and AllIn by their target_bb in this ctx.
    int lo = -1, hi = -1;
    double lo_v = -1e18, hi_v = 1e18;
    for (int i = 0; i < size(); ++i) {
        const auto& a = actions_[i];
        if (a.type != ActionType::Bet && a.type != ActionType::AllIn) continue;
        double v = target_bb(i, ctx);
        if (v <= bet_to_bb && v > lo_v) { lo = i; lo_v = v; }
        if (v >= bet_to_bb && v < hi_v) { hi = i; hi_v = v; }
    }
    if (lo == -1) return hi;
    if (hi == -1) return lo;
    if (lo == hi || lo_v == hi_v) return lo;
    // Pseudo-harmonic (Ganzfried & Sandholm): P(map to lo) for x in [A,B]:
    //   p = (B - x) * (1 + A) / ((B - A) * (1 + x))
    double A = lo_v, B = hi_v, x = bet_to_bb;
    double p = (B - x) * (1.0 + A) / ((B - A) * (1.0 + x));
    return (u < p) ? lo : hi;
}

}  // namespace sixmax
