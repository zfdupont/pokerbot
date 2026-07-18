#include "vocab/vocab.h"
#include <algorithm>
#include <cmath>

namespace sixmax {

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
    : actions_(std::move(actions)), hash_(fnv1a(actions_)) {}

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

int ActionVocab::nearest(double bet_to_bb, const BetContext& ctx, double u) const {
    // Candidates: Pot-sized bets and AllIn by their target_bb in this ctx.
    int lo = -1, hi = -1;
    double lo_v = -1e18, hi_v = 1e18;
    for (int i = 0; i < size(); ++i) {
        const auto& a = actions_[i];
        if (a.type != ActionType::Bet && a.type != ActionType::AllIn) continue;
        // Skip BB-sized bets; only consider Pot-sized bets and AllIn
        if (a.type == ActionType::Bet && a.unit == SizeUnit::BB) continue;
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
