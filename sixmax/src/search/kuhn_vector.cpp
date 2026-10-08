#include "search/kuhn_vector.h"

namespace sixmax {

double KuhnVectorState::utility(int player, int h0, int h1) const {
    if (h0 == h1) return 0.0;                 // impossible pair (card removal)
    // Fold terminals are card-independent.
    if (h_ == "pbp") return player == 0 ? -1.0 : 1.0;
    if (h_ == "bp")  return player == 0 ?  1.0 : -1.0;
    const double stake = (h_ == "pp") ? 1.0 : 2.0;
    const bool p0_wins = h0 > h1;
    const double u0 = p0_wins ? stake : -stake;
    return player == 0 ? u0 : -u0;
}

uint64_t KuhnVectorState::id() const {
    uint64_t code = 1469598103934665603ull;   // FNV-1a over the history
    for (char c : h_) { code ^= static_cast<uint8_t>(c); code *= 1099511628211ull; }
    return code;
}

}  // namespace sixmax
