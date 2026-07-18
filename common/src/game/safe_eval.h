#pragma once
#include <array>

namespace safe_eval {

// Opaque hand rank. The raw evaluator score (lower = better, inverted
// kickers) never leaves common/; comparisons go through beats()/ties().
class HandRank {
public:
    bool beats(const HandRank& o) const { return score_ < o.score_; }
    bool ties(const HandRank& o) const { return score_ == o.score_; }
private:
    explicit HandRank(unsigned int s) : score_(s) {}
    unsigned int score_;
    friend HandRank rank7(const std::array<int, 7>& codes);
};

// codes: 0-51, code = (rank-2)*4 + suit  (project convention)
HandRank rank7(const std::array<int, 7>& codes);

}  // namespace safe_eval
