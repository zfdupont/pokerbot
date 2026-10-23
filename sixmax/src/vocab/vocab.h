#pragma once
#include <cstdint>
#include <vector>

namespace sixmax {

enum class ActionType { Fold, Check, Call, Bet, AllIn };
enum class SizeUnit { BB, Pot };

struct AbstractAction {
    ActionType type;
    double size;      // BB unit: raise-to in BB. Pot unit: pot fraction.
    SizeUnit unit;
};

// All quantities in big blinds (big_blind == 1.0 frame).
struct BetContext {
    double pot;
    double current_bet;
    double to_call;
    double stack;
};

class ActionVocab {
public:
    explicit ActionVocab(std::vector<AbstractAction> actions);
    int size() const { return (int)actions_.size(); }
    const AbstractAction& at(int i) const { return actions_[i]; }
    // Bet/raise-to total in BB for action idx in this context; capped by stack.
    double target_bb(int idx, const BetContext& ctx) const;
    // Size class for the infoset key: 0 for fold/check/call, 4 for all-in,
    // else 1 + the index's rank among Bet entries of the same SizeUnit (the
    // vocab lists sizes ascending). Throws std::invalid_argument if that unit
    // has more than 3 Bet entries (3 bits cannot encode the ranks).
    int raise_class(int idx) const;
    // Map an observed bet (raise-to, BB) onto the grid with randomized
    // pseudo-harmonic weighting; u in [0,1) supplied by caller.
    int nearest(double bet_to_bb, const BetContext& ctx, double u) const;
    uint64_t hash() const { return hash_; }

private:
    std::vector<AbstractAction> actions_;
    uint64_t hash_;
};

}  // namespace sixmax
