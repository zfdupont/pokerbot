#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace sixmax {

// A two-player zero-sum game in VECTOR form: the public tree is explicit and
// each player holds one private "hand" drawn from a fixed index space. The
// solver never sees cards — only indices and a utility matrix per leaf.
//
// Card removal is the caller's contract: a hand pair that cannot co-occur
// (h0 == h1 sharing a card) MUST contribute utility 0, and the range vectors
// fed to the solver must already be zero on blocked hands.
class VectorState {
public:
    virtual ~VectorState() = default;
    virtual int num_players() const = 0;   // 2 in v1
    virtual int actor() const = 0;         // 0/1, or -1 at a leaf
    virtual bool is_leaf() const = 0;
    virtual int num_actions() const = 0;
    virtual void legal_mask(std::vector<uint8_t>& mask) const = 0;
    virtual std::unique_ptr<VectorState> apply(int action) const = 0;
    // Payoff to `player` when the two players hold h0 / h1. Leaf matrices for
    // NLHE are precomputed by LeafEval and cached inside the state; Kuhn
    // computes them directly.
    virtual double utility(int player, int h0, int h1) const = 0;
    // Stable identity for this public state (regret-table key).
    virtual uint64_t id() const = 0;
};

class VectorGame {
public:
    virtual ~VectorGame() = default;
    virtual int num_hands(int player) const = 0;   // 3 (Kuhn), <=1326 (NLHE)
    virtual std::unique_ptr<VectorState> root() const = 0;
    // True if hero hand h0 and opponent hand h1 can co-occur (share no card).
    // Kuhn: h0 != h1. Subgame: the two combos share no card. Used both to zero
    // conflicting pairs in the leaf and to normalise the root game value.
    virtual bool compatible(int h0, int h1) const = 0;
    // Prior reach for each of `player`'s hands at the root — this is how the
    // tracked opponent range ENTERS the solve. Kuhn: uniform (1/num_hands).
    // Subgame: the hero seat is a point mass [1.0] over its one exact hand; the
    // opponent seat is the RangeTracker posterior, renormalised over surviving
    // combos. Need not be normalised; the solver treats it as a weight vector.
    virtual std::vector<double> prior_reach(int player) const = 0;
};

}  // namespace sixmax
