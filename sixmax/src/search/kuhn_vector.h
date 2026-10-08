#pragma once
#include <memory>
#include <string>
#include "search/vector_game.h"

namespace sixmax {

// Standard 3-card Kuhn poker as a VectorGame. History alphabet:
//   'p' = pass/check, 'b' = bet/call. Terminals:
//   "pp"  showdown, stakes 1
//   "pbp" P0 folded            -> u0 = -1
//   "pbb" showdown, stakes 2
//   "bp"  P1 folded            -> u0 = +1
//   "bb"  showdown, stakes 2
class KuhnVectorState : public VectorState {
public:
    explicit KuhnVectorState(std::string history) : h_(std::move(history)) {}
    int num_players() const override { return 2; }
    int actor() const override { return h_.size() % 2; }      // P0 first
    bool is_leaf() const override {
        return h_ == "pp" || h_ == "pbp" || h_ == "pbb" || h_ == "bp" || h_ == "bb";
    }
    int num_actions() const override { return 2; }            // 0 = pass, 1 = bet
    void legal_mask(std::vector<uint8_t>& m) const override { m.assign({1, 1}); }
    std::unique_ptr<VectorState> apply(int action) const override {
        return std::make_unique<KuhnVectorState>(h_ + (action == 1 ? 'b' : 'p'));
    }
    double utility(int player, int h0, int h1) const override;
    uint64_t id() const override;                              // history code
private:
    std::string h_;
};

class KuhnVectorGame : public VectorGame {
public:
    int num_hands(int) const override { return 3; }
    bool compatible(int h0, int h1) const override { return h0 != h1; }
    std::vector<double> prior_reach(int) const override { return {1.0/3, 1.0/3, 1.0/3}; }
    std::unique_ptr<VectorState> root() const override {
        return std::make_unique<KuhnVectorState>("");
    }
};

}  // namespace sixmax
