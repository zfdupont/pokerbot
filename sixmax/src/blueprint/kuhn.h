#pragma once
#include "blueprint/game.h"

namespace sixmax {

// Kuhn poker: cards {0=J,1=Q,2=K}, ante 1 BB each, bet size 1 BB.
// Actions: 0 = check/fold ("pass"), 1 = bet/call. Closed-form equilibrium:
// P0 bets J with alpha in [0,1/3], K with 3*alpha, never Q; value -1/18.
class KuhnState : public GameState {
public:
    KuhnState(int card0, int card1) { cards_[0] = card0; cards_[1] = card1; }
    bool is_terminal() const override;
    int current_player() const override { return (int)history_.size() % 2; }
    void legal_mask(std::vector<uint8_t>& mask) const override { mask.assign(2, 1); }
    uint64_t infoset_key() const override;
    void apply(int action) override { history_.push_back(action); }
    double utility(int player) const override;
    std::unique_ptr<GameState> clone() const override {
        return std::make_unique<KuhnState>(*this);
    }
    // History codes for keys: 0="" 1="check" 2="bet" 3="check,bet".
    static uint64_t key_for(int card, int history_code) {
        return (uint64_t)card * 4 + (uint64_t)history_code;
    }

private:
    int cards_[2];
    std::vector<int> history_;
};

class KuhnGame : public Game {
public:
    int num_players() const override { return 2; }
    int num_actions() const override { return 2; }
    std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) override;
};

}  // namespace sixmax
