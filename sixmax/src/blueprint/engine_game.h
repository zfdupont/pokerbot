#pragma once
#include "blueprint/game.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

// Adapts HandState + ActionVocab to the MCCFR Game interface. Legality is
// masking only — vocab order is never filtered or reordered. infoset_key()
// is a NAIVE exact-information hash (hole cards + board + action history);
// Phase 1b replaces it with the card/history abstraction. Everything else
// here (masking, translation) is final.
class EngineGameState : public GameState {
public:
    EngineGameState(HandState hand, const ActionVocab* vocab)
        : hand_(std::move(hand)), vocab_(vocab) {}
    bool is_terminal() const override { return hand_.is_terminal(); }
    int current_player() const override { return hand_.current_player(); }
    void legal_mask(std::vector<uint8_t>& mask) const override;
    uint64_t infoset_key() const override;
    void apply(int action) override;
    double utility(int player) const override {
        return hand_.payoffs()[player];
    }
    std::unique_ptr<GameState> clone() const override {
        return std::make_unique<EngineGameState>(*this);
    }
    BetContext bet_context() const;  // exposed for tests and Phase 2 search

private:
    bool size_class_ok(const AbstractAction& a) const;
    HandState hand_;
    const ActionVocab* vocab_;
    std::vector<int> history_;  // applied vocab indices, all seats (public)
};

class EngineGame : public Game {
public:
    EngineGame(EngineConfig cfg, const ActionVocab* vocab)
        : cfg_(cfg), vocab_(vocab) {}
    int num_players() const override { return cfg_.num_players; }
    int num_actions() const override { return vocab_->size(); }
    std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) override;

private:
    EngineConfig cfg_;
    const ActionVocab* vocab_;
    int button_ = 0;  // rotates every hand for positional balance
};

}  // namespace sixmax
