#pragma once
#include "abstraction/abstraction.h"
#include "blueprint/game.h"
#include "engine/engine.h"
#include "vocab/vocab.h"

namespace sixmax {

// Adapts HandState + ActionVocab to the MCCFR Game interface. Legality is
// masking only — vocab order is never filtered or reordered. infoset_key()
// delegates to abstract_key() (bit-packed card/history abstraction) when an
// Abstraction is attached; otherwise it falls back to a NAIVE exact-information
// FNV-1a hash (hole cards + board + action history). Everything else here
// (masking, translation) is final.
class EngineGameState : public GameState {
public:
    EngineGameState(HandState hand, const ActionVocab* vocab,
                    const Abstraction* abstraction = nullptr)
        : hand_(std::move(hand)), vocab_(vocab), abstraction_(abstraction) {}
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
    // Bit-packed abstraction key (layout in the Phase 1b plan). Public and
    // parameterized so eval-time strategies with their own abstractions can
    // key the same public state independently.
    uint64_t abstract_key(const Abstraction& abs) const;
    BetContext bet_context() const;  // exposed for tests and Phase 2 search

private:
    bool size_class_ok(const AbstractAction& a) const;
    HandState hand_;
    const ActionVocab* vocab_;
    const Abstraction* abstraction_ = nullptr;
    std::vector<int> history_;          // naive-keyer input (unchanged)
    std::array<uint8_t, 4> raises_{};   // per-street raise count, capped 3
};

class EngineGame : public Game {
public:
    EngineGame(EngineConfig cfg, const ActionVocab* vocab,
               const Abstraction* abstraction = nullptr)
        : cfg_(cfg), vocab_(vocab), abstraction_(abstraction) {}
    int num_players() const override { return cfg_.num_players; }
    int num_actions() const override { return vocab_->size(); }
    std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) override;

private:
    EngineConfig cfg_;
    const ActionVocab* vocab_;
    const Abstraction* abstraction_ = nullptr;
    int button_ = 0;  // rotates every hand for positional balance
};

}  // namespace sixmax
