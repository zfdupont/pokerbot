#include "blueprint/engine_game.h"
#include "abstraction/abstract_key.h"

namespace sixmax {

BetContext EngineGameState::bet_context() const {
    const PlayerState& p = hand_.player(hand_.current_player());
    double owe = hand_.current_bet() - p.street_bet;  // uncapped
    return BetContext{hand_.pot() - owe, hand_.current_bet(), owe,
                      p.street_bet + p.stack};
}

bool EngineGameState::size_class_ok(const AbstractAction& a) const {
    bool unopened_preflop = hand_.street() == Street::Preflop &&
                            hand_.current_bet() <= 1.0 + kChipEps;
    // BB-unit sizes are preflop opens; Pot-unit sizes are everything after.
    return (a.unit == SizeUnit::BB) == unopened_preflop;
}

void EngineGameState::legal_mask(std::vector<uint8_t>& mask) const {
    int n = vocab_->size();
    mask.assign(n, 0);
    BetContext ctx = bet_context();
    bool facing = ctx.to_call > kChipEps;
    for (int i = 0; i < n; ++i) {
        const AbstractAction& a = vocab_->at(i);
        switch (a.type) {
            case ActionType::Fold:
                mask[i] = facing ? 1 : 0;
                break;
            case ActionType::Check:
                mask[i] = facing ? 0 : 1;
                break;
            case ActionType::Call:
                mask[i] = facing ? 1 : 0;
                break;
            case ActionType::Bet: {
                if (!hand_.can_raise() || !size_class_ok(a)) break;
                double t = vocab_->target_bb(i, ctx);
                // ctx.stack is the all-in TARGET (street_bet + remaining chips),
                // not the player's remaining chips; t < ctx.stack guards that
                // this Bet entry does not collapse into the AllIn entry.
                mask[i] = (t >= hand_.min_raise_to() - kChipEps &&
                           t < ctx.stack - kChipEps)
                              ? 1
                              : 0;  // jam entry owns the stack-off
                break;
            }
            case ActionType::AllIn:
                mask[i] = hand_.can_raise() ? 1 : 0;
                break;
        }
    }
}

uint64_t EngineGameState::infoset_key() const {
    if (abstraction_) return abstract_key(*abstraction_);
    // FNV-1a over exact private+public information (naive fallback,
    // used by tests and any vocab-only construction).
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    int p = hand_.current_player();
    auto hc = hand_.hole_cards(p);
    mix((uint64_t)std::min(hc[0], hc[1]));
    mix((uint64_t)std::max(hc[0], hc[1]));
    // +64 shifts board-card codes out of the hole-card 0–51 range, preventing
    // hash collisions between a board card and a hole card with the same code.
    for (int c : hand_.board()) mix((uint64_t)(c + 64));
    mix(0xFFFFull);  // separator: board cards vs action history
    for (int a : history_) mix((uint64_t)(a + 128));
    return h;
}

void EngineGameState::apply(int action) {
    const AbstractAction& a = vocab_->at(action);
    switch (a.type) {
        case ActionType::Fold:
            hand_.apply({EngineActionType::Fold, 0.0});
            break;
        case ActionType::Check:
        case ActionType::Call:
            hand_.apply({EngineActionType::CheckCall, 0.0});
            break;
        case ActionType::Bet:
        case ActionType::AllIn: {
            int st = (int)hand_.street();
            ++raises_[st];
            double target = vocab_->target_bb(action, bet_context());
            hand_.apply({EngineActionType::RaiseTo, target});
            break;
        }
    }
    history_.push_back(action);
}

uint64_t EngineGameState::abstract_key(const Abstraction& abs) const {
    const int p = hand_.current_player();
    const int n = hand_.num_players();
    const int street = (int)hand_.street();
    auto hole = hand_.hole_cards(p);
    const int card = street == 0 ? preflop_class(hole)
                                 : abs.bucket(hole, hand_.board());
    // Canonical action-order start: preflop = seat after BB (HU: button);
    // postflop = seat after button.
    const int start = street == 0
        ? (n == 2 ? hand_.button() : (hand_.button() + 3) % n)
        : (hand_.button() + 1) % n;
    auto order = [&](int seat) { return (seat - start + n) % n; };
    int live = 0, after = 0;
    for (int s = 0; s < n; ++s) {
        if (s == p || hand_.player(s).folded) continue;
        ++live;
        if (!hand_.player(s).all_in && order(s) > order(p)) ++after;
    }
    std::array<uint8_t, 4> capped_raises;
    for (int i = 0; i < 4; ++i)
        capped_raises[i] = std::min(raises_[i], uint8_t(3));
    uint64_t key = pack_abstract_key((int)card, street, capped_raises,
                                     hand_.pot(), live, after);
    return key;
}

std::unique_ptr<GameState> EngineGame::new_hand(std::mt19937_64& rng) {
    button_ = (button_ + 1) % cfg_.num_players;
    return std::make_unique<EngineGameState>(
        HandState::deal(cfg_, button_, rng), vocab_, abstraction_);
}

}  // namespace sixmax
