#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace sixmax {

// All chip quantities are doubles denominated in big blinds:
// big_blind = 1.0, small_blind = 0.5. No absolute-chip frame anywhere.
constexpr double kChipEps = 1e-9;

struct EngineConfig {
    int num_players = 6;             // 2..6
    double starting_stack = 100.0;   // BB; per-seat override via HandState ctor
};

enum class Street { Preflop = 0, Flop = 1, Turn = 2, River = 3 };

enum class EngineActionType { Fold, CheckCall, RaiseTo };
struct EngineAction {
    EngineActionType type;
    double amount = 0.0;  // RaiseTo: total street commitment ("raise to"), BB
};

struct PlayerState {
    double stack = 0.0;        // remaining behind
    double street_bet = 0.0;   // committed this street
    double total_bet = 0.0;    // committed this hand
    bool folded = false;
    bool all_in = false;
};

// Pure side-pot settlement from total contributions. rank_order[i] is the
// dense showdown rank of player i (0 = best, ties share a value); entries
// for folded players are ignored. Returns gross payouts (winnings only).
std::vector<double> settle_pots(const std::vector<double>& total_bets,
                                const std::vector<uint8_t>& folded,
                                const std::vector<int>& rank_order);

class HandState {
public:
    // deck: >= 2n+5 card codes, code = (rank-2)*4 + suit. Deal order:
    // player i holds deck[2i], deck[2i+1]; board = deck[2n..2n+4]; no burn.
    // stacks: per-seat starting stacks; empty = cfg.starting_stack for all.
    // Precondition: every stack > 1.0 BB (blind posting cannot go all-in).
    HandState(const EngineConfig& cfg, int button, std::vector<int> deck,
              std::vector<double> stacks = {});
    static HandState deal(const EngineConfig& cfg, int button,
                          std::mt19937_64& rng);

    bool is_terminal() const { return terminal_; }
    int current_player() const { return next_; }
    Street street() const { return street_; }
    int num_players() const { return (int)players_.size(); }
    int button() const { return button_; }
    double pot() const;                // total contributed by everyone
    double current_bet() const { return current_bet_; }
    double to_call() const;            // current player's owe, capped by stack
    double min_raise_to() const { return current_bet_ + last_raise_; }
    bool can_raise() const;            // current player may bet/raise
    const PlayerState& player(int i) const { return players_[i]; }
    std::array<int, 2> hole_cards(int i) const;
    std::vector<int> board() const;    // cards revealed so far (0/3/4/5)

    void apply(const EngineAction& a);
    // Net result per player (final stack minus starting stack), BB.
    // Valid iff is_terminal().
    const std::vector<double>& payoffs() const { return payoffs_; }

private:
    void commit(int seat, double amount);
    void finish_action(int seat);
    void close_round_or_advance();
    int next_active_after(int seat) const;  // next seat that can still act
    int num_can_act() const;
    std::vector<int> showdown_order() const;
    void settle();

    EngineConfig cfg_;
    int button_;
    std::vector<int> deck_;
    std::vector<PlayerState> players_;
    std::vector<double> start_stacks_;
    Street street_ = Street::Preflop;
    int next_ = -1;
    int to_act_ = 0;               // players still owed an action this round
    double current_bet_ = 0.0;     // highest street_bet on the table
    double last_raise_ = 1.0;      // last raise increment (min-raise basis)
    bool terminal_ = false;
    std::vector<double> payoffs_;
};

}  // namespace sixmax
