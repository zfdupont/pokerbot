#pragma once
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace sixmax {

// Abstract sequential game for MCCFR. Chance (the deal) happens once in
// Game::new_hand(); states after that are deterministic in applied actions.
class GameState {
public:
    virtual ~GameState() = default;
    virtual bool is_terminal() const = 0;
    virtual int current_player() const = 0;      // valid iff !is_terminal()
    // mask is assigned to size Game::num_actions(); 1 = legal.
    virtual void legal_mask(std::vector<uint8_t>& mask) const = 0;
    virtual uint64_t infoset_key() const = 0;    // for current_player()
    virtual void apply(int action) = 0;          // action must be legal
    virtual double utility(int player) const = 0;  // BB; valid iff terminal
    virtual std::unique_ptr<GameState> clone() const = 0;
};

class Game {
public:
    virtual ~Game() = default;
    virtual int num_players() const = 0;
    virtual int num_actions() const = 0;         // fixed action-vector width
    virtual std::unique_ptr<GameState> new_hand(std::mt19937_64& rng) = 0;
};

}  // namespace sixmax
