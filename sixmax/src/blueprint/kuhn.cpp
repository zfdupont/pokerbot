#include "blueprint/kuhn.h"

namespace sixmax {

bool KuhnState::is_terminal() const {
    const auto& h = history_;
    if (h.size() < 2) return false;
    if (h.size() == 2) return !(h[0] == 0 && h[1] == 1);  // "check,bet" continues
    return true;  // length 3: "check,bet,{fold|call}"
}

uint64_t KuhnState::infoset_key() const {
    int code = 0;
    if (history_.size() == 1) code = history_[0] == 0 ? 1 : 2;
    else if (history_.size() == 2) code = 3;  // must be "check,bet"
    return key_for(cards_[current_player()], code);
}

double KuhnState::utility(int player) const {
    const auto& h = history_;
    int winner = cards_[0] > cards_[1] ? 0 : 1;
    double p0;
    if (h.size() == 2 && h[0] == 0 && h[1] == 0) p0 = winner == 0 ? 1 : -1;  // cc
    else if (h.size() == 2 && h[0] == 1 && h[1] == 0) p0 = 1;                // b,f
    else if (h.size() == 2) p0 = winner == 0 ? 2 : -2;                       // b,c
    else if (h[2] == 0) p0 = -1;                                             // c,b,f
    else p0 = winner == 0 ? 2 : -2;                                          // c,b,c
    return player == 0 ? p0 : -p0;
}

std::unique_ptr<GameState> KuhnGame::new_hand(std::mt19937_64& rng) {
    int c0 = (int)(rng() % 3);
    int c1 = (int)(rng() % 2);
    if (c1 >= c0) ++c1;  // uniform ordered pair without replacement
    return std::make_unique<KuhnState>(c0, c1);
}

}  // namespace sixmax
