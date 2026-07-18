#include "game/safe_eval.h"
#include "game/card.h"

namespace safe_eval {

HandRank rank7(const std::array<int, 7>& codes) {
    std::array<Card, 7> cards;
    for (int i = 0; i < 7; ++i)
        cards[i] = codes[i];  // Card = int; code = (rank-2)*4 + suit matches project convention
    return HandRank(evaluate_7card(cards));
}

}  // namespace safe_eval
