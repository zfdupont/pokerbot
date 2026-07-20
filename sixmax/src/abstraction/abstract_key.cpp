#include "abstraction/abstract_key.h"

namespace sixmax {

int pot_bucket(double pot_bb) {
    if (pot_bb <= 7.0) return 0;
    if (pot_bb <= 15.0) return 1;
    if (pot_bb <= 40.0) return 2;
    return 3;
}

uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after) {
    uint64_t key = (uint64_t)card;                       // bits 0-7
    key |= (uint64_t)street << 8;                        // bits 8-9
    for (int st = 0; st < 4; ++st)
        key |= (uint64_t)raises[st] << (10 + 2 * st);    // bits 10-17
    key |= (uint64_t)pot_bucket(pot_bb) << 18;           // bits 18-19
    key |= (uint64_t)live << 20;                         // bits 20-22
    key |= (uint64_t)after << 23;                        // bits 23-25
    return key;
}

}  // namespace sixmax
