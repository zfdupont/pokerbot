#include "abstraction/abstract_key.h"
#include <stdexcept>

namespace sixmax {

namespace {

constexpr int kLegacyPotShift = 18;
constexpr int kPotShift = 26;

}  // namespace

int pot_bucket(double pot_bb) {
    if (pot_bb <= 7.0) return 0;
    if (pot_bb <= 15.0) return 1;
    if (pot_bb <= 40.0) return 2;
    return 3;
}

bool PotLayout::is_legacy() const {
    return mode == PotMode::BB && edges.size() == 3 && edges[0] == 7.0 &&
           edges[1] == 15.0 && edges[2] == 40.0;
}

int PotLayout::num_buckets() const { return (int)edges.size() + 1; }

void validate_pot_layout(const PotLayout& layout) {
    const auto& edges = layout.edges;
    if (edges.empty())
        throw std::invalid_argument("pot_edges must not be empty");
    if (edges.size() > 15)
        throw std::invalid_argument("pot_edges must have at most 15 entries");
    for (size_t i = 0; i < edges.size(); ++i) {
        if (!(edges[i] > 0.0))
            throw std::invalid_argument("pot_edges must be > 0");
        if (i > 0 && !(edges[i] > edges[i - 1]))
            throw std::invalid_argument(
                "pot_edges must be strictly increasing");
    }
}

int pot_bucket(const PotLayout& layout, double pot_bb, double eff_stack_bb) {
    double value = pot_bb;
    if (layout.mode == PotMode::SPR) {
        // Blinds make pot_bb > 0, but never divide by zero if a caller
        // constructs a degenerate state.
        if (pot_bb <= 0.0)
            return eff_stack_bb > 0.0 ? (int)layout.edges.size() : 0;
        value = eff_stack_bb / pot_bb;
    }
    for (size_t i = 0; i < layout.edges.size(); ++i)
        if (value <= layout.edges[i]) return (int)i;
    return (int)layout.edges.size();
}

uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           int pot_idx, bool legacy_layout, int live,
                           int after) {
    uint64_t key = (uint64_t)card;                       // bits 0-7
    key |= (uint64_t)street << 8;                        // bits 8-9
    for (int st = 0; st < 4; ++st)
        key |= (uint64_t)raises[st] << (10 + 2 * st);    // bits 10-17
    key |= (uint64_t)pot_idx
           << (legacy_layout ? kLegacyPotShift : kPotShift);
    key |= (uint64_t)live << 20;                         // bits 20-22
    key |= (uint64_t)after << 23;                        // bits 23-25
    return key;
}

uint64_t pack_abstract_key(int card, int street,
                           const std::array<uint8_t, 4>& raises,
                           double pot_bb, int live, int after) {
    return pack_abstract_key(card, street, raises, pot_bucket(pot_bb), true,
                             live, after);
}

}  // namespace sixmax
