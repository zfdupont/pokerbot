#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace sixmax {

// Version of the suit-canonical hand index. The indexer's output is part of
// the bucket-map / feature-store artifact contract: bump this whenever the
// mapping changes, because existing tables would no longer be addressable.
constexpr uint32_t kHandIndexVersion = 1;

// Bijection between (hole, board) on a postflop street and a dense index.
//
// Hands that are identical up to a relabeling of the four suits, and up to
// ordering within each betting round (hole, flop, turn, river), share one
// index. Street is 1=flop, 2=turn, 3=river (matching
// Abstraction::num_buckets); board sizes are 3, 4 and 5 respectively.
//
// Implemented with Waugh 2013, "A Fast and Optimal Hand Isomorphism
// Algorithm", using rounds {2,3} / {2,4} / {2,5}: the hole is one round and
// the whole board is one unordered round. See hand_index.cpp for the
// upstream copyright notice.
class HandIndexer {
public:
    static const HandIndexer& instance();

    uint64_t size(int street) const;
    uint64_t index(int street, const std::array<int, 2>& hole,
                   const std::vector<int>& board) const;
    void unindex(int street, uint64_t idx, std::array<int, 2>& hole,
                 std::vector<int>& board) const;
    // Number of raw (hole, board) deals this canonical class stands for,
    // with card order within each round irrelevant.
    uint32_t multiplicity(int street, uint64_t idx) const;

private:
    HandIndexer() = default;
};

}  // namespace sixmax
