#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sixmax {

// Exact equity vs one uniformly random opponent hand on the river, with ties
// counting 0.5. Board must have 5 cards. Enumerates all 990 non-colliding
// opponent hole-card combos through safe_eval::rank7.
double river_equity_exact(const std::array<int, 2>& hole,
                          const std::vector<int>& board);

// Dense river table indexed by HandIndexer::index(3, hole, board).
//
// Built per canonical board: for each board, the 1081 hole-pair ranks are
// computed once, sorted, and each pair's win/tie count is corrected for the 91
// opponent pairs that share one of its cards. That is ~145M rank7 calls instead
// of the ~1.2e11 a per-canonical-hand enumeration would need.
//
// num_threads <= 0 means std::thread::hardware_concurrency().
std::vector<float> build_river_equity(int num_threads);

// Exact equity for street 1 (flop) or 2 (turn) from the next street's dense
// table: the mean over every next card not in hole/board of
// next_street[index(street + 1, hole, board + card)].
std::vector<float> derive_equity(int street,
                                 const std::vector<float>& next_street,
                                 int num_threads);

// Exact equity for one hand, computed recursively from river_equity_exact.
// Test helper: a turn hand costs 46 river evaluations, a flop hand 47*46.
double derive_equity_one(int street, const std::array<int, 2>& hole,
                         const std::vector<int>& board);

// Feature store: eq[0] = flop, eq[1] = turn, eq[2] = river, each a dense
// float32 array addressed by the corresponding HandIndexer index.
struct EquityStore {
    std::array<std::vector<float>, 3> eq;
    uint64_t content_hash = 0;
};

uint64_t equity_content_hash(const EquityStore& store);

// Atomic save (.tmp + rename, like save_blueprint). strict_sizes verifies each
// array against the indexer size; test-only stores may pass false.
void save_equity_store(const std::string& path, const EquityStore& store,
                       bool strict_sizes = true);

// Strict load: magic, hand-index version, exact indexer sizes, content hash.
EquityStore load_equity_store(const std::string& path);

// Same checks except the indexer-size equality, for tiny test stores. Magic,
// version and content hash are still enforced.
EquityStore load_equity_store_skip_size_check(const std::string& path);

// One entry without loading the whole file (test helper; does not verify the
// content hash). Bounds-checks against the sizes stored in the header.
float equity_store_lookup(const std::string& path, int street, uint64_t idx);

// Test helper: equities for every hole pair on one river board, in pair-index
// order over the 47 remaining cards (see pair_index in equity_table.cpp).
std::vector<float> river_equities_for_board(
    const std::vector<int>& board);

}  // namespace sixmax
