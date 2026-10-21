#include "abstraction/equity_table.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "abstraction/hand_index.h"
#include "game/safe_eval.h"

namespace sixmax {
namespace {

constexpr uint32_t kCards = 52;
constexpr uint32_t kRiverBoardCards = 5;
constexpr int kRiverHolePairs = 47 * 46 / 2;  // 1081
constexpr int kRiverOpponentPairs = 45 * 44 / 2;  // 990

int resolve_threads(int num_threads) {
    if (num_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw == 0 ? 1 : static_cast<int>(hw);
    }
    return std::max(1, num_threads);
}

template <typename Fn>
void run_parallel(uint64_t total, int num_threads, const char* label,
                  Fn&& fn) {
    if (total == 0) return;
    const int threads = resolve_threads(num_threads);
    const uint64_t report_every = std::max<uint64_t>(1, total / 100);
    std::atomic<uint64_t> next{0};
    std::atomic<uint64_t> done{0};
    std::exception_ptr error;
    std::mutex error_mutex;

    auto worker = [&] {
        for (;;) {
            const uint64_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= total) break;
            try {
                fn(i);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                }
                next.store(total, std::memory_order_relaxed);
                break;
            }
            const uint64_t d =
                done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (d == total || d % report_every == 0) {
                std::fprintf(stderr, "[equity] %s %3llu%% (%llu/%llu)\n",
                             label,
                             static_cast<unsigned long long>(d * 100 / total),
                             static_cast<unsigned long long>(d),
                             static_cast<unsigned long long>(total));
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) workers.emplace_back(worker);
    for (auto& w : workers) w.join();
    if (error) std::rethrow_exception(error);
}

void validate_river_board(const std::vector<int>& board) {
    if (board.size() != kRiverBoardCards)
        throw std::invalid_argument("river board must have 5 cards");
}

// Validate 5 board cards and return the 47-card remaining deck in ascending
// card order. The caller has already checked the board size.
void remaining_cards(const std::array<int, 5>& board,
                     std::array<int, 47>& out) {
    bool used[kCards] = {false};
    for (int c : board) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument("card out of range");
        if (used[c]) throw std::invalid_argument("duplicate card");
        used[c] = true;
    }
    int n = 0;
    for (int c = 0; c < static_cast<int>(kCards); ++c)
        if (!used[c]) out[n++] = c;
}

void validate_distinct_cards(const std::array<int, 2>& hole,
                             const std::array<int, 5>& board) {
    bool used[kCards] = {false};
    for (int c : {hole[0], hole[1]}) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument("card out of range");
        if (used[c]) throw std::invalid_argument("duplicate card");
        used[c] = true;
    }
    for (int c : board) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument("card out of range");
        if (used[c]) throw std::invalid_argument("duplicate card");
        used[c] = true;
    }
}

// Index of pair (i, j), i < j, in the nested-loop order over 47 cards.
inline int pair_index(int i, int j) {
    return 46 * i - i * (i - 1) / 2 + (j - i - 1);
}

const std::array<std::array<int, 4>, 24>& suit_permutations() {
    static const auto perms = [] {
        std::array<std::array<int, 4>, 24> out{};
        std::array<int, 4> p{0, 1, 2, 3};
        int n = 0;
        do {
            out[n++] = p;
        } while (std::next_permutation(p.begin(), p.end()));
        return out;
    }();
    return perms;
}

// True when `board` (sorted ascending) is the lexicographically smallest
// representative of its orbit under the 24 suit relabelings.
bool is_canonical_board(const std::array<int, 5>& board) {
    std::array<int, 5> best = board;
    std::array<int, 5> transformed{};
    for (const auto& perm : suit_permutations()) {
        for (int i = 0; i < 5; ++i) {
            const int c = board[i];
            transformed[i] = (c / 4) * 4 + perm[c % 4];
        }
        std::sort(transformed.begin(), transformed.end());
        if (transformed < best) best = transformed;
    }
    return best == board;
}

// One representative for every 5-card board orbit under suit relabeling.
// Distinct board orbits cover disjoint joint (hole, board) hand classes, so
// processing all holes on these boards fills every river index exactly.
std::vector<std::array<int, 5>> canonical_river_boards() {
    std::vector<std::array<int, 5>> boards;
    boards.reserve(120000);
    for (int a = 0; a < 48; ++a)
        for (int b = a + 1; b < 49; ++b)
            for (int c = b + 1; c < 50; ++c)
                for (int d = c + 1; d < 51; ++d)
                    for (int e = d + 1; e < 52; ++e) {
                        const std::array<int, 5> board{{a, b, c, d, e}};
                        if (is_canonical_board(board)) boards.push_back(board);
                    }
    return boards;
}

// Fill `out` (size 1081) with the exact equity of every hole pair on `board`,
// in pair_index order over the 47 remaining cards.
void compute_board_equities(const std::array<int, 5>& board,
                            std::vector<float>& out) {
    std::array<int, 47> remaining{};
    remaining_cards(board, remaining);

    std::vector<safe_eval::HandRank> ranks;
    ranks.reserve(kRiverHolePairs);
    for (int i = 0; i < 47; ++i)
        for (int j = i + 1; j < 47; ++j) {
            const std::array<int, 7> cards{{board[0], board[1], board[2],
                                            board[3], board[4], remaining[i],
                                            remaining[j]}};
            ranks.push_back(safe_eval::rank7(cards));
        }

    const auto less = [](const safe_eval::HandRank& a,
                         const safe_eval::HandRank& b) { return a.beats(b); };
    std::vector<safe_eval::HandRank> sorted = ranks;
    std::sort(sorted.begin(), sorted.end(), less);

    out.resize(kRiverHolePairs);
    for (int i = 0; i < 47; ++i) {
        for (int j = i + 1; j < 47; ++j) {
            const safe_eval::HandRank& mine = ranks[pair_index(i, j)];
            const auto lo =
                std::lower_bound(sorted.begin(), sorted.end(), mine, less);
            const auto hi = std::upper_bound(lo, sorted.end(), mine, less);
            // Counts over all 1081 hole pairs, including the 91 that are not
            // legal opponents for this hole.
            const uint32_t all_better =
                static_cast<uint32_t>(lo - sorted.begin());
            const uint32_t all_ties = static_cast<uint32_t>(hi - lo);

            uint32_t removed_better = 0;
            uint32_t removed_ties = 0;
            for (int k = 0; k < 47; ++k) {
                if (k == i || k == j) continue;
                const safe_eval::HandRank& ri =
                    ranks[pair_index(std::min(i, k), std::max(i, k))];
                if (ri.beats(mine)) ++removed_better;
                else if (ri.ties(mine)) ++removed_ties;
                const safe_eval::HandRank& rj =
                    ranks[pair_index(std::min(j, k), std::max(j, k))];
                if (rj.beats(mine)) ++removed_better;
                else if (rj.ties(mine)) ++removed_ties;
            }
            ++removed_ties;  // the (i, j) pair itself is also removed

            const uint32_t losses = all_better - removed_better;
            const uint32_t ties = all_ties - removed_ties;
            const uint32_t wins =
                kRiverOpponentPairs - losses - ties;
            out[pair_index(i, j)] = static_cast<float>(
                (wins + 0.5 * ties) / kRiverOpponentPairs);
        }
    }
}

std::vector<std::array<int, 5>> board_from_vector(const std::vector<int>& b) {
    validate_river_board(b);
    return {std::array<int, 5>{{b[0], b[1], b[2], b[3], b[4]}}};
}

// ---------------------------------------------------------------------------
// Store format: magic[8] version[4] sizes[3*8] content_hash[8] then the three
// dense float arrays in flop/turn/river order.
// ---------------------------------------------------------------------------

constexpr char kMagic[8] = {'S', 'I', 'X', 'E', 'Q', '0', '0', '1'};
constexpr std::streamoff kHeaderBytes = 8 + 4 + 8 * 3 + 8;

template <typename T>
void put(std::ofstream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
T get(std::ifstream& i) {
    T v;
    i.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!i) throw std::runtime_error("equity store: truncated file");
    return v;
}

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hash_mix(uint64_t& h, uint64_t v) {
    h ^= v;
    h *= kFnvPrime;
}

uint64_t hash_floats(uint64_t h, const std::vector<float>& values) {
    hash_mix(h, static_cast<uint64_t>(values.size()));
    const auto* bytes =
        reinterpret_cast<const unsigned char*>(values.data());
    const size_t n = values.size() * sizeof(float);
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= n; i += sizeof(uint64_t)) {
        uint64_t word = 0;
        std::memcpy(&word, bytes + i, sizeof word);
        hash_mix(h, word);
    }
    if (i < n) {
        uint64_t word = 0;
        std::memcpy(&word, bytes + i, n - i);
        hash_mix(h, word);
    }
    return h;
}

void read_header(std::ifstream& i, const std::string& path,
                 std::array<uint64_t, 3>& sizes, uint64_t& stored_hash) {
    char magic[8];
    i.read(magic, 8);
    if (!i || std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("not an equity store: " + path);

    const uint32_t version = get<uint32_t>(i);
    if (version != kHandIndexVersion)
        throw std::runtime_error(
            "equity store hand-index version mismatch: file has " +
            std::to_string(version) + ", expected " +
            std::to_string(kHandIndexVersion));

    for (auto& s : sizes) s = get<uint64_t>(i);
    stored_hash = get<uint64_t>(i);
}

EquityStore load_impl(const std::string& path, bool check_sizes) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open " + path);

    std::array<uint64_t, 3> sizes{};
    uint64_t stored_hash = 0;
    read_header(i, path, sizes, stored_hash);

    if (check_sizes) {
        for (int street = 1; street <= 3; ++street) {
            const uint64_t expected = HandIndexer::instance().size(street);
            if (sizes[static_cast<size_t>(street - 1)] != expected)
                throw std::runtime_error(
                    "equity store size mismatch on street " +
                    std::to_string(street) + ": file has " +
                    std::to_string(sizes[static_cast<size_t>(street - 1)]) +
                    ", expected " + std::to_string(expected));
        }
    }

    // Refuse a corrupt size before allocating; a truncated file must not
    // trigger a huge allocation.
    i.seekg(0, std::ios::end);
    const std::streamoff file_size = i.tellg();
    const uint64_t max_entries =
        file_size < kHeaderBytes
            ? 0
            : static_cast<uint64_t>((file_size - kHeaderBytes) /
                                    static_cast<std::streamoff>(sizeof(float)));
    uint64_t total_entries = 0;
    for (uint64_t s : sizes) {
        if (s > max_entries || total_entries > max_entries - s)
            throw std::runtime_error("equity store: truncated file " + path);
        total_entries += s;
    }
    i.seekg(kHeaderBytes);

    EquityStore out;
    for (int k = 0; k < 3; ++k) {
        out.eq[static_cast<size_t>(k)].resize(
            static_cast<size_t>(sizes[static_cast<size_t>(k)]));
        auto& values = out.eq[static_cast<size_t>(k)];
        i.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!i) throw std::runtime_error("equity store: truncated file " + path);
    }

    out.content_hash = equity_content_hash(out);
    if (out.content_hash != stored_hash)
        throw std::runtime_error("equity store content hash mismatch: " + path);
    return out;
}

}  // namespace

double river_equity_exact(const std::array<int, 2>& hole,
                          const std::vector<int>& board) {
    validate_river_board(board);
    const std::array<int, 5> board_arr{
        {board[0], board[1], board[2], board[3], board[4]}};
    validate_distinct_cards(hole, board_arr);

    bool used[kCards] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int c : board_arr) used[c] = true;

    const std::array<int, 7> mine_cards{{hole[0], hole[1], board_arr[0],
                                         board_arr[1], board_arr[2],
                                         board_arr[3], board_arr[4]}};
    const safe_eval::HandRank mine = safe_eval::rank7(mine_cards);

    uint32_t wins = 0;
    uint32_t ties = 0;
    for (int a = 0; a < static_cast<int>(kCards); ++a) {
        if (used[a]) continue;
        for (int b = a + 1; b < static_cast<int>(kCards); ++b) {
            if (used[b]) continue;
            const std::array<int, 7> theirs_cards{
                {a, b, board_arr[0], board_arr[1], board_arr[2],
                 board_arr[3], board_arr[4]}};
            const safe_eval::HandRank theirs =
                safe_eval::rank7(theirs_cards);
            if (mine.beats(theirs)) ++wins;
            else if (mine.ties(theirs)) ++ties;
        }
    }
    return (wins + 0.5 * ties) / kRiverOpponentPairs;
}

std::vector<float> river_equities_for_board(const std::vector<int>& board) {
    const auto boards = board_from_vector(board);
    std::vector<float> out;
    compute_board_equities(boards[0], out);
    return out;
}

std::vector<float> build_river_equity(int num_threads) {
    const auto& indexer = HandIndexer::instance();
    std::vector<float> out(
        static_cast<size_t>(indexer.size(3)),
        std::numeric_limits<float>::quiet_NaN());
    const auto boards = canonical_river_boards();

    run_parallel(boards.size(), num_threads, "river",
                 [&](uint64_t board_idx) {
                     const auto& board = boards[static_cast<size_t>(board_idx)];
                     std::vector<float> equities;
                     compute_board_equities(board, equities);

                     std::array<int, 47> remaining{};
                     remaining_cards(board, remaining);
                     std::vector<int> board_vec(board.begin(), board.end());

                     size_t p = 0;
                     for (int i = 0; i < 47; ++i)
                         for (int j = i + 1; j < 47; ++j) {
                             const std::array<int, 2> hole{
                                 {remaining[i], remaining[j]}};
                             out[static_cast<size_t>(
                                 indexer.index(3, hole, board_vec))] =
                                 equities[p++];
                         }
                 });

    for (size_t i = 0; i < out.size(); ++i) {
        if (std::isnan(out[i]))
            throw std::runtime_error(
                "river equity build left index " + std::to_string(i) +
                " unfilled");
    }
    return out;
}

std::vector<float> derive_equity(int street,
                                 const std::vector<float>& next_street,
                                 int num_threads) {
    if (street != 1 && street != 2)
        throw std::invalid_argument(
            "derive_equity: street must be 1 (flop) or 2 (turn)");
    const auto& indexer = HandIndexer::instance();
    const uint64_t street_size = indexer.size(street);
    if (next_street.size() != indexer.size(street + 1))
        throw std::invalid_argument(
            "derive_equity: next-street table size does not match");

    std::vector<float> out(static_cast<size_t>(street_size));
    run_parallel(street_size, num_threads, street == 2 ? "turn" : "flop",
                 [&](uint64_t idx) {
                     std::array<int, 2> hole{};
                     std::vector<int> board;
                     indexer.unindex(street, idx, hole, board);

                     bool used[kCards] = {false};
                     used[hole[0]] = used[hole[1]] = true;
                     for (int c : board) used[c] = true;

                     double sum = 0.0;
                     int count = 0;
                     for (int c = 0; c < static_cast<int>(kCards); ++c) {
                         if (used[c]) continue;
                         board.push_back(c);
                         sum += next_street[static_cast<size_t>(
                             indexer.index(street + 1, hole, board))];
                         board.pop_back();
                         ++count;
                     }
                     out[static_cast<size_t>(idx)] =
                         static_cast<float>(sum / count);
                 });
    return out;
}

double derive_equity_one(int street, const std::array<int, 2>& hole,
                         const std::vector<int>& board) {
    if (street == 3) return river_equity_exact(hole, board);
    if (street != 1 && street != 2)
        throw std::invalid_argument(
            "derive_equity_one: street must be 1, 2 or 3");
    if (board.size() != static_cast<size_t>(street + 2))
        throw std::invalid_argument(
            "derive_equity_one: board size does not match street");

    bool used[kCards] = {false};
    for (int c : {hole[0], hole[1]}) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument("card out of range");
        if (used[c]) throw std::invalid_argument("duplicate card");
        used[c] = true;
    }
    for (int c : board) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument("card out of range");
        if (used[c]) throw std::invalid_argument("duplicate card");
        used[c] = true;
    }

    double sum = 0.0;
    int count = 0;
    std::vector<int> next_board = board;
    for (int c = 0; c < static_cast<int>(kCards); ++c) {
        if (used[c]) continue;
        next_board.push_back(c);
        if (street == 2)
            sum += river_equity_exact(hole, next_board);
        else
            sum += derive_equity_one(2, hole, next_board);
        next_board.pop_back();
        ++count;
    }
    return sum / count;
}

uint64_t equity_content_hash(const EquityStore& store) {
    uint64_t h = kFnvOffset;
    for (const auto& values : store.eq) h = hash_floats(h, values);
    return h;
}

void save_equity_store(const std::string& path, const EquityStore& store,
                       bool strict_sizes) {
    if (strict_sizes) {
        for (int street = 1; street <= 3; ++street) {
            const uint64_t expected = HandIndexer::instance().size(street);
            if (store.eq[static_cast<size_t>(street - 1)].size() != expected)
                throw std::invalid_argument(
                    "save_equity_store: array size does not match street " +
                    std::to_string(street));
        }
    }

    const uint64_t content_hash = equity_content_hash(store);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open " + tmp);
        o.write(kMagic, 8);
        put<uint32_t>(o, kHandIndexVersion);
        for (const auto& values : store.eq)
            put<uint64_t>(o, static_cast<uint64_t>(values.size()));
        put<uint64_t>(o, content_hash);
        for (const auto& values : store.eq)
            o.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() *
                                                 sizeof(float)));
        if (!o) {
            std::remove(tmp.c_str());
            throw std::runtime_error("write failed: " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("atomic rename failed: " + path);
    }
}

EquityStore load_equity_store(const std::string& path) {
    return load_impl(path, true);
}

EquityStore load_equity_store_skip_size_check(const std::string& path) {
    return load_impl(path, false);
}

float equity_store_lookup(const std::string& path, int street, uint64_t idx) {
    if (street < 1 || street > 3)
        throw std::invalid_argument(
            "equity_store_lookup: street must be 1, 2 or 3");
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open " + path);

    std::array<uint64_t, 3> sizes{};
    uint64_t stored_hash = 0;
    read_header(i, path, sizes, stored_hash);
    (void)stored_hash;

    const uint64_t size = sizes[static_cast<size_t>(street - 1)];
    if (idx >= size)
        throw std::out_of_range("equity_store_lookup: index out of range");

    std::streamoff offset = kHeaderBytes;
    for (int k = 0; k < street - 1; ++k)
        offset += static_cast<std::streamoff>(
            sizes[static_cast<size_t>(k)] * sizeof(float));
    offset += static_cast<std::streamoff>(idx * sizeof(float));

    i.seekg(offset);
    float value = 0.0f;
    i.read(reinterpret_cast<char*>(&value), sizeof value);
    if (!i) throw std::runtime_error("equity store: truncated file " + path);
    return value;
}

}  // namespace sixmax
