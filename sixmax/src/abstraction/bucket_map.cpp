#include "abstraction/bucket_map.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "abstraction/equity_table.h"
#include "abstraction/hand_index.h"

namespace sixmax {
namespace {

constexpr uint32_t kCards = 52;
constexpr int kTurnCards = 46;  // river cards left after hole + 4-card board
constexpr int kFlopCards = 47;  // turn cards left after hole + 3-card board
constexpr int kShards = 64;     // fixed work/summation granularity
constexpr int kPotentialIters = 20;

// Work is split into a fixed number of contiguous shards, independent of the
// thread count. Each shard owns a disjoint output range and, where a sum is
// needed, a private accumulator; merging in shard order then makes results
// identical for any number of threads.
int resolve_threads(int num_threads) {
    if (num_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw == 0 ? 1 : static_cast<int>(hw);
    }
    return std::max(1, num_threads);
}

template <typename Fn>
void run_shards(uint64_t n, int shards, int num_threads, Fn&& fn) {
    if (n == 0) return;
    const int count = static_cast<int>(
        std::max<uint64_t>(1, std::min<uint64_t>(
                                  static_cast<uint64_t>(shards), n)));
    const int threads = std::min(resolve_threads(num_threads), count);
    std::atomic<int> next_shard{0};
    std::exception_ptr error;
    std::mutex error_mutex;

    auto worker = [&] {
        for (;;) {
            const int shard =
                next_shard.fetch_add(1, std::memory_order_relaxed);
            if (shard >= count) break;
            const uint64_t begin = n * static_cast<uint64_t>(shard) /
                                   static_cast<uint64_t>(count);
            const uint64_t end = n * static_cast<uint64_t>(shard + 1) /
                                 static_cast<uint64_t>(count);
            try {
                fn(shard, begin, end);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                }
                next_shard.store(count, std::memory_order_relaxed);
                break;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) workers.emplace_back(worker);
    for (auto& w : workers) w.join();
    if (error) std::rethrow_exception(error);
}

// Validate a hand plus a board of `expected_board` cards and report the used
// cards. Used by the per-hand feature helpers; index()/unindex() validate the
// dense bulk paths.
std::array<bool, kCards> validate_hand(const std::array<int, 2>& hole,
                                       const std::vector<int>& board,
                                       size_t expected_board,
                                       const char* what) {
    if (board.size() != expected_board)
        throw std::invalid_argument(std::string(what) +
                                    ": board size does not match street");
    std::array<bool, kCards> used{};
    for (int c : {hole[0], hole[1]}) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument(std::string(what) +
                                        ": card out of range");
        if (used[static_cast<size_t>(c)])
            throw std::invalid_argument(std::string(what) + ": duplicate card");
        used[static_cast<size_t>(c)] = true;
    }
    for (int c : board) {
        if (c < 0 || c >= static_cast<int>(kCards))
            throw std::invalid_argument(std::string(what) +
                                        ": card out of range");
        if (used[static_cast<size_t>(c)])
            throw std::invalid_argument(std::string(what) + ": duplicate card");
        used[static_cast<size_t>(c)] = true;
    }
    return used;
}

int equity_bin(float equity, int bins) {
    int b = static_cast<int>(equity * static_cast<float>(bins));
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return b;
}

// ---------------------------------------------------------------------------
// percentile buckets
// ---------------------------------------------------------------------------

template <typename WeightFn>
std::vector<uint8_t> percentile_buckets_impl(const std::vector<float>& eq,
                                             WeightFn weight, int k,
                                             int num_threads) {
    if (k < 1 || k > kMaxBucketCount)
        throw std::invalid_argument(
            "percentile_buckets: k must be in [1, 255]");
    const uint64_t n = eq.size();
    if (n == 0)
        throw std::invalid_argument("percentile_buckets: empty equity array");

    std::vector<float> sorted(eq);
    std::sort(sorted.begin(), sorted.end());
    std::vector<float> distinct;
    distinct.reserve(std::min<uint64_t>(n, 1u << 16));
    for (float v : sorted)
        if (distinct.empty() || v != distinct.back()) distinct.push_back(v);
    std::vector<float>().swap(sorted);

    const uint64_t d = distinct.size();
    if (d < static_cast<uint64_t>(k))
        throw std::invalid_argument(
            "percentile_buckets: fewer than k distinct values");

    // Total weight per distinct value. Weights are integers, so the
    // accumulation order does not matter.
    std::vector<uint64_t> distinct_weight(d, 0);
    for (uint64_t i = 0; i < n; ++i) {
        const auto it =
            std::lower_bound(distinct.begin(), distinct.end(), eq[i]);
        distinct_weight[static_cast<uint64_t>(it - distinct.begin())] +=
            static_cast<uint64_t>(weight(i));
    }
    std::vector<uint64_t> prefix(d + 1, 0);
    for (uint64_t b = 0; b < d; ++b)
        prefix[b + 1] = prefix[b] + distinct_weight[b];
    const uint64_t total = prefix[d];

    // Cut at the first distinct value whose cumulative weight reaches j/k of
    // the total, but never so late that a later bucket would run out of
    // distinct values and never earlier than the previous cut + 1.
    std::vector<uint8_t> distinct_bucket(d, 0);
    uint64_t first = 0;
    for (int j = 1; j < k; ++j) {
        const unsigned __int128 scaled =
            static_cast<unsigned __int128>(total) * static_cast<uint64_t>(j);
        const uint64_t target =
            static_cast<uint64_t>((scaled + static_cast<uint64_t>(k) - 1) /
                                  static_cast<uint64_t>(k));
        const uint64_t last_allowed = d - static_cast<uint64_t>(k - j);
        const auto it = std::lower_bound(prefix.begin() + (first + 1),
                                         prefix.end(), target);
        uint64_t cut = static_cast<uint64_t>(it - prefix.begin());
        if (cut < first + 1) cut = first + 1;
        if (cut > last_allowed) cut = last_allowed;
        for (uint64_t b = first; b < cut; ++b)
            distinct_bucket[b] = static_cast<uint8_t>(j - 1);
        first = cut;
    }
    for (uint64_t b = first; b < d; ++b)
        distinct_bucket[b] = static_cast<uint8_t>(k - 1);

    std::vector<uint8_t> out(n);
    run_shards(n, kShards, num_threads,
               [&](int, uint64_t begin, uint64_t end) {
                   for (uint64_t i = begin; i < end; ++i) {
                       const auto it =
                           std::lower_bound(distinct.begin(), distinct.end(),
                                            eq[i]);
                       out[i] = distinct_bucket[static_cast<uint64_t>(
                           it - distinct.begin())];
                   }
               });
    return out;
}

// ---------------------------------------------------------------------------
// k-means over cumulative histograms
// ---------------------------------------------------------------------------

// `cdf` holds n rows of `dims` floats, each row the cumulative sum of a
// histogram along its ordered bins (so |row_a - row_b| summed over bins is the
// 1-D EMD between the histograms, up to the constant bin width).
template <typename WeightFn>
std::vector<uint8_t> kmeans_from_cdf_impl(const float* cdf, uint64_t n,
                                          int dims, WeightFn weight, int k,
                                          uint64_t seed, int iters,
                                          int num_threads) {
    if (k < 1 || k > kMaxBucketCount)
        throw std::invalid_argument("kmeans_buckets: k must be in [1, 255]");
    if (iters < 1)
        throw std::invalid_argument("kmeans_buckets: iters must be >= 1");
    if (dims < 1)
        throw std::invalid_argument(
            "kmeans_buckets: histograms must be non-empty");
    if (n < static_cast<uint64_t>(k))
        throw std::invalid_argument(
            "kmeans_buckets: fewer points than buckets");
    if (n > (std::numeric_limits<uint64_t>::max() / sizeof(float)) /
                static_cast<uint64_t>(dims))
        throw std::invalid_argument("kmeans_buckets: too many points");

    const size_t row = static_cast<size_t>(dims);
    const size_t kk = static_cast<size_t>(k);
    const int shards = static_cast<int>(
        std::max<uint64_t>(1, std::min<uint64_t>(kShards, n)));
    std::mt19937_64 rng(seed);

    std::vector<float> centroids(kk * row, 0.0f);

    uint64_t total_weight = 0;
    for (uint64_t i = 0; i < n; ++i) total_weight += weight(i);
    const auto sample_by_weight = [&]() -> uint64_t {
        if (total_weight == 0) return rng() % n;
        const uint64_t r = rng() % total_weight;
        uint64_t acc = 0;
        for (uint64_t i = 0; i < n; ++i) {
            acc += weight(i);
            if (acc > r) return i;
        }
        return n - 1;
    };

    // k-means++: first centre proportional to weight, then each new centre
    // proportional to weight * (squared EMD to the nearest chosen centre).
    const uint64_t first = sample_by_weight();
    std::copy(cdf + static_cast<size_t>(first) * row,
              cdf + (static_cast<size_t>(first) + 1) * row,
              centroids.begin());
    std::vector<double> min_dist(n, 0.0);

    for (int j = 1; j < k; ++j) {
        std::vector<double> shard_total(static_cast<size_t>(shards), 0.0);
        const float* centre = &centroids[static_cast<size_t>(j - 1) * row];
        run_shards(n, shards, num_threads,
                   [&](int shard, uint64_t begin, uint64_t end) {
                       double acc = 0.0;
                       for (uint64_t i = begin; i < end; ++i) {
                           const float* p = cdf + static_cast<size_t>(i) * row;
                           double d = 0.0;
                           for (size_t t = 0; t < row; ++t)
                               d += std::fabs(static_cast<double>(p[t]) -
                                              centre[t]);
                           if (j == 1 || d < min_dist[i]) min_dist[i] = d;
                           const double wi = static_cast<double>(weight(i));
                           acc += wi * min_dist[i] * min_dist[i];
                       }
                       shard_total[static_cast<size_t>(shard)] = acc;
                   });
        double total = 0.0;
        for (double v : shard_total) total += v;

        uint64_t pick = n - 1;
        if (total <= 0.0) {
            pick = rng() % n;
        } else {
            const double unit =
                static_cast<double>(rng() >> 11) / 9007199254740992.0;
            const double r = unit * total;
            double acc = 0.0;
            pick = n - 1;
            for (uint64_t i = 0; i < n; ++i) {
                acc += static_cast<double>(weight(i)) * min_dist[i] *
                       min_dist[i];
                if (acc > r) {
                    pick = i;
                    break;
                }
            }
        }
        std::copy(cdf + static_cast<size_t>(pick) * row,
                  cdf + (static_cast<size_t>(pick) + 1) * row,
                  centroids.begin() + static_cast<size_t>(j) * row);
    }

    // Lloyd iterations. Per-shard sums are merged in shard order, so the
    // result is identical for any thread count.
    std::vector<uint8_t> labels(n, 0);
    std::vector<uint8_t> new_labels(n, 0);
    std::vector<double> shard_sums(static_cast<size_t>(shards) * kk * row, 0.0);
    std::vector<double> shard_cluster_weight(
        static_cast<size_t>(shards) * kk, 0.0);
    std::vector<uint64_t> shard_changed(static_cast<size_t>(shards), 0);
    std::vector<double> sums(kk * row, 0.0);
    std::vector<double> cluster_weight(kk, 0.0);
    std::vector<float> new_centroids(kk * row, 0.0f);

    // Re-seed each empty cluster at the point farthest from its assigned
    // centre, choosing distinct points greedily (ties -> lowest index). This
    // runs single-threaded in index order, so the result is identical for any
    // thread count.
    const auto reseed_empty = [&](std::vector<float>& target,
                                  const std::vector<uint8_t>& assigned) {
        std::vector<size_t> empty;
        for (size_t c = 0; c < kk; ++c)
            if (cluster_weight[c] == 0.0) empty.push_back(c);
        if (empty.empty()) return false;
        std::vector<uint64_t> chosen;
        chosen.reserve(empty.size());
        for (size_t c : empty) {
            uint64_t best = 0;
            double best_dist = -1.0;
            for (uint64_t i = 0; i < n; ++i) {
                if (std::find(chosen.begin(), chosen.end(), i) !=
                    chosen.end())
                    continue;
                const float* p = cdf + static_cast<size_t>(i) * row;
                const float* centre =
                    &centroids[static_cast<size_t>(assigned[i]) * row];
                double d = 0.0;
                for (size_t t = 0; t < row; ++t)
                    d += std::fabs(static_cast<double>(p[t]) - centre[t]);
                if (d > best_dist) {  // ties keep the lowest index
                    best_dist = d;
                    best = i;
                }
            }
            chosen.push_back(best);
            std::copy(cdf + static_cast<size_t>(best) * row,
                      cdf + (static_cast<size_t>(best) + 1) * row,
                      target.begin() + static_cast<size_t>(c) * row);
        }
        return true;
    };

    bool reseeded_last = false;
    for (int iteration = 0; iteration < iters; ++iteration) {
        std::fill(shard_sums.begin(), shard_sums.end(), 0.0);
        std::fill(shard_cluster_weight.begin(), shard_cluster_weight.end(),
                  0.0);
        std::fill(shard_changed.begin(), shard_changed.end(), 0);
        run_shards(n, shards, num_threads,
                   [&](int shard, uint64_t begin, uint64_t end) {
                       double* sums_s = &shard_sums[static_cast<size_t>(shard) *
                                                    kk * row];
                       double* cw_s =
                           &shard_cluster_weight[static_cast<size_t>(shard) *
                                                 kk];
                       uint64_t changed = 0;
                       for (uint64_t i = begin; i < end; ++i) {
                           const float* p = cdf + static_cast<size_t>(i) * row;
                           int best = 0;
                           float best_dist = std::numeric_limits<float>::max();
                           for (int c = 0; c < k; ++c) {
                               const float* centre =
                                   &centroids[static_cast<size_t>(c) * row];
                               float d = 0.0f;
                               for (int t = 0; t < dims; ++t)
                                   d += std::fabs(p[t] - centre[t]);
                               if (d < best_dist) {
                                   best_dist = d;
                                   best = c;
                               }
                           }
                           new_labels[i] = static_cast<uint8_t>(best);
                           if (new_labels[i] != labels[i]) ++changed;
                           const double wi = static_cast<double>(weight(i));
                           cw_s[best] += wi;
                           double* target =
                               sums_s + static_cast<size_t>(best) * row;
                           for (int t = 0; t < dims; ++t)
                               target[t] += wi * p[t];
                       }
                       shard_changed[static_cast<size_t>(shard)] = changed;
                   });

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(cluster_weight.begin(), cluster_weight.end(), 0.0);
        for (int shard = 0; shard < shards; ++shard) {
            const double* sums_s =
                &shard_sums[static_cast<size_t>(shard) * kk * row];
            const double* cw_s =
                &shard_cluster_weight[static_cast<size_t>(shard) * kk];
            for (size_t c = 0; c < kk; ++c) {
                cluster_weight[c] += cw_s[c];
                for (size_t t = 0; t < row; ++t)
                    sums[c * row + t] += sums_s[c * row + t];
            }
        }
        for (size_t c = 0; c < kk; ++c) {
            if (cluster_weight[c] > 0.0) {
                for (size_t t = 0; t < row; ++t)
                    new_centroids[c * row + t] = static_cast<float>(
                        sums[c * row + t] / cluster_weight[c]);
            } else {
                // Empty cluster: keep the previous centre for now; the
                // re-seed below picks a real point when possible.
                std::copy(&centroids[c * row], &centroids[(c + 1) * row],
                          &new_centroids[c * row]);
            }
        }
        reseeded_last = reseed_empty(new_centroids, new_labels);
        centroids.swap(new_centroids);
        labels.swap(new_labels);

        uint64_t changed = 0;
        for (uint64_t v : shard_changed) changed += v;
        if (changed == 0 && !reseeded_last) break;
    }

    // If the final iteration re-seeded a centre, `labels` still reflects the
    // pre-seed centres: do one assignment pass with the final centres.
    if (reseeded_last) {
        run_shards(n, shards, num_threads,
                   [&](int, uint64_t begin, uint64_t end) {
                       for (uint64_t i = begin; i < end; ++i) {
                           const float* p = cdf + static_cast<size_t>(i) * row;
                           int best = 0;
                           float best_dist =
                               std::numeric_limits<float>::max();
                           for (int c = 0; c < k; ++c) {
                               const float* centre =
                                   &centroids[static_cast<size_t>(c) * row];
                               float d = 0.0f;
                               for (int t = 0; t < dims; ++t)
                                   d += std::fabs(p[t] - centre[t]);
                               if (d < best_dist) {
                                   best_dist = d;
                                   best = c;
                               }
                           }
                           labels[i] = static_cast<uint8_t>(best);
                       }
                   });
    }

    // Final guarantee: every bucket is non-empty. Degenerate inputs (fewer
    // distinct histograms than k, or exact duplicate rows) can leave a
    // re-seeded centre coincident with an existing one, so move the farthest
    // member of the largest cluster into each empty cluster. Deterministic:
    // lowest empty cluster, largest cluster (ties by lowest index), farthest
    // point (ties by lowest index).
    std::vector<uint64_t> cluster_size(kk, 0);
    for (uint64_t i = 0; i < n; ++i) ++cluster_size[labels[i]];
    for (;;) {
        size_t empty = kk;
        for (size_t c = 0; c < kk; ++c)
            if (cluster_size[c] == 0) {
                empty = c;
                break;
            }
        if (empty == kk) break;
        size_t source = 0;
        for (size_t c = 1; c < kk; ++c)
            if (cluster_size[c] > cluster_size[source]) source = c;
        // n >= k and at least one empty cluster => the largest has >= 2.
        uint64_t best = 0;
        double best_dist = -1.0;
        for (uint64_t i = 0; i < n; ++i) {
            if (labels[i] != source) continue;
            const float* p = cdf + static_cast<size_t>(i) * row;
            const float* centre = &centroids[source * row];
            double d = 0.0;
            for (size_t t = 0; t < row; ++t)
                d += std::fabs(static_cast<double>(p[t]) - centre[t]);
            if (d > best_dist) {  // ties keep the lowest index
                best_dist = d;
                best = i;
            }
        }
        labels[best] = static_cast<uint8_t>(empty);
        --cluster_size[source];
        ++cluster_size[empty];
    }
    return labels;
}

// ---------------------------------------------------------------------------
// Potential-aware features
// ---------------------------------------------------------------------------

template <typename RiverEquity>
std::vector<float> turn_hist_impl(const std::array<int, 2>& hole,
                                  const std::vector<int>& board, int bins,
                                  RiverEquity river_equity) {
    if (bins < 1)
        throw std::invalid_argument("turn_potential_hist: bins must be >= 1");
    const auto used = validate_hand(hole, board, 4, "turn_potential_hist");

    std::vector<float> hist(static_cast<size_t>(bins), 0.0f);
    std::vector<int> river_board = board;
    int cards = 0;
    for (int c = 0; c < static_cast<int>(kCards); ++c) {
        if (used[static_cast<size_t>(c)]) continue;
        river_board.push_back(c);
        const int bin = equity_bin(river_equity(river_board), bins);
        hist[static_cast<size_t>(bin)] +=
            1.0f / static_cast<float>(kTurnCards);
        river_board.pop_back();
        ++cards;
    }
    if (cards != kTurnCards)
        throw std::runtime_error("turn_potential_hist: unexpected deck size");
    return hist;
}

// ---------------------------------------------------------------------------
// File format helpers (FNV-1a, matching the equity store's style)
// ---------------------------------------------------------------------------

constexpr char kBucketMagic[8] = {'S', 'I', 'X', 'B', 'M', '0', '0', '1'};
constexpr std::streamoff kHeaderBytes = 8 + 4 + 8 * 3 + 4 * 3 + 8 + 8 + 4 + 4;

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hash_mix(uint64_t& h, uint64_t v) {
    h ^= v;
    h *= kFnvPrime;
}

uint64_t hash_bytes(uint64_t h, const uint8_t* bytes, size_t n) {
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

uint64_t bucket_content_hash(const std::array<std::vector<uint8_t>, 3>& b) {
    uint64_t h = kFnvOffset;
    for (const auto& values : b) {
        hash_mix(h, static_cast<uint64_t>(values.size()));
        h = hash_bytes(h, values.data(), values.size());
    }
    return h;
}

std::array<uint32_t, 3> bucket_counts(
    const std::array<std::vector<uint8_t>, 3>& b) {
    std::array<uint32_t, 3> counts{};
    for (size_t s = 0; s < 3; ++s) {
        if (b[s].empty())
            throw std::invalid_argument(
                "write_bucket_map: bucket array must not be empty");
        const uint8_t max_bucket = *std::max_element(b[s].begin(), b[s].end());
        if (static_cast<uint32_t>(max_bucket) + 1 > kMaxBucketCount)
            throw std::invalid_argument(
                "write_bucket_map: more than 255 buckets on a street");
        counts[s] = static_cast<uint32_t>(max_bucket) + 1;
    }
    return counts;
}

template <typename T>
void put(std::ofstream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
T read_field(const unsigned char* base, uint64_t offset) {
    T v;
    std::memcpy(&v, base + offset, sizeof v);
    return v;
}

// Multiplicity weights for one street, in the uint8 range the builder needs.
std::vector<uint8_t> street_weights(int street, int num_threads) {
    const uint64_t n = HandIndexer::instance().size(street);
    std::vector<uint8_t> weight(n);
    run_shards(n, kShards, num_threads,
               [&](int, uint64_t begin, uint64_t end) {
                   for (uint64_t i = begin; i < end; ++i) {
                       const uint32_t m =
                           HandIndexer::instance().multiplicity(street, i);
                       if (m > kMaxBucketCount)
                           throw std::runtime_error(
                               "bucket map: multiplicity out of range");
                       weight[i] = static_cast<uint8_t>(m);
                   }
               });
    return weight;
}

void report_population(const char* label, const std::vector<uint8_t>& buckets,
                       const std::vector<uint8_t>& weight, int k,
                       int num_threads) {
    std::vector<uint64_t> shard_pop(static_cast<size_t>(kShards) *
                                        static_cast<size_t>(k),
                                    0);
    run_shards(buckets.size(), kShards, num_threads,
               [&](int shard, uint64_t begin, uint64_t end) {
                   uint64_t* pop =
                       &shard_pop[static_cast<size_t>(shard) * k];
                   for (uint64_t i = begin; i < end; ++i)
                       pop[buckets[i]] += weight[i];
               });
    std::vector<uint64_t> population(static_cast<size_t>(k), 0);
    for (int shard = 0; shard < kShards; ++shard)
        for (int c = 0; c < k; ++c)
            population[c] += shard_pop[static_cast<size_t>(shard) * k + c];
    uint64_t min_pop = population[0];
    uint64_t max_pop = population[0];
    for (uint64_t p : population) {
        min_pop = std::min(min_pop, p);
        max_pop = std::max(max_pop, p);
    }
    const double ratio =
        min_pop == 0 ? std::numeric_limits<double>::infinity()
                     : static_cast<double>(max_pop) /
                           static_cast<double>(min_pop);
    std::fprintf(stderr,
                 "[bucket] %s k=%d population min=%llu max=%llu max/min=%.4f\n",
                 label, k, static_cast<unsigned long long>(min_pop),
                 static_cast<unsigned long long>(max_pop), ratio);
}

}  // namespace

// ---------------------------------------------------------------------------
// percentile_buckets
// ---------------------------------------------------------------------------

std::vector<uint8_t> percentile_buckets(const std::vector<float>& eq,
                                        const std::vector<uint32_t>& weight,
                                        int k) {
    if (eq.size() != weight.size())
        throw std::invalid_argument(
            "percentile_buckets: weight size does not match eq");
    return percentile_buckets_impl(
        eq, [&weight](uint64_t i) { return weight[i]; }, k, 0);
}

// ---------------------------------------------------------------------------
// kmeans_buckets
// ---------------------------------------------------------------------------

std::vector<uint8_t> kmeans_buckets(
    const std::vector<std::vector<float>>& hists,
    const std::vector<uint32_t>& weight, int k, uint64_t seed, int iters,
    int num_threads) {
    if (hists.size() != weight.size())
        throw std::invalid_argument(
            "kmeans_buckets: weight size does not match hists");
    if (hists.empty())
        throw std::invalid_argument("kmeans_buckets: no histograms");
    const int dims = static_cast<int>(hists[0].size());
    if (dims < 1)
        throw std::invalid_argument(
            "kmeans_buckets: histograms must be non-empty");
    const uint64_t n = hists.size();
    std::vector<float> cdf(static_cast<size_t>(n) * static_cast<size_t>(dims));
    for (uint64_t i = 0; i < n; ++i) {
        if (static_cast<int>(hists[static_cast<size_t>(i)].size()) != dims)
            throw std::invalid_argument(
                "kmeans_buckets: histograms must all have the same length");
        float running = 0.0f;
        float* out = &cdf[static_cast<size_t>(i) * static_cast<size_t>(dims)];
        for (int t = 0; t < dims; ++t) {
            running += hists[static_cast<size_t>(i)][static_cast<size_t>(t)];
            out[t] = running;
        }
    }
    return kmeans_from_cdf_impl(
        cdf.data(), n, dims, [&weight](uint64_t i) { return weight[i]; }, k,
        seed, iters, num_threads);
}

// ---------------------------------------------------------------------------
// Potential-aware features
// ---------------------------------------------------------------------------

std::vector<float> turn_potential_hist(const std::array<int, 2>& hole,
                                       const std::vector<int>& board,
                                       const std::vector<float>& river,
                                       int bins) {
    const auto& indexer = HandIndexer::instance();
    if (river.size() != indexer.size(3))
        throw std::invalid_argument(
            "turn_potential_hist: river table size does not match");
    return turn_hist_impl(hole, board, bins,
                          [&](const std::vector<int>& river_board) {
                              return river[static_cast<size_t>(indexer.index(
                                  3, hole, river_board))];
                          });
}

std::vector<float> turn_potential_hist_exact(const std::array<int, 2>& hole,
                                             const std::vector<int>& board,
                                             int bins) {
    return turn_hist_impl(hole, board, bins,
                          [&](const std::vector<int>& river_board) {
                              return static_cast<float>(
                                  river_equity_exact(hole, river_board));
                          });
}

std::vector<float> flop_potential_hist(
    const std::array<int, 2>& hole, const std::vector<int>& board,
    const std::vector<uint8_t>& turn_buckets, int num_buckets,
    const std::vector<int>& order) {
    if (num_buckets < 1 || num_buckets > kMaxBucketCount)
        throw std::invalid_argument(
            "flop_potential_hist: num_buckets must be in [1, 255]");
    const auto& indexer = HandIndexer::instance();
    if (turn_buckets.size() != indexer.size(2))
        throw std::invalid_argument(
            "flop_potential_hist: turn bucket array size does not match");
    const auto used = validate_hand(hole, board, 3, "flop_potential_hist");

    std::vector<int> dimension(static_cast<size_t>(num_buckets));
    if (order.empty()) {
        for (int b = 0; b < num_buckets; ++b) dimension[b] = b;
    } else {
        if (static_cast<int>(order.size()) != num_buckets)
            throw std::invalid_argument(
                "flop_potential_hist: order length does not match num_buckets");
        std::vector<bool> seen(static_cast<size_t>(num_buckets), false);
        for (int b = 0; b < num_buckets; ++b) {
            const int dim = order[static_cast<size_t>(b)];
            if (dim < 0 || dim >= num_buckets || seen[static_cast<size_t>(dim)])
                throw std::invalid_argument(
                    "flop_potential_hist: order must be a permutation of "
                    "[0, num_buckets)");
            seen[static_cast<size_t>(dim)] = true;
            dimension[static_cast<size_t>(b)] = dim;
        }
    }

    std::vector<float> hist(static_cast<size_t>(num_buckets), 0.0f);
    std::vector<int> turn_board = board;
    int cards = 0;
    for (int c = 0; c < static_cast<int>(kCards); ++c) {
        if (used[static_cast<size_t>(c)]) continue;
        turn_board.push_back(c);
        const uint64_t idx = indexer.index(2, hole, turn_board);
        const uint8_t b = turn_buckets[static_cast<size_t>(idx)];
        if (b >= num_buckets)
            throw std::invalid_argument(
                "flop_potential_hist: turn bucket out of range");
        hist[static_cast<size_t>(dimension[b])] +=
            1.0f / static_cast<float>(kFlopCards);
        turn_board.pop_back();
        ++cards;
    }
    if (cards != kFlopCards)
        throw std::runtime_error("flop_potential_hist: unexpected deck size");
    return hist;
}

// ---------------------------------------------------------------------------
// SIXBM001 writer
// ---------------------------------------------------------------------------

void write_bucket_map(const std::string& path,
                      const std::array<std::vector<uint8_t>, 3>& buckets,
                      const std::string& method, const std::string& params,
                      uint64_t equity_hash) {
    if (method.empty())
        throw std::invalid_argument(
            "write_bucket_map: method must not be empty");
    if (method.size() > 0xFFFFFFFFull || params.size() > 0xFFFFFFFFull)
        throw std::invalid_argument("write_bucket_map: method/params too long");
    for (int street = 1; street <= 3; ++street) {
        const uint64_t expected = HandIndexer::instance().size(street);
        if (buckets[static_cast<size_t>(street - 1)].size() != expected)
            throw std::invalid_argument(
                "write_bucket_map: array size does not match street " +
                std::to_string(street));
    }
    const auto counts = bucket_counts(buckets);
    const uint64_t content_hash = bucket_content_hash(buckets);

    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open " + tmp);
        o.write(kBucketMagic, 8);
        put<uint32_t>(o, kHandIndexVersion);
        for (const auto& values : buckets)
            put<uint64_t>(o, static_cast<uint64_t>(values.size()));
        for (uint32_t c : counts) put<uint32_t>(o, c);
        put<uint64_t>(o, equity_hash);
        put<uint64_t>(o, content_hash);
        put<uint32_t>(o, static_cast<uint32_t>(method.size()));
        put<uint32_t>(o, static_cast<uint32_t>(params.size()));
        o.write(method.data(), static_cast<std::streamsize>(method.size()));
        o.write(params.data(), static_cast<std::streamsize>(params.size()));
        for (const auto& values : buckets)
            o.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size()));
        // close() flushes, so a full-disk failure at the final flush is seen
        // here rather than after the rename.
        o.close();
        if (o.fail()) {
            std::remove(tmp.c_str());
            throw std::runtime_error("write failed: " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("atomic rename failed: " + path);
    }
}

// ---------------------------------------------------------------------------
// BucketMap
// ---------------------------------------------------------------------------

BucketMap::~BucketMap() {
    if (base_ != nullptr) {
        ::munmap(const_cast<unsigned char*>(base_), length_);
    }
}

std::shared_ptr<const BucketMap> BucketMap::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat " + path);
    }
    const uint64_t length = static_cast<uint64_t>(st.st_size);
    if (length < static_cast<uint64_t>(kHeaderBytes)) {
        ::close(fd);
        throw std::runtime_error("bucket map: truncated file " + path);
    }
    void* mapped = ::mmap(nullptr, static_cast<size_t>(length), PROT_READ,
                          MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps the file alive
    if (mapped == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);

    try {
        const auto* base = static_cast<const unsigned char*>(mapped);
        char magic[8];
        std::memcpy(magic, base, 8);
        if (std::memcmp(magic, kBucketMagic, 8) != 0)
            throw std::runtime_error("not a bucket map: " + path);

        const uint32_t version = read_field<uint32_t>(base, 8);
        if (version != kHandIndexVersion)
            throw std::runtime_error(
                "bucket map hand-index version mismatch: file has " +
                std::to_string(version) + ", expected " +
                std::to_string(kHandIndexVersion));

        std::array<uint64_t, 3> sizes{};
        std::array<uint32_t, 3> counts{};
        for (size_t s = 0; s < 3; ++s)
            sizes[s] = read_field<uint64_t>(base, 12 + 8 * s);
        for (size_t s = 0; s < 3; ++s)
            counts[s] = read_field<uint32_t>(base, 36 + 4 * s);
        const uint64_t equity_hash = read_field<uint64_t>(base, 48);
        const uint64_t stored_hash = read_field<uint64_t>(base, 56);
        const uint32_t method_len = read_field<uint32_t>(base, 64);
        const uint32_t params_len = read_field<uint32_t>(base, 68);

        for (int street = 1; street <= 3; ++street) {
            const uint64_t expected = HandIndexer::instance().size(street);
            const uint64_t stored = sizes[static_cast<size_t>(street - 1)];
            if (stored != expected)
                throw std::runtime_error(
                    "bucket map size mismatch on street " +
                    std::to_string(street) + ": file has " +
                    std::to_string(stored) + ", expected " +
                    std::to_string(expected));
        }
        for (int street = 1; street <= 3; ++street) {
            const uint32_t c = counts[static_cast<size_t>(street - 1)];
            if (c < 1 || c > kMaxBucketCount)
                throw std::runtime_error(
                    "bucket map has an invalid bucket count on street " +
                    std::to_string(street) + ": " + std::to_string(c));
        }

        const uint64_t strings =
            static_cast<uint64_t>(method_len) +
            static_cast<uint64_t>(params_len);
        const uint64_t data_bytes =
            sizes[0] + sizes[1] + sizes[2];
        if (length != static_cast<uint64_t>(kHeaderBytes) + strings +
                          data_bytes)
            throw std::runtime_error(
                "bucket map file length mismatch: " + path);

        const uint8_t* data = base + kHeaderBytes + strings;
        uint64_t h = kFnvOffset;
        for (size_t s = 0; s < 3; ++s) {
            // The content hash does not cover the header, so the declared
            // bucket count is cross-checked against the data here, in the
            // same pass that hashes it.
            uint8_t max_bucket = 0;
            for (uint64_t i = 0; i < sizes[s]; ++i)
                max_bucket = std::max(max_bucket, data[i]);
            if (static_cast<uint32_t>(max_bucket) + 1 != counts[s])
                throw std::runtime_error(
                    "bucket map bucket count mismatch on street " +
                    std::to_string(s + 1) + ": header has " +
                    std::to_string(counts[s]) + ", data has " +
                    std::to_string(static_cast<uint32_t>(max_bucket) + 1));
            hash_mix(h, sizes[s]);
            h = hash_bytes(h, data, static_cast<size_t>(sizes[s]));
            data += sizes[s];
        }
        if (h != stored_hash)
            throw std::runtime_error("bucket map content hash mismatch: " +
                                     path);

        // Validate and copy the variable-length header strings before the
        // mapping changes hands (their bounds were checked by the length
        // equality above).
        const std::string method(
            reinterpret_cast<const char*>(base + kHeaderBytes), method_len);
        const std::string params(
            reinterpret_cast<const char*>(base + kHeaderBytes + method_len),
            params_len);

        // All validation passed. Build the object first and hand the mapping
        // over last: the BucketMap destructor is then the only owner of the
        // munmap on every path (a throw above leaves base_ null).
        auto out = std::shared_ptr<BucketMap>(new BucketMap());
        out->length_ = static_cast<size_t>(length);
        out->sizes_ = sizes;
        out->buckets_ = counts;
        out->content_hash_ = stored_hash;
        out->equity_hash_ = equity_hash;
        out->method_ = method;
        out->params_ = params;
        out->path_ = path;
        const uint8_t* cursor = base + kHeaderBytes + strings;
        for (size_t s = 0; s < 3; ++s) {
            out->data_[s] = cursor;
            cursor += sizes[s];
        }
        out->base_ = base;
        return out;
    } catch (...) {
        ::munmap(mapped, static_cast<size_t>(length));
        throw;
    }
}

uint8_t BucketMap::bucket(int street, uint64_t idx) const {
    if (street < 1 || street > 3)
        throw std::invalid_argument("bucket: street must be 1, 2 or 3");
    const size_t s = static_cast<size_t>(street - 1);
    if (idx >= sizes_[s])
        throw std::out_of_range("bucket map: index out of range");
    return data_[s][idx];
}

int BucketMap::num_buckets(int street) const {
    if (street < 1 || street > 3)
        throw std::invalid_argument(
            "num_buckets: street must be 1, 2 or 3");
    return static_cast<int>(buckets_[static_cast<size_t>(street - 1)]);
}

// ---------------------------------------------------------------------------
// build_bucket_map
// ---------------------------------------------------------------------------

void build_bucket_map(const std::string& equity_path, const std::string& out,
                      const std::string& method, std::array<int, 3> counts,
                      uint64_t seed, int num_threads) {
    if (method != "percentile" && method != "potential")
        throw std::invalid_argument(
            "build_bucket_map: method must be 'percentile' or 'potential'");
    for (int c : counts)
        if (c < 1 || c > kMaxBucketCount)
            throw std::invalid_argument(
                "build_bucket_map: bucket counts must be in [1, 255]");

    const auto& indexer = HandIndexer::instance();
    EquityStore store = load_equity_store(equity_path);
    const uint64_t equity_hash = store.content_hash;
    std::array<std::vector<uint8_t>, 3> buckets;

    const auto counts_text = [&counts] {
        return std::to_string(counts[0]) + "," + std::to_string(counts[1]) +
               "," + std::to_string(counts[2]);
    };

    if (method == "percentile") {
        for (int street = 1; street <= 3; ++street) {
            const size_t s = static_cast<size_t>(street - 1);
            auto weight = street_weights(street, num_threads);
            std::fprintf(stderr, "[bucket] percentile street %d (k=%d)\n",
                         street, counts[s]);
            buckets[s] = percentile_buckets_impl(
                store.eq[s],
                [&weight](uint64_t i) {
                    return static_cast<uint32_t>(weight[i]);
                },
                counts[s], num_threads);
            report_population(street == 1 ? "flop"
                                          : (street == 2 ? "turn" : "river"),
                              buckets[s], weight, counts[s], num_threads);
            std::vector<float>().swap(store.eq[s]);
        }
        write_bucket_map(out, buckets, method, "counts=" + counts_text(),
                         equity_hash);
    } else {
        // River: no future cards, so potential == percentile.
        auto river_weight = street_weights(3, num_threads);
        std::fprintf(stderr, "[bucket] potential river (percentile, k=%d)\n",
                     counts[2]);
        buckets[2] = percentile_buckets_impl(
            store.eq[2],
            [&river_weight](uint64_t i) {
                return static_cast<uint32_t>(river_weight[i]);
            },
            counts[2], num_threads);
        report_population("river", buckets[2], river_weight, counts[2],
                          num_threads);
        std::vector<uint8_t>().swap(river_weight);

        // Turn: k-means over each hand's 46-river-equity histogram.
        const uint64_t n_turn = indexer.size(2);
        const int bins = kPotentialTurnBins;
        auto turn_weight = street_weights(2, num_threads);
        std::fprintf(stderr,
                     "[bucket] potential turn features (%llu hands, %d bins)\n",
                     static_cast<unsigned long long>(n_turn), bins);
        std::vector<float> turn_cdf;
        turn_cdf.resize(static_cast<size_t>(n_turn) *
                        static_cast<size_t>(bins));
        run_shards(n_turn, kShards, num_threads,
                   [&](int, uint64_t begin, uint64_t end) {
                       std::array<int, 2> hole{};
                       std::vector<int> board;
                       for (uint64_t idx = begin; idx < end; ++idx) {
                           indexer.unindex(2, idx, hole, board);
                           const auto hist = turn_potential_hist(
                               hole, board, store.eq[2], bins);
                           float running = 0.0f;
                           float* row =
                               &turn_cdf[static_cast<size_t>(idx) *
                                         static_cast<size_t>(bins)];
                           for (int t = 0; t < bins; ++t) {
                               running += hist[static_cast<size_t>(t)];
                               row[t] = running;
                           }
                       }
                   });
        std::vector<float>().swap(store.eq[2]);
        std::vector<float>().swap(store.eq[0]);
        std::fprintf(stderr,
                     "[bucket] potential turn k-means (k=%d, iters=%d)\n",
                     counts[1], kPotentialIters);
        buckets[1] = kmeans_from_cdf_impl(
            turn_cdf.data(), n_turn, bins,
            [&turn_weight](uint64_t i) {
                return static_cast<uint32_t>(turn_weight[i]);
            },
            counts[1], seed, kPotentialIters, num_threads);
        std::vector<float>().swap(turn_cdf);
        report_population("turn", buckets[1], turn_weight, counts[1],
                          num_threads);

        // Feature dimensions of the flop histogram are turn buckets ordered
        // by mean equity, so the ordered-outcome EMD stays meaningful.
        std::vector<double> bucket_equity_sum(static_cast<size_t>(counts[1]),
                                              0.0);
        std::vector<double> bucket_weight(static_cast<size_t>(counts[1]), 0.0);
        for (uint64_t idx = 0; idx < n_turn; ++idx) {
            const uint8_t b = buckets[1][static_cast<size_t>(idx)];
            const double w = static_cast<double>(turn_weight[idx]);
            bucket_equity_sum[b] += w * store.eq[1][static_cast<size_t>(idx)];
            bucket_weight[b] += w;
        }
        std::vector<int> order(static_cast<size_t>(counts[1]));
        for (int b = 0; b < counts[1]; ++b) order[b] = b;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const double wa = bucket_weight[static_cast<size_t>(a)];
            const double wb = bucket_weight[static_cast<size_t>(b)];
            const double ma =
                wa > 0.0 ? bucket_equity_sum[static_cast<size_t>(a)] / wa : 0.0;
            const double mb =
                wb > 0.0 ? bucket_equity_sum[static_cast<size_t>(b)] / wb : 0.0;
            if (ma != mb) return ma < mb;
            return a < b;
        });
        std::vector<int> dimension(static_cast<size_t>(counts[1]));
        for (int d = 0; d < counts[1]; ++d)
            dimension[static_cast<size_t>(order[static_cast<size_t>(d)])] = d;
        std::vector<double>().swap(bucket_equity_sum);
        std::vector<double>().swap(bucket_weight);
        std::vector<float>().swap(store.eq[1]);

        // Flop: k-means over each hand's distribution over turn buckets.
        const uint64_t n_flop = indexer.size(1);
        const int flop_dims = counts[1];
        auto flop_weight = street_weights(1, num_threads);
        std::fprintf(stderr,
                     "[bucket] potential flop features (%llu hands, %d dims)\n",
                     static_cast<unsigned long long>(n_flop), flop_dims);
        std::vector<float> flop_cdf;
        flop_cdf.resize(static_cast<size_t>(n_flop) *
                        static_cast<size_t>(flop_dims));
        run_shards(n_flop, kShards, num_threads,
                   [&](int, uint64_t begin, uint64_t end) {
                       std::array<int, 2> hole{};
                       std::vector<int> board;
                       for (uint64_t idx = begin; idx < end; ++idx) {
                           indexer.unindex(1, idx, hole, board);
                           const auto hist = flop_potential_hist(
                               hole, board, buckets[1], counts[1], dimension);
                           float running = 0.0f;
                           float* row =
                               &flop_cdf[static_cast<size_t>(idx) *
                                         static_cast<size_t>(flop_dims)];
                           for (int t = 0; t < flop_dims; ++t) {
                               running += hist[static_cast<size_t>(t)];
                               row[t] = running;
                           }
                       }
                   });
        std::fprintf(stderr,
                     "[bucket] potential flop k-means (k=%d, iters=%d)\n",
                     counts[0], kPotentialIters);
        buckets[0] = kmeans_from_cdf_impl(
            flop_cdf.data(), n_flop, flop_dims,
            [&flop_weight](uint64_t i) {
                return static_cast<uint32_t>(flop_weight[i]);
            },
            counts[0], seed, kPotentialIters, num_threads);
        std::vector<float>().swap(flop_cdf);
        report_population("flop", buckets[0], flop_weight, counts[0],
                          num_threads);

        const std::string params =
            "counts=" + counts_text() +
            ",iters=" + std::to_string(kPotentialIters) +
            ",seed=" + std::to_string(seed) +
            ",turn_bins=" + std::to_string(bins);
        write_bucket_map(out, buckets, method, params, equity_hash);
    }
    std::fprintf(stderr, "[bucket] wrote %s (content hash %016llx)\n",
                 out.c_str(),
                 static_cast<unsigned long long>(
                     bucket_content_hash(buckets)));
}

}  // namespace sixmax
