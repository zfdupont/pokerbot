#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sixmax {

// The infoset key's card field is 8 bits (pack_abstract_key), so a map can
// never address more than 255 buckets per street.
constexpr uint8_t kMaxBucketCount = 255;

// Fixed turn feature geometry for the `potential` builder: each turn hand is
// summarised by the histogram of its 46 river equities over this many
// equal-width bins on [0, 1] (each card weighing 1/46).
constexpr int kPotentialTurnBins = 50;

// Equal-population buckets by multiplicity-weighted equity rank.
//
// Values are walked in ascending order and a bucket boundary is cut once the
// cumulative weight reaches j/k of the total, so bucket populations are equal
// wherever ties allow. Hands with identical equity always share a bucket
// (identical float values are never split). When a heavy tie block would
// swallow a boundary the cut is forced to the next distinct value, so every
// bucket still contains at least one distinct equity and exactly `k` buckets
// are used whenever there are at least `k` distinct values.
//
// Throws std::invalid_argument if k < 1, k > kMaxBucketCount, eq and weight
// differ in length, eq is empty, or there are fewer than k distinct values.
std::vector<uint8_t> percentile_buckets(const std::vector<float>& eq,
                                        const std::vector<uint32_t>& weight,
                                        int k);

// Multiplicity-weighted k-means over histograms whose bins are ordered
// outcomes. Distance is the exact 1-D earth mover's distance between the
// histograms' cumulative distributions (sum of |CDF difference| over bins),
// which is O(bins) and exact for ordered bins. Initialisation is
// multiplicity-weighted k-means++ seeded from `seed`; `iters` Lloyd
// iterations follow (stopping early when no label changes). Results are
// deterministic for a given seed and independent of the thread count;
// num_threads <= 0 means hardware concurrency.
//
// Histograms must all have the same number of bins and are treated as
// distributions over those bins (they need not be normalised). Throws
// std::invalid_argument on ragged input, k < 1, k > kMaxBucketCount, iters < 1,
// fewer points than buckets, or a weight-length mismatch.
std::vector<uint8_t> kmeans_buckets(
    const std::vector<std::vector<float>>& hists,
    const std::vector<uint32_t>& weight, int k, uint64_t seed, int iters,
    int num_threads = 0);

// ---------------------------------------------------------------------------
// Potential-aware features (exposed per hand so they can be unit-tested).
// ---------------------------------------------------------------------------

// Histogram (kPotentialTurnBins entries, summing to 1) of the 46 river
// equities of this turn hand, read from a dense street-3 table. Board must
// have 4 cards.
std::vector<float> turn_potential_hist(const std::array<int, 2>& hole,
                                       const std::vector<int>& board,
                                       const std::vector<float>& river,
                                       int bins = kPotentialTurnBins);

// Same feature computed from `river_equity_exact` instead of a dense table.
// Test helper: a turn hand costs 46 exact river evaluations.
std::vector<float> turn_potential_hist_exact(const std::array<int, 2>& hole,
                                             const std::vector<int>& board,
                                             int bins = kPotentialTurnBins);

// Distribution (num_buckets entries, summing to 1) of the 47 turn cards of
// this flop hand over the turn bucket map. `order` maps a turn bucket id to
// its feature dimension (bins are ordered by bucket mean equity); an empty
// `order` means the identity. Board must have 3 cards, `turn_buckets` must be
// a dense street-2 array, and `order` a permutation of [0, num_buckets).
std::vector<float> flop_potential_hist(
    const std::array<int, 2>& hole, const std::vector<int>& board,
    const std::vector<uint8_t>& turn_buckets, int num_buckets,
    const std::vector<int>& order);

// ---------------------------------------------------------------------------
// SIXBM001 file format
// ---------------------------------------------------------------------------
//
// Header (little-endian, native layout): magic[8] "SIXBM001", uint32
// kHandIndexVersion, uint64 size[3], uint32 buckets[3], uint64 equity_hash,
// uint64 content_hash, uint32 method_len, uint32 params_len; then the method
// and params bytes, then the three dense uint8 arrays in flop/turn/river
// order. `content_hash` is an FNV-1a hash over the arrays only, so a
// bit-flipped bucket is refused at open.

// Atomic save (.tmp + rename). Each array must be exactly the indexer size for
// its street; bucket counts are derived as max(array) + 1.
void write_bucket_map(const std::string& path,
                      const std::array<std::vector<uint8_t>, 3>& buckets,
                      const std::string& method, const std::string& params,
                      uint64_t equity_hash);

// Read-only mmap view of a SIXBM001 file. open() refuses a wrong magic, hand
// index version, street size, bucket count (in the header or in the data),
// file length or content hash, and throws std::runtime_error otherwise.
class BucketMap {
public:
    static std::shared_ptr<const BucketMap> open(const std::string& path);
    ~BucketMap();

    BucketMap(const BucketMap&) = delete;
    BucketMap& operator=(const BucketMap&) = delete;

    // street 1=flop 2=turn 3=river. Throws std::out_of_range for idx >= size
    // and std::invalid_argument for a bad street.
    uint8_t bucket(int street, uint64_t idx) const;
    // std::invalid_argument for a bad street.
    int num_buckets(int street) const;
    uint64_t content_hash() const { return content_hash_; }
    // Header metadata: the equity store this map was built from, and the
    // method/params strings the builder recorded.
    uint64_t equity_hash() const { return equity_hash_; }
    const std::string& method() const { return method_; }
    const std::string& params() const { return params_; }
    const std::string& path() const { return path_; }

private:
    BucketMap() = default;

    const unsigned char* base_ = nullptr;
    size_t length_ = 0;
    std::array<const uint8_t*, 3> data_{};
    std::array<uint64_t, 3> sizes_{};
    std::array<uint32_t, 3> buckets_{};
    uint64_t content_hash_ = 0;
    uint64_t equity_hash_ = 0;
    std::string method_;
    std::string params_;
    std::string path_;
};

// Build a map from an equity store and save it as SIXBM001.
//
// method "percentile" buckets each street on its own exact equity. method
// "potential" keeps the river percentile, derives the turn from k-means over
// each turn hand's 46-river-equity histogram (kPotentialTurnBins equal-width
// bins) and the flop from k-means over each flop hand's distribution over the
// 47 turn cards' turn buckets (bins ordered by bucket mean equity). Weights
// are always HandIndexer::multiplicity. num_threads <= 0 means hardware
// concurrency; results do not depend on the thread count.
//
// Throws std::invalid_argument for an unknown method or a count outside
// [1, kMaxBucketCount], and std::runtime_error if the equity store is missing,
// corrupt or sized for a different indexer.
void build_bucket_map(const std::string& equity_path, const std::string& out,
                      const std::string& method, std::array<int, 3> counts,
                      uint64_t seed, int num_threads);

}  // namespace sixmax
