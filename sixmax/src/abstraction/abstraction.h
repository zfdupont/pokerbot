#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "abstraction/bucket_cache.h"

namespace sixmax {

// Lossless 169-class preflop index (0..168): pairs on the diagonal
// (r*13+r), suited above it (hi*13+lo), offsuit below (lo*13+hi).
// Pure function of the two card codes; card order is irrelevant.
int preflop_class(const std::array<int, 2>& hole);

// Deterministic Monte-Carlo equity of hole vs ONE uniform random opponent
// hand, rolling out the remaining board with the shared evaluator. The RNG
// is seeded from (sorted hole, sorted board, salt): identical inputs always
// return the identical estimate, so bucket assignments derived from it are
// stable across visits, threads, runs, and machines — no cache needed.
double hand_equity(const std::array<int, 2>& hole,
                   const std::vector<int>& board, int rollouts, uint64_t salt);

struct AbstractionConfig {
    int flop_buckets = 50;
    int turn_buckets = 50;
    int river_buckets = 20;
    int equity_rollouts = 100;     // MC rollouts per equity estimate
    int quantile_samples = 10000;  // per street, for percentile edges
    uint64_t seed = 20260719;      // edge sampling + equity salt
};

// Postflop equity-percentile buckets. Edges are per-street equity quantiles
// estimated once from quantile_samples random (hole, board) draws, so
// buckets are (approximately) equally populated. Street inferred from board
// size (3=flop, 4=turn, 5=river).
class Abstraction {
public:
    // Per-shard cap for the memo cache (see bucket_cache.h). A runtime-only
    // tuning knob: never hashed, serialised, or otherwise part of the artifact.
    static constexpr size_t kDefaultBucketCacheCap = 4096;

    explicit Abstraction(const AbstractionConfig& cfg,
                         size_t cache_cap = kDefaultBucketCacheCap);  // builds edges
    Abstraction(const AbstractionConfig& cfg,
                std::array<std::vector<double>, 3> edges,
                size_t cache_cap = kDefaultBucketCacheCap);  // from artifact
    // Moves transfer the cache pointer alongside cfg_/edges_. That is always
    // correct: the cache is a deterministic accelerator (a stale or shared
    // entry still equals the value it would recompute). Copies stay deleted;
    // nothing copies an Abstraction.
    Abstraction(Abstraction&& other) noexcept;
    Abstraction& operator=(Abstraction&& other) noexcept;
    Abstraction(const Abstraction&) = delete;
    Abstraction& operator=(const Abstraction&) = delete;
    int bucket(const std::array<int, 2>& hole,
               const std::vector<int>& board) const;
    int num_buckets(int street) const;  // street 1=flop 2=turn 3=river
    const AbstractionConfig& config() const { return cfg_; }
    const std::array<std::vector<double>, 3>& edges() const { return edges_; }
    uint64_t hash() const;  // config + edges; part of the artifact contract
    size_t bucket_cache_size() const;  // diagnostics / tests

private:
    AbstractionConfig cfg_;
    std::array<std::vector<double>, 3> edges_;  // [0]=flop [1]=turn [2]=river

    // Deterministic (hole, board) -> bucket memo, sharded + size-bounded. An
    // accelerator only: NOT hashed, serialised, or reconstructed from an
    // artifact (the owning pointer moves with the object; see hash()).
    std::unique_ptr<BucketCache> bucket_cache_;
};

}  // namespace sixmax
