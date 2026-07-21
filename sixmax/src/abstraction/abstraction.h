#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    explicit Abstraction(const AbstractionConfig& cfg);  // builds edges
    Abstraction(const AbstractionConfig& cfg,
                std::array<std::vector<double>, 3> edges);  // from artifact
    // The runtime bucket cache below holds a std::mutex per shard, which would
    // implicitly delete the move constructor and break the by-value/std::move
    // handoff in BlueprintStrategy. Provide moves that transfer only the
    // artifact state (cfg_/edges_) and leave the destination's cache freshly
    // empty — the cache is a deterministic accelerator, so an empty cache after
    // a move is always correct. Copies stay deleted (mutex is non-copyable);
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

private:
    AbstractionConfig cfg_;
    std::array<std::vector<double>, 3> edges_;  // [0]=flop [1]=turn [2]=river

    // Deterministic (hole, board) -> bucket memo. bucket() is const and hit
    // concurrently by up to 8 traversal workers, so the cache is mutable and
    // sharded (mutex+map per shard, keyed by cache_key % kBucketCacheShards)
    // to keep lock contention low — a single global lock would serialise every
    // lookup and defeat the optimisation. Purely a runtime accelerator: NOT
    // hashed, serialised, copied, or moved with the artifact (see hash()).
    static constexpr size_t kBucketCacheShards = 64;
    struct BucketCacheShard {
        std::mutex mu;
        std::unordered_map<uint64_t, int> map;
    };
    mutable std::array<BucketCacheShard, kBucketCacheShards> bucket_cache_;
};

}  // namespace sixmax
