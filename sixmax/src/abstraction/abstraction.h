#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "abstraction/abstract_key.h"
#include "abstraction/bucket_cache.h"

namespace sixmax {

class BucketMap;

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

// Postflop equity bucketing, in two kinds:
//
//  * legacy (default): equity-percentile buckets. Edges are per-street equity
//    quantiles estimated once from quantile_samples random (hole, board)
//    draws, so buckets are (approximately) equally populated.
//  * map-backed: a memory-mapped SIXBM001 bucket map. bucket() is
//    map[street][HandIndexer::index(hole, board)] — no rollouts, no memo
//    cache, and num_buckets() comes from the map header.
//
// Street is inferred from the board size (3=flop, 4=turn, 5=river) either way.
// Preflop is not bucketed by this class (the caller uses preflop_class).
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
    // Map-backed kind. The map is immutable after open, so a shared read-only
    // mmap view can be handed to several Abstraction owners.
    explicit Abstraction(std::shared_ptr<const BucketMap> map);
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
    bool is_map() const { return map_ != nullptr; }
    const BucketMap* bucket_map() const { return map_.get(); }
    const AbstractionConfig& config() const { return cfg_; }
    const std::array<std::vector<double>, 3>& edges() const { return edges_; }
    // Pot layout part of the infoset key. Defaults to the legacy 7/15/40 BB
    // layout; non-legacy layouts are validated and hashed into hash().
    void set_pot_layout(PotLayout layout);
    const PotLayout& pot_layout() const { return pot_layout_; }
    // Opt-in last-raise-class key field. Off (default) keeps keys and the
    // hash byte-identical to the historical artifacts; LastRaise adds bits
    // 30-41 and mixes a tag into hash(). Throws std::invalid_argument for an
    // unknown mode.
    void set_bet_history(BetHistory mode);
    BetHistory bet_history() const { return bet_history_; }
    uint64_t pack_key(int card, int street,
                      const std::array<uint8_t, 4>& raises, double pot_bb,
                      double eff_stack_bb, int live, int after,
                      const std::array<uint8_t, 4>& last_raise = {}) const;
    // Config + edges (legacy) or kind tag + map content hash (map-backed);
    // part of the artifact contract.
    uint64_t hash() const;
    size_t bucket_cache_size() const;  // diagnostics / tests

private:
    AbstractionConfig cfg_;
    std::array<std::vector<double>, 3> edges_;  // [0]=flop [1]=turn [2]=river
    PotLayout pot_layout_;                      // legacy by default
    BetHistory bet_history_ = BetHistory::Off;  // opt-in key field

    // Non-null for the map-backed kind; cfg_/edges_/bucket_cache_ unused then.
    std::shared_ptr<const BucketMap> map_;

    // Deterministic (hole, board) -> bucket memo, sharded + size-bounded. An
    // accelerator only: NOT hashed, serialised, or reconstructed from an
    // artifact (the owning pointer moves with the object; see hash()).
    std::unique_ptr<BucketCache> bucket_cache_;
};

}  // namespace sixmax
