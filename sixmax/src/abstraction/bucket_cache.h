#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace sixmax {

// How a shard is shrunk when it exceeds the per-shard cap. Called under the
// shard's mutex. Policy is a swappable seam so the eviction strategy can change
// (and be unit-tested) without touching Abstraction or the artifact — the cache
// never affects outputs, only memory/throughput.
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual void on_insert(std::unordered_map<uint64_t, int>& shard) = 0;
    // Recency hook for future clock/LRU policies; default no-op keeps the
    // per-hit path free for the clear-on-full default.
    virtual void on_hit(uint64_t /*key*/) {}
};

// Clear the whole shard on overflow. Zero per-hit cost, which matters because
// the cache is hit frequently by up to 8 threads under the shard lock, and the
// useful working set is small/short-lived. Documented upgrade if clear-storms
// appear: clock/second-chance (near-LRU quality, no per-hit splice) — not LRU,
// whose move-to-front would land on the hot path.
class ClearOnFullPolicy final : public EvictionPolicy {
public:
    void on_insert(std::unordered_map<uint64_t, int>& shard) override {
        shard.clear();
    }
};

// Sharded, size-bounded, thread-safe memo keyed by an opaque 64-bit id.
// Purely an accelerator: its cap/policy are deliberately absent from every
// artifact (never hashed, serialised, or reconstructed from a checkpoint).
class BucketCache {
public:
    BucketCache(size_t cap_per_shard, std::unique_ptr<EvictionPolicy> policy)
        : cap_(cap_per_shard ? cap_per_shard : 1),
          policy_(std::move(policy)) {}

    bool get(uint64_t key, int& out) {
        Shard& sh = shard_for(key);
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.map.find(key);
        if (it == sh.map.end()) return false;
        out = it->second;
        policy_->on_hit(key);
        return true;
    }

    void put(uint64_t key, int value) {
        Shard& sh = shard_for(key);
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.map[key] = value;
        if (sh.map.size() > cap_) policy_->on_insert(sh.map);
    }

    size_t size() const {  // diagnostics / tests
        size_t n = 0;
        for (const Shard& sh : shards_) {
            std::lock_guard<std::mutex> lk(sh.mu);
            n += sh.map.size();
        }
        return n;
    }

private:
    static constexpr size_t kShards = 64;
    struct Shard {
        mutable std::mutex mu;
        std::unordered_map<uint64_t, int> map;
    };
    Shard& shard_for(uint64_t key) {
        // Same coarse distribution as before; the exact shard is irrelevant to
        // outputs (a lookup only ever returns the value it would recompute).
        return shards_[key % kShards];
    }

    std::array<Shard, kShards> shards_{};
    size_t cap_;
    std::unique_ptr<EvictionPolicy> policy_;
};

}  // namespace sixmax
