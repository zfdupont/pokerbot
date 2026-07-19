#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>
#include "blueprint/game.h"
#include "blueprint/mccfr.h"

namespace sixmax {

struct TrainerConfig {
    int num_threads = 1;  // <=0 means hardware_concurrency
    uint64_t seed = 1;
};

using GameFactory = std::function<std::unique_ptr<Game>()>;

// Multithreaded external-sampling MCCFR with linear weighting. Identical
// update rules to MCCFRTrainer; the table is sharded under mutexes and the
// linear-CFR weight comes from one atomic global iteration counter (the
// counter is cumulative across train() calls and resumes). Each worker
// thread owns its own Game instance from the factory — EngineGame mutates
// per-hand state (button rotation) and must never be shared across threads.
class BlueprintTrainer {
public:
    BlueprintTrainer(GameFactory factory, TrainerConfig cfg)
        : factory_(std::move(factory)), cfg_(cfg) {}
    void train(uint64_t iterations);
    uint64_t iterations() const {
        return iter_.load(std::memory_order_relaxed);
    }
    size_t num_infosets() const;
    std::vector<double> average_strategy(uint64_t key) const;
    std::vector<uint64_t> keys() const;
    // Checkpoint seam: merged copies of the sharded table.
    std::unordered_map<uint64_t, InfosetData> export_table() const;
    void import_table(std::unordered_map<uint64_t, InfosetData> table,
                      uint64_t iterations);

private:
    static constexpr int kShards = 64;
    struct Shard {
        mutable std::mutex mu;
        std::unordered_map<uint64_t, InfosetData> map;
    };
    int shard_of(uint64_t key) const {
        return (int)((key * 0x9E3779B97F4A7C15ull) >> 58);
    }
    double traverse(GameState& s, int traverser, double weight,
                    std::mt19937_64& rng, int num_actions);

    GameFactory factory_;
    TrainerConfig cfg_;
    std::array<Shard, kShards> shards_;
    std::atomic<uint64_t> iter_{0};
};

}  // namespace sixmax
