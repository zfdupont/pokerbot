#pragma once
#include <mutex>
#include <random>
#include <torch/torch.h>

namespace sixmax {

// Thread-safe reservoir with weighted replacement sampling.
// Capacity-bounded: when full, each new entry replaces a random existing
// entry with probability capacity/n_seen (standard reservoir algorithm).
class WeightedReservoir {
public:
    explicit WeightedReservoir(size_t capacity) : capacity_(capacity) {}

    // Add one (features, target, weight) triple. Thread-safe.
    void add(torch::Tensor features, torch::Tensor target, float weight,
             std::mt19937_64& rng);

    // Sample batch_size entries uniformly. Returns (features, targets, weights)
    // as stacked tensors. Caller must hold no lock.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_batch(size_t batch_size, std::mt19937_64& rng) const;

    size_t size() const;
    void clear();

private:
    struct Entry { torch::Tensor features, target; float weight; };
    size_t capacity_;
    size_t n_seen_ = 0;
    std::vector<Entry> entries_;
    mutable std::mutex mu_;
};

}  // namespace sixmax
