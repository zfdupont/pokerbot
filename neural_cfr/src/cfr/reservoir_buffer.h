#pragma once
#include <vector>
#include <array>
#include <random>
#include <cstddef>

template<typename T>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(size_t max_size)
        : max_size_(max_size), n_seen_(0),
          rng_(std::random_device{}()) {
        data_.reserve(max_size);
    }

    void add(T item) {
        ++n_seen_;
        if (data_.size() < max_size_) {
            data_.push_back(std::move(item));
        } else {
            // Uniform random replacement
            size_t idx = std::uniform_int_distribution<size_t>(0, n_seen_ - 1)(rng_);
            if (idx < max_size_)
                data_[idx] = std::move(item);
        }
    }

    const std::vector<T>& data() const { return data_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void clear() { data_.clear(); n_seen_ = 0; }

private:
    size_t max_size_;
    size_t n_seen_;
    std::vector<T> data_;
    std::mt19937_64 rng_;
};

// Entry stored in advantage and strategy buffers.
struct BufferEntry {
    std::vector<float> features;  // 134 floats (stored as vector for easy stacking)
    std::array<float, 6> targets; // advantages or strategy probs, per action
    float weight;                 // iteration number t (linear CFR weighting)
};
