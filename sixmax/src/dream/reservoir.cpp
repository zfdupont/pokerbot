#include "dream/reservoir.h"
#include <stdexcept>

namespace sixmax {

void WeightedReservoir::add(torch::Tensor features, torch::Tensor target,
                            float weight, std::mt19937_64& rng) {
    std::lock_guard<std::mutex> lock(mu_);
    ++n_seen_;
    Entry e{features.clone(), target.clone(), weight};
    if (entries_.size() < capacity_) {
        entries_.push_back(std::move(e));
    } else {
        // Replace random existing entry with probability capacity/n_seen
        size_t idx = std::uniform_int_distribution<size_t>(0, n_seen_ - 1)(rng);
        if (idx < capacity_) entries_[idx] = std::move(e);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
WeightedReservoir::sample_batch(size_t batch_size, std::mt19937_64& rng) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.empty())
        throw std::runtime_error("WeightedReservoir::sample_batch: empty reservoir");
    size_t n = entries_.size();
    std::vector<torch::Tensor> fs, ts;
    std::vector<float> ws;
    fs.reserve(batch_size); ts.reserve(batch_size); ws.reserve(batch_size);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        const Entry& e = entries_[dist(rng)];
        fs.push_back(e.features);
        ts.push_back(e.target);
        ws.push_back(e.weight);
    }
    return {torch::stack(fs), torch::stack(ts),
            torch::tensor(ws, torch::kFloat32)};
}

size_t WeightedReservoir::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

void WeightedReservoir::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
    n_seen_ = 0;
}

}  // namespace sixmax
