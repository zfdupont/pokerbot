// sixmax/src/dream/checkpoint.h
#pragma once
#include <string>
#include <torch/torch.h>
#include "blueprint/engine_game.h"
#include "dream/nets.h"
#include "vocab/vocab.h"

namespace sixmax {

// Save SIXDM001 checkpoint. Atomically writes to path (tmp+rename).
// Embeds vocab_hash and iteration count in "meta" sub-archive.
void save_dream_checkpoint(const std::string& path,
                           const DreamMLP& adv_net,
                           const DreamMLP& strat_net,
                           uint64_t iterations,
                           uint64_t vocab_hash);

// Inference-only strategy loaded from a SIXDM001 checkpoint.
// Loads only the "strat" sub-archive.
class DreamStrategy {
public:
    static DreamStrategy load(const std::string& path, torch::Device device,
                              const ActionVocab& vocab);

    // Returns softmax probabilities over all vocab actions.
    // Illegal actions (per legal_mask) are zeroed and the result renormalized.
    std::vector<double> get_probs(const EngineGameState& state) const;

    uint64_t iterations() const { return iterations_; }

private:
    DreamStrategy() = default;
    mutable DreamMLP strat_net_{nullptr};
    torch::Device device_{torch::kCPU};
    uint64_t iterations_ = 0;
    int n_actions_ = 0;
    const ActionVocab* vocab_ = nullptr;
};

}  // namespace sixmax
