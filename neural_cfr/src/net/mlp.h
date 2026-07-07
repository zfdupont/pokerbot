#pragma once
#include <torch/torch.h>

constexpr int64_t DEFAULT_HIDDEN_DIM = 256;

// Shared MLP architecture for both advantage and strategy networks.
// Output activation differs: advantage net uses raw logits,
// strategy net applies softmax at inference (handled by caller).
struct MLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    MLP(int64_t input_dim = 134, int64_t hidden_dim = DEFAULT_HIDDEN_DIM, int64_t output_dim = 6);
    torch::Tensor forward(torch::Tensor x);  // returns raw logits
    // Reinitialize all layers in place (Deep CFR retrains nets from scratch
    // each CFR iteration). Pair with a fresh optimizer.
    void reset_parameters();
};
