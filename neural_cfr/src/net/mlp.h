#pragma once
#include <torch/torch.h>

// Shared MLP architecture for both advantage and strategy networks.
// Output activation differs: advantage net uses raw logits,
// strategy net applies softmax at inference (handled by caller).
struct MLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    MLP(int64_t input_dim = 134, int64_t hidden_dim = 256, int64_t output_dim = 6);
    torch::Tensor forward(torch::Tensor x);  // returns raw logits
};
