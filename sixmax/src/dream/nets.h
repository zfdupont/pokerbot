#pragma once
#include <torch/torch.h>

namespace sixmax {

// Fully-connected MLP with ReLU activations.
// Architecture: input_dim -> [hidden_size]*n_layers -> output_dim
// Output is raw logits (no softmax/sigmoid).
struct DreamMLPImpl : torch::nn::Module {
    DreamMLPImpl(int input_dim, int hidden_size, int n_layers, int output_dim);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::Sequential layers_{nullptr};
};
TORCH_MODULE(DreamMLP);

}  // namespace sixmax
