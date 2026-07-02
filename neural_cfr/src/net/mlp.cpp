#include "net/mlp.h"

MLP::MLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
    fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, hidden_dim));
    fc4 = register_module("fc4", torch::nn::Linear(hidden_dim, output_dim));
}

torch::Tensor MLP::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    return fc4->forward(x);  // raw logits — no output activation
}
