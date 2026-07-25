#include "dream/nets.h"

namespace sixmax {

DreamMLPImpl::DreamMLPImpl(int input_dim, int hidden_size,
                          int n_layers, int output_dim) {
    torch::nn::Sequential seq;
    int in = input_dim;
    for (int i = 0; i < n_layers; ++i) {
        seq->push_back(torch::nn::Linear(in, hidden_size));
        seq->push_back(torch::nn::ReLU());
        in = hidden_size;
    }
    seq->push_back(torch::nn::Linear(in, output_dim));
    layers_ = register_module("layers", seq);
}

torch::Tensor DreamMLPImpl::forward(torch::Tensor x) {
    return layers_->forward(x);
}

}  // namespace sixmax
