// sixmax/src/dream/checkpoint.cpp
#include "dream/checkpoint.h"
#include "dream/features.h"
#include <filesystem>
#include <stdexcept>

namespace sixmax {

static constexpr uint32_t kMagic = 0x5349584D;  // 'SIXM' (SIXDM001)

void save_dream_checkpoint(const std::string& path,
                           const DreamMLP& adv_net,
                           const DreamMLP& strat_net,
                           uint64_t iterations,
                           uint64_t vocab_hash) {
    std::string tmp = path + ".tmp";
    torch::serialize::OutputArchive root;

    // Save advantage and strategy nets
    torch::serialize::OutputArchive adv_ar, strat_ar;
    adv_net->save(adv_ar);
    strat_net->save(strat_ar);
    root.write("adv", adv_ar);
    root.write("strat", strat_ar);

    // Save meta sub-archive with magic, iteration count, and vocab hash
    torch::serialize::OutputArchive meta;
    meta.write("magic",      torch::tensor((int64_t)kMagic));
    meta.write("iterations", torch::tensor((int64_t)iterations));
    meta.write("vocab_hash", torch::tensor((int64_t)vocab_hash));
    root.write("meta", meta);

    root.save_to(tmp);
    std::filesystem::rename(tmp, path);
}

DreamStrategy DreamStrategy::load(const std::string& path,
                                  torch::Device device,
                                  const ActionVocab& vocab) {
    torch::serialize::InputArchive root;
    root.load_from(path);

    // Read and verify meta
    torch::serialize::InputArchive meta;
    root.read("meta", meta);

    torch::Tensor magic_t;
    meta.read("magic", magic_t);
    if ((uint32_t)magic_t.item<int64_t>() != kMagic)
        throw std::runtime_error("DreamStrategy::load: not a SIXDM001 checkpoint");

    torch::Tensor vh_t;
    meta.read("vocab_hash", vh_t);
    uint64_t saved_hash = (uint64_t)vh_t.item<int64_t>();
    if (saved_hash != vocab.hash())
        throw std::runtime_error("DreamStrategy::load: vocab hash mismatch — "
                                 "checkpoint was trained with a different action set");

    torch::Tensor iter_t;
    meta.read("iterations", iter_t);

    DreamStrategy s;
    s.device_    = device;
    s.vocab_     = &vocab;
    s.n_actions_ = (int)vocab.size();
    s.iterations_ = (uint64_t)iter_t.item<int64_t>();

    // Load strategy net only (not "adv")
    s.strat_net_ = DreamMLP(FEATURE_DIM, 256, 3, s.n_actions_);
    torch::serialize::InputArchive strat_ar;
    root.read("strat", strat_ar);
    s.strat_net_->load(strat_ar);
    s.strat_net_->to(device);
    s.strat_net_->eval();

    return s;
}

std::vector<double> DreamStrategy::get_probs(const EngineGameState& state) const {
    torch::NoGradGuard no_grad;
    auto feat   = encode_state(state).to(device_).unsqueeze(0);
    auto logits = strat_net_->forward(feat).squeeze(0).cpu();

    // legal_mask() takes a non-const ref; use a local mutable variable
    std::vector<uint8_t> mask;
    state.legal_mask(mask);

    // Zero illegal logits before softmax
    auto logits_a = logits.accessor<float, 1>();
    for (int i = 0; i < n_actions_; ++i)
        if (!mask[i]) logits_a[i] = -1e9f;

    auto probs   = torch::softmax(logits, 0);
    auto probs_a = probs.accessor<float, 1>();

    std::vector<double> result(n_actions_);
    for (int i = 0; i < n_actions_; ++i)
        result[i] = (double)probs_a[i];
    return result;
}

}  // namespace sixmax
