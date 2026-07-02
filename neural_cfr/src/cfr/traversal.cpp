#include "cfr/traversal.h"
#include "net/features.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>

static const std::array<std::string, 6> ALL_ACTIONS =
    {"fold", "check", "call", "b0.5", "b1.0", "allin"};

// Regret matching: given raw advantage logits and legal action indices,
// return a probability distribution via ReLU + normalize.
static std::vector<float> regret_match(
    const std::array<float, 6>& advantages,
    const std::vector<int>& legal_indices)
{
    std::vector<float> pos(legal_indices.size());
    float total = 0.0f;
    for (size_t i = 0; i < legal_indices.size(); ++i) {
        pos[i] = std::max(0.0f, advantages[legal_indices[i]]);
        total += pos[i];
    }
    if (total > 0.0f)
        for (auto& p : pos) p /= total;
    else
        for (auto& p : pos) p = 1.0f / pos.size();
    return pos;
}

// Sample index from probability distribution
static int sample_action(const std::vector<float>& probs) {
    static std::mt19937 rng{std::random_device{}()};
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Map action string to index in ALL_ACTIONS
static int action_idx(const std::string& a) {
    for (int i = 0; i < 6; ++i)
        if (ALL_ACTIONS[i] == a) return i;
    return -1;
}

float external_sample(
    const AbstractState& state,
    int traversing_player,
    MLP& adv_net,
    MLP& opp_adv_net,
    MLP& strat_net,
    ReservoirBuffer<BufferEntry>& adv_buffer,
    ReservoirBuffer<BufferEntry>& strat_buffer,
    int iteration)
{
    if (state.is_terminal())
        return state.payoff(traversing_player);

    // Chance node: advance street
    if (state.to_act.empty())
        return external_sample(state.advance_street(), traversing_player,
                               adv_net, opp_adv_net, strat_net, adv_buffer, strat_buffer, iteration);

    int acting = state.acting_player();
    auto legal_strs = state.legal_actions();
    std::vector<int> legal_idx;
    for (auto& a : legal_strs) legal_idx.push_back(action_idx(a));

    // Encode features for acting player
    auto feat_tensor = encode_features(state, acting);
    auto feat_vec = std::vector<float>(
        feat_tensor.data_ptr<float>(),
        feat_tensor.data_ptr<float>() + FEATURE_DIM);

    if (acting == traversing_player) {
        // Query advantage net → regret match → traverse ALL actions
        torch::NoGradGuard no_grad;
        auto logits = adv_net.forward(feat_tensor.unsqueeze(0)).squeeze(0);
        std::array<float, 6> advantages{};
        for (int i = 0; i < 6; ++i) advantages[i] = logits[i].item<float>();

        auto strategy = regret_match(advantages, legal_idx);

        // Traverse all legal actions
        std::array<float, 6> action_values{};
        float node_value = 0.0f;
        for (size_t i = 0; i < legal_strs.size(); ++i) {
            float v = external_sample(state.apply_action(legal_strs[i]),
                                      traversing_player, adv_net, opp_adv_net, strat_net,
                                      adv_buffer, strat_buffer, iteration);
            action_values[legal_idx[i]] = v;
            node_value += strategy[i] * v;
        }

        // Compute advantages and store in M_v
        std::array<float, 6> adv_targets{};
        for (size_t i = 0; i < legal_strs.size(); ++i)
            adv_targets[legal_idx[i]] = action_values[legal_idx[i]] - node_value;

        adv_buffer.add({feat_vec, adv_targets, static_cast<float>(iteration)});

        // Also accumulate strategy for M_π
        std::array<float, 6> strat_targets{};
        for (size_t i = 0; i < legal_strs.size(); ++i)
            strat_targets[legal_idx[i]] = strategy[i];
        strat_buffer.add({feat_vec, strat_targets, static_cast<float>(iteration)});

        return node_value;

    } else {
        // Opponent: query opponent's advantage net → regret-match → sample ONE action
        // (Brown et al. 2019: M_π populated at both traverser and opponent nodes)
        torch::NoGradGuard no_grad;
        auto logits = opp_adv_net.forward(feat_tensor.unsqueeze(0)).squeeze(0);
        std::array<float, 6> advantages{};
        for (int i = 0; i < 6; ++i) advantages[i] = logits[i].item<float>();

        auto strategy = regret_match(advantages, legal_idx);

        // Accumulate strategy for M_π
        std::array<float, 6> strat_targets{};
        for (size_t i = 0; i < legal_idx.size(); ++i)
            strat_targets[legal_idx[i]] = strategy[i];
        strat_buffer.add({feat_vec, strat_targets, static_cast<float>(iteration)});

        int chosen = sample_action(strategy);
        return external_sample(state.apply_action(legal_strs[chosen]),
                               traversing_player, adv_net, opp_adv_net, strat_net,
                               adv_buffer, strat_buffer, iteration);
    }
}
