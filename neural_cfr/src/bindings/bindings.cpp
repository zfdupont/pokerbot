#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "cfr/trainer.h"
#include "cfr/traversal.h"
#include "net/mlp.h"
#include "net/features.h"
#include "net/inference.h"
#include "game/abstract_state.h"

namespace py = pybind11;

// Strategy wrapper: loads a strat_ network from checkpoint, provides inference.
class Strategy {
public:
    explicit Strategy(const std::string& checkpoint_path) {
        std::ifstream f(checkpoint_path);
        if (!f.good())
            throw std::runtime_error("Checkpoint not found: " + checkpoint_path);
        torch::serialize::InputArchive root, s;
        root.load_from(checkpoint_path);
        root.read("strat", s);
        net_.load(s);
        net_.eval();
    }

    // Returns action->probability map for legal actions.
    // hole_cards: [c0, c1] as 0-51 ints (player's own cards)
    // board_cards: 0-5 cards as 0-51 ints
    // street: 0-3
    // pot: total pot INCLUDING all street bets (same as during training traversal)
    // stack: acting player's remaining stack (normalized internally)
    // to_call: amount player must call (0 if facing check / acting first)
    // raises_per_street: list of 4 ints (clamped to [0,2] internally)
    // position: 0 or 1
    // my_street_bet: chips committed this street by the acting player (-1 = unknown)
    // opp_street_bet: chips committed this street by the opponent (-1 = unknown)
    // If either sentinel is -1, falls back to player_bets={0, to_call}.
    py::dict get_action_probs(
        std::vector<int> hole_cards,
        std::vector<int> board_cards,
        int street, float pot, float stack,
        float to_call,
        std::vector<int> raises_per_street,
        int position,
        float my_street_bet  = -1.0f,
        float opp_street_bet = -1.0f)
    {
        auto s = make_inference_state(hole_cards, board_cards, street, pot,
                                      stack, to_call, raises_per_street,
                                      position, my_street_bet, opp_street_bet);
        auto feat = encode_features(s, position);

        torch::NoGradGuard no_grad;
        auto logits = net_.forward(feat.unsqueeze(0)).squeeze(0);

        // Get legal actions for this state
        auto legal = s.legal_actions();

        // Softmax over legal actions
        std::vector<float> legal_logits;
        for (auto& a : legal)
            legal_logits.push_back(logits[action_idx(a)].item<float>());
        float maxl = *std::max_element(legal_logits.begin(), legal_logits.end());
        float sum = 0.0f;
        for (auto& l : legal_logits) { l = std::exp(l - maxl); sum += l; }
        for (auto& l : legal_logits) l /= sum;

        py::dict result;
        for (size_t i = 0; i < legal.size(); ++i)
            result[py::str(legal[i])] = legal_logits[i];
        return result;
    }

private:
    MLP net_;

    static int action_idx(const std::string& a) {
        static const std::array<std::string, 6> ALL =
            {"fold","check","call","b0.5","b1.0","allin"};
        for (int i = 0; i < 6; ++i) if (ALL[i] == a) return i;
        throw std::runtime_error("unknown action: " + a);
    }
};

// AdvantageProbe: loads adv0_ and adv1_ nets, exposes regret-matched action probs
// and raw advantage values. Used to diagnose whether the advantage nets have
// learned hand-strength discrimination even when the strategy net hasn't.
class AdvantageProbe {
public:
    explicit AdvantageProbe(const std::string& checkpoint_path) {
        std::ifstream f(checkpoint_path);
        if (!f.good())
            throw std::runtime_error("Checkpoint not found: " + checkpoint_path);
        torch::serialize::InputArchive root, a0, a1;
        root.load_from(checkpoint_path);
        root.read("adv0", a0);
        root.read("adv1", a1);
        adv0_.load(a0);
        adv1_.load(a1);
        adv0_.eval();
        adv1_.eval();
    }

    // Returns action->probability via regret matching (ReLU + normalize),
    // identical to what traversal.cpp does at traverser nodes.
    py::dict get_advantage_probs(
        int player,
        std::vector<int> hole_cards,
        std::vector<int> board_cards,
        int street, float pot, float stack,
        float to_call,
        std::vector<int> raises_per_street,
        int position,
        float my_street_bet  = -1.0f,
        float opp_street_bet = -1.0f)
    {
        auto legal = make_inference_state(hole_cards, board_cards, street, pot, stack,
                                         to_call, raises_per_street, position,
                                         my_street_bet, opp_street_bet).legal_actions();
        auto logits = _forward(player, hole_cards, board_cards, street, pot, stack,
                               to_call, raises_per_street, position,
                               my_street_bet, opp_street_bet);

        std::array<float, 6> adv{};
        for (int i = 0; i < 6; ++i) adv[i] = logits[i].item<float>();
        std::vector<int> legal_idx;
        for (auto& a : legal) legal_idx.push_back(action_idx(a));
        // Same regret matching (incl. argmax fallback) as traversal.
        auto probs = regret_match(adv, legal_idx);

        py::dict result;
        for (size_t i = 0; i < legal.size(); ++i)
            result[py::str(legal[i])] = probs[i];
        return result;
    }

    // Returns raw advantage logits (before regret matching) for each legal action.
    py::dict get_raw_advantages(
        int player,
        std::vector<int> hole_cards,
        std::vector<int> board_cards,
        int street, float pot, float stack,
        float to_call,
        std::vector<int> raises_per_street,
        int position,
        float my_street_bet  = -1.0f,
        float opp_street_bet = -1.0f)
    {
        auto legal = make_inference_state(hole_cards, board_cards, street, pot, stack,
                                         to_call, raises_per_street, position,
                                         my_street_bet, opp_street_bet).legal_actions();
        auto logits = _forward(player, hole_cards, board_cards, street, pot, stack,
                               to_call, raises_per_street, position,
                               my_street_bet, opp_street_bet);

        py::dict result;
        for (auto& a : legal)
            result[py::str(a)] = logits[action_idx(a)].item<float>();
        return result;
    }

private:
    MLP adv0_, adv1_;

    static int action_idx(const std::string& a) {
        static const std::array<std::string, 6> ALL =
            {"fold","check","call","b0.5","b1.0","allin"};
        for (int i = 0; i < 6; ++i) if (ALL[i] == a) return i;
        throw std::runtime_error("unknown action: " + a);
    }

    torch::Tensor _forward(
        int player,
        const std::vector<int>& hole_cards,
        const std::vector<int>& board_cards,
        int street, float pot, float stack,
        float to_call,
        const std::vector<int>& raises_per_street,
        int position,
        float my_street_bet  = -1.0f,
        float opp_street_bet = -1.0f)
    {
        auto s = make_inference_state(hole_cards, board_cards, street, pot, stack,
                                      to_call, raises_per_street, position,
                                      my_street_bet, opp_street_bet);
        auto feat = encode_features(s, position);
        MLP& net = (player == 0) ? adv0_ : adv1_;
        torch::NoGradGuard no_grad;
        return net.forward(feat.unsqueeze(0)).squeeze(0);
    }
};

PYBIND11_MODULE(neural_cfr, m) {
    m.doc() = "Deep CFR neural network strategy -- C++ core via libtorch";

    // evaluate_hand(cards) -> int  (lower = better; pass 5 or 7 card ints 0-51)
    m.def("evaluate_hand", [](std::vector<int> cards) -> uint32_t {
        if (cards.size() == 7) {
            std::array<Card, 7> a; std::copy(cards.begin(), cards.end(), a.begin());
            return evaluate_7card(a);
        } else if (cards.size() == 5) {
            std::array<Card, 5> a; std::copy(cards.begin(), cards.end(), a.begin());
            return evaluate_5card(a);
        }
        throw std::runtime_error("evaluate_hand requires 5 or 7 cards");
    }, py::arg("cards"), "Evaluate a 5- or 7-card hand (lower = better)");

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<size_t, size_t, float, int, int, float, int, bool>(),
             py::arg("reservoir_size")  = DEFAULT_RESERVOIR_SIZE,
             py::arg("batch_size")      = DEFAULT_BATCH_SIZE,
             py::arg("lr")              = DEFAULT_LR,
             py::arg("train_interval")  = DEFAULT_TRAIN_INTERVAL,
             py::arg("num_threads")     = DEFAULT_NUM_THREADS,
             py::arg("epsilon")         = DEFAULT_EPSILON,
             py::arg("sgd_steps")       = DEFAULT_SGD_STEPS,
             py::arg("reinit_adv")      = DEFAULT_REINIT_ADV)
        .def("run",            &Trainer::run,            py::arg("iterations"))
        .def("train_strategy", &Trainer::train_strategy, py::arg("sgd_steps") = -1,
             "Retrain the strategy net on M_pi (called automatically by checkpoint)")
        .def("checkpoint",     &Trainer::checkpoint,     py::arg("path"))
        .def("load",           &Trainer::load,           py::arg("path"))
        .def("total_iterations", &Trainer::total_iterations,
             "Cumulative traversal-pair count (persists across checkpoints)");

    py::class_<Strategy>(m, "Strategy")
        .def(py::init<const std::string&>(), py::arg("checkpoint_path"))
        .def("get_action_probs", &Strategy::get_action_probs,
             py::arg("hole_cards"),
             py::arg("board_cards"),
             py::arg("street"),
             py::arg("pot"),
             py::arg("stack"),
             py::arg("to_call"),
             py::arg("raises_per_street"),
             py::arg("position"),
             py::arg("my_street_bet")  = -1.0f,
             py::arg("opp_street_bet") = -1.0f);

    py::class_<AdvantageProbe>(m, "AdvantageProbe")
        .def(py::init<const std::string&>(), py::arg("checkpoint_path"))
        .def("get_advantage_probs", &AdvantageProbe::get_advantage_probs,
             py::arg("player"),
             py::arg("hole_cards"),
             py::arg("board_cards"),
             py::arg("street"),
             py::arg("pot"),
             py::arg("stack"),
             py::arg("to_call"),
             py::arg("raises_per_street"),
             py::arg("position"),
             py::arg("my_street_bet")  = -1.0f,
             py::arg("opp_street_bet") = -1.0f,
             "Regret-matched action probs from the advantage net (same as traversal-time strategy)")
        .def("get_raw_advantages", &AdvantageProbe::get_raw_advantages,
             py::arg("player"),
             py::arg("hole_cards"),
             py::arg("board_cards"),
             py::arg("street"),
             py::arg("pot"),
             py::arg("stack"),
             py::arg("to_call"),
             py::arg("raises_per_street"),
             py::arg("position"),
             py::arg("my_street_bet")  = -1.0f,
             py::arg("opp_street_bet") = -1.0f,
             "Raw advantage logits before regret matching — negative = action is being avoided");
}
