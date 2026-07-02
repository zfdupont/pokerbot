#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "cfr/trainer.h"
#include "net/mlp.h"
#include "net/features.h"
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
    // pot, stack: raw float values (normalized internally)
    // to_call: amount player must call (0 if facing check / acting first)
    // raises_per_street: list of 4 ints
    // position: 0 or 1
    py::dict get_action_probs(
        std::vector<int> hole_cards,
        std::vector<int> board_cards,
        int street, float pot, float stack,
        float to_call,
        std::vector<int> raises_per_street,
        int position)
    {
        // Build a synthetic AbstractState for feature encoding + legal action derivation
        AbstractState s{};
        s.hole_cards[position][0] = hole_cards[0];
        s.hole_cards[position][1] = hole_cards[1];
        s.board = std::vector<Card>(board_cards.begin(), board_cards.end());
        s.street = street;
        s.pot = pot;
        s.stacks[position] = stack;
        s.stacks[1 - position] = stack;  // approximation
        // Set current_bet and player_bets so that to_call = current_bet - player_bets[position]
        s.player_bets = {0.0f, 0.0f};
        s.current_bet = to_call;  // player_bets[position]=0, so to_call = current_bet - 0
        s.betting_history = {0, 0, 0, 0};
        for (int i = 0; i < 4 && i < (int)raises_per_street.size(); ++i)
            s.betting_history[i] = raises_per_street[i];
        s.folded = {false, false};
        s.to_act = {position};

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

PYBIND11_MODULE(neural_cfr, m) {
    m.doc() = "Deep CFR neural network strategy -- C++ core via libtorch";

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<size_t, size_t, float, int>(),
             py::arg("reservoir_size")  = DEFAULT_RESERVOIR_SIZE,
             py::arg("batch_size")      = DEFAULT_BATCH_SIZE,
             py::arg("lr")              = DEFAULT_LR,
             py::arg("train_interval")  = DEFAULT_TRAIN_INTERVAL)
        .def("run",        &Trainer::run,        py::arg("iterations"))
        .def("checkpoint", &Trainer::checkpoint, py::arg("path"))
        .def("load",       &Trainer::load,       py::arg("path"));

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
             py::arg("position"));
}
