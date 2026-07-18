#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/safe_eval.h"
#include "vocab/vocab.h"
#include "blueprint/game.h"
#include "blueprint/kuhn.h"
#include "blueprint/mccfr.h"

namespace py = pybind11;

PYBIND11_MODULE(sixmax, m) {
    m.doc() = "Six-max blueprint + search subsystem";
    py::class_<safe_eval::HandRank>(m, "HandRank")
        .def("beats", &safe_eval::HandRank::beats)
        .def("ties", &safe_eval::HandRank::ties);
    m.def("rank7", [](const std::vector<int>& codes) {
        if (codes.size() != 7) throw py::value_error("rank7 expects 7 cards");
        std::array<int, 7> a;
        std::copy(codes.begin(), codes.end(), a.begin());
        return safe_eval::rank7(a);
    });
    py::enum_<sixmax::ActionType>(m, "ActionType")
        .value("Fold", sixmax::ActionType::Fold).value("Check", sixmax::ActionType::Check)
        .value("Call", sixmax::ActionType::Call).value("Bet", sixmax::ActionType::Bet)
        .value("AllIn", sixmax::ActionType::AllIn);
    py::enum_<sixmax::SizeUnit>(m, "SizeUnit")
        .value("BB", sixmax::SizeUnit::BB).value("Pot", sixmax::SizeUnit::Pot);
    py::class_<sixmax::AbstractAction>(m, "AbstractAction")
        .def(py::init<sixmax::ActionType, double, sixmax::SizeUnit>())
        .def_readonly("type", &sixmax::AbstractAction::type)
        .def_readonly("size", &sixmax::AbstractAction::size)
        .def_readonly("unit", &sixmax::AbstractAction::unit);
    py::class_<sixmax::BetContext>(m, "BetContext")
        .def(py::init([](double pot, double current_bet, double to_call, double stack) {
            return sixmax::BetContext{pot, current_bet, to_call, stack};
        }), py::kw_only(), py::arg("pot"), py::arg("current_bet"),
            py::arg("to_call"), py::arg("stack"));
    py::class_<sixmax::ActionVocab>(m, "ActionVocab")
        .def(py::init<std::vector<sixmax::AbstractAction>>())
        .def("size", &sixmax::ActionVocab::size)
        .def("at", &sixmax::ActionVocab::at)
        .def("target_bb", &sixmax::ActionVocab::target_bb)
        .def("nearest", &sixmax::ActionVocab::nearest)
        .def("hash", &sixmax::ActionVocab::hash);
    // --- MCCFR game interface (Task 1) ---
    py::class_<sixmax::GameState>(m, "GameState")
        .def("is_terminal", &sixmax::GameState::is_terminal)
        .def("current_player", &sixmax::GameState::current_player)
        .def("legal_mask", [](const sixmax::GameState& s) {
            std::vector<uint8_t> mask;
            s.legal_mask(mask);
            return std::vector<int>(mask.begin(), mask.end());
        })
        .def("infoset_key", &sixmax::GameState::infoset_key)
        .def("apply", &sixmax::GameState::apply)
        .def("utility", &sixmax::GameState::utility);
    py::class_<sixmax::Game>(m, "Game")
        .def("num_players", &sixmax::Game::num_players)
        .def("num_actions", &sixmax::Game::num_actions);
    py::class_<sixmax::KuhnState, sixmax::GameState>(m, "KuhnState")
        .def(py::init<int, int>(), py::arg("card0"), py::arg("card1"));
    py::class_<sixmax::KuhnGame, sixmax::Game>(m, "KuhnGame")
        .def(py::init<>());
    m.def("kuhn_infoset_key", &sixmax::KuhnState::key_for,
          py::arg("card"), py::arg("history_code"));
    // --- MCCFR trainer (Task 2) ---
    py::class_<sixmax::MCCFRTrainer>(m, "MCCFRTrainer")
        .def(py::init<sixmax::Game&, uint64_t>(),
             py::arg("game"), py::arg("seed"),
             py::keep_alive<1, 2>())  // trainer holds Game&; keep game alive
        .def("train", &sixmax::MCCFRTrainer::train, py::arg("iterations"))
        .def("iterations", &sixmax::MCCFRTrainer::iterations)
        .def("num_infosets", &sixmax::MCCFRTrainer::num_infosets)
        .def("average_strategy", &sixmax::MCCFRTrainer::average_strategy,
             py::arg("key"));
    m.def("kuhn_exact_value", &sixmax::kuhn_exact_value);
}
