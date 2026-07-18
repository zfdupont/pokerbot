#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/safe_eval.h"
#include "vocab/vocab.h"

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
}
