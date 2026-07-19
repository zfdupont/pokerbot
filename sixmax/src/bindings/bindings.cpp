#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/safe_eval.h"
#include "vocab/vocab.h"
#include "blueprint/game.h"
#include "blueprint/kuhn.h"
#include "blueprint/mccfr.h"
#include "engine/engine.h"
#include "blueprint/engine_game.h"
#include "abstraction/abstraction.h"

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
            py::arg("to_call"), py::arg("stack"))
        .def_readonly("pot", &sixmax::BetContext::pot)
        .def_readonly("current_bet", &sixmax::BetContext::current_bet)
        .def_readonly("to_call", &sixmax::BetContext::to_call)
        .def_readonly("stack", &sixmax::BetContext::stack);
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
    // --- NLHE engine (Task 3) ---
    py::enum_<sixmax::Street>(m, "Street")
        .value("Preflop", sixmax::Street::Preflop)
        .value("Flop", sixmax::Street::Flop)
        .value("Turn", sixmax::Street::Turn)
        .value("River", sixmax::Street::River);
    py::class_<sixmax::EngineConfig>(m, "EngineConfig")
        .def(py::init([](int n, double stack) {
                 return sixmax::EngineConfig{n, stack};
             }),
             py::arg("num_players"), py::arg("starting_stack") = 100.0)
        .def_readonly("num_players", &sixmax::EngineConfig::num_players)
        .def_readonly("starting_stack", &sixmax::EngineConfig::starting_stack);
    py::class_<sixmax::HandState>(m, "HandState")
        .def(py::init<const sixmax::EngineConfig&, int, std::vector<int>,
                      std::vector<double>>(),
             py::arg("cfg"), py::arg("button"), py::arg("deck"),
             py::arg("stacks") = std::vector<double>{})
        .def_static("deal",
                    [](const sixmax::EngineConfig& c, int button, uint64_t seed) {
                        std::mt19937_64 rng(seed);
                        return sixmax::HandState::deal(c, button, rng);
                    },
                    py::arg("cfg"), py::arg("button"), py::arg("seed"))
        .def("is_terminal", &sixmax::HandState::is_terminal)
        .def("current_player", &sixmax::HandState::current_player)
        .def("street", &sixmax::HandState::street)
        .def("button", &sixmax::HandState::button)
        .def("pot", &sixmax::HandState::pot)
        .def("to_call", &sixmax::HandState::to_call)
        .def("current_bet", &sixmax::HandState::current_bet)
        .def("min_raise_to", &sixmax::HandState::min_raise_to)
        .def("can_raise", &sixmax::HandState::can_raise)
        .def("board", &sixmax::HandState::board)
        .def("hole_cards",
             [](const sixmax::HandState& s, int i) {
                 auto hc = s.hole_cards(i);
                 return std::vector<int>{hc[0], hc[1]};
             })
        .def("stack", [](const sixmax::HandState& s, int i) {
            return s.player(i).stack;
        })
        .def("folded", [](const sixmax::HandState& s, int i) {
            return s.player(i).folded;
        })
        .def("all_in", [](const sixmax::HandState& s, int i) {
            return s.player(i).all_in;
        })
        .def("apply_fold", [](sixmax::HandState& s) {
            s.apply({sixmax::EngineActionType::Fold, 0.0});
        })
        .def("apply_check_call", [](sixmax::HandState& s) {
            s.apply({sixmax::EngineActionType::CheckCall, 0.0});
        })
        .def("apply_raise_to", [](sixmax::HandState& s, double amount) {
            s.apply({sixmax::EngineActionType::RaiseTo, amount});
        })
        .def("payoffs", &sixmax::HandState::payoffs);
    m.def("settle_pots",
          [](const std::vector<double>& totals, const std::vector<bool>& folded,
             const std::vector<int>& ranks) {
              std::vector<uint8_t> f(folded.begin(), folded.end());
              return sixmax::settle_pots(totals, f, ranks);
          },
          py::arg("total_bets"), py::arg("folded"), py::arg("rank_order"));
    // --- Engine <-> vocab bridge (Task 5) ---
    py::class_<sixmax::EngineGameState, sixmax::GameState>(m, "EngineGameState")
        .def(py::init([](const sixmax::EngineConfig& cfg, int button,
                         std::vector<int> deck, const sixmax::ActionVocab* v,
                         std::vector<double> stacks,
                         const sixmax::Abstraction* abstraction) {
                 return sixmax::EngineGameState(
                     sixmax::HandState(cfg, button, std::move(deck),
                                       std::move(stacks)),
                     v, abstraction);
             }),
             py::arg("cfg"), py::arg("button"), py::arg("deck"),
             py::arg("vocab"), py::arg("stacks") = std::vector<double>{},
             py::arg("abstraction") = nullptr,
             py::keep_alive<1, 5>(),   // state holds ActionVocab*
             py::keep_alive<1, 7>())   // state holds Abstraction*
        .def("bet_context", &sixmax::EngineGameState::bet_context)
        .def("abstract_key", &sixmax::EngineGameState::abstract_key);
    py::class_<sixmax::EngineGame, sixmax::Game>(m, "EngineGame")
        .def(py::init<sixmax::EngineConfig, const sixmax::ActionVocab*,
                      const sixmax::Abstraction*>(),
             py::arg("cfg"), py::arg("vocab"),
             py::arg("abstraction") = nullptr,
             py::keep_alive<1, 3>(),   // game holds ActionVocab*
             py::keep_alive<1, 4>())   // game holds Abstraction*
        .def("new_hand", [](sixmax::EngineGame& g, uint64_t seed) {
            std::mt19937_64 rng(seed);
            return g.new_hand(rng);
        }, py::arg("seed"),
           py::keep_alive<0, 1>());  // returned state holds game-owned ptrs
    // --- Card abstraction (Phase 1b Task 1) ---
    m.def("preflop_class", [](const std::vector<int>& hole) {
        if (hole.size() != 2) throw py::value_error("expects 2 cards");
        return sixmax::preflop_class({hole[0], hole[1]});
    });
    m.def("hand_equity",
          [](const std::vector<int>& hole, const std::vector<int>& board,
             int rollouts, uint64_t salt) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::hand_equity({hole[0], hole[1]}, board, rollouts,
                                         salt);
          },
          py::arg("hole"), py::arg("board"), py::arg("rollouts"),
          py::arg("salt"));
    py::class_<sixmax::Abstraction>(m, "Abstraction")
        .def(py::init([](int flop_buckets, int turn_buckets, int river_buckets,
                         int equity_rollouts, int quantile_samples,
                         uint64_t seed) {
                 return sixmax::Abstraction(sixmax::AbstractionConfig{
                     flop_buckets, turn_buckets, river_buckets,
                     equity_rollouts, quantile_samples, seed});
             }),
             py::kw_only(), py::arg("flop_buckets") = 50,
             py::arg("turn_buckets") = 50, py::arg("river_buckets") = 20,
             py::arg("equity_rollouts") = 100,
             py::arg("quantile_samples") = 10000,
             py::arg("seed") = 20260719)
        .def("bucket",
             [](const sixmax::Abstraction& a, const std::vector<int>& hole,
                const std::vector<int>& board) {
                 if (hole.size() != 2) throw py::value_error("expects 2 cards");
                 return a.bucket({hole[0], hole[1]}, board);
             })
        .def("num_buckets", &sixmax::Abstraction::num_buckets)
        .def("edges", &sixmax::Abstraction::edges)
        .def("hash", &sixmax::Abstraction::hash);
}
