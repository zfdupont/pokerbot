#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include "game/safe_eval.h"
#include "vocab/vocab.h"
#include "blueprint/game.h"
#include "blueprint/kuhn.h"
#include "blueprint/mccfr.h"
#include "blueprint/trainer.h"
#include "engine/engine.h"
#include "blueprint/engine_game.h"
#include "abstraction/abstraction.h"
#include "abstraction/abstract_key.h"
#include "abstraction/bucket_map.h"
#include "abstraction/equity_table.h"
#include "abstraction/hand_index.h"
#include "blueprint/checkpoint.h"
#include "exploit/policy.h"
#include "exploit/lbr.h"
#include "search/vector_cfr.h"
#include "search/kuhn_vector.h"
#include "search/range_tracker.h"
#include "search/subgame.h"
#include "search/leaf_eval.h"
#include "search/search_strategy.h"

namespace py = pybind11;

namespace {

bool same_pot_layout(const sixmax::PotLayout& a,
                     const sixmax::PotLayout& b) {
    if (a.mode != b.mode || a.edges.size() != b.edges.size()) return false;
    for (size_t i = 0; i < a.edges.size(); ++i)
        if (a.edges[i] != b.edges[i]) return false;
    return true;
}

// A bare HandState keeps no per-street raise history, so a blueprint key built
// from it is wrong once anyone has bet beyond the blinds. Only the uniform
// opponent model ignores the key, so the shortcut is safe there.
void require_keyable_hand(const sixmax::LbrEvaluator& e,
                          const sixmax::HandState& s) {
    if (e.lbr_config().uniform_opponents) return;
    bool bet_beyond_blinds = false;
    for (int p = 0; p < s.num_players(); ++p)
        if (s.player(p).total_bet > 1.0 + sixmax::kChipEps)
            bet_beyond_blinds = true;
    const double threshold =
        s.street() == sixmax::Street::Preflop ? 1.0 : 0.0;
    if (s.current_bet() > threshold + sixmax::kChipEps)
        bet_beyond_blinds = true;
    if (bet_beyond_blinds)
        throw py::value_error(
            "LbrEvaluator: a bare HandState carries no per-street raise "
            "history, so blueprint keys are wrong after a bet; pass an "
            "EngineGameState built by applying actions, or set "
            "uniform_opponents");
}

sixmax::EngineGameState wrap_hand(const sixmax::LbrEvaluator& e,
                                  const sixmax::HandState& s) {
    require_keyable_hand(e, s);
    return sixmax::EngineGameState(s, &e.vocab(), &e.abstraction());
}

}  // namespace

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
        .def("raise_class", &sixmax::ActionVocab::raise_class)
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
        .def(py::init<int, int>(), py::arg("card0"), py::arg("card1"))
        .def("card", &sixmax::KuhnState::card, py::arg("player"),
             "Return the card held by the given player (0=J, 1=Q, 2=K).")
        .def("history_code", &sixmax::KuhnState::history_code,
             "History code: 0=empty, 1=check, 2=bet, 3=check+bet.")
        .def("acting_player", &sixmax::KuhnState::acting_player,
             "Index of the player whose turn it is (0 or 1).");
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
        .def("payoffs", &sixmax::HandState::payoffs)
        .def("clone", &sixmax::HandState::clone)
        .def("reroll_hidden",
             [](sixmax::HandState& s, int hero, std::vector<int> pool,
                uint64_t seed) {
                 std::mt19937_64 rng(seed);
                 s.reroll_hidden(hero, std::move(pool), rng);
             },
             py::arg("hero"), py::arg("pool"), py::arg("seed") = 1);
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
        .def("abstract_key", &sixmax::EngineGameState::abstract_key)
        .def("last_raise_classes", [](const sixmax::EngineGameState& s) {
            auto classes = s.last_raise_classes();
            return std::vector<int>(classes.begin(), classes.end());
        })
        .def("street", [](const sixmax::EngineGameState& s) { return (int)s.street(); })
        .def("board", &sixmax::EngineGameState::board)
        .def("hole_cards", [](const sixmax::EngineGameState& s, int p) {
            auto hc = s.hole_cards(p);
            return std::vector<int>{hc[0], hc[1]};
        });
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
    // --- Suit-canonical hand indexer ---
    m.attr("HAND_INDEX_VERSION") = sixmax::kHandIndexVersion;
    m.def("hand_index_size", [](int street) {
        return sixmax::HandIndexer::instance().size(street);
    }, py::arg("street"));
    m.def("hand_index",
          [](int street, const std::vector<int>& hole,
             const std::vector<int>& board) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::HandIndexer::instance().index(
                  street, {hole[0], hole[1]}, board);
          },
          py::arg("street"), py::arg("hole"), py::arg("board"));
    m.def("hand_unindex",
          [](int street, uint64_t idx) {
              std::array<int, 2> hole{};
              std::vector<int> board;
              sixmax::HandIndexer::instance().unindex(street, idx, hole, board);
              return py::make_tuple(
                  std::vector<int>{hole[0], hole[1]}, board);
          },
          py::arg("street"), py::arg("idx"));
    m.def("hand_multiplicity",
          [](int street, uint64_t idx) {
              return sixmax::HandIndexer::instance().multiplicity(street, idx);
          },
          py::arg("street"), py::arg("idx"));
    // --- Exact equity feature store ---
    m.def("river_equity_exact",
          [](const std::vector<int>& hole, const std::vector<int>& board) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::river_equity_exact({hole[0], hole[1]}, board);
          },
          py::arg("hole"), py::arg("board"));
    m.def("derive_equity_one",
          [](int street, const std::vector<int>& hole,
             const std::vector<int>& board) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::derive_equity_one(street, {hole[0], hole[1]},
                                               board);
          },
          py::arg("street"), py::arg("hole"), py::arg("board"));
    m.def("derive_equity",
          [](int street, const std::vector<float>& next_street,
             int num_threads) {
              return sixmax::derive_equity(street, next_street, num_threads);
          },
          py::arg("street"), py::arg("next_street"),
          py::arg("num_threads") = 0);
    m.def("build_equity_store",
          [](const std::string& path, int num_threads) {
              std::fprintf(stderr, "[equity] river table\n");
              auto river = sixmax::build_river_equity(num_threads);
              std::fprintf(stderr, "[equity] turn table (from river)\n");
              auto turn = sixmax::derive_equity(2, river, num_threads);
              std::fprintf(stderr, "[equity] flop table (from turn)\n");
              auto flop = sixmax::derive_equity(1, turn, num_threads);
              sixmax::EquityStore store;
              store.eq = {std::move(flop), std::move(turn),
                          std::move(river)};
              const uint64_t hash = sixmax::equity_content_hash(store);
              store.content_hash = hash;
              sixmax::save_equity_store(path, store);
              std::fprintf(stderr, "[equity] wrote %s (hash %016llx)\n",
                           path.c_str(),
                           static_cast<unsigned long long>(hash));
              return hash;
          },
          py::arg("path"), py::arg("num_threads") = 0);
    m.def("equity_store_lookup",
          [](const std::string& path, int street, uint64_t idx) {
              return sixmax::equity_store_lookup(path, street, idx);
          },
          py::arg("path"), py::arg("street"), py::arg("idx"));
    m.def("_write_equity_store_for_test",
          [](const std::string& path, const std::vector<float>& flop,
             const std::vector<float>& turn,
             const std::vector<float>& river) {
              sixmax::EquityStore store;
              store.eq = {flop, turn, river};
              sixmax::save_equity_store(path, store, false);
              return sixmax::equity_content_hash(store);
          },
          py::arg("path"), py::arg("flop"), py::arg("turn"),
          py::arg("river"));
    m.def("_load_equity_store_for_test",
          [](const std::string& path, bool strict) {
              try {
                  auto store = strict
                      ? sixmax::load_equity_store(path)
                      : sixmax::load_equity_store_skip_size_check(path);
                  return py::make_tuple(store.eq[0], store.eq[1], store.eq[2],
                                        store.content_hash);
              } catch (const std::exception& e) {
                  throw py::value_error(e.what());
              }
          },
          py::arg("path"), py::arg("strict") = false);
    m.def("_river_equities_for_board",
          [](const std::vector<int>& board) {
              return sixmax::river_equities_for_board(board);
          },
          py::arg("board"));
    // --- Bucket maps (percentile / potential-aware builders + mmap reader) ---
    m.def("_percentile_buckets",
          [](const std::vector<float>& eq, const std::vector<uint32_t>& weight,
             int k) {
              return sixmax::percentile_buckets(eq, weight, k);
          },
          py::arg("eq"), py::arg("weight"), py::arg("k"));
    m.def("_kmeans_buckets",
          [](const std::vector<std::vector<float>>& hists,
             const std::vector<uint32_t>& weight, int k, uint64_t seed,
             int iters, int num_threads) {
              return sixmax::kmeans_buckets(hists, weight, k, seed, iters,
                                            num_threads);
          },
          py::arg("hists"), py::arg("weight"), py::arg("k"),
          py::arg("seed"), py::arg("iters"), py::arg("num_threads") = 0);
    m.def("_turn_potential_hist",
          [](const std::vector<int>& hole, const std::vector<int>& board,
             int bins) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              return sixmax::turn_potential_hist_exact({hole[0], hole[1]},
                                                       board, bins);
          },
          py::arg("hole"), py::arg("board"),
          py::arg("bins") = sixmax::kPotentialTurnBins);
    m.def("_flop_potential_hist",
          [](const std::vector<int>& hole, const std::vector<int>& board,
             const py::array_t<uint8_t, py::array::c_style>& turn_buckets,
             int num_buckets, const std::vector<int>& order) {
              if (hole.size() != 2) throw py::value_error("expects 2 cards");
              const auto info = turn_buckets.request();
              if (info.ndim != 1)
                  throw py::value_error("expected a 1-D uint8 array");
              const auto* data = static_cast<const uint8_t*>(info.ptr);
              return sixmax::flop_potential_hist(
                  {hole[0], hole[1]}, board,
                  std::vector<uint8_t>(data, data + info.shape[0]),
                  num_buckets, order);
          },
          py::arg("hole"), py::arg("board"), py::arg("turn_buckets"),
          py::arg("num_buckets"), py::arg("order"));
    m.def("_bucket_spread",
          [](const std::vector<std::vector<float>>& features,
             const std::vector<uint32_t>& weight,
             const std::vector<uint8_t>& buckets, int k, int num_threads) {
              try {
                  const auto s = sixmax::bucket_spread(features, weight,
                                                       buckets, k, num_threads);
                  return py::make_tuple(s.error, s.baseline);
              } catch (const std::invalid_argument& e) {
                  throw py::value_error(e.what());
              }
          },
          py::arg("features"), py::arg("weight"), py::arg("buckets"),
          py::arg("k"), py::arg("num_threads") = 0);
    m.def("score_bucket_maps",
          [](const std::string& equity_path,
             const std::vector<std::string>& map_paths, int num_threads) {
              std::vector<sixmax::MapScore> scores;
              try {
                  py::gil_scoped_release release;
                  scores = sixmax::score_bucket_maps(equity_path, map_paths,
                                                     num_threads);
              } catch (const std::exception& e) {
                  throw py::value_error(e.what());
              }
              py::list out;
              for (const auto& s : scores) {
                  py::dict row;
                  row["map"] = s.path;
                  row["street"] = s.street;
                  row["buckets"] = s.buckets;
                  row["error"] = s.spread.error;
                  row["baseline"] = s.spread.baseline;
                  out.append(row);
              }
              return out;
          },
          py::arg("equity_path"), py::arg("map_paths"),
          py::arg("num_threads") = 0);
    py::class_<sixmax::BucketMap, std::shared_ptr<sixmax::BucketMap>>(
        m, "BucketMap")
        .def_static("open",
                    [](const std::string& path) {
                        try {
                            // The map is immutable after open; const_cast only
                            // adapts to pybind's shared_ptr<T> holder.
                            return std::const_pointer_cast<sixmax::BucketMap>(
                                sixmax::BucketMap::open(path));
                        } catch (const std::exception& e) {
                            throw py::value_error(e.what());
                        }
                    },
                    py::arg("path"))
        .def("bucket", &sixmax::BucketMap::bucket, py::arg("street"),
             py::arg("idx"))
        .def("num_buckets", &sixmax::BucketMap::num_buckets,
             py::arg("street"))
        .def("content_hash", &sixmax::BucketMap::content_hash)
        .def("equity_hash", &sixmax::BucketMap::equity_hash)
        .def("method", &sixmax::BucketMap::method,
             py::return_value_policy::reference_internal)
        .def("params", &sixmax::BucketMap::params,
             py::return_value_policy::reference_internal)
        .def("path", &sixmax::BucketMap::path,
             py::return_value_policy::reference_internal);
    m.def("build_bucket_map",
          [](const std::string& equity_path, const std::string& out,
             const std::string& method, const std::vector<int>& counts,
             uint64_t seed, int num_threads) {
              if (counts.size() != 3)
                  throw py::value_error("counts must have length 3");
              try {
                  sixmax::build_bucket_map(
                      equity_path, out, method,
                      std::array<int, 3>{counts[0], counts[1], counts[2]},
                      seed, num_threads);
              } catch (const std::exception& e) {
                  throw py::value_error(e.what());
              }
          },
          py::arg("equity_path"), py::arg("out"), py::arg("method"),
          py::arg("counts"), py::arg("seed") = 0,
          py::arg("num_threads") = 0);
    m.def("_write_bucket_map_for_test",
          [](const std::string& path,
             const py::array_t<uint8_t, py::array::c_style>& flop,
             const py::array_t<uint8_t, py::array::c_style>& turn,
             const py::array_t<uint8_t, py::array::c_style>& river,
             const std::string& method) {
              const auto to_vector =
                  [](const py::array_t<uint8_t, py::array::c_style>& array) {
                  const auto info = array.request();
                  if (info.ndim != 1)
                      throw py::value_error("expected a 1-D uint8 array");
                  const auto* data = static_cast<const uint8_t*>(info.ptr);
                  return std::vector<uint8_t>(data,
                                              data + info.shape[0]);
              };
              sixmax::write_bucket_map(
                  path, {to_vector(flop), to_vector(turn), to_vector(river)},
                  method, "", 0);
          },
          py::arg("path"), py::arg("flop"), py::arg("turn"),
          py::arg("river"), py::arg("method"));
    m.def("pot_bucket", [](double pot_bb) { return sixmax::pot_bucket(pot_bb); },
          py::arg("pot_bb"));
    m.def("pack_abstract_key",
          [](int card, int street, const std::vector<int>& raises,
             double pot_bb, int live, int after) {
              if (raises.size() != 4)
                  throw py::value_error("raises must have length 4");
              std::array<uint8_t, 4> r{(uint8_t)raises[0], (uint8_t)raises[1],
                                       (uint8_t)raises[2], (uint8_t)raises[3]};
              return sixmax::pack_abstract_key(card, street, r, pot_bb, live,
                                               after);
          },
          py::arg("card"), py::arg("street"), py::arg("raises"),
          py::arg("pot_bb"), py::arg("live"), py::arg("after"));
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
                         uint64_t seed, size_t cache_cap) {
                 return sixmax::Abstraction(sixmax::AbstractionConfig{
                     flop_buckets, turn_buckets, river_buckets,
                     equity_rollouts, quantile_samples, seed}, cache_cap);
             }),
             py::kw_only(), py::arg("flop_buckets") = 50,
             py::arg("turn_buckets") = 50, py::arg("river_buckets") = 20,
             py::arg("equity_rollouts") = 100,
             py::arg("quantile_samples") = 10000,
             py::arg("seed") = 20260719,
             py::arg("cache_cap") = sixmax::Abstraction::kDefaultBucketCacheCap)
        .def_static("from_bucket_map",
                    [](const std::string& path) {
                        try {
                            return sixmax::Abstraction(
                                sixmax::BucketMap::open(path));
                        } catch (const std::exception& e) {
                            throw py::value_error(e.what());
                        }
                    },
                    py::arg("path"))
        .def("is_map", &sixmax::Abstraction::is_map)
        .def("bucket",
             [](const sixmax::Abstraction& a, const std::vector<int>& hole,
                const std::vector<int>& board) {
                 if (hole.size() != 2) throw py::value_error("expects 2 cards");
                 return a.bucket({hole[0], hole[1]}, board);
             })
        .def("num_buckets", &sixmax::Abstraction::num_buckets)
        .def("edges", &sixmax::Abstraction::edges)
        .def("bucket_cache_size", &sixmax::Abstraction::bucket_cache_size)
        .def("hash", &sixmax::Abstraction::hash)
        .def("set_pot_layout",
             [](sixmax::Abstraction& a, const std::string& mode,
                const std::vector<double>& edges) {
                 sixmax::PotLayout layout;
                 if (mode == "bb")
                     layout.mode = sixmax::PotMode::BB;
                 else if (mode == "spr")
                     layout.mode = sixmax::PotMode::SPR;
                 else
                     throw py::value_error(
                         "pot mode must be 'bb' or 'spr'");
                 layout.edges = edges;
                 try {
                     a.set_pot_layout(std::move(layout));
                 } catch (const std::invalid_argument& e) {
                     throw py::value_error(e.what());
                 }
             },
             py::arg("mode"), py::arg("edges"))
        .def("pot_layout",
             [](const sixmax::Abstraction& a) {
                 const auto& layout = a.pot_layout();
                 const std::string mode =
                     layout.mode == sixmax::PotMode::SPR ? "spr" : "bb";
                 return py::make_tuple(mode, layout.edges);
             })
        .def("set_bet_history",
             [](sixmax::Abstraction& a, const std::string& mode) {
                 if (mode == "off")
                     a.set_bet_history(sixmax::BetHistory::Off);
                 else if (mode == "last_raise")
                     a.set_bet_history(sixmax::BetHistory::LastRaise);
                 else
                     throw py::value_error(
                         "bet_history must be 'off' or 'last_raise'");
             },
             py::arg("mode"))
        .def("bet_history", [](const sixmax::Abstraction& a) {
            return a.bet_history() == sixmax::BetHistory::LastRaise
                       ? std::string("last_raise")
                       : std::string("off");
        })
        .def("pack_key",
             [](const sixmax::Abstraction& a, int card, int street,
                const std::vector<int>& raises, double pot_bb,
                double eff_stack_bb, int live, int after,
                const py::object& last_raise) {
                 if (raises.size() != 4)
                     throw py::value_error("raises must have length 4");
                 std::array<uint8_t, 4> r{(uint8_t)raises[0],
                                          (uint8_t)raises[1],
                                          (uint8_t)raises[2],
                                          (uint8_t)raises[3]};
                 std::array<uint8_t, 4> lr{};  // None = zeros
                 if (!last_raise.is_none()) {
                     auto v = last_raise.cast<std::vector<int>>();
                     if (v.size() != 4)
                         throw py::value_error(
                             "last_raise must have length 4");
                     for (int i = 0; i < 4; ++i) lr[i] = (uint8_t)v[i];
                 }
                 return a.pack_key(card, street, r, pot_bb, eff_stack_bb,
                                   live, after, lr);
             },
             py::arg("card"), py::arg("street"), py::arg("raises"),
             py::arg("pot_bb"), py::arg("eff_stack_bb"), py::arg("live"),
             py::arg("after"), py::arg("last_raise") = py::none());
    m.def("blueprint_probs",
          [](const sixmax::BlueprintStrategy& st,
             const sixmax::EngineGameState& s) {
              return sixmax::blueprint_probs(st, s);
          },
          py::arg("strategy"), py::arg("state"));
    m.def("sample_blueprint",
          [](const sixmax::BlueprintStrategy& st,
             const sixmax::EngineGameState& s, bool uniform, uint64_t seed) {
              std::mt19937_64 rng(seed);
              return sixmax::sample_blueprint(st, s, uniform, rng);
          },
          py::arg("strategy"), py::arg("state"), py::arg("uniform") = false,
          py::arg("seed") = 1);
    py::class_<sixmax::LbrConfig>(m, "LbrConfig")
        .def(py::init([](const std::vector<int>& act_streets,
                         bool fold_call_only, int preflop_wp_samples,
                         bool uniform_opponents, int num_threads) {
                 sixmax::LbrConfig cfg;
                 cfg.act_streets = {false, false, false, false};
                 for (int street : act_streets) {
                     if (street < 0 || street > 3)
                         throw py::value_error(
                             "act_streets entries must be 0-3");
                     cfg.act_streets[street] = true;
                 }
                 cfg.fold_call_only = fold_call_only;
                 if (preflop_wp_samples < 1)
                     throw py::value_error(
                         "preflop_wp_samples must be at least 1");
                 cfg.preflop_wp_samples = preflop_wp_samples;
                 cfg.uniform_opponents = uniform_opponents;
                 if (num_threads < 0)
                     throw py::value_error("num_threads must not be negative");
                 cfg.num_threads = num_threads;
                 return cfg;
             }),
             py::kw_only(), py::arg("act_streets") = std::vector<int>{2, 3},
             py::arg("fold_call_only") = false,
             py::arg("preflop_wp_samples") = 500,
             py::arg("uniform_opponents") = false,
             py::arg("num_threads") = 0);
    py::class_<sixmax::LbrEvaluator>(m, "LbrEvaluator")
        .def(py::init<const sixmax::BlueprintStrategy&,
                      const sixmax::ActionVocab&, const sixmax::EngineConfig&,
                      sixmax::LbrConfig>(),
             py::arg("strategy"), py::arg("vocab"), py::arg("cfg"),
             py::arg("lbr") = sixmax::LbrConfig{},
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("win_probability",
             [](const sixmax::LbrEvaluator& e,
                const std::array<int, 2>& hero_hole,
                const std::vector<int>& board,
                const std::vector<double>& range, uint64_t seed) {
                 std::mt19937_64 rng(seed);
                 return e.win_probability(hero_hole, board, range, rng);
             },
             py::arg("hero_hole"), py::arg("board"), py::arg("range"),
             py::arg("seed") = 1)
        .def("candidate_utilities",
             [](const sixmax::LbrEvaluator& e,
                const sixmax::EngineGameState& state, int hero,
                const std::vector<double>& range, uint64_t seed) {
                 std::mt19937_64 rng(seed);
                 py::list out;
                 for (const auto& cand :
                      e.candidate_utilities(state, hero, range, rng))
                     out.append(py::make_tuple(cand.first, cand.second));
                 return out;
             },
             py::arg("state"), py::arg("hero"), py::arg("range"),
             py::arg("seed") = 1)
        .def("candidate_utilities",
             [](const sixmax::LbrEvaluator& e, const sixmax::HandState& s,
                int hero, const std::vector<double>& range, uint64_t seed) {
                 const sixmax::EngineGameState state = wrap_hand(e, s);
                 std::mt19937_64 rng(seed);
                 py::list out;
                 for (const auto& cand :
                      e.candidate_utilities(state, hero, range, rng))
                     out.append(py::make_tuple(cand.first, cand.second));
                 return out;
             },
             py::arg("hand"), py::arg("hero"), py::arg("range"),
             py::arg("seed") = 1)
        .def("decide",
             [](const sixmax::LbrEvaluator& e,
                const sixmax::EngineGameState& state, int hero,
                const std::vector<double>& range, uint64_t seed) {
                 // Keys go through the state's own raise history, exactly as
                 // the blueprint saw it while training.
                 std::mt19937_64 rng(seed);
                 return e.decide(state, hero, range, rng);
             },
             py::arg("state"), py::arg("hero"), py::arg("range"),
             py::arg("seed") = 1)
        .def("decide",
             [](const sixmax::LbrEvaluator& e, const sixmax::HandState& s,
                int hero, const std::vector<double>& range, uint64_t seed) {
                 const sixmax::EngineGameState state = wrap_hand(e, s);
                 std::mt19937_64 rng(seed);
                 return e.decide(state, hero, range, rng);
             },
             py::arg("hand"), py::arg("hero"), py::arg("range"),
             py::arg("seed") = 1)
        .def("run",
             [](const sixmax::LbrEvaluator& e, int decks, uint64_t seed) {
                 sixmax::LbrReport r;
                 {
                     // The evaluation loop is long; release the GIL only for
                     // the C++ run, never while building the Python result.
                     py::gil_scoped_release release;
                     r = e.run(decks, seed);
                 }
                 py::dict out;
                 out["decks"] = r.decks;
                 out["seat_bb100"] = r.seat_bb100;
                 out["seat_stderr"] = r.seat_std_err;
                 out["bb100"] = r.bb100;
                 out["stderr"] = r.std_err;
                 out["opp_unseen"] = r.opp_unseen;
                 out["opp_decisions"] = r.opp_decisions;
                 out["per_deck"] = r.per_deck;
                 return out;
             },
             py::arg("decks"), py::arg("seed") = 1);
    // --- Multithreaded blueprint trainer (Phase 1b Task 3) ---
    py::class_<sixmax::BlueprintTrainer>(m, "BlueprintTrainer")
        .def(py::init([](const sixmax::EngineConfig& cfg,
                         const sixmax::ActionVocab* vocab,
                         const sixmax::Abstraction* abstraction,
                         int num_threads, uint64_t seed) {
                 sixmax::GameFactory f = [cfg, vocab, abstraction]() {
                     return std::make_unique<sixmax::EngineGame>(cfg, vocab,
                                                                 abstraction);
                 };
                 return new sixmax::BlueprintTrainer(
                     std::move(f),
                     sixmax::TrainerConfig{num_threads, seed});
             }),
             py::arg("cfg"), py::arg("vocab"), py::arg("abstraction"),
             py::arg("num_threads") = 1, py::arg("seed") = 1,
             py::keep_alive<1, 3>(),   // trainer's factory holds vocab*
             py::keep_alive<1, 4>())   // trainer's factory holds abstraction*
        .def_static("kuhn", [](int num_threads, uint64_t seed) {
            sixmax::GameFactory f = []() {
                return std::make_unique<sixmax::KuhnGame>();
            };
            return new sixmax::BlueprintTrainer(
                std::move(f), sixmax::TrainerConfig{num_threads, seed});
        }, py::arg("num_threads") = 1, py::arg("seed") = 1)
        .def("train", &sixmax::BlueprintTrainer::train, py::arg("iterations"),
             py::call_guard<py::gil_scoped_release>())
        .def("iterations", &sixmax::BlueprintTrainer::iterations)
        .def("num_infosets", &sixmax::BlueprintTrainer::num_infosets)
        .def("average_strategy", &sixmax::BlueprintTrainer::average_strategy,
             py::arg("key"))
        .def("keys", &sixmax::BlueprintTrainer::keys)
        .def("save",
             [](const sixmax::BlueprintTrainer& t, const std::string& path,
                const sixmax::ActionVocab* vocab,
                const sixmax::EngineConfig& cfg,
                const sixmax::Abstraction* abstraction) {
                 sixmax::save_blueprint(
                     path,
                     sixmax::BlueprintMeta{vocab->hash(), cfg.num_players,
                                           cfg.starting_stack, vocab->size()},
                     *abstraction, t);
             },
             py::arg("path"), py::arg("vocab"), py::arg("cfg"),
             py::arg("abstraction"));
    m.def("blueprint_kuhn_value", [](const sixmax::BlueprintTrainer& t) {
        return sixmax::kuhn_exact_value_lookup(
            [&](uint64_t k) { return t.average_strategy(k); });
    });
    // --- Blueprint checkpoints + strategy (Phase 1b Task 4) ---
    m.def("load_abstraction", [](const std::string& path,
                                const std::string& bucket_map) {
        // Peek only the vocab hash: reuse the loader with the stored hash so
        // it cannot mismatch, then rebuild the abstraction (stored edges for
        // SIXBP001, resolved + hash-checked bucket map for SIXBP002/SIXBP003).
        // SIXBP003 restores its stored pot layout; SIXBP004 also restores its
        // bet-history mode. `bucket_map` is an optional explicit map path
        // (config/CLI override).
        auto loaded = sixmax::load_blueprint(
            path, [&] {
                std::ifstream i(path, std::ios::binary);
                i.seekg(8);
                uint64_t h;
                i.read(reinterpret_cast<char*>(&h), sizeof h);
                return h;
            }());
        return sixmax::load_abstraction_for(path, loaded, bucket_map);
    }, py::arg("path"), py::arg("bucket_map") = "");
    // --- Read-only autopsy dump: recovers per-infoset visit-weight and regret
    //     that the normalized accessors discard (plateau diagnosis). ---
    m.def("dump_infosets", [](const std::string& path) {
        uint64_t h;
        {
            std::ifstream i(path, std::ios::binary);
            i.seekg(8);
            i.read(reinterpret_cast<char*>(&h), sizeof h);
        }
        auto loaded = sixmax::load_blueprint(path, h);
        py::list records;
        for (const auto& [key, data] : loaded.table) {
            double mass = 0.0, reg = 0.0;
            for (double v : data.strategy_sum) mass += v;
            for (double v : data.regret) reg += std::fabs(v);
            std::vector<double> probs(data.strategy_sum.size(), 0.0);
            if (mass > 0.0)
                for (size_t i = 0; i < probs.size(); ++i)
                    probs[i] = data.strategy_sum[i] / mass;
            records.append(py::make_tuple(key, probs, mass, reg));
        }
        return py::make_tuple(loaded.iterations, records);
    }, py::arg("path"));
    m.def("resume_blueprint",
          [](const std::string& path, const sixmax::EngineConfig& cfg,
             const sixmax::ActionVocab* vocab,
             const sixmax::Abstraction* abstraction, int num_threads,
             uint64_t seed) {
              auto loaded = sixmax::load_blueprint(path, vocab->hash());
              // The abstraction is supplied by the caller, so refuse one that
              // does not match the artifact: resuming into the wrong key space
              // would silently corrupt the imported table. The bet-history
              // setting is checked first because it changes every key too, then
              // the pot layout, which a layout difference also changes
              // wholesale.
              if (abstraction->bet_history() != loaded.bet_history)
                  throw std::runtime_error(
                      "bet history mismatch: checkpoint " + path +
                      " was trained with a different bet-history setting");
              if (!same_pot_layout(abstraction->pot_layout(),
                                   loaded.pot_layout))
                  throw std::runtime_error(
                      "pot layout mismatch: checkpoint " + path +
                      " was trained with a different pot layout");
              if (loaded.abs_kind == 0) {
                  if (abstraction->is_map())
                      throw std::runtime_error(
                          "abstraction kind mismatch: checkpoint " + path +
                          " stores legacy quantile edges, not a bucket map");
                  const sixmax::Abstraction stored(loaded.abs_cfg,
                                                   loaded.edges);
                  if (abstraction->hash() != stored.hash())
                      throw std::runtime_error(
                          "abstraction hash mismatch: checkpoint " + path +
                          " was trained with different legacy edges/config");
              } else {
                  if (!abstraction->is_map())
                      throw std::runtime_error(
                          "abstraction kind mismatch: checkpoint " + path +
                          " is map-backed but a legacy abstraction was "
                          "passed");
                  if (abstraction->bucket_map()->content_hash() !=
                      loaded.map_hash)
                      throw std::runtime_error(
                          "map hash mismatch: checkpoint " + path +
                          " was trained with a different bucket map");
              }
              sixmax::GameFactory f = [cfg, vocab, abstraction]() {
                  return std::make_unique<sixmax::EngineGame>(cfg, vocab,
                                                              abstraction);
              };
              auto* t = new sixmax::BlueprintTrainer(
                  std::move(f), sixmax::TrainerConfig{num_threads, seed});
              t->import_table(std::move(loaded.table), loaded.iterations);
              return t;
          },
          py::arg("path"), py::arg("cfg"), py::arg("vocab"),
          py::arg("abstraction"), py::arg("num_threads") = 1,
          py::arg("seed") = 1,
          py::keep_alive<0, 3>(),   // returned trainer holds vocab*
          py::keep_alive<0, 4>());  // returned trainer holds abstraction*
    py::class_<sixmax::BlueprintStrategy>(m, "BlueprintStrategy")
        .def_static("load", &sixmax::BlueprintStrategy::load,
                    py::arg("path"), py::arg("vocab"))
        .def("probs", &sixmax::BlueprintStrategy::probs, py::arg("key"))
        .def("probs_for", &sixmax::BlueprintStrategy::probs_for)
        .def("iterations", &sixmax::BlueprintStrategy::iterations)
        .def("num_infosets", &sixmax::BlueprintStrategy::num_infosets)
        .def("num_players", &sixmax::BlueprintStrategy::num_players)
        .def("starting_stack", &sixmax::BlueprintStrategy::starting_stack)
        .def("abstraction", &sixmax::BlueprintStrategy::abstraction,
             py::return_value_policy::reference_internal);
    // --- Phase 2 search: vector CFR+ core (Task 1) ---
    py::class_<sixmax::VectorCfrConfig>(m, "VectorCfrConfig")
        .def(py::init<>())
        .def_readwrite("seed", &sixmax::VectorCfrConfig::seed)
        .def_readwrite("max_iters", &sixmax::VectorCfrConfig::max_iters)
        .def_readwrite("budget_ms", &sixmax::VectorCfrConfig::budget_ms);

    py::class_<sixmax::VectorCfrResult>(m, "VectorCfrResult")
        .def_readonly("root_value", &sixmax::VectorCfrResult::root_value)
        .def_readonly("iterations", &sixmax::VectorCfrResult::iterations)
        .def_readonly("deadline_hit", &sixmax::VectorCfrResult::deadline_hit);

    // Register the abstract bases so derived classes (KuhnVectorGame, Subgame)
    // and their states are usable from Python. VectorState exposes just the
    // read-only tree accessors the Subgame tests touch.
    py::class_<sixmax::VectorState>(m, "VectorState")
        .def("actor", &sixmax::VectorState::actor)
        .def("is_leaf", &sixmax::VectorState::is_leaf)
        .def("num_actions", &sixmax::VectorState::num_actions)
        .def("legal_mask", [](const sixmax::VectorState& s) {
            std::vector<uint8_t> m2;
            s.legal_mask(m2);
            return std::vector<int>(m2.begin(), m2.end());
        })
        .def("id", &sixmax::VectorState::id);
    py::class_<sixmax::VectorGame>(m, "VectorGame");

    py::class_<sixmax::KuhnVectorGame, sixmax::VectorGame>(m, "KuhnVectorGame")
        .def(py::init<>());

    py::class_<sixmax::VectorCfr>(m, "VectorCfr")
        .def(py::init<sixmax::VectorCfrConfig>(), py::arg("cfg"))
        .def("solve", &sixmax::VectorCfr::solve,
             py::arg("game"), py::arg("query_player"), py::arg("query_hand"))
        .def("average_strategy", &sixmax::VectorCfr::average_strategy,
             py::arg("node_id"), py::arg("hand"))
        .def("rank_root_action", &sixmax::VectorCfr::rank_root_action,
             py::arg("query_player"), py::arg("hand"))
        .def("root_id", &sixmax::VectorCfr::root_id);
    // --- Phase 2 search: HU range tracker (Task 2) ---
    py::class_<sixmax::RangeTracker>(m, "RangeTracker")
        .def(py::init<sixmax::EngineConfig, const sixmax::ActionVocab&>(),
             py::arg("cfg"), py::arg("vocab"),
             py::keep_alive<1, 3>())   // tracker holds ActionVocab&
        .def_static("combo_index", &sixmax::RangeTracker::combo_index)
        .def_static("combo_cards", &sixmax::RangeTracker::combo_cards)
        .def("reset", &sixmax::RangeTracker::reset)
        .def("range", [](const sixmax::RangeTracker& t, int p) { return t.range(p); })
        .def("combo_prob", &sixmax::RangeTracker::combo_prob)
        .def("remove_blocked", &sixmax::RangeTracker::remove_blocked)
        .def("set_range_for_test", &sixmax::RangeTracker::set_range_for_test)
        .def("observe_with_probs", &sixmax::RangeTracker::observe_with_probs)
        .def("set_prior", &sixmax::RangeTracker::set_prior);
    // --- Phase 2 search: street subgame builder (Task 3) ---
    py::class_<sixmax::SubgameConfig>(m, "SubgameConfig")
        .def(py::init<>())
        .def_readwrite("max_nodes", &sixmax::SubgameConfig::max_nodes);

    // --- Phase 2 search: leaf evaluation (Task 4) ---
    // Registered before Subgame: its ctor takes a LeafEvalConfig default.
    py::class_<sixmax::LeafEvalConfig>(m, "LeafEvalConfig")
        .def(py::init<>())
        .def_readwrite("samples", &sixmax::LeafEvalConfig::samples)
        .def_readwrite("seed", &sixmax::LeafEvalConfig::seed)
        .def_readwrite("threads", &sixmax::LeafEvalConfig::threads);

    py::class_<sixmax::LeafEval>(m, "LeafEval")
        .def(py::init<sixmax::LeafEvalConfig>(), py::arg("cfg"))   // exact-only
        .def(py::init<const sixmax::BlueprintStrategy&, sixmax::LeafEvalConfig>(),
             py::arg("bp"), py::arg("cfg"))
        .def("matrix", &sixmax::LeafEval::matrix,
             py::arg("leaf"), py::arg("actor"), py::arg("our_hands"),
             py::arg("opp_hands"));

    py::class_<sixmax::Subgame, sixmax::VectorGame>(m, "Subgame")
        .def(py::init([](const sixmax::EngineGameState& sr,
                         const sixmax::ActionVocab& v,
                         const sixmax::RangeTracker& r, int hero,
                         std::array<int, 2> hh, sixmax::SubgameConfig c,
                         const sixmax::BlueprintStrategy* bp,
                         sixmax::LeafEvalConfig lc) {
                 // Python callers never pass a deadline; search sets it C++-side.
                 return sixmax::Subgame(
                     sr, v, r, hero, hh, c, bp, lc,
                     std::chrono::steady_clock::time_point::max());
             }),
             py::arg("street_root"), py::arg("vocab"), py::arg("ranges"),
             py::arg("hero"), py::arg("hero_hole"), py::arg("cfg"),
             py::arg("bp") = nullptr,
             py::arg("leaf_cfg") = sixmax::LeafEvalConfig{},
             py::keep_alive<1, 3>())   // subgame holds ActionVocab*
        .def("num_hands", &sixmax::Subgame::num_hands)
        .def("compatible", &sixmax::Subgame::compatible)
        .def("prior_reach", &sixmax::Subgame::prior_reach)
        .def("root", &sixmax::Subgame::root)
        .def("node_count", &sixmax::Subgame::node_count)
        .def("node_ids", &sixmax::Subgame::node_ids)
        .def("complete", &sixmax::Subgame::complete)
        .def("query_node_id", &sixmax::Subgame::query_node_id);
    // --- Phase 2 search: orchestration (Task 5) ---
    py::class_<sixmax::SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &sixmax::SearchConfig::enabled)
        .def_readwrite("budget_ms", &sixmax::SearchConfig::budget_ms)
        .def_readwrite("max_nodes", &sixmax::SearchConfig::max_nodes)
        .def_readwrite("max_iters", &sixmax::SearchConfig::max_iters)
        .def_readwrite("leaf_samples", &sixmax::SearchConfig::leaf_samples)
        .def_readwrite("seed", &sixmax::SearchConfig::seed)
        .def_readwrite("leaf_threads", &sixmax::SearchConfig::leaf_threads);

    py::class_<sixmax::SearchStrategy>(m, "SearchStrategy")
        .def(py::init<const sixmax::BlueprintStrategy&, const sixmax::ActionVocab&,
                      const sixmax::EngineConfig&, sixmax::SearchConfig>(),
             py::arg("blueprint"), py::arg("vocab"), py::arg("cfg"),
             py::arg("search_cfg"),
             py::keep_alive<1, 2>(),   // strategy holds BlueprintStrategy&
             py::keep_alive<1, 3>())   // strategy holds ActionVocab&
        .def("probs", &sixmax::SearchStrategy::probs,
             py::arg("decision_state"), py::arg("hero"),
             py::arg("observed_actions"),
             py::arg("opp_prior") = std::vector<double>())
        .def("last_decision_searched",
             &sixmax::SearchStrategy::last_decision_searched)
        .def("searches", &sixmax::SearchStrategy::searches)
        .def("fallbacks", &sixmax::SearchStrategy::fallbacks);

}
