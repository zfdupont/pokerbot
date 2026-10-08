#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <cmath>
#include <random>
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
#include "blueprint/checkpoint.h"
#include "exploit/policy.h"
#include "exploit/lbr.h"
#include "search/vector_cfr.h"
#include "search/kuhn_vector.h"
#include "search/range_tracker.h"
#include "search/subgame.h"

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
    m.def("pot_bucket", &sixmax::pot_bucket, py::arg("pot_bb"));
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
        .def("bucket",
             [](const sixmax::Abstraction& a, const std::vector<int>& hole,
                const std::vector<int>& board) {
                 if (hole.size() != 2) throw py::value_error("expects 2 cards");
                 return a.bucket({hole[0], hole[1]}, board);
             })
        .def("num_buckets", &sixmax::Abstraction::num_buckets)
        .def("edges", &sixmax::Abstraction::edges)
        .def("bucket_cache_size", &sixmax::Abstraction::bucket_cache_size)
        .def("hash", &sixmax::Abstraction::hash);
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
        .def(py::init([](int samples, bool condition_on_history,
                         bool uniform_opponents, int hero_action_limit,
                         int max_steps, uint64_t seed) {
                 return sixmax::LbrConfig{samples, condition_on_history,
                                          uniform_opponents, hero_action_limit,
                                          max_steps, seed};
             }),
             py::kw_only(), py::arg("samples") = 64,
             py::arg("condition_on_history") = true,
             py::arg("uniform_opponents") = false,
             py::arg("hero_action_limit") = 0,
             py::arg("max_steps") = 4096, py::arg("seed") = 1);
    py::class_<sixmax::LbrEvaluator>(m, "LbrEvaluator")
        .def(py::init<const sixmax::BlueprintStrategy&,
                      const sixmax::ActionVocab&, const sixmax::EngineConfig&,
                      sixmax::LbrConfig>(),
             py::arg("strategy"), py::arg("vocab"), py::arg("cfg"),
             py::arg("lbr") = sixmax::LbrConfig{},
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("decide",
             [](sixmax::LbrEvaluator& e, const sixmax::HandState& s, int hero,
                const std::vector<int>& actors,
                const std::vector<int>& actions, uint64_t seed) {
                 std::mt19937_64 rng(seed);
                 return e.decide(s, hero, actors, actions, rng);
             },
             py::arg("hand"), py::arg("hero"),
             py::arg("actors") = std::vector<int>{},
             py::arg("actions") = std::vector<int>{}, py::arg("seed") = 1)
        .def("sampled_rollouts_exhausted",
             &sixmax::LbrEvaluator::sampled_rollouts_exhausted)
        .def("debug_sample_deals",
             [](sixmax::LbrEvaluator& e, const sixmax::HandState& s, int hero,
                const std::vector<int>& actors,
                const std::vector<int>& actions, uint64_t seed) {
                 std::mt19937_64 rng(seed);
                 py::list hands, weights;
                 for (auto& smp : e.debug_sample_deals(s, hero, actors, actions,
                                                       rng)) {
                     hands.append(smp.current);
                     weights.append(smp.weight);
                 }
                 return py::make_tuple(hands, weights);
             },
             py::arg("hand"), py::arg("hero"),
             py::arg("actors") = std::vector<int>{},
             py::arg("actions") = std::vector<int>{}, py::arg("seed") = 1)
        .def("run",
             [](sixmax::LbrEvaluator& e, int decks, uint64_t seed) {
                 auto r = e.run(decks, seed);
                 py::dict out;
                 py::list gains, ses;
                 for (auto& s : r.seats) {
                     gains.append(s.gain_bb100);
                     ses.append(s.stderr);
                 }
                 out["gains"] = gains;
                 out["stderrs"] = ses;
                 out["max_gain"] = r.max_gain_bb100;
                 out["mean_gain"] = r.mean_gain_bb100;
                 out["decks"] = r.decks;
                 out["num_players"] = r.num_players;
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
    m.def("load_abstraction", [](const std::string& path) {
        // Peek only the abstraction block: reuse the loader with the
        // stored hash so it cannot mismatch, then rebuild from edges.
        auto loaded = sixmax::load_blueprint(
            path, [&] {
                std::ifstream i(path, std::ios::binary);
                i.seekg(8);
                uint64_t h;
                i.read(reinterpret_cast<char*>(&h), sizeof h);
                return h;
            }());
        return sixmax::Abstraction(loaded.abs_cfg, std::move(loaded.edges));
    }, py::arg("path"));
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
        .def("set_range_for_test", &sixmax::RangeTracker::set_range_for_test)
        .def("observe_with_probs", &sixmax::RangeTracker::observe_with_probs);
    // --- Phase 2 search: street subgame builder (Task 3) ---
    py::class_<sixmax::SubgameConfig>(m, "SubgameConfig")
        .def(py::init<>())
        .def_readwrite("max_nodes", &sixmax::SubgameConfig::max_nodes);

    py::class_<sixmax::Subgame, sixmax::VectorGame>(m, "Subgame")
        .def(py::init<const sixmax::EngineGameState&, const sixmax::ActionVocab&,
                      const sixmax::RangeTracker&, int, std::array<int, 2>,
                      sixmax::SubgameConfig>(),
             py::arg("street_root"), py::arg("vocab"), py::arg("ranges"),
             py::arg("hero"), py::arg("hero_hole"), py::arg("cfg"),
             py::keep_alive<1, 3>())   // subgame holds ActionVocab*
        .def("num_hands", &sixmax::Subgame::num_hands)
        .def("compatible", &sixmax::Subgame::compatible)
        .def("prior_reach", &sixmax::Subgame::prior_reach)
        .def("root", &sixmax::Subgame::root)
        .def("node_count", &sixmax::Subgame::node_count)
        .def("node_ids", &sixmax::Subgame::node_ids)
        .def("query_node_id", &sixmax::Subgame::query_node_id);

}
