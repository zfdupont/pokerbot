#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vocab/vocab.h"
#include "blueprint/engine_game.h"
#include "abstraction/abstraction.h"
#include "dream/features.h"
#include "dream/nets.h"
#include "dream/reservoir.h"
#include "dream/trainer.h"
#include "dream/checkpoint.h"

namespace py = pybind11;

PYBIND11_MODULE(sixmax_dream, m) {
    m.doc() = "Six-max DREAM neural CFR (isolated libtorch extension)";

    // --- Dream neural CFR (Task 7) ---

    // DreamMLP: thin wrapper around the TORCH_MODULE holder
    struct PyDreamMLP {
        sixmax::DreamMLP net;
        int input_dim_, output_dim_;
        PyDreamMLP(int in, int h, int n, int out)
            : net(in, h, n, out), input_dim_(in), output_dim_(out) {}
        torch::Tensor forward(torch::Tensor x) { return net->forward(x); }
        // Python-list overload: accepts flat vector (batch*in), returns flat (batch*out)
        std::vector<float> forward_vec(const std::vector<float>& flat, int batch) {
            auto x = torch::from_blob(
                const_cast<float*>(flat.data()),
                {batch, input_dim_},
                torch::kFloat32).clone();
            auto out = net->forward(x);
            out = out.contiguous();
            const float* ptr = out.data_ptr<float>();
            return std::vector<float>(ptr, ptr + batch * output_dim_);
        }
    };
    py::class_<PyDreamMLP>(m, "DreamMLP")
        .def(py::init<int, int, int, int>(),
             py::arg("input_dim"), py::arg("hidden_size"),
             py::arg("n_layers"), py::arg("output_dim"))
        .def("forward_vec", &PyDreamMLP::forward_vec,
             py::arg("flat_input"), py::arg("batch_size"),
             "Run forward pass via Python lists. flat_input is a flat float list of "
             "length batch_size*input_dim; returns flat float list of length "
             "batch_size*output_dim.");

    // WeightedReservoir: wrapper with internal rng
    struct PyWeightedReservoir {
        sixmax::WeightedReservoir r;
        std::mt19937_64 rng;
        PyWeightedReservoir(size_t cap, uint64_t seed = 42) : r(cap), rng(seed) {}
        void add(torch::Tensor f, torch::Tensor t, float w) { r.add(f, t, w, rng); }
        // Python-list overload: accepts flat vectors, avoids Python-tensor bridge
        void add_vec(const std::vector<float>& feat,
                     const std::vector<float>& tgt, float w) {
            auto f = torch::from_blob(
                const_cast<float*>(feat.data()),
                {(long)feat.size()}, torch::kFloat32).clone();
            auto t = torch::from_blob(
                const_cast<float*>(tgt.data()),
                {(long)tgt.size()}, torch::kFloat32).clone();
            r.add(f, t, w, rng);
        }
        // Returns (feat_flat, tgt_flat, weights) as Python lists
        std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
        sample_batch_vec(size_t n) {
            auto [feat, tgt, w] = r.sample_batch(n, rng);
            feat = feat.contiguous(); tgt = tgt.contiguous(); w = w.contiguous();
            const float* fp = feat.data_ptr<float>();
            const float* tp = tgt.data_ptr<float>();
            const float* wp = w.data_ptr<float>();
            long fsz = feat.numel(), tsz = tgt.numel();
            return {
                std::vector<float>(fp, fp + fsz),
                std::vector<float>(tp, tp + tsz),
                std::vector<float>(wp, wp + n)
            };
        }
        auto sample_batch(size_t n) { return r.sample_batch(n, rng); }
        size_t size() const { return r.size(); }
        void clear() { r.clear(); }
    };
    py::class_<PyWeightedReservoir>(m, "WeightedReservoir")
        .def(py::init<size_t, uint64_t>(),
             py::arg("capacity"), py::arg("seed") = 42)
        .def("add_vec", &PyWeightedReservoir::add_vec,
             py::arg("feat"), py::arg("target"), py::arg("weight"),
             "Add a sample via Python float lists instead of torch.Tensor.")
        .def("sample_batch_vec", &PyWeightedReservoir::sample_batch_vec,
             py::arg("n"),
             "Sample n items; returns (feat_flat, tgt_flat, weights) as Python lists.")
        .def("size", &PyWeightedReservoir::size)
        .def("clear", &PyWeightedReservoir::clear);

    // DreamConfig
    py::class_<sixmax::DreamConfig>(m, "DreamConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &sixmax::DreamConfig::hidden_size)
        .def_readwrite("hidden_layers", &sixmax::DreamConfig::hidden_layers)
        .def_readwrite("lr", &sixmax::DreamConfig::lr)
        .def_readwrite("batch_size", &sixmax::DreamConfig::batch_size)
        .def_readwrite("reservoir_size", &sixmax::DreamConfig::reservoir_size)
        .def_readwrite("train_interval", &sixmax::DreamConfig::train_interval)
        .def_readwrite("sgd_steps", &sixmax::DreamConfig::sgd_steps)
        .def_readwrite("epsilon", &sixmax::DreamConfig::epsilon)
        .def_readwrite("num_threads", &sixmax::DreamConfig::num_threads)
        .def_readwrite("seed", &sixmax::DreamConfig::seed)
        .def_readwrite("stack_min", &sixmax::DreamConfig::stack_min)
        .def_readwrite("stack_max", &sixmax::DreamConfig::stack_max)
        .def_readwrite("players_min", &sixmax::DreamConfig::players_min)
        .def_readwrite("players_max", &sixmax::DreamConfig::players_max)
        .def_readwrite("stack_log_mean", &sixmax::DreamConfig::stack_log_mean)
        .def_readwrite("stack_log_std",  &sixmax::DreamConfig::stack_log_std);

    // DreamTrainer
    py::class_<sixmax::DreamTrainer>(m, "DreamTrainer")
        .def(py::init([](int n_actions, const sixmax::ActionVocab* vocab,
                         const sixmax::Abstraction* abstraction,
                         sixmax::DreamConfig cfg,
                         const std::string& device_str) {
            return std::make_unique<sixmax::DreamTrainer>(
                n_actions, vocab, abstraction, cfg,
                torch::Device(device_str));
        }),
        py::arg("n_actions"), py::arg("vocab"), py::arg("abstraction"),
        py::arg("cfg"), py::arg("device") = "cpu",
        py::keep_alive<0, 2>(),   // trainer holds ActionVocab*
        py::keep_alive<0, 3>())   // trainer holds Abstraction*
        .def("train", &sixmax::DreamTrainer::train, py::arg("iterations"),
             py::call_guard<py::gil_scoped_release>())
        .def("total_iterations", &sixmax::DreamTrainer::total_iterations)
        .def("adv_reservoir_size", &sixmax::DreamTrainer::adv_reservoir_size,
             "Number of entries currently in the advantage reservoir (M_v). "
             "Used by tests to verify all seats contribute advantage samples.")
        .def("avg_traverse_ns", &sixmax::DreamTrainer::avg_traverse_ns,
             "Average nanoseconds per traverse() call (both players combined). "
             "Zero until the first retraining cycle fires.")
        .def("save", [](const sixmax::DreamTrainer& t, const std::string& path,
                        uint64_t vocab_hash) {
            sixmax::save_dream_checkpoint(path, t.adv_net(), t.strat_net(),
                                          t.total_iterations(), vocab_hash);
        }, py::arg("path"), py::arg("vocab_hash"));

    // DreamStrategy
    py::class_<sixmax::DreamStrategy>(m, "DreamStrategy")
        .def_static("load",
            [](const std::string& path, const std::string& device_str,
               const sixmax::ActionVocab& vocab) {
                auto device = torch::Device(device_str);
                return sixmax::DreamStrategy::load(path, device, vocab);
            },
            py::arg("path"), py::arg("device") = "cpu", py::arg("vocab"))
        .def("get_probs",
            [](const sixmax::DreamStrategy& s, const sixmax::EngineGameState& state) {
                return s.get_probs(state);
            })
        .def_property_readonly("iterations", &sixmax::DreamStrategy::iterations);

    // Module-level save function
    m.def("save_dream_checkpoint",
        [](const std::string& path, PyDreamMLP& adv, PyDreamMLP& strat,
           uint64_t iterations, uint64_t vocab_hash) {
            sixmax::save_dream_checkpoint(path, adv.net, strat.net,
                                          iterations, vocab_hash);
        },
        py::arg("path"), py::arg("adv_net"), py::arg("strat_net"),
        py::arg("iterations"), py::arg("vocab_hash"));

    // encode_state_vec: Python-callable overload avoiding at::Tensor ABI mismatch
    m.def("encode_state_vec", [](const sixmax::EngineGameState& state) {
        auto t = sixmax::encode_state(state);
        auto a = t.accessor<float, 1>();
        std::vector<float> out(sixmax::FEATURE_DIM);
        for (int i = 0; i < sixmax::FEATURE_DIM; ++i) out[i] = a[i];
        return out;
    }, py::arg("state"),
       "Encode game state to a flat float vector of length FEATURE_DIM. "
       "Use instead of encode_state() to avoid the Python<->C++ torch ABI mismatch.");

    // Constants
    m.attr("FEATURE_DIM") = sixmax::FEATURE_DIM;
    m.attr("CHIP_NORM")   = sixmax::CHIP_NORM;
    m.attr("RAISE_NORM")  = sixmax::RAISE_NORM;
}
