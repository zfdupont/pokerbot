#include "blueprint/checkpoint.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sixmax {

namespace {

constexpr char kMagic[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '1'};

template <typename T>
void put(std::ofstream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
T get(std::ifstream& i) {
    T v;
    i.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
    return v;
}

void put_doubles(std::ofstream& o, const std::vector<double>& v) {
    put<uint64_t>(o, v.size());
    o.write(reinterpret_cast<const char*>(v.data()),
            (std::streamsize)(v.size() * sizeof(double)));
}

std::vector<double> get_doubles(std::ifstream& i) {
    auto n = get<uint64_t>(i);
    std::vector<double> v(n);
    i.read(reinterpret_cast<char*>(v.data()),
           (std::streamsize)(n * sizeof(double)));
    if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
    return v;
}

}  // namespace

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open " + tmp);
        o.write(kMagic, 8);
        put<uint64_t>(o, meta.vocab_hash);
        put<int32_t>(o, meta.num_players);
        put<int32_t>(o, meta.action_dim);
        put<double>(o, meta.starting_stack);
        const AbstractionConfig& c = abs.config();
        put<int32_t>(o, c.flop_buckets);
        put<int32_t>(o, c.turn_buckets);
        put<int32_t>(o, c.river_buckets);
        put<int32_t>(o, c.equity_rollouts);
        put<int32_t>(o, c.quantile_samples);
        put<uint64_t>(o, c.seed);
        for (const auto& e : abs.edges()) put_doubles(o, e);
        auto table = trainer.export_table();
        put<uint64_t>(o, trainer.iterations());
        put<uint64_t>(o, table.size());
        for (const auto& [k, d] : table) {
            put<uint64_t>(o, k);
            o.write(reinterpret_cast<const char*>(d.regret.data()),
                    (std::streamsize)(meta.action_dim * sizeof(double)));
            o.write(reinterpret_cast<const char*>(d.strategy_sum.data()),
                    (std::streamsize)(meta.action_dim * sizeof(double)));
        }
        if (!o) throw std::runtime_error("write failed: " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("atomic rename failed: " + path);
}

LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open " + path);
    char magic[8];
    i.read(magic, 8);
    if (!i || std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("not a blueprint checkpoint: " + path);
    LoadedBlueprint out;
    out.meta.vocab_hash = get<uint64_t>(i);
    if (out.meta.vocab_hash != expected_vocab_hash)
        throw std::runtime_error(
            "vocab hash mismatch: checkpoint was trained with a different "
            "action vocabulary");
    out.meta.num_players = get<int32_t>(i);
    out.meta.action_dim = get<int32_t>(i);
    out.meta.starting_stack = get<double>(i);
    out.abs_cfg.flop_buckets = get<int32_t>(i);
    out.abs_cfg.turn_buckets = get<int32_t>(i);
    out.abs_cfg.river_buckets = get<int32_t>(i);
    out.abs_cfg.equity_rollouts = get<int32_t>(i);
    out.abs_cfg.quantile_samples = get<int32_t>(i);
    out.abs_cfg.seed = get<uint64_t>(i);
    for (auto& e : out.edges) e = get_doubles(i);
    out.iterations = get<uint64_t>(i);
    const auto n_infosets = get<uint64_t>(i);
    const int dim = out.meta.action_dim;
    out.table.reserve(n_infosets);
    for (uint64_t k = 0; k < n_infosets; ++k) {
        const uint64_t key = get<uint64_t>(i);
        InfosetData d;
        d.regret.resize(dim);
        d.strategy_sum.resize(dim);
        i.read(reinterpret_cast<char*>(d.regret.data()),
               (std::streamsize)(dim * sizeof(double)));
        i.read(reinterpret_cast<char*>(d.strategy_sum.data()),
               (std::streamsize)(dim * sizeof(double)));
        if (!i) throw std::runtime_error("blueprint checkpoint: truncated file");
        out.table.emplace(key, std::move(d));
    }
    return out;
}

BlueprintStrategy BlueprintStrategy::load(const std::string& path,
                                          const ActionVocab& vocab) {
    auto loaded = load_blueprint(path, vocab.hash());
    BlueprintStrategy s(Abstraction(loaded.abs_cfg, std::move(loaded.edges)),
                        loaded.iterations, loaded.meta.num_players);
    for (const auto& [k, d] : loaded.table) {
        double total = 0.0;
        for (double v : d.strategy_sum) total += v;
        if (total <= 0.0) continue;
        std::vector<double> p(d.strategy_sum.size());
        for (size_t a = 0; a < p.size(); ++a) p[a] = d.strategy_sum[a] / total;
        s.probs_.emplace(k, std::move(p));
    }
    return s;
}

std::vector<double> BlueprintStrategy::probs(uint64_t key) const {
    auto it = probs_.find(key);
    return it == probs_.end() ? std::vector<double>{} : it->second;
}

std::vector<double> BlueprintStrategy::probs_for(
        const EngineGameState& s) const {
    return probs(s.abstract_key(abs_));
}

}  // namespace sixmax
