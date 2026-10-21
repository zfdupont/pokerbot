#include "blueprint/checkpoint.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "abstraction/bucket_map.h"

namespace sixmax {

namespace {

constexpr char kMagicV1[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '1'};
constexpr char kMagicV2[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '2'};
constexpr size_t kMaxHintLength = 1u << 16;

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

// Directory part of a path, or "." when the path has no directory.
std::string dir_name(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    if (pos == 0) return path.substr(0, 1);  // root: "/x" -> "/"
    return path.substr(0, pos);
}

std::string base_name(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

}  // namespace

std::string resolve_bucket_map(const std::string& ckpt_path,
                               const std::string& hint,
                               const std::string& explicit_path) {
    std::vector<std::string> candidates;
    // Process-wide override: applies to every SIXBP002 load in this process.
    if (const char* env = std::getenv("SIXMAX_BUCKET_MAP"); env && *env)
        candidates.push_back(env);
    // Config/CLI override for THIS load. Warn when the file is not the one the
    // checkpoint recorded: the content-hash check below still decides.
    if (!explicit_path.empty()) {
        if (!hint.empty() && base_name(explicit_path) != hint)
            std::fprintf(stderr,
                         "[checkpoint] warning: explicit bucket map '%s' "
                         "does not match the checkpoint's map hint '%s'; "
                         "the content hash decides\n",
                         explicit_path.c_str(), hint.c_str());
        candidates.push_back(explicit_path);
    }
    if (!hint.empty()) {
        candidates.push_back(dir_name(ckpt_path) + "/" + hint);
        const char* env_dir = std::getenv("SIXMAX_TABLES_DIR");
        const std::string tables_dir =
            (env_dir && *env_dir) ? env_dir : "sixmax/tables";
        candidates.push_back(tables_dir + "/" + hint);
    }
    for (const auto& candidate : candidates)
        if (file_exists(candidate)) return candidate;

    std::string message =
        "bucket map '" + hint + "' not found; tried:";
    for (const auto& candidate : candidates) message += " " + candidate;
    throw std::runtime_error(message);
}

Abstraction load_abstraction_for(const std::string& ckpt_path,
                                 const LoadedBlueprint& loaded,
                                 const std::string& explicit_map) {
    if (loaded.abs_kind == 0)
        return Abstraction(loaded.abs_cfg, loaded.edges);

    const std::string map_path =
        resolve_bucket_map(ckpt_path, loaded.map_hint, explicit_map);
    auto map = BucketMap::open(map_path);
    if (map->content_hash() != loaded.map_hash)
        throw std::runtime_error(
            "map hash mismatch: checkpoint " + ckpt_path +
            " was trained with bucket map '" + loaded.map_hint +
            "' but " + map_path + " has different content");
    return Abstraction(std::move(map));
}

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer) {
    const bool mapped = abs.is_map();
    std::string hint;
    if (mapped) {
        hint = base_name(abs.bucket_map()->path());
        if (hint.size() > kMaxHintLength)
            throw std::invalid_argument(
                "blueprint checkpoint: bucket map path too long");
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open " + tmp);
        o.write(mapped ? kMagicV2 : kMagicV1, 8);
        put<uint64_t>(o, meta.vocab_hash);
        put<int32_t>(o, meta.num_players);
        put<int32_t>(o, meta.action_dim);
        put<double>(o, meta.starting_stack);
        if (mapped) {
            const BucketMap* map = abs.bucket_map();
            put<int32_t>(o, 1);  // kind: 1 = bucket map
            put<uint64_t>(o, map->content_hash());
            put<uint32_t>(o, (uint32_t)hint.size());
            o.write(hint.data(), (std::streamsize)hint.size());
        } else {
            const AbstractionConfig& c = abs.config();
            put<int32_t>(o, c.flop_buckets);
            put<int32_t>(o, c.turn_buckets);
            put<int32_t>(o, c.river_buckets);
            put<int32_t>(o, c.equity_rollouts);
            put<int32_t>(o, c.quantile_samples);
            put<uint64_t>(o, c.seed);
            for (const auto& e : abs.edges()) put_doubles(o, e);
        }
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
        if (!o) {
            std::remove(tmp.c_str());  // never leave an orphaned partial
            throw std::runtime_error("write failed: " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("atomic rename failed: " + path);
    }
}

LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open " + path);
    char magic[8];
    i.read(magic, 8);
    const bool mapped = i && std::memcmp(magic, kMagicV2, 8) == 0;
    if (!i || (!mapped && std::memcmp(magic, kMagicV1, 8) != 0))
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
    if (mapped) {
        out.abs_kind = get<int32_t>(i);
        if (out.abs_kind != 1)
            throw std::runtime_error(
                "blueprint checkpoint: unknown abstraction kind " +
                std::to_string(out.abs_kind));
        out.map_hash = get<uint64_t>(i);
        const auto hint_len = get<uint32_t>(i);
        if (hint_len > kMaxHintLength)
            throw std::runtime_error(
                "blueprint checkpoint: map hint too long: " + path);
        out.map_hint.resize(hint_len);
        if (hint_len > 0)
            i.read(&out.map_hint[0], (std::streamsize)hint_len);
        if (!i)
            throw std::runtime_error("blueprint checkpoint: truncated file");
    } else {
        out.abs_cfg.flop_buckets = get<int32_t>(i);
        out.abs_cfg.turn_buckets = get<int32_t>(i);
        out.abs_cfg.river_buckets = get<int32_t>(i);
        out.abs_cfg.equity_rollouts = get<int32_t>(i);
        out.abs_cfg.quantile_samples = get<int32_t>(i);
        out.abs_cfg.seed = get<uint64_t>(i);
        for (auto& e : out.edges) e = get_doubles(i);
    }
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
    BlueprintStrategy s(load_abstraction_for(path, loaded),
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
