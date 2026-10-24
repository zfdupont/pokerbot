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
constexpr char kMagicV3[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '3'};
constexpr char kMagicV4[8] = {'S', 'I', 'X', 'B', 'P', '0', '0', '4'};
constexpr size_t kMaxHintLength = 1u << 16;
constexpr uint32_t kMaxPotEdges = 15;

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
    Abstraction abs = [&] {
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
    }();
    // SIXBP001/SIXBP002 imply the legacy layout; SIXBP003 stores its own.
    abs.set_pot_layout(loaded.pot_layout);
    // SIXBP001..003 imply the bet-history field off; SIXBP004 stores its mode.
    abs.set_bet_history(loaded.bet_history);
    return abs;
}

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer) {
    const bool mapped = abs.is_map();
    const bool layout_v3 = !abs.pot_layout().is_legacy();
    const bool bet_history_on = abs.bet_history() != BetHistory::Off;
    if (layout_v3 && !mapped)
        throw std::invalid_argument(
            "blueprint checkpoint: non-legacy pot layout requires a "
            "bucket map");
    if (bet_history_on && !mapped)
        throw std::invalid_argument(
            "blueprint checkpoint: bet_history requires a bucket map");
    // V4 marks the opt-in key field; it always writes the pot layout so a
    // single reader handles every v4 artifact. Off keeps the old magics.
    const bool v4 = bet_history_on;
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
        o.write(v4 ? kMagicV4
                   : (layout_v3 ? kMagicV3
                                : (mapped ? kMagicV2 : kMagicV1)),
                8);
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
            if (layout_v3 || v4) {
                const PotLayout& layout = abs.pot_layout();
                put<int32_t>(o, (int32_t)layout.mode);
                put<uint32_t>(o, (uint32_t)layout.edges.size());
                for (double edge : layout.edges) put<double>(o, edge);
            }
            if (v4) put<int32_t>(o, (int32_t)abs.bet_history());
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
    // V4 keeps every V3 field (incl. the pot layout) and appends the
    // bet-history mode, so it reads through the same v3 branch below.
    const bool v4 = i && std::memcmp(magic, kMagicV4, 8) == 0;
    const bool v3 = v4 || (i && std::memcmp(magic, kMagicV3, 8) == 0);
    const bool mapped = v3 || (i && std::memcmp(magic, kMagicV2, 8) == 0);
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
        if (v3) {
            const int32_t mode = get<int32_t>(i);
            if (mode != (int32_t)PotMode::BB &&
                mode != (int32_t)PotMode::SPR)
                throw std::runtime_error(
                    "blueprint checkpoint: unknown pot mode " +
                    std::to_string(mode) + ": " + path);
            out.pot_layout.mode = (PotMode)mode;
            const uint32_t n_edges = get<uint32_t>(i);
            if (n_edges > kMaxPotEdges)
                throw std::runtime_error(
                    "blueprint checkpoint: pot layout too long: " + path);
            out.pot_layout.edges.resize(n_edges);
            for (auto& edge : out.pot_layout.edges) edge = get<double>(i);
            try {
                validate_pot_layout(out.pot_layout);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error(
                    "blueprint checkpoint: invalid pot layout: " +
                    std::string(e.what()));
            }
        }
        if (v4) {
            const int32_t bh = get<int32_t>(i);
            if (bh != (int32_t)BetHistory::Off &&
                bh != (int32_t)BetHistory::LastRaise)
                throw std::runtime_error(
                    "blueprint checkpoint: unknown bet history " +
                    std::to_string(bh) + ": " + path);
            out.bet_history = (BetHistory)bh;
        }
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
                        loaded.iterations, loaded.meta.num_players,
                        loaded.meta.starting_stack);
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
