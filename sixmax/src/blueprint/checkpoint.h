#pragma once
#include <array>
#include <string>
#include <unordered_map>
#include "abstraction/abstraction.h"
#include "blueprint/engine_game.h"
#include "blueprint/trainer.h"
#include "vocab/vocab.h"

namespace sixmax {

// Binary blueprint artifact. Self-describing per the artifact contract:
// embeds the vocab hash and the full abstraction (config + quantile edges),
// so loaders can refuse mismatches and reconstruct the exact keyer without
// re-sampling. Native-endian; artifacts move between same-arch machines.
struct BlueprintMeta {
    uint64_t vocab_hash;
    int num_players;
    double starting_stack;
    int action_dim;
};

void save_blueprint(const std::string& path, const BlueprintMeta& meta,
                    const Abstraction& abs, const BlueprintTrainer& trainer);

struct LoadedBlueprint {
    BlueprintMeta meta;
    AbstractionConfig abs_cfg;
    std::array<std::vector<double>, 3> edges;
    uint64_t iterations;
    std::unordered_map<uint64_t, InfosetData> table;
};

// Throws std::runtime_error on bad magic, truncation, or vocab-hash
// mismatch against expected_vocab_hash.
LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash);

// Read-only average-strategy view of an artifact, for eval and deployment.
// Owns its Abstraction (reconstructed from the stored edges).
class BlueprintStrategy {
public:
    static BlueprintStrategy load(const std::string& path,
                                  const ActionVocab& vocab);
    std::vector<double> probs(uint64_t key) const;  // {} if unseen/empty
    // Keys the state through THIS strategy's abstraction — two strategies
    // with different abstractions can evaluate the same public state.
    std::vector<double> probs_for(const EngineGameState& s) const;
    uint64_t iterations() const { return iterations_; }
    size_t num_infosets() const { return probs_.size(); }
    int num_players() const { return num_players_; }
    const Abstraction& abstraction() const { return abs_; }

private:
    BlueprintStrategy(Abstraction abs, uint64_t iters, int num_players)
        : abs_(std::move(abs)), iterations_(iters), num_players_(num_players) {}
    Abstraction abs_;
    uint64_t iterations_;
    int num_players_;
    std::unordered_map<uint64_t, std::vector<double>> probs_;
};

}  // namespace sixmax
