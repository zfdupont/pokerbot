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
// embeds the vocab hash and the full abstraction, so loaders can refuse
// mismatches and reconstruct the exact keyer without re-sampling.
//
// Three magics:
//  * SIXBP001 — legacy kind: config + quantile edges, as before.
//  * SIXBP002 — map-backed kind: same layout up to starting_stack, then
//    int32 kind=1, uint64 map content hash and a length-prefixed map filename
//    hint, then the table as v1. The map itself is resolved at load time
//    (see resolve_bucket_map) and refused if its content hash differs.
//  * SIXBP003 — map-backed kind plus a non-legacy pot layout: the SIXBP002
//    prefix through the map hint, then int32 mode, uint32 edge count and that
//    many doubles, then the table. Legacy layouts keep the SIXBP001/SIXBP002
//    bytes exactly.
// Native-endian; artifacts move between same-arch machines.
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
    int abs_kind = 0;  // 0 = legacy edges, 1 = bucket map
    uint64_t map_hash = 0;   // SIXBP002/SIXBP003 only
    std::string map_hint;    // SIXBP002/SIXBP003 only: the map's basename
    PotLayout pot_layout;    // legacy default for SIXBP001/SIXBP002
    AbstractionConfig abs_cfg;                    // legacy kind only
    std::array<std::vector<double>, 3> edges;     // legacy kind only
    uint64_t iterations;
    std::unordered_map<uint64_t, InfosetData> table;
};

// Throws std::runtime_error on bad magic, truncation, or vocab-hash
// mismatch against expected_vocab_hash.
LoadedBlueprint load_blueprint(const std::string& path,
                               uint64_t expected_vocab_hash);

// Locate a SIXBP002 checkpoint's bucket map. Search order:
//   1. $SIXMAX_BUCKET_MAP (full path) if set,
//   2. `explicit_path` if non-empty (config/CLI override),
//   3. dirname(ckpt_path)/hint,
//   4. $SIXMAX_TABLES_DIR/hint (default `sixmax/tables`, relative to cwd).
//
// $SIXMAX_BUCKET_MAP applies to EVERY SIXBP002 resolution in the process, so
// a head-to-head between checkpoints that live on different maps must leave
// it unset (use per-load explicit paths instead). A wrong map is still refused
// downstream by the content-hash check, never used silently.
//
// Throws std::runtime_error listing every path tried.
std::string resolve_bucket_map(const std::string& ckpt_path,
                               const std::string& hint,
                               const std::string& explicit_path = "");

// Rebuild the abstraction a loaded checkpoint was trained with: the stored
// edges for the legacy kind, the resolved bucket map (refusing a content-hash
// mismatch) for the map-backed kind. `explicit_map` (optional) is a config/CLI
// candidate inserted into the map search order (see resolve_bucket_map).
Abstraction load_abstraction_for(const std::string& ckpt_path,
                                 const LoadedBlueprint& loaded,
                                 const std::string& explicit_map = "");

// Read-only average-strategy view of an artifact, for eval and deployment.
// Owns its Abstraction (reconstructed from the stored edges, or from the
// resolved bucket map for a SIXBP002 checkpoint).
class BlueprintStrategy {
public:
    static BlueprintStrategy load(const std::string& path,
                                  const ActionVocab& vocab);
    std::vector<double> probs(uint64_t key) const;  // {} if unseen/empty
    // Keys the state through THIS strategy's abstraction — two strategies
    // with different abstractions can evaluate the same public state.
    std::vector<double> probs_for(const EngineGameState& s) const;
    uint64_t iterations() const { return iterations_; }
    // Count of QUERYABLE infosets (those with a positive strategy_sum, hence a
    // normalized policy). Stored entries never visited as an opponent are
    // dropped at load, so this can be < the trainer's num_infosets().
    size_t num_infosets() const { return probs_.size(); }
    int num_players() const { return num_players_; }
    // Training stack in BB, from the checkpoint metadata. Evaluators must use
    // it rather than assume 100 (a 200 BB blueprint keys pots and stacks in
    // its own frame).
    double starting_stack() const { return starting_stack_; }
    const Abstraction& abstraction() const { return abs_; }

private:
    BlueprintStrategy(Abstraction abs, uint64_t iters, int num_players,
                      double starting_stack)
        : abs_(std::move(abs)), iterations_(iters), num_players_(num_players),
          starting_stack_(starting_stack) {}
    Abstraction abs_;
    uint64_t iterations_;
    int num_players_;
    double starting_stack_;
    std::unordered_map<uint64_t, std::vector<double>> probs_;
};

}  // namespace sixmax
