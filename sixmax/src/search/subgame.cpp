#include "search/subgame.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sixmax {

Subgame::Subgame(const EngineGameState& street_root, const ActionVocab& vocab,
                 const RangeTracker& ranges, int hero, std::array<int, 2> hero_hole,
                 SubgameConfig cfg, const BlueprintStrategy* bp,
                 LeafEvalConfig leaf_cfg,
                 std::chrono::steady_clock::time_point deadline)
    : hero_seat_(hero), street_(street_root.street()), vocab_(&vocab),
      leaf_eval_(bp, leaf_cfg), cfg_(cfg) {
    leaf_eval_.set_deadline(deadline);
    // Asymmetric hand spaces: the hero holds one exact combo; the opponent
    // ranges over the combos the posterior left alive, and that posterior becomes
    // the opponent prior (renormalised over the surviving support).
    hero_combo_ = RangeTracker::combo_index(hero_hole[0], hero_hole[1]);
    const std::vector<double>& post = ranges.range(0);   // opponent posterior
    double total = 0.0;
    for (int i = 0; i < 1326; ++i) {
        if (post[i] > 0.0 && i != hero_combo_) {
            opp_hands_.push_back(i);
            opp_prior_.push_back(post[i]);
            total += post[i];
        }
    }
    if (total > 0.0)
        for (double& v : opp_prior_) v /= total;
    build(street_root, 0);
}

Subgame::~Subgame() = default;

int Subgame::build(const EngineGameState& s, int depth) {
    const int idx = (int)nodes_.size();
    nodes_.push_back(Node{});
    {
        Node& n = nodes_[idx];
        n.id = make_node_id(s, depth);
        s.legal_mask(n.mask);
        n.children.assign(vocab_->size(), -1);
        n.leaf = s.is_terminal() || s.street() != street_;
        if (!n.leaf) {
            // Node cap: expand only if every legal child fits. Checking BEFORE
            // pushing children keeps node_count() <= max_nodes and degrades the
            // node to an evaluated leaf instead of throwing. The leaf keeps the
            // full engine legal mask, so the public action space is preserved.
            int legal = 0;
            for (uint8_t m : n.mask) legal += m ? 1 : 0;
            const int room = cfg_.max_nodes - (int)nodes_.size();
            if (legal > room) n.leaf = true;
        }
        if (n.leaf) {
            n.actor = -1;
            // Fill the leaf's value matrix (hero row x opponent columns) unless
            // it needs the sampled continuation and no blueprint was supplied.
            if (!leaf_eval_.expired() &&
                (leaf_eval_.has_blueprint() || s.board().size() == 5))
                n.leaf_matrix = leaf_eval_.matrix(s, hero_seat_, {hero_combo_},
                                                  opp_hands_);
            return idx;
        }
        n.actor = s.current_player();
    }
    const std::vector<uint8_t> mask = nodes_[idx].mask;
    for (int a = 0; a < (int)mask.size(); ++a) {
        if (!mask[a]) continue;
        auto child = s.clone_engine();
        child->apply(a);
        // Re-index `nodes_` after the recursive push_back may have reallocated.
        nodes_[idx].children[a] = build(*child, depth + 1);
    }
    return idx;
}

uint64_t Subgame::make_node_id(const EngineGameState& s, int depth) const {
    uint64_t h = 1469598103934665603ull;   // FNV-1a, deterministic (no pointers)
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix((uint64_t)s.street());
    std::vector<int> board = s.board();
    std::sort(board.begin(), board.end());
    for (int c : board) mix((uint64_t)c + 1);
    mix((uint64_t)std::llround(s.pot() * 1e6));
    for (int p = 0; p < s.num_players(); ++p) {
        const PlayerState& ps = s.player_state(p);
        mix((uint64_t)std::llround(ps.stack * 1e6));
        mix((uint64_t)std::llround(ps.street_bet * 1e6));
    }
    mix((uint64_t)(s.is_terminal() ? 0xFFFFFFu : (uint64_t)s.current_player() + 1));
    mix((uint64_t)depth);
    return h;
}

bool Subgame::compatible(int h0, int h1) const {
    const int c0 = combo_for(0, h0), c1 = combo_for(1, h1);
    auto a = RangeTracker::combo_cards(c0);
    auto b = RangeTracker::combo_cards(c1);
    return a[0] != b[0] && a[0] != b[1] && a[1] != b[0] && a[1] != b[1];
}

std::vector<double> Subgame::prior_reach(int player) const {
    if (player == hero_seat_) return {1.0};   // point mass on the hero's one hand
    return opp_prior_;                         // tracked posterior over opp_hands_
}

std::unique_ptr<VectorState> Subgame::root() const {
    return std::make_unique<SubgameState>(this, 0);
}

std::vector<uint64_t> Subgame::node_ids() const {
    std::vector<uint64_t> out;
    out.reserve(nodes_.size());
    for (const Node& n : nodes_) out.push_back(n.id);
    return out;
}

uint64_t Subgame::query_node_id(const std::vector<int>& actions) const {
    int node = 0;
    for (int a : actions) {
        if (node < 0 || node >= (int)nodes_.size()) break;
        const Node& n = nodes_[node];
        if (n.leaf || a < 0 || a >= (int)n.children.size() || n.children[a] < 0)
            break;
        node = n.children[a];
    }
    return nodes_.empty() ? 0 : nodes_[node].id;
}

std::unique_ptr<VectorState> SubgameState::apply(int action) const {
    const Subgame::Node& n = sg_->nodes_[node_];
    if (n.leaf) throw std::logic_error("apply() on a subgame leaf");
    if (action < 0 || action >= (int)n.children.size() || n.children[action] < 0)
        throw std::out_of_range("illegal action at subgame node");
    return std::make_unique<SubgameState>(sg_, n.children[action]);
}

double SubgameState::utility(int player, int h0, int h1) const {
    const Subgame::Node& n = sg_->nodes_[node_];
    if (n.leaf_matrix.empty()) return 0.0;   // leaf not yet evaluated (Task 4)
    // leaf_matrix is one row over the opponent's compact hands, payoff to hero.
    const int hero_idx = sg_->hero_seat_;
    const int opp_hand = (hero_idx == 0) ? h1 : h0;
    const double hero_payoff = n.leaf_matrix[opp_hand];
    return (player == hero_idx) ? hero_payoff : -hero_payoff;
}

}  // namespace sixmax
