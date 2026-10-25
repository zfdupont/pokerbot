#include "engine/engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include "game/safe_eval.h"

namespace sixmax {

void HandState::reroll_hidden(int hero, std::vector<int> pool,
                              std::mt19937_64& rng) {
    std::shuffle(pool.begin(), pool.end(), rng);
    size_t p = 0;
    const int n = num_players();
    for (int i = 0; i < n; ++i) {
        if (i == hero) continue;
        deck_[2 * i] = pool[p++];
        deck_[2 * i + 1] = pool[p++];
    }
    const int revealed = (int)board().size();
    for (int j = revealed; j < 5; ++j) deck_[2 * n + j] = pool[p++];
}

void settle_pots_into(const double* total_bets, const uint8_t* folded,
                      const int* rank_order, int n, double* payout) {
    assert(n >= 0 && n <= kMaxPlayers);
    for (int i = 0; i < n; ++i) payout[i] = 0.0;
    std::array<double, kMaxPlayers> levels{};
    int nlevels = 0;
    for (int i = 0; i < n; ++i)
        if (total_bets[i] > kChipEps) levels[nlevels++] = total_bets[i];
    std::sort(levels.begin(), levels.begin() + nlevels);
    nlevels = (int)(std::unique(levels.begin(), levels.begin() + nlevels,
                                [](double a, double b) {
                                    return std::abs(a - b) < kChipEps;
                                }) -
                     levels.begin());
    std::array<int, kMaxPlayers> winners{};
    double prev = 0.0;
    for (int k = 0; k < nlevels; ++k) {
        const double level = levels[k];
        double amount = 0.0;
        for (int i = 0; i < n; ++i)
            amount += std::max(0.0, std::min(total_bets[i], level) - prev);
        int best = std::numeric_limits<int>::max();
        for (int i = 0; i < n; ++i)
            if (!folded[i] && total_bets[i] >= level - kChipEps)
                best = std::min(best, rank_order[i]);
        int nwinners = 0;
        if (best != std::numeric_limits<int>::max()) {
            for (int i = 0; i < n; ++i)
                if (!folded[i] && total_bets[i] >= level - kChipEps &&
                    rank_order[i] == best)
                    winners[nwinners++] = i;
        } else {
            // Defensive: every contributor at this level folded. Unreachable
            // from valid play (the closing aggressor never folds); dead money
            // goes to all live players so chips are conserved.
            for (int i = 0; i < n; ++i)
                if (!folded[i]) winners[nwinners++] = i;
        }
        for (int j = 0; j < nwinners; ++j)
            payout[winners[j]] += amount / (double)nwinners;
        prev = level;
    }
}

std::vector<double> settle_pots(const std::vector<double>& total_bets,
                                const std::vector<uint8_t>& folded,
                                const std::vector<int>& rank_order) {
    if (total_bets.size() > (size_t)kMaxPlayers)
        throw std::invalid_argument(
            "settle_pots: at most kMaxPlayers players are supported");
    int n = (int)total_bets.size();
    std::vector<double> payout(n, 0.0);
    settle_pots_into(total_bets.data(), folded.data(), rank_order.data(), n,
                     payout.data());
    return payout;
}

HandState::HandState(const EngineConfig& cfg, int button, std::vector<int> deck,
                     std::vector<double> stacks)
    : cfg_(cfg), button_(button), deck_(std::move(deck)) {
    int n = cfg_.num_players;
    assert(n >= 2 && n <= 6);
    assert((int)deck_.size() >= 2 * n + 5);
    players_.resize(n);
    start_stacks_.resize(n);
    for (int i = 0; i < n; ++i) {
        double st = stacks.empty() ? cfg_.starting_stack : stacks[i];
        assert(st > 1.0 + kChipEps);  // blinds may not force an all-in
        players_[i].stack = st;
        start_stacks_[i] = st;
    }
    int sb = (n == 2) ? button_ : (button_ + 1) % n;
    int bb = (n == 2) ? (button_ + 1) % n : (button_ + 2) % n;
    commit(sb, 0.5);
    commit(bb, 1.0);
    current_bet_ = 1.0;
    last_raise_ = 1.0;
    next_ = (n == 2) ? button_ : (bb + 1) % n;
    to_act_ = num_can_act();  // everyone, including the BB option
}

HandState HandState::deal(const EngineConfig& cfg, int button,
                          std::mt19937_64& rng) {
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    return HandState(cfg, button, std::move(deck));
}

double HandState::pot() const {
    double p = 0.0;
    for (const auto& pl : players_) p += pl.total_bet;
    return p;
}

double HandState::to_call() const {
    const PlayerState& p = players_[next_];
    return std::min(std::max(0.0, current_bet_ - p.street_bet), p.stack);
}

bool HandState::can_raise() const {
    const PlayerState& p = players_[next_];
    if (p.stack <= (current_bet_ - p.street_bet) + kChipEps) return false;
    for (int i = 0; i < (int)players_.size(); ++i)
        if (i != next_ && !players_[i].folded && !players_[i].all_in)
            return true;  // someone can respond, so a bet has meaning
    return false;
}

std::array<int, 2> HandState::hole_cards(int i) const {
    return {deck_[2 * i], deck_[2 * i + 1]};
}

int HandState::board_into(std::array<int, 5>& out) const {
    static const int reveal[4] = {0, 3, 4, 5};
    int n = (int)players_.size();
    int count = reveal[(int)street_];
    for (int j = 0; j < count; ++j) out[j] = deck_[2 * n + j];
    return count;
}

std::vector<int> HandState::board() const {
    std::array<int, 5> cards{};
    int count = board_into(cards);
    return std::vector<int>(cards.begin(), cards.begin() + count);
}

void HandState::commit(int seat, double amount) {
    PlayerState& p = players_[seat];
    amount = std::min(amount, p.stack);
    p.stack -= amount;
    p.street_bet += amount;
    p.total_bet += amount;
    if (p.stack < kChipEps) {
        p.stack = 0.0;
        p.all_in = true;
    }
}

int HandState::next_active_after(int seat) const {
    int n = (int)players_.size();
    for (int k = 1; k <= n; ++k) {
        int i = (seat + k) % n;
        if (!players_[i].folded && !players_[i].all_in) return i;
    }
    return -1;
}

int HandState::num_can_act() const {
    int c = 0;
    for (const auto& p : players_)
        if (!p.folded && !p.all_in) ++c;
    return c;
}

void HandState::apply(const EngineAction& a) {
    assert(!terminal_);
    int seat = next_;
    PlayerState& p = players_[seat];
    switch (a.type) {
        case EngineActionType::Fold:
            p.folded = true;
            --to_act_;
            break;
        case EngineActionType::CheckCall:
            commit(seat, std::max(0.0, current_bet_ - p.street_bet));
            --to_act_;
            break;
        case EngineActionType::RaiseTo: {
            double target = std::min(a.amount, p.street_bet + p.stack);
            // Must meet min-raise OR be a short jam (all-in below min-raise is legal).
            assert(target >= min_raise_to() - kChipEps ||
                   target >= p.street_bet + p.stack - kChipEps);
            assert(target > current_bet_ + kChipEps);
            double raise_size = target - current_bet_;
            if (raise_size > last_raise_ - kChipEps) last_raise_ = raise_size;
            current_bet_ = target;
            commit(seat, target - p.street_bet);
            to_act_ = 0;  // everyone else live owes a response
            for (int i = 0; i < (int)players_.size(); ++i)
                if (i != seat && !players_[i].folded && !players_[i].all_in)
                    ++to_act_;
            break;
        }
    }
    finish_action(seat);
}

void HandState::finish_action(int seat) {
    int unfolded = 0;
    for (const auto& p : players_)
        if (!p.folded) ++unfolded;
    if (unfolded == 1) {
        terminal_ = true;
        settle();
        return;
    }
    if (to_act_ > 0) {
        next_ = next_active_after(seat);
        return;
    }
    close_round_or_advance();
}

void HandState::close_round_or_advance() {
    while (true) {
        if (street_ == Street::River) {
            terminal_ = true;
            settle();
            return;
        }
        street_ = (Street)((int)street_ + 1);
        for (auto& p : players_) p.street_bet = 0.0;
        current_bet_ = 0.0;
        last_raise_ = 1.0;  // min bet is 1 BB on a fresh street
        if (num_can_act() >= 2) {
            next_ = next_active_after(button_);
            to_act_ = num_can_act();
            return;
        }
        // Fewer than 2 players can act: no betting, run out the board.
    }
}

void HandState::showdown_order(std::array<int, kMaxPlayers>& order) const {
    int n = (int)players_.size();
    order.fill(0);
    std::array<int, kMaxPlayers> alive{};
    int nalive = 0;
    for (int i = 0; i < n; ++i)
        if (!players_[i].folded) alive[nalive++] = i;
    if (nalive < 2) return;  // fold-win: ranks are irrelevant
    std::array<int, 5> bd{};
    board_into(bd);  // river reached: 5 cards
    // HandRank is opaque and not default-constructible, so slots stay empty
    // until filled.
    std::array<std::optional<safe_eval::HandRank>, kMaxPlayers> hr{};
    for (int k = 0; k < nalive; ++k) {
        auto hc = hole_cards(alive[k]);
        std::array<int, 7> codes = {hc[0], hc[1], bd[0], bd[1],
                                    bd[2], bd[3], bd[4]};
        hr[k] = safe_eval::rank7(codes);
    }
    std::array<int, kMaxPlayers> pos{};
    for (int k = 0; k < nalive; ++k) pos[k] = k;
    std::sort(pos.begin(), pos.begin() + nalive, [&](int a, int b) {
        return hr[a]->beats(*hr[b]);
    });
    int rank = 0;
    for (int k = 0; k < nalive; ++k) {
        if (k > 0 && !hr[pos[k - 1]]->ties(*hr[pos[k]])) rank = k;
        order[alive[pos[k]]] = rank;
    }
}

void HandState::settle() {
    int n = (int)players_.size();
    std::array<double, kMaxPlayers> totals{};
    std::array<uint8_t, kMaxPlayers> folded{};
    std::array<int, kMaxPlayers> ranks{};
    for (int i = 0; i < n; ++i) {
        totals[i] = players_[i].total_bet;
        folded[i] = players_[i].folded ? 1 : 0;
    }
    showdown_order(ranks);
    std::array<double, kMaxPlayers> payout{};
    settle_pots_into(totals.data(), folded.data(), ranks.data(), n,
                     payout.data());
    payoffs_.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        players_[i].stack += payout[i];
        payoffs_[i] = players_[i].stack - start_stacks_[i];
    }
    next_ = -1;
}

}  // namespace sixmax
