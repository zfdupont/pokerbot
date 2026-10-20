// Suit-canonical hand indexing, derived from kdub0/hand-isomorphism
// (Kevin Waugh's reference implementation of "A Fast and Optimal Hand
// Isomorphism Algorithm", 2013).
//
// Copyright (c) 2013 Kevin Waugh (waugh@cs.cmu.edu).
// All rights reserved.
//
// Redistribution and use in source and binary forms are permitted
// provided that the above copyright notice and this paragraph are
// duplicated in all such forms and that any documentation,
// advertising materials, and other materials related to such
// distribution and use acknowledge that the software was developed
// by the <organization>.  The name of the
// <organization> may not be used to endorse or promote products derived
// from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#include "abstraction/hand_index.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sixmax {
namespace {

constexpr uint32_t kSuits = 4;
constexpr uint32_t kRanks = 13;
constexpr uint32_t kCards = 52;
constexpr uint32_t kMaxRounds = 8;
constexpr uint32_t kMaxCardsPerRound = 15;
constexpr uint32_t kMaxGroupIndex = 0x100000;
constexpr uint32_t kRoundShift = 4;
constexpr uint32_t kRoundMask = 0xf;

uint8_t nth_unset[1 << kRanks][kRanks];
bool equal[1 << (kSuits - 1)][kSuits];
uint32_t nCr_ranks[kRanks + 1][kRanks + 1];
uint32_t rank_set_to_index[1 << kRanks];
uint32_t index_to_rank_set[kRanks + 1][1 << kRanks];
uint32_t suit_permutations[24][kSuits];
uint64_t nCr_groups[kMaxGroupIndex][kSuits + 1];

void init_globals() {
    for (uint32_t i = 0; i < 1u << (kSuits - 1); ++i) {
        for (uint32_t j = 1; j < kSuits; ++j) {
            equal[i][j] = i & (1u << (j - 1));
        }
    }

    for (uint32_t i = 0; i < 1u << kRanks; ++i) {
        for (uint32_t j = 0, set = ~i & ((1u << kRanks) - 1); j < kRanks;
             ++j, set &= set - 1) {
            nth_unset[i][j] = set ? __builtin_ctz(set) : 0xff;
        }
    }

    nCr_ranks[0][0] = 1;
    for (uint32_t i = 1; i < kRanks + 1; ++i) {
        nCr_ranks[i][0] = nCr_ranks[i][i] = 1;
        for (uint32_t j = 1; j < i; ++j) {
            nCr_ranks[i][j] = nCr_ranks[i - 1][j - 1] + nCr_ranks[i - 1][j];
        }
    }

    nCr_groups[0][0] = 1;
    for (uint32_t i = 1; i < kMaxGroupIndex; ++i) {
        nCr_groups[i][0] = 1;
        if (i < kSuits + 1) {
            nCr_groups[i][i] = 1;
        }
        for (uint32_t j = 1; j < (i < kSuits + 1 ? i : kSuits + 1); ++j) {
            nCr_groups[i][j] = nCr_groups[i - 1][j - 1] + nCr_groups[i - 1][j];
        }
    }

    for (uint32_t i = 0; i < 1u << kRanks; ++i) {
        for (uint32_t set = i, j = 1; set; ++j, set &= set - 1) {
            rank_set_to_index[i] += nCr_ranks[__builtin_ctz(set)][j];
        }
        index_to_rank_set[__builtin_popcount(i)][rank_set_to_index[i]] = i;
    }

    uint32_t num_permutations = 1;
    for (uint32_t i = 2; i <= kSuits; ++i) {
        num_permutations *= i;
    }
    assert(num_permutations == 24);

    for (uint32_t i = 0; i < num_permutations; ++i) {
        for (uint32_t j = 0, index = i, used = 0; j < kSuits; ++j) {
            uint32_t suit = index % (kSuits - j);
            index /= kSuits - j;
            uint32_t shifted_suit = nth_unset[used][suit];
            suit_permutations[i][j] = shifted_suit;
            used |= 1u << shifted_suit;
        }
    }
}

void ensure_globals() {
    static const bool initialized = [] {
        init_globals();
        return true;
    }();
    (void)initialized;
}

inline uint32_t deck_get_suit(uint32_t card) { return card & 3; }
inline uint32_t deck_get_rank(uint32_t card) { return card >> 2; }
inline uint32_t deck_make_card(uint32_t suit, uint32_t rank) {
    return rank << 2 | suit;
}

struct hand_indexer_t {
    uint8_t cards_per_round[kMaxRounds], round_start[kMaxRounds];
    uint32_t rounds, configurations[kMaxRounds], permutations[kMaxRounds];
    uint64_t round_size[kMaxRounds];

    uint32_t* permutation_to_configuration[kMaxRounds];
    uint32_t* permutation_to_pi[kMaxRounds];
    uint32_t* configuration_to_equal[kMaxRounds];
    uint32_t (*configuration[kMaxRounds])[kSuits];
    uint32_t (*configuration_to_suit_size[kMaxRounds])[kSuits];
    uint64_t* configuration_to_offset[kMaxRounds];
};

struct hand_indexer_state_t {
    uint32_t suit_index[kSuits];
    uint32_t suit_multiplier[kSuits];
    uint32_t round, permutation_index, permutation_multiplier;
    uint32_t used_ranks[kSuits];
};

void enumerate_configurations_r(
    uint32_t rounds, const uint8_t cards_per_round[], uint32_t round,
    uint32_t remaining, uint32_t suit, uint32_t equal, uint32_t used[],
    uint32_t configuration[],
    void (*observe)(uint32_t, uint32_t[], void*), void* data) {
    if (suit == kSuits) {
        observe(round, configuration, data);

        if (round + 1 < rounds) {
            enumerate_configurations_r(rounds, cards_per_round, round + 1,
                                       cards_per_round[round + 1], 0, equal,
                                       used, configuration, observe, data);
        }
    } else {
        uint32_t min = 0;
        if (suit == kSuits - 1) {
            min = remaining;
        }

        uint32_t max = kRanks - used[suit];
        if (remaining < max) {
            max = remaining;
        }

        uint32_t previous = kRanks + 1;
        bool was_equal = equal & (1u << suit);
        if (was_equal) {
            previous = configuration[suit - 1] >>
                       (kRoundShift * (rounds - round - 1)) & kRoundMask;
            if (previous < max) {
                max = previous;
            }
        }

        uint32_t old_configuration = configuration[suit];
        uint32_t old_used = used[suit];
        for (uint32_t i = min; i <= max; ++i) {
            uint32_t new_configuration =
                old_configuration | i << (kRoundShift * (rounds - round - 1));
            uint32_t new_equal =
                (equal & ~(1u << suit)) | (was_equal & (i == previous)) << suit;

            used[suit] = old_used + i;
            configuration[suit] = new_configuration;
            enumerate_configurations_r(rounds, cards_per_round, round,
                                       remaining - i, suit + 1, new_equal, used,
                                       configuration, observe, data);
            configuration[suit] = old_configuration;
            used[suit] = old_used;
        }
    }
}

void enumerate_configurations(
    uint32_t rounds, const uint8_t cards_per_round[],
    void (*observe)(uint32_t, uint32_t[], void*), void* data) {
    uint32_t used[kSuits] = {0}, configuration[kSuits] = {0};
    enumerate_configurations_r(rounds, cards_per_round, 0, cards_per_round[0],
                               0, (1u << kSuits) - 2, used, configuration,
                               observe, data);
}

void count_configurations(uint32_t round, uint32_t configuration[], void* data) {
    uint32_t* counts = static_cast<uint32_t*>(data);
    ++counts[round];
    (void)configuration;
}

void tabulate_configurations(uint32_t round, uint32_t configuration[],
                             void* data) {
    hand_indexer_t* indexer = static_cast<hand_indexer_t*>(data);

    uint32_t id = indexer->configurations[round]++;
    for (; id > 0; --id) {
        for (uint32_t i = 0; i < kSuits; ++i) {
            if (configuration[i] < indexer->configuration[round][id - 1][i]) {
                break;
            } else if (configuration[i] >
                       indexer->configuration[round][id - 1][i]) {
                goto out;
            }
        }
        for (uint32_t i = 0; i < kSuits; ++i) {
            indexer->configuration[round][id][i] =
                indexer->configuration[round][id - 1][i];
            indexer->configuration_to_suit_size[round][id][i] =
                indexer->configuration_to_suit_size[round][id - 1][i];
        }
        indexer->configuration_to_offset[round][id] =
            indexer->configuration_to_offset[round][id - 1];
        indexer->configuration_to_equal[round][id] =
            indexer->configuration_to_equal[round][id - 1];
    }
out:;

    indexer->configuration_to_offset[round][id] = 1;
    for (uint32_t i = 0; i < kSuits; ++i) {
        indexer->configuration[round][id][i] = configuration[i];
    }

    uint32_t equal_bits = 0;
    for (uint32_t i = 0; i < kSuits;) {
        uint64_t size = 1;
        for (uint32_t j = 0, remaining = kRanks; j <= round; ++j) {
            uint32_t ranks =
                configuration[i] >>
                (kRoundShift * (indexer->rounds - j - 1)) & kRoundMask;
            size *= nCr_ranks[remaining][ranks];
            remaining -= ranks;
        }
        assert(size + kSuits - 1 < kMaxGroupIndex);

        uint32_t j = i + 1;
        for (; j < kSuits && configuration[j] == configuration[i]; ++j) {
        }
        for (uint32_t k = i; k < j; ++k) {
            indexer->configuration_to_suit_size[round][id][k] = size;
        }

        indexer->configuration_to_offset[round][id] *=
            nCr_groups[size + j - i - 1][j - i];

        for (uint32_t k = i + 1; k < j; ++k) {
            equal_bits |= 1u << k;
        }

        i = j;
    }

    indexer->configuration_to_equal[round][id] = equal_bits >> 1;
}

void enumerate_permutations_r(
    uint32_t rounds, const uint8_t cards_per_round[], uint32_t round,
    uint32_t remaining, uint32_t suit, uint32_t used[], uint32_t count[],
    void (*observe)(uint32_t, uint32_t[], void*), void* data) {
    if (suit == kSuits) {
        observe(round, count, data);

        if (round + 1 < rounds) {
            enumerate_permutations_r(rounds, cards_per_round, round + 1,
                                     cards_per_round[round + 1], 0, used, count,
                                     observe, data);
        }
    } else {
        uint32_t min = 0;
        if (suit == kSuits - 1) {
            min = remaining;
        }
        uint32_t max = kRanks - used[suit];
        if (remaining < max) {
            max = remaining;
        }

        uint32_t old_count = count[suit], old_used = used[suit];
        for (uint32_t i = min; i <= max; ++i) {
            uint32_t new_count =
                old_count | i << (kRoundShift * (rounds - round - 1));

            used[suit] = old_used + i;
            count[suit] = new_count;
            enumerate_permutations_r(rounds, cards_per_round, round,
                                     remaining - i, suit + 1, used, count,
                                     observe, data);
            count[suit] = old_count;
            used[suit] = old_used;
        }
    }
}

void enumerate_permutations(
    uint32_t rounds, const uint8_t cards_per_round[],
    void (*observe)(uint32_t, uint32_t[], void*), void* data) {
    uint32_t used[kSuits] = {0}, count[kSuits] = {0};
    enumerate_permutations_r(rounds, cards_per_round, 0, cards_per_round[0], 0,
                             used, count, observe, data);
}

void count_permutations(uint32_t round, uint32_t count[], void* data) {
    hand_indexer_t* indexer = static_cast<hand_indexer_t*>(data);

    uint32_t idx = 0, mult = 1;
    for (uint32_t i = 0; i <= round; ++i) {
        for (uint32_t j = 0, remaining = indexer->cards_per_round[i];
             j < kSuits - 1; ++j) {
            uint32_t size = count[j] >>
                            ((indexer->rounds - i - 1) * kRoundShift) &
                            kRoundMask;
            idx += mult * size;
            mult *= remaining + 1;
            remaining -= size;
        }
    }

    if (indexer->permutations[round] < idx + 1) {
        indexer->permutations[round] = idx + 1;
    }
}

void tabulate_permutations(uint32_t round, uint32_t count[], void* data) {
    hand_indexer_t* indexer = static_cast<hand_indexer_t*>(data);

    uint32_t idx = 0, mult = 1;
    for (uint32_t i = 0; i <= round; ++i) {
        for (uint32_t j = 0, remaining = indexer->cards_per_round[i];
             j < kSuits - 1; ++j) {
            uint32_t size = count[j] >>
                            ((indexer->rounds - i - 1) * kRoundShift) &
                            kRoundMask;
            idx += mult * size;
            mult *= remaining + 1;
            remaining -= size;
        }
    }

    uint32_t pi[kSuits];
    for (uint32_t i = 0; i < kSuits; ++i) {
        pi[i] = i;
    }

    for (uint32_t i = 1; i < kSuits; ++i) {
        uint32_t j = i, pi_i = pi[i];
        for (; j > 0; --j) {
            if (count[pi_i] > count[pi[j - 1]]) {
                pi[j] = pi[j - 1];
            } else {
                break;
            }
        }
        pi[j] = pi_i;
    }

    uint32_t pi_idx = 0, pi_mult = 1, pi_used = 0;
    for (uint32_t i = 0; i < kSuits; ++i) {
        uint32_t this_bit = 1u << pi[i];
        uint32_t smaller = __builtin_popcount((this_bit - 1) & pi_used);
        pi_idx += (pi[i] - smaller) * pi_mult;
        pi_mult *= kSuits - i;
        pi_used |= this_bit;
    }

    indexer->permutation_to_pi[round][idx] = pi_idx;

    uint32_t low = 0, high = indexer->configurations[round];
    while (low < high) {
        uint32_t mid = (low + high) / 2;

        int32_t compare = 0;
        for (uint32_t i = 0; i < kSuits; ++i) {
            uint32_t this_count = count[pi[i]];
            uint32_t other = indexer->configuration[round][mid][i];
            if (other > this_count) {
                compare = -1;
                break;
            } else if (other < this_count) {
                compare = 1;
                break;
            }
        }

        if (compare == -1) {
            high = mid;
        } else if (compare == 0) {
            low = high = mid;
        } else {
            low = mid + 1;
        }
    }

    indexer->permutation_to_configuration[round][idx] = low;
}

bool hand_indexer_init(uint32_t rounds, const uint8_t cards_per_round[],
                       hand_indexer_t* indexer) {
    if (rounds == 0) {
        return false;
    }
    if (rounds > kMaxRounds) {
        return false;
    }
    for (uint32_t i = 0, count = 0; i < rounds; ++i) {
        count += cards_per_round[i];
        if (count > kCards) {
            return false;
        }
    }

    std::memset(indexer, 0, sizeof(hand_indexer_t));

    indexer->rounds = rounds;
    std::memcpy(indexer->cards_per_round, cards_per_round, rounds);
    for (uint32_t i = 0, j = 0; i < rounds; ++i) {
        indexer->round_start[i] = j;
        j += cards_per_round[i];
    }

    std::memset(indexer->configurations, 0, sizeof(indexer->configurations));
    enumerate_configurations(rounds, cards_per_round, count_configurations,
                             indexer->configurations);

    for (uint32_t i = 0; i < rounds; ++i) {
        indexer->configuration_to_equal[i] =
            static_cast<uint32_t*>(std::calloc(indexer->configurations[i],
                                               sizeof(uint32_t)));
        indexer->configuration_to_offset[i] =
            static_cast<uint64_t*>(std::calloc(indexer->configurations[i],
                                               sizeof(uint64_t)));
        indexer->configuration[i] = static_cast<uint32_t(*)[kSuits]>(
            std::calloc(indexer->configurations[i], kSuits * sizeof(uint32_t)));
        indexer->configuration_to_suit_size[i] =
            static_cast<uint32_t(*)[kSuits]>(std::calloc(
                indexer->configurations[i], kSuits * sizeof(uint32_t)));
        if (!indexer->configuration_to_equal[i] ||
            !indexer->configuration_to_offset[i] || !indexer->configuration[i] ||
            !indexer->configuration_to_suit_size[i]) {
            return false;
        }
    }

    std::memset(indexer->configurations, 0, sizeof(indexer->configurations));
    enumerate_configurations(rounds, cards_per_round, tabulate_configurations,
                             indexer);

    for (uint32_t i = 0; i < rounds; ++i) {
        uint64_t accum = 0;
        for (uint32_t j = 0; j < indexer->configurations[i]; ++j) {
            uint64_t next = accum + indexer->configuration_to_offset[i][j];
            indexer->configuration_to_offset[i][j] = accum;
            accum = next;
        }
        indexer->round_size[i] = accum;
    }

    std::memset(indexer->permutations, 0, sizeof(indexer->permutations));
    enumerate_permutations(rounds, cards_per_round, count_permutations, indexer);

    for (uint32_t i = 0; i < rounds; ++i) {
        indexer->permutation_to_configuration[i] =
            static_cast<uint32_t*>(std::calloc(indexer->permutations[i],
                                               sizeof(uint32_t)));
        indexer->permutation_to_pi[i] = static_cast<uint32_t*>(
            std::calloc(indexer->permutations[i], sizeof(uint32_t)));
        if (!indexer->permutation_to_configuration[i] ||
            !indexer->permutation_to_pi[i]) {
            return false;
        }
    }

    enumerate_permutations(rounds, cards_per_round, tabulate_permutations,
                           indexer);

    return true;
}

void hand_indexer_free(hand_indexer_t* indexer) {
    for (uint32_t i = 0; i < indexer->rounds; ++i) {
        std::free(indexer->permutation_to_configuration[i]);
        std::free(indexer->permutation_to_pi[i]);
        std::free(indexer->configuration_to_equal[i]);
        std::free(indexer->configuration_to_offset[i]);
        std::free(indexer->configuration[i]);
        std::free(indexer->configuration_to_suit_size[i]);
    }
}

void hand_indexer_state_init(hand_indexer_t* indexer,
                             hand_indexer_state_t* state) {
    (void)indexer;
    std::memset(state, 0, sizeof(hand_indexer_state_t));

    state->permutation_multiplier = 1;
    for (uint32_t i = 0; i < kSuits; ++i) {
        state->suit_multiplier[i] = 1;
    }
}

uint64_t hand_index_next_round(const hand_indexer_t* indexer,
                               const uint8_t cards[],
                               hand_indexer_state_t* state) {
    uint32_t round = state->round++;
    assert(round < indexer->rounds);

    uint32_t ranks[kSuits] = {0}, shifted_ranks[kSuits] = {0};
    for (uint32_t i = 0; i < indexer->cards_per_round[round]; ++i) {
        assert(cards[i] < kCards); /* valid card */

        uint32_t rank = deck_get_rank(cards[i]);
        uint32_t suit = deck_get_suit(cards[i]);
        uint32_t rank_bit = 1u << rank;
        assert(!(ranks[suit] & rank_bit));
        ranks[suit] |= rank_bit;
        shifted_ranks[suit] |=
            rank_bit >>
            __builtin_popcount((rank_bit - 1) & state->used_ranks[suit]);
    }

    for (uint32_t i = 0; i < kSuits; ++i) {
        assert(!(state->used_ranks[i] & ranks[i])); /* no duplicate cards */

        uint32_t used_size = __builtin_popcount(state->used_ranks[i]);
        uint32_t this_size = __builtin_popcount(ranks[i]);
        state->suit_index[i] +=
            state->suit_multiplier[i] * rank_set_to_index[shifted_ranks[i]];
        state->suit_multiplier[i] *= nCr_ranks[kRanks - used_size][this_size];
        state->used_ranks[i] |= ranks[i];
    }

    for (uint32_t i = 0, remaining = indexer->cards_per_round[round];
         i < kSuits - 1; ++i) {
        uint32_t this_size = __builtin_popcount(ranks[i]);
        state->permutation_index += state->permutation_multiplier * this_size;
        state->permutation_multiplier *= remaining + 1;
        remaining -= this_size;
    }

    uint32_t configuration =
        indexer->permutation_to_configuration[round][state->permutation_index];
    uint32_t pi_index =
        indexer->permutation_to_pi[round][state->permutation_index];
    uint32_t equal_index = indexer->configuration_to_equal[round][configuration];
    uint64_t offset = indexer->configuration_to_offset[round][configuration];
    const uint32_t* pi = suit_permutations[pi_index];

    uint64_t suit_index[kSuits], suit_multiplier[kSuits];
    for (uint32_t i = 0; i < kSuits; ++i) {
        suit_index[i] = state->suit_index[pi[i]];
        suit_multiplier[i] = state->suit_multiplier[pi[i]];
    }

/* sort using an optimal sorting network */
#define swap(u, v)                                  \
    do {                                            \
        if (suit_index[u] > suit_index[v]) {        \
            suit_index[u] ^= suit_index[v];         \
            suit_index[v] ^= suit_index[u];         \
            suit_index[u] ^= suit_index[v];         \
        }                                           \
    } while (0)

    uint64_t index = offset, multiplier = 1;
    for (uint32_t i = 0; i < kSuits;) {
        uint64_t part, size;

        if (i + 1 < kSuits && equal[equal_index][i + 1]) {
            if (i + 2 < kSuits && equal[equal_index][i + 2]) {
                if (i + 3 < kSuits && equal[equal_index][i + 3]) {
                    /* four equal suits */
                    swap(i, i + 1);
                    swap(i + 2, i + 3);
                    swap(i, i + 2);
                    swap(i + 1, i + 3);
                    swap(i + 1, i + 2);
                    part = suit_index[i] +
                           nCr_groups[suit_index[i + 1] + 1][2] +
                           nCr_groups[suit_index[i + 2] + 2][3] +
                           nCr_groups[suit_index[i + 3] + 3][4];
                    size = nCr_groups[suit_multiplier[i] + 3][4];
                    i += 4;
                } else {
                    /* three equal suits */
                    swap(i, i + 1);
                    swap(i, i + 2);
                    swap(i + 1, i + 2);
                    part = suit_index[i] +
                           nCr_groups[suit_index[i + 1] + 1][2] +
                           nCr_groups[suit_index[i + 2] + 2][3];
                    size = nCr_groups[suit_multiplier[i] + 2][3];
                    i += 3;
                }
            } else {
                /* two equal suits */
                swap(i, i + 1);
                part = suit_index[i] + nCr_groups[suit_index[i + 1] + 1][2];
                size = nCr_groups[suit_multiplier[i] + 1][2];
                i += 2;
            }
        } else {
            /* no equal suits */
            part = suit_index[i];
            size = suit_multiplier[i];
            i += 1;
        }

        index += multiplier * part;
        multiplier *= size;
    }

#undef swap

    return index;
}

uint64_t hand_index_last(const hand_indexer_t* indexer,
                         const uint8_t cards[]) {
    hand_indexer_state_t state;
    hand_indexer_state_init(const_cast<hand_indexer_t*>(indexer), &state);

    uint64_t index = 0;
    for (uint32_t i = 0, j = 0; i < indexer->rounds;
         j += indexer->cards_per_round[i++]) {
        index = hand_index_next_round(indexer, cards + j, &state);
    }
    return index;
}

bool hand_unindex(const hand_indexer_t* indexer, uint32_t round,
                  uint64_t index, uint8_t cards[]) {
    if (round >= indexer->rounds || index >= indexer->round_size[round]) {
        return false;
    }

    uint32_t low = 0, high = indexer->configurations[round],
             configuration_idx = 0;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (indexer->configuration_to_offset[round][mid] <= index) {
            configuration_idx = mid;
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    index -= indexer->configuration_to_offset[round][configuration_idx];

    uint64_t suit_index[kSuits];
    for (uint32_t i = 0; i < kSuits;) {
        uint32_t j = i + 1;
        for (; j < kSuits &&
               indexer->configuration[round][configuration_idx][j] ==
                   indexer->configuration[round][configuration_idx][i];
             ++j) {
        }

        uint32_t suit_size =
            indexer->configuration_to_suit_size[round][configuration_idx][i];
        uint64_t group_size = nCr_groups[suit_size + j - i - 1][j - i];
        uint64_t group_index = index % group_size;
        index /= group_size;

        for (; i < j - 1; ++i) {
            // Avoid log(0) in the estimate below; a zero group index means
            // every remaining suit in this group takes index 0.
            if (group_index == 0) {
                suit_index[i] = 0;
                continue;
            }
            const double group_width = static_cast<double>(j - i);
            const double log_group_index =
                std::log(static_cast<double>(group_index));
            double low_estimate = std::floor(
                std::exp(log_group_index / group_width - 1.0 +
                         std::log(group_width)) -
                group_width);
            double high_estimate = std::ceil(
                std::exp(log_group_index / group_width +
                         std::log(group_width)) -
                group_width + 1.0);
            if (low_estimate < 0.0) {
                low_estimate = 0.0;
            }
            if (high_estimate < 0.0) {
                high_estimate = 0.0;
            }
            low = static_cast<uint32_t>(low_estimate);
            high = static_cast<uint32_t>(high_estimate);
            if (high > suit_size) {
                high = suit_size;
            }
            if (high <= low) {
                low = 0;
                suit_index[i] = 0;
            } else {
                suit_index[i] = low;
            }
            while (low < high) {
                uint32_t mid = (low + high) / 2;
                if (nCr_groups[mid + j - i - 1][j - i] <= group_index) {
                    suit_index[i] = mid;
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            group_index -= nCr_groups[suit_index[i] + j - i - 1][j - i];
        }

        suit_index[i] = group_index;
        ++i;
    }

    uint8_t location[kMaxRounds];
    std::memcpy(location, indexer->round_start, kMaxRounds);
    for (uint32_t i = 0; i < kSuits; ++i) {
        uint32_t used = 0, m = 0;
        for (uint32_t j = 0; j < indexer->rounds; ++j) {
            uint32_t n =
                indexer->configuration[round][configuration_idx][i] >>
                (kRoundShift * (indexer->rounds - j - 1)) & kRoundMask;
            uint32_t round_size = nCr_ranks[kRanks - m][n];
            m += n;
            uint32_t round_idx = suit_index[i] % round_size;
            suit_index[i] /= round_size;
            uint32_t shifted_cards = index_to_rank_set[n][round_idx],
                     rank_set = 0;
            for (uint32_t k = 0; k < n; ++k) {
                uint32_t shifted_card = shifted_cards & -shifted_cards;
                shifted_cards ^= shifted_card;
                uint32_t card =
                    nth_unset[used][__builtin_ctz(shifted_card)];
                rank_set |= 1u << card;
                cards[location[j]++] = static_cast<uint8_t>(
                    deck_make_card(i, card));
            }
            used |= rank_set;
        }
    }

    return true;
}

// One Waugh indexer per street: rounds {2,3}, {2,4}, {2,5}. The board is a
// single unordered round: turn/river order never affects a Hold'em hand, and
// these are the round sets whose canonical sizes are the required
// 1,286,792 / 13,960,050 / 123,156,254. (Splitting the board into
// {2,3,1} / {2,3,1,1} rounds would instead produce 55,190,538 and
// 2,428,287,420 classes by distinguishing the turn and river cards.)
struct Tables {
    hand_indexer_t indexers[3];

    Tables() {
        ensure_globals();
        static const uint8_t cards_per_round[3][2] = {
            {2, 3},
            {2, 4},
            {2, 5},
        };
        for (uint32_t i = 0; i < 3; ++i) {
            if (!hand_indexer_init(2, cards_per_round[i], &indexers[i])) {
                throw std::runtime_error("failed to build hand index tables");
            }
        }
    }

    ~Tables() {
        for (uint32_t i = 0; i < 3; ++i) {
            hand_indexer_free(&indexers[i]);
        }
    }

    const hand_indexer_t& at(int street) const { return indexers[street - 1]; }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

constexpr uint32_t kFactorial[5] = {1, 1, 2, 6, 24};

void validate_street(int street) {
    if (street < 1 || street > 3) {
        throw std::invalid_argument(
            "street must be 1 (flop), 2 (turn), or 3 (river)");
    }
}

void validate_cards(const std::array<int, 2>& hole,
                    const std::vector<int>& board, int street) {
    const int expected_board = street + 2;
    if (static_cast<int>(board.size()) != expected_board) {
        throw std::invalid_argument("board size does not match street");
    }
    bool seen[kCards] = {false};
    for (int c : {hole[0], hole[1]}) {
        if (c < 0 || c >= static_cast<int>(kCards)) {
            throw std::invalid_argument("card out of range");
        }
        if (seen[c]) {
            throw std::invalid_argument("duplicate card");
        }
        seen[c] = true;
    }
    for (int c : board) {
        if (c < 0 || c >= static_cast<int>(kCards)) {
            throw std::invalid_argument("card out of range");
        }
        if (seen[c]) {
            throw std::invalid_argument("duplicate card");
        }
        seen[c] = true;
    }
}

uint32_t raw_multiplicity(const hand_indexer_t& indexer, uint32_t round,
                          const uint8_t* cards) {
    // Suits with identical rank sets in every round are interchangeable, so
    // the stabilizer is the product of the factorials of those groups.
    uint64_t signature[kSuits] = {0, 0, 0, 0};
    uint32_t offset = 0;
    for (uint32_t r = 0; r <= round; ++r) {
        for (uint32_t i = 0; i < indexer.cards_per_round[r]; ++i) {
            uint32_t card = cards[offset++];
            uint64_t bit = 1ull << (deck_get_rank(card) + kRanks * r);
            signature[deck_get_suit(card)] |= bit;
        }
    }

    uint32_t stabilizer = 1;
    bool grouped[kSuits] = {false, false, false, false};
    for (uint32_t i = 0; i < kSuits; ++i) {
        if (grouped[i]) continue;
        uint32_t group = 0;
        for (uint32_t j = i; j < kSuits; ++j) {
            if (signature[j] == signature[i]) {
                grouped[j] = true;
                ++group;
            }
        }
        stabilizer *= kFactorial[group];
    }
    return 24 / stabilizer;
}

}  // namespace

const HandIndexer& HandIndexer::instance() {
    static const HandIndexer indexer;
    return indexer;
}

uint64_t HandIndexer::size(int street) const {
    validate_street(street);
    const hand_indexer_t& indexer = tables().at(street);
    return indexer.round_size[indexer.rounds - 1];
}

uint64_t HandIndexer::index(int street, const std::array<int, 2>& hole,
                            const std::vector<int>& board) const {
    validate_street(street);
    validate_cards(hole, board, street);

    uint8_t cards[7];
    cards[0] = static_cast<uint8_t>(hole[0]);
    cards[1] = static_cast<uint8_t>(hole[1]);
    for (size_t i = 0; i < board.size(); ++i) {
        cards[2 + i] = static_cast<uint8_t>(board[i]);
    }
    return hand_index_last(&tables().at(street), cards);
}

void HandIndexer::unindex(int street, uint64_t idx, std::array<int, 2>& hole,
                          std::vector<int>& board) const {
    validate_street(street);
    const hand_indexer_t& indexer = tables().at(street);
    if (idx >= indexer.round_size[indexer.rounds - 1]) {
        throw std::invalid_argument("hand index out of range");
    }

    uint8_t cards[7] = {0};
    if (!hand_unindex(&indexer, indexer.rounds - 1, idx, cards)) {
        throw std::invalid_argument("hand index out of range");
    }
    hole[0] = cards[0];
    hole[1] = cards[1];
    board.resize(2 + street);
    for (int i = 0; i < 2 + street; ++i) {
        board[i] = cards[2 + i];
    }
}

uint32_t HandIndexer::multiplicity(int street, uint64_t idx) const {
    validate_street(street);
    const hand_indexer_t& indexer = tables().at(street);
    if (idx >= indexer.round_size[indexer.rounds - 1]) {
        throw std::invalid_argument("hand index out of range");
    }

    uint8_t cards[7] = {0};
    if (!hand_unindex(&indexer, indexer.rounds - 1, idx, cards)) {
        throw std::invalid_argument("hand index out of range");
    }
    return raw_multiplicity(indexer, indexer.rounds - 1, cards);
}

}  // namespace sixmax
