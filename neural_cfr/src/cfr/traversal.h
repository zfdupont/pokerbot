#pragma once
#include "game/abstract_state.h"
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"
#include <random>

float external_sample(
    const AbstractState& state,
    int traversing_player,
    MLP& adv_net,           // advantage net for traversing player
    MLP& opp_adv_net,       // advantage net for opponent (fixed for whole traversal)
    MLP& strat_net,         // strategy net (trained on strat_buf offline; not used in traversal)
    ReservoirBuffer<BufferEntry>& adv_buffer,   // M_v[traversing_player]
    ReservoirBuffer<BufferEntry>& strat_buffer, // M_π
    int iteration,           // t — used as linear CFR weight
    std::mt19937& rng        // caller-owned RNG — thread-safe, no static state
);
