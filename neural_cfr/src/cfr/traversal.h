#pragma once
#include "game/abstract_state.h"
#include "net/mlp.h"
#include "cfr/reservoir_buffer.h"

float external_sample(
    const AbstractState& state,
    int traversing_player,
    MLP& adv_net,           // advantage net for traversing player
    MLP& strat_net,         // strategy net (used for opponent sampling)
    ReservoirBuffer<BufferEntry>& adv_buffer,   // M_v[traversing_player]
    ReservoirBuffer<BufferEntry>& strat_buffer, // M_π
    int iteration            // t — used as linear CFR weight
);
