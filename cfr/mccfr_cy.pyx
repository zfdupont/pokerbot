# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated external_sample. The cpdef recursion eliminates Python
argument-packing overhead on the ~320k recursive calls per 1000 iterations.
Everything else (RegretTable dict ops, AbstractState method calls, InfoSet
hashing) is Python-object work that Cython cannot reach without restructuring.
Expected gain: 1.1-1.2x at warm-cache steady state.
"""
import random

from cfr.info_set import InfoSet, stack_bucket
from cfr.abstraction import _hand_to_bucket_cached, _board_to_bucket_cached


cdef inline str _weighted_choice_cy(list actions, list probs):
    """Weighted sample from a small list — C loop, no numpy overhead."""
    cdef double r = random.random()
    cdef double w = 0.0
    cdef int i, n = len(actions)
    for i in range(n):
        w += <double>probs[i]
        if w > r:
            return actions[i]
    return actions[n - 1]


cdef inline object _encode_infoset_cy(state, int player):
    cdef int street = state.street
    return InfoSet(
        player=player,
        hand_bucket=_hand_to_bucket_cached(state.hole_keys[player], state.board_key, street),
        street=street,
        board_bucket=_board_to_bucket_cached(state.board_key, street),
        betting_history=state.betting_history,
        stack_bucket=stack_bucket(state.stacks[player]),
    )


cpdef double external_sample_cy(state, int traversing_player, table):
    """
    External Sampling MCCFR — cpdef so recursive calls use C calling convention.
    Returns EV in BB for traversing_player.
    """
    cdef double node_value
    cdef int i, n_legal
    cdef list strat_list

    if state.is_terminal():
        return state.payoff(traversing_player)

    if len(state.to_act) == 0:
        return external_sample_cy(state.advance_street(), traversing_player, table)

    cdef int acting = state.acting_player()
    legal = state.legal_actions()
    infoset = _encode_infoset_cy(state, acting)
    strategy = table.get_strategy(infoset, legal)
    strat_list = strategy.tolist()
    n_legal = len(legal)

    if acting == traversing_player:
        action_values = {}
        for i in range(n_legal):
            action_values[legal[i]] = external_sample_cy(
                state.apply_action(legal[i]), traversing_player, table
            )
        node_value = 0.0
        for i in range(n_legal):
            node_value += strat_list[i] * action_values[legal[i]]
        table.update_regrets(infoset, action_values, node_value, legal)
        return node_value
    else:
        action = _weighted_choice_cy(legal, strat_list)
        table.accumulate_strategy(infoset, strategy, legal)
        return external_sample_cy(state.apply_action(action), traversing_player, table)
