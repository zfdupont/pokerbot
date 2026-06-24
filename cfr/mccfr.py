import random
from typing import Dict, List
import numpy as np

from cfr.abstract_state import AbstractState, deal_heads_up
from cfr.regret_table import RegretTable
from cfr.info_set import InfoSet, stack_bucket
from cfr.abstraction import hand_to_bucket, board_to_bucket


def _encode_infoset(state: AbstractState, player: int) -> InfoSet:
    board = list(state.board)
    street = state.street
    return InfoSet(
        player=player,
        hand_bucket=hand_to_bucket(list(state.hole_cards[player]), board, street),
        street=street,
        board_bucket=board_to_bucket(board, street),
        betting_history=state.betting_history,
        stack_bucket=stack_bucket(state.stacks[player]),
    )


def external_sample(
    state: AbstractState,
    traversing_player: int,
    table: RegretTable,
) -> float:
    """
    External Sampling MCCFR.
    Returns EV in BB for traversing_player.
    Traverses all of traversing_player's actions; samples one from opponent.
    """
    if state.is_terminal():
        return state.payoff(traversing_player)

    # Chance node: advance to next street
    if len(state.to_act) == 0:
        return external_sample(state.advance_street(), traversing_player, table)

    acting = state.acting_player()
    legal = state.legal_actions()
    infoset = _encode_infoset(state, acting)
    strategy = table.get_strategy(infoset, legal)

    if acting == traversing_player:
        action_values = {}
        for action in legal:
            action_values[action] = external_sample(
                state.apply_action(action), traversing_player, table
            )
        node_value = float(np.dot(strategy, [action_values[a] for a in legal]))
        table.update_regrets(infoset, action_values, node_value, legal)
        return node_value
    else:
        # Sample one opponent action
        action = np.random.choice(legal, p=strategy)
        table.accumulate_strategy(infoset, strategy, legal)
        return external_sample(state.apply_action(action), traversing_player, table)


def best_response(
    state: AbstractState,
    br_player: int,
    table: RegretTable,
) -> float:
    """
    Compute best-response value for br_player against opponent's average strategy.
    Used to measure exploitability.
    """
    if state.is_terminal():
        return state.payoff(br_player)

    if len(state.to_act) == 0:
        return best_response(state.advance_street(), br_player, table)

    acting = state.acting_player()
    legal = state.legal_actions()

    if acting == br_player:
        values = [
            best_response(state.apply_action(a), br_player, table)
            for a in legal
        ]
        return max(values)
    else:
        infoset = _encode_infoset(state, acting)
        strategy = table.get_average_strategy(infoset, legal)
        return float(np.dot(strategy, [
            best_response(state.apply_action(a), br_player, table)
            for a in legal
        ]))


def compute_exploitability(table: RegretTable, num_samples: int = 10_000) -> float:
    """
    Estimate exploitability in milli-big-blinds per hand (mbb/h).
    exploitability = average of best-response values for each player / 2 * 1000
    """
    br_values = []
    for br_player in [0, 1]:
        total = sum(
            best_response(deal_heads_up(), br_player, table)
            for _ in range(num_samples)
        )
        br_values.append(total / num_samples)
    return (br_values[0] + br_values[1]) / 2.0 * 1000.0
