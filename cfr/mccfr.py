import random
from typing import List
from tqdm import tqdm

from cfr.abstract_state import AbstractState, deal_heads_up
from cfr.regret_table import RegretTable
from cfr.info_set import InfoSet, stack_bucket
from cfr.abstraction import _hand_to_bucket_cached, _board_to_bucket_cached


def _weighted_choice(actions: List[str], probs: list) -> str:
    """Sample one action proportional to probs (pure-Python list, no numpy overhead)."""
    r = random.random()
    w = 0.0
    for i in range(len(actions)):
        w += probs[i]
        if w > r:
            return actions[i]
    return actions[-1]  # floating-point fallback: probs may sum to 0.9999...


def _encode_infoset(state: AbstractState, player: int) -> InfoSet:
    street = state.street
    return InfoSet(
        player=player,
        hand_bucket=_hand_to_bucket_cached(state.hole_keys[player], state.board_key, street),
        street=street,
        board_bucket=_board_to_bucket_cached(state.board_key, street),
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

    strat_list = strategy.tolist()
    if acting == traversing_player:
        action_values = {}
        for action in legal:
            action_values[action] = external_sample(
                state.apply_action(action), traversing_player, table
            )
        node_value = sum(strat_list[i] * action_values[legal[i]] for i in range(len(legal)))
        table.update_regrets(infoset, action_values, node_value, legal)
        return node_value
    else:
        # Sample one opponent action
        action = _weighted_choice(legal, strat_list)
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
        strategy = table.get_average_strategy(infoset, legal).tolist()
        child_values = [best_response(state.apply_action(a), br_player, table) for a in legal]
        return sum(strategy[i] * child_values[i] for i in range(len(legal)))


def compute_exploitability(
    table: RegretTable, num_samples: int = 10_000, show_progress: bool = False
) -> float:
    """
    Estimate exploitability in milli-big-blinds per hand (mbb/h).
    exploitability = average of best-response values for each player / 2 * 1000
    """
    br_values = []
    for br_player in [0, 1]:
        samples = (
            best_response(deal_heads_up(), br_player, table)
            for _ in range(num_samples)
        )
        if show_progress:
            samples = tqdm(
                samples,
                total=num_samples,
                desc=f"BR player {br_player}",
                unit="sample",
                leave=False,
            )
        br_values.append(sum(samples) / num_samples)
    return (br_values[0] + br_values[1]) / 2.0 * 1000.0
