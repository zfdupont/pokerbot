import random
from dataclasses import dataclass
from typing import List, Tuple

from models.card import Card
from models.enums import Suit
from cfr.abstraction import legal_abstract_actions
from util.util import hand_value, hand_value_from_key
from util.index52c7 import index52c7, cards_to_bitmask

_DECK = [Card(r, s) for r in range(2, 15) for s in list(Suit)]

BIG_BLIND = 1.0
SMALL_BLIND = 0.5
STARTING_STACK = 100.0  # in BB


@dataclass
class AbstractState:
    hole_cards: Tuple[Tuple[Card, ...], ...]   # (p0_cards, p1_cards)
    hole_keys: Tuple[Tuple, ...]               # precomputed sorted c.key tuples per player
    board: Tuple[Card, ...]
    board_key: Tuple                           # precomputed tuple(sorted(c.key for c in board))
    deck: Tuple[Card, ...]                     # remaining for runout
    street: int                                # 0=preflop .. 3=river
    pot: float
    stacks: Tuple[float, ...]                  # [p0, p1] in BB
    current_bet: float
    player_bets: Tuple[float, ...]             # [p0, p1] this street
    to_act: Tuple[int, ...]                    # action queue
    betting_history: Tuple[int, int, int, int]  # raise counts per street, capped at 2
    folded: Tuple[bool, ...]                   # [p0_folded, p1_folded]
    showdown_keys: Tuple                       # per-player precomputed binhand_keys at river; () otherwise

    def is_terminal(self) -> bool:
        if self.folded[0] or self.folded[1]:
            return True
        if len(self.to_act) == 0 and self.street >= 3:
            return True
        return False

    def acting_player(self) -> int:
        return self.to_act[0]

    def payoff(self, player: int) -> float:
        """Return net chip gain/loss in BB for player at a terminal state."""
        invest = STARTING_STACK - self.stacks[player]
        if self.folded[0]:
            return (self.pot - invest) if player == 1 else -invest
        if self.folded[1]:
            return (self.pot - invest) if player == 0 else -invest
        # Showdown: use precomputed keys when available (skip sorted() on cache hits)
        if self.showdown_keys:
            ranks = [
                (i, hand_value_from_key(self.showdown_keys[i], self.hole_cards[i], self.board))
                for i in range(2)
            ]
        else:
            ranks = [(i, hand_value(self.hole_cards[i], self.board)) for i in range(2)]
        best_rank = min(r for _, r in ranks)
        winners = [i for i, r in ranks if r == best_rank]
        split = self.pot / len(winners)
        return (split - invest) if player in winners else -invest

    def legal_actions(self) -> List[str]:
        p = self.acting_player()
        to_call = self.current_bet - self.player_bets[p]
        return legal_abstract_actions(
            to_call=to_call,
            pot=self.pot,
            stack=self.stacks[p],
            current_bet=self.current_bet,
            player_bet=self.player_bets[p],
        )

    def apply_action(self, action: str) -> "AbstractState":
        p = self.acting_player()
        pot = self.pot
        current_bet = self.current_bet
        if action in ("b0.5", "b1.0", "allin"):
            counts = list(self.betting_history)
            counts[self.street] = min(counts[self.street] + 1, 2)
            history = tuple(counts)
        else:
            history = self.betting_history

        stacks = list(self.stacks)
        player_bets = list(self.player_bets)
        folded = list(self.folded)

        if action == "fold":
            folded[p] = True
            to_act = ()

        elif action == "check":
            to_act = self.to_act[1:]

        elif action == "call":
            to_call = min(current_bet - player_bets[p], stacks[p])
            stacks[p] -= to_call
            player_bets[p] += to_call
            pot += to_call
            to_act = self.to_act[1:]

        elif action in ("b0.5", "b1.0", "allin"):
            to_call = current_bet - player_bets[p]
            if action == "allin":
                additional = stacks[p]
            else:
                size = float(action[1:])
                effective_pot = pot + to_call * 2
                additional = min(to_call + size * effective_pot, stacks[p])
            stacks[p] -= additional
            player_bets[p] += additional
            pot += additional
            current_bet = max(current_bet, player_bets[p])
            num = len(self.hole_cards)
            to_act = tuple(
                i for i in range(num)
                if not folded[i] and stacks[i] > 0 and i != p
            )
        else:
            raise ValueError(f"Unknown action: {action}")

        return AbstractState(
            hole_cards=self.hole_cards,
            hole_keys=self.hole_keys,
            board=self.board,
            board_key=self.board_key,
            deck=self.deck,
            street=self.street,
            pot=pot,
            stacks=stacks,
            current_bet=current_bet,
            player_bets=player_bets,
            to_act=to_act,
            betting_history=history,
            folded=folded,
            showdown_keys=self.showdown_keys,
        )

    def advance_street(self) -> "AbstractState":
        """Deal community cards and start next betting round."""
        assert len(self.to_act) == 0 and not all(self.folded)
        deck = list(self.deck)
        board = list(self.board)
        next_street = self.street + 1

        if next_street == 1:
            board += [deck.pop() for _ in range(3)]
        else:
            board.append(deck.pop())

        new_board_key = tuple(sorted(c.key for c in board))

        # At river, precompute the index52c7 integer key for each player's 7-card hand.
        # payoff can then do a direct dict lookup without recomputing cards_to_bitmask.
        if next_street == 3:
            board_mask = cards_to_bitmask(board)
            new_showdown_keys = tuple(
                index52c7(cards_to_bitmask(self.hole_cards[i]) | board_mask)
                for i in range(len(self.hole_cards))
            )
        else:
            new_showdown_keys = ()

        # Postflop: player 1 (BB/OOP) acts first in HU, then player 0
        num = len(self.hole_cards)
        to_act = tuple(
            i for i in ([1, 0] if num == 2 else list(range(num)))
            if not self.folded[i] and self.stacks[i] > 0
        )

        return AbstractState(
            hole_cards=self.hole_cards,
            hole_keys=self.hole_keys,
            board=tuple(board),
            board_key=new_board_key,
            deck=tuple(deck),
            street=next_street,
            pot=self.pot,
            stacks=self.stacks,
            current_bet=0.0,
            player_bets=(0.0,) * num,
            to_act=to_act,
            betting_history=self.betting_history,
            folded=self.folded,
            showdown_keys=new_showdown_keys,
        )


def deal_heads_up(
    starting_stack: float = STARTING_STACK,
    big_blind: float = BIG_BLIND,
) -> AbstractState:
    """Create initial heads-up state with blinds posted."""
    deck = list(_DECK)
    random.shuffle(deck)
    p0_cards = tuple(deck[:2])
    p1_cards = tuple(deck[2:4])
    remaining = tuple(deck[4:9])  # pre-deal runout (5 community cards)

    sb = min(SMALL_BLIND, starting_stack)
    bb = min(big_blind, starting_stack)

    return AbstractState(
        hole_cards=(p0_cards, p1_cards),
        hole_keys=(
            tuple(sorted(c.key for c in p0_cards)),
            tuple(sorted(c.key for c in p1_cards)),
        ),
        board=(),
        board_key=(),
        deck=remaining,
        street=0,
        pot=sb + bb,
        stacks=(starting_stack - sb, starting_stack - bb),
        current_bet=bb,
        player_bets=(sb, bb),
        to_act=(0, 1),
        betting_history=(0, 0, 0, 0),
        folded=(False, False),
        showdown_keys=(),
    )
