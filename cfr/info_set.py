from dataclasses import dataclass


@dataclass(frozen=True)
class InfoSet:
    player: int                       # 0 or 1; extensible to 0..N-1
    hand_bucket: int                  # equity bucket for this player's hole cards
    street: int                       # 0=preflop, 1=flop, 2=turn, 3=river
    board_bucket: int                 # board texture bucket (0 preflop)
    betting_history: tuple            # abstract action codes, "/" separates streets
    stack_bucket: int                 # discretized stack depth (0-4)


def stack_bucket(stack_in_bb: float, big_blind: float = 1.0) -> int:
    """Map stack size to one of 5 tiers: <10BB=0, 10-25=1, 25-50=2, 50-100=3, 100+=4."""
    bb = stack_in_bb / big_blind
    if bb < 10:   return 0
    if bb < 25:   return 1
    if bb < 50:   return 2
    if bb < 100:  return 3
    return 4
