# Base Game Implementation — Design Spec

**Date:** 2026-06-16
**Scope:** Single-hand correctness — fix engine bugs, wire observer, swap in fast evaluator, add side pot support.
**Out of scope:** Multi-hand loop, tournament elimination, ML agents (future).

---

## Goals

1. Fix three correctness bugs in the betting engine (chip accounting, round termination, all-in stack underflow)
2. Add proper side pot calculation and distribution
3. Wire the `GameObserver` event system into `PokerGame` with a multi-observer list
4. Swap in the fast bitwise hand evaluator (`util/util.py`) to replace the pure-Python one

---

## Module Changes

| File | Type | Change |
|---|---|---|
| `game/pot_manager.py` | New | Side pot calculation and pot distribution |
| `game/poker.py` | Modify | Fix betting round, chip math, all-in, observer calls |
| `models/player.py` | Modify | Add `is_all_in: bool` and `total_contributed: int` |
| `models/card.py` | Modify | Add `suit_index: int` property for fast evaluator |
| `util/util.py` | Modify | Fix imports, adapt to our `Card`/`Suit` types |
| `util/evaluator.py` | Modify | Wire fast evaluator, return int ranks, update `rank_to_hand_name` |
| `util/observer.py` | Modify | `GameObserver` → no-op base class; current logger → `ConsoleObserver` |

`models/state.py`, `models/enums.py`, `models/hand.py`, `agents/` — untouched.

---

## Section 1: Observer Event System

### Base class

`GameObserver` in `util/observer.py` becomes a no-op base class. Subclass and override only what you need:

```python
class GameObserver:
    def on_hand_start(self, players: List[Player], button_pos: int) -> None: pass
    def on_hole_cards_dealt(self, player: Player, cards: List[Card]) -> None: pass
    def on_street_start(self, street_name: str, community_cards: List[Card]) -> None: pass
    def on_player_action(self, player: Player, action: Action, amount: Optional[int]) -> None: pass
    def on_hand_complete(self, winners: List[Tuple[Player, int]], pot: int) -> None: pass
```

### ConsoleObserver

The existing print-based logger moves into `ConsoleObserver(GameObserver)` in the same file, overriding each event method.

### PokerGame wiring

```python
class PokerGame:
    def __init__(self, players, small_blind=1, observers: Optional[List[GameObserver]] = None):
        self.observers = observers or []
        ...

    def _emit(self, event, *args, **kwargs):
        for obs in self.observers:
            getattr(obs, event)(*args, **kwargs)
```

`_emit` is the single dispatch point — `poker.py` calls `self._emit("on_player_action", player, action, amount)` etc.

### ML-readiness

`on_player_action` fires for every action in the hand, not just the current agent's turn. A future training recorder subscribes here and collects full game trajectories without modifying the engine.

---

## Section 2: Fast Evaluator

### Fixes to `util/util.py`

1. **Import**: `from card import Hand, Card` → `from models.card import Card` (drop `Hand` — unused as type)
2. **Suit indexing**: `card_to_binary` uses `card.suit - 1` expecting an integer. Add a `suit_index: int` property to `models/card.py`:

```python
_SUIT_INDEX = {Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3, Suit.SPADES: 4}

@property
def suit_index(self) -> int:
    return _SUIT_INDEX[self.suit]
```

Replace all `card.suit` references in `card_to_binary` with `card.suit_index`.

3. **Signature change**: `hand_value(hole: Hand, community)` → `hand_value(hole_cards: List[Card], community: List[Card]) -> int`

Return value: integer rank where **lower = better** (standard Cactus Kev convention).

### HandEvaluator changes (`util/evaluator.py`)

```python
@staticmethod
def evaluate_hands(players, community_cards) -> List[Tuple[Player, int]]:
    results = [
        (p, hand_value(p.hole_cards, community_cards))
        for p in players if p.is_active
    ]
    return sorted(results, key=lambda x: x[1])  # ascending = best first
```

Return type changes from `List[Tuple[Player, Hand]]` to `List[Tuple[Player, int]]`.

```python
@staticmethod
def rank_to_hand_name(rank: int) -> str:
    # Cactus Kev rank ranges (lower = better):
    # 1          → Royal Flush
    # 2–10       → Straight Flush
    # 11–166     → Four of a Kind
    # 167–322    → Full House
    # 323–1599   → Flush
    # 1600–1609  → Straight
    # 1610–2467  → Three of a Kind
    # 2468–3325  → Two Pair
    # 3326–6185  → Pair
    # 6186–7462  → High Card
```

`get_hand_description` is replaced by `rank_to_hand_name(rank: int) -> str`.

---

## Section 3: PotManager and Side Pots

### Data structure

```python
@dataclass
class SidePot:
    amount: int
    eligible: List[Player]
```

### PotManager interface

```python
class PotManager:
    def contribute(self, player: Player, amount: int) -> None
    def calculate_side_pots(self) -> List[SidePot]
    def award(self, player_ranks: List[Tuple[Player, int]]) -> None
    @property
    def total(self) -> int
```

### Side pot algorithm

At showdown, `calculate_side_pots()`:

1. Collect each player's `total_contributed` (tracked by `contribute()`)
2. Sort all-in contribution levels ascending
3. For each level, carve out: `min(each_player.total_contributed, level) * num_contributors` — only players who contributed at least `level` are eligible
4. Remainder above the last all-in level → main pot, all active players eligible

Example: A all-in for 50, B and C each put in 200.
- Side pot 1: `50 × 3 = 150`, eligible: A, B, C
- Side pot 2: `150 × 2 = 300`, eligible: B, C

### Instantiation

`PotManager` is created fresh at the start of each `play_hand()` call — not in `PokerGame.__init__`. This ensures per-hand state (contributions, pots) is always clean. `calculate_side_pots()` is called internally by `award()` — callers only call `award()`.

### Award logic

`award(player_ranks)` calls `calculate_side_pots()` then loops pots from first to last:
- Filter `player_ranks` to eligible players for this pot
- Find lowest rank value (= best hand)
- Split pot among all tied players; remainder chips go to earliest position

### Player fields added (`models/player.py`)

```python
self.is_all_in: bool = False
self.total_contributed: int = 0
```

`total_contributed` is incremented by `PotManager.contribute()` — `Player` itself doesn't manage it.

---

## Section 4: Betting Round Fixes

### Bug 1 — BET/RAISE chip accounting

Current code deducts `amount` and adds `amount` to pot, ignoring already-committed chips:

```python
# Fix: only move the delta
additional = amount - player.current_bet
player.stack -= additional
player.current_bet = amount
self.state.current_bet = amount
pot_manager.contribute(player, additional)
```

### Bug 2 — All-in handling

When a player's stack can't cover a call/bet:
- Cap the deduction at `player.stack`
- Set `player.is_all_in = True`
- Player stays `is_active = True` (still eligible for their pot level)
- `_betting_round` skips players where `is_all_in` is true

### Bug 3 — Betting round termination

Replace the current `_round_complete` (which only checks equal bets) with `last_aggressor_idx` tracking:

- On street start, `last_aggressor_idx` = first active player left of the button (SB post-flop, first left of BB pre-flop)
- On any BET or RAISE, `last_aggressor_idx` = that player's index
- Loop terminates when the current player index equals `last_aggressor_idx` AND all non-folded, non-all-in players have acted at least once this street
- Special case: if all remaining players are all-in, skip remaining betting and deal out streets

### Street start reset

At the beginning of each street (`_betting_round` call), reset:
- `player.current_bet = 0` for all players
- `self.state.current_player_idx` to first active player left of button

---

## Data Flow

```
main.py
  └── PokerGame(players, small_blind, observers=[ConsoleObserver()])
        ├── play_hand()
        │     ├── _emit("on_hand_start", ...)
        │     ├── deal_hole_cards() → _emit("on_hole_cards_dealt", ...)
        │     ├── _post_blinds() → pot_manager.contribute(...)
        │     └── for each street:
        │           ├── _emit("on_street_start", ...)
        │           └── _betting_round()
        │                 └── _process_action() → pot_manager.contribute()
        │                                       → _emit("on_player_action", ...)
        └── _determine_winners()
              ├── HandEvaluator.evaluate_hands() → hand_value() [fast evaluator]
              └── pot_manager.award(player_ranks)
                    └── _emit("on_hand_complete", ...)
```

---

## Notes

- `models/hand.py` (pure-Python evaluator) stays in the codebase but is no longer in the hot path. Keep as reference/fallback.
- `util/lookup_table.py` is untouched — it's precomputed data used by `util/util.py`.
- No tests exist yet; this spec doesn't add them, but `PotManager` is deliberately isolated so unit tests can be added independently.
