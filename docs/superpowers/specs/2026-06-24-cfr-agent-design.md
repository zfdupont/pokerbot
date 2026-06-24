# CFR Poker Agent Design

**Goal:** Build a heads-up No-Limit Hold'em agent using Monte Carlo CFR (External Sampling) with hand/bet abstraction, targeting convergence toward Nash equilibrium. Designed for N-player extension without structural changes.

**Architecture:** Tabular MCCFR with NumPy-backed regret storage. Abstraction layer (equity buckets + discretized bet sizes) makes the game tree tractable on CPU. CFR agent bridges trained strategy to the existing simulation engine via the `PokerAgent` interface. Neural value estimation (Phase 2) replaces the tabular regret table without changing any other module.

**Tech Stack:** Python 3.10+, NumPy (regret tables), tqdm (progress bars for training loop and equity bucket precomputation), PyTorch MPS (Phase 2 neural), existing `models/` + `game/` engine unchanged.

---

## Scope

This spec covers **Phase 1: tabular MCCFR for heads-up cash game**. It explicitly excludes:
- Multi-way (6-max) play — interfaces are designed for it, training is not
- Tournament / ICM objective — deferred to a separate spec
- Neural value estimation — deferred to Phase 2 spec

---

## Module Structure

New `cfr/` package alongside existing `agents/`, `game/`, `models/`, `util/`:

```
cfr/
  abstraction.py    — equity bucketing, bet size discretization
  info_set.py       — InfoSet dataclass, state → abstract encoding
  regret_table.py   — NumPy regret + strategy storage, regret matching
  mccfr.py          — External Sampling MCCFR, best-response computation
  trainer.py        — training loop, checkpointing, exploitability logging

agents/
  cfr_agent.py      — loads strategy tables, translates concrete ↔ abstract
```

**Key boundary:** `cfr/` never imports from `game/poker.py`. It has its own lightweight abstract game state optimized for tree traversal (cloneable, deterministic, fast). The existing `PokerGame` is used only for simulation and evaluation, not training.

---

## Abstraction Layer (`cfr/abstraction.py`)

### Hand Bucketing

Hands are mapped to equity buckets at each street using Monte Carlo sampling (N=1000 rollouts per hand). Bucket counts:

| Street | Buckets | Method |
|--------|---------|--------|
| Preflop | 8 | Equity vs 1 random hand |
| Flop | 5 | Equity vs 1 random hand + random runout |
| Turn | 5 | Equity vs 1 random hand + random runout |
| River | 5 | Equity vs 1 random hand (no runout) |

Initial implementation uses **simple equity buckets** (mean equity only). Phase 1.5 upgrade: **EMD clustering** (Earth Mover's Distance over equity distributions), which separates draws from made hands at the same equity level. The interface (`hand_to_bucket(hole_cards, board, street) -> int`) does not change between implementations.

Bucket assignments are computed once at startup and cached to `cfr/data/buckets.pkl`.

N-player extension: equity computation changes from "vs 1 random hand" to "vs N-1 random hands". No other change needed.

### Bet Size Discretization

Allowed actions per decision point:

```python
BET_SIZES = {
    "fold":   None,
    "check":  0,
    "call":   None,       # amount computed from game state
    "b0.5":   0.5,        # 0.5x pot
    "b1.0":   1.0,        # 1.0x pot
    "allin":  None,       # player's remaining stack
}
```

Legal actions filtered per state (e.g., `check` unavailable if there's a bet to call). Configurable — adding `"b0.33"` or `"b2.0"` grows the tree but improves play quality.

---

## Information Set Encoding (`cfr/info_set.py`)

```python
@dataclass(frozen=True)
class InfoSet:
    player: int                      # 0 or 1; extensible to 0..N-1
    hand_bucket: int                 # equity bucket for this player's hole cards
    street: int                      # 0=preflop, 1=flop, 2=turn, 3=river
    board_bucket: int                # equity bucket of community cards (0 preflop)
    betting_history: tuple[str, ...] # e.g. ("b0.5", "c", "b1.0", "f") — full hand
    stack_bucket: int                # discretized stack depth (5 tiers)
```

**Stack buckets:** `<10BB=0, 10-25BB=1, 25-50BB=2, 50-100BB=3, 100BB+=4`. Tournament ICM agents need per-player stack awareness — the field is intentionally a single int now (own stack) but will become `tuple[int, ...]` for multi-player without changing the hash/lookup interface.

**`frozen=True`** makes `InfoSet` directly hashable as a dict key. Two info sets with identical fields are identical states regardless of path taken.

`betting_history` encodes all streets in sequence using `/` as a street separator: `("b0.5", "c", "/", "check", "b1.0", "c")`. This makes the full hand history unambiguous and avoids needing a per-street field.

---

## Regret Table (`cfr/regret_table.py`)

```python
class RegretTable:
    regrets:  dict[InfoSet, np.ndarray]   # shape (num_actions,) per infoset
    strategy: dict[InfoSet, np.ndarray]   # cumulative strategy sum

    def get_strategy(self, infoset: InfoSet, legal_actions: list[str]) -> np.ndarray:
        """Regret matching: proportional to positive cumulative regrets."""

    def update_regrets(self, infoset: InfoSet, action_values: dict[str, float], node_value: float) -> None:

    def get_average_strategy(self, infoset: InfoSet) -> np.ndarray:
        """Normalized cumulative strategy — what converges to Nash."""

    def save(self, path: str) -> None:   # pickle to disk
    def load(self, path: str) -> None:
```

**Regret matching rule:** `σ(I, a) = max(0, R(I, a)) / Σ max(0, R(I, b))`. If all regrets are ≤ 0, play uniformly over legal actions.

**Memory estimate:** ~10^6 info sets × 6 actions × 2 arrays (regret + strategy) × 8 bytes = ~96MB. Well within MacBook RAM.

---

## MCCFR Algorithm (`cfr/mccfr.py`)

**Variant:** External Sampling. Fully traverses the acting player's subtree; samples one action from the opponent's current strategy. Lower variance than Outcome Sampling; converges faster in practice.

**One iteration (heads-up):**

```
for player in [0, 1]:
    deal random hole cards + runout (pre-sampled for efficiency)
    traverse abstract game tree:
        our decision nodes  → visit ALL legal actions, accumulate regrets
        opponent nodes      → sample ONE action from their average strategy
        chance nodes        → already fixed by pre-sampled deal
        terminal nodes      → return chip payoff (normalized to BB)
```

**N-player extension:** iterate over all N players per iteration, sampling N-1 opponents each time. Algorithm structure unchanged.

**Best-response computation** (for exploitability):

```
for player in [0, 1]:
    compute best-response strategy against opponent's current average strategy
    br_value = expected payoff of best response
exploitability = (br_value[0] + br_value[1]) / 2
```

Exploitability in milli-big-blinds per hand (mbb/h). Target: below 250 mbb/h (human pro level) as a meaningful milestone.

---

## Trainer (`cfr/trainer.py`)

```python
class Trainer:
    def train(
        self,
        num_iterations: int,
        checkpoint_interval: int = 100_000,
        checkpoint_dir: str = "cfr/checkpoints/",
    ) -> None:
```

**Training loop:**
1. Run one MCCFR iteration (both players), advancing a `tqdm` progress bar
2. Every `checkpoint_interval` iterations:
   - Save `RegretTable` to disk (allows resuming)
   - Compute exploitability estimate over 10,000 sampled hands
   - Update tqdm postfix: iteration count, exploitability (mbb/h), wall time

**Convergence target:** 10^7 iterations takes ~3 hours with NumPy arrays. Exploitability should be below 500 mbb/h at that point with the 8/5/5/5 abstraction.

---

## CFR Agent (`agents/cfr_agent.py`)

Bridges trained strategy to the existing `PokerAgent` interface:

```python
class CFRAgent(PokerAgent):
    def __init__(self, strategy_path: str): ...

    def get_action(self, player, game_state) -> Tuple[Action, Optional[int]]:
        infoset = self._encode(player, game_state)
        probs = self.table.get_average_strategy(infoset)
        abstract_action = np.random.choice(legal_actions, p=probs)  # mixed strategy
        return self._translate(abstract_action, game_state)
```

**Encoding (`_encode`):** maps live betting history to abstract action codes by snapping each bet to the nearest discretized size. Hand and board bucketed via `abstraction.hand_to_bucket()`.

**Translation (`_translate`):** maps abstract action back to concrete `(Action, amount)`. `"b0.5"` → `(Action.BET, pot * 0.5)`.

**Out-of-abstraction handling:** opponent bet sizes not in the discrete set are mapped to nearest neighbor (e.g., 73% pot → 50% pot or 100% pot, whichever is closer). Known weakness — acceptable at this stage.

**Inference uses mixed strategy** (sampling from probability distribution), not argmax. This is required for Nash equilibrium properties — argmax is exploitable.

---

## Evaluation

| Method | Frequency | Metric |
|--------|-----------|--------|
| Exploitability (best-response) | Every 100K training iterations | mbb/h vs Nash |
| BB/100 vs `HandStrengthAgent` | After training | Practical strength |
| BB/100 vs `SimpleAgent` | After training | Sanity check |
| Self-play (CFR vs CFR) | After training | Should be ~0 net |

A correctly converging agent should beat `HandStrengthAgent` comfortably and show declining exploitability over training iterations.

---

## Phase 2 Transition (Neural CFR)

When switching from tabular to neural:

- `RegretTable` → replaced by a PyTorch value network (`cfr/value_net.py`)
- `InfoSet` encoding → becomes a float feature vector (input to the network)
- `mccfr.py` → adds batched tree traversal for GPU utilization
- Everything else (abstraction, trainer interface, CFR agent, evaluation) → unchanged

The `InfoSet` dataclass becomes a feature vector by design: all fields are integers, so `np.array([player, hand_bucket, street, board_bucket, *stack_bucket, *history_encoding])` is the natural neural input.

---

## What This Spec Does Not Cover

- Multi-way (6-max) MCCFR — separate spec when heads-up is validated
- Tournament ICM objective — separate spec
- Neural value network architecture — Phase 2 spec
- EMD clustering for postflop buckets — Phase 1.5 improvement, same interface
