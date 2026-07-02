# Neural CFR (Deep CFR) — Design Spec
Date: 2026-07-02

## Overview

Replace the tabular `RegretTable` with two MLPs implementing Deep CFR (Brown et al. 2019).
The C++ core (traversal, buffers, networks) is built with libtorch and exposed to Python via
pybind11. Existing eval and play scripts remain usable with a single constructor swap.

This is a learning exercise: `neural_cfr/` is a self-contained module; the existing `cfr/`
package is unchanged and serves as the reference implementation.

---

## Goals

- Eliminate card/hand abstraction ceiling — network sees raw cards, not equity buckets
- Keep existing eval infrastructure (`eval_openspiel.py`, `play.py`, `openpoker_bot.py`) working
- Performance-first: traversal, buffers, and network training all in C++
- Faithful to Brown et al. 2019 for learning value

---

## Algorithm: Deep CFR (Brown et al. 2019)

External Sampling MCCFR with two networks per player replacing the regret table.

For each iteration `t = 1 … T`, for each player `p ∈ {0, 1}`:

1. **Traversal** — run external sampling with player `p` as traversing player:
   - At `p`'s infosets: query `AdvNet_p` → regret-match → strategy σ → traverse ALL actions
     → compute advantages → store `(features, advantages, t)` in `M_v[p]`
     → store `(features, σ, t)` in `M_π`
   - At opponent's infosets: query `StratNet` → sample ONE action

2. **Training**:
   - Train `AdvNet_p` on `M_v[p]` with weighted MSE (weight = iteration `t`)
   - Train `StratNet` on `M_π` with weighted cross-entropy (weight = iteration `t`)

Linear CFR weighting (samples weighted by `t`) ensures later, more accurate samples
dominate over noisy early ones.

---

## Repository Layout

```
neural_cfr/
  BUCK
  src/
    game/
      abstract_state.h/.cpp   # C++ port of cfr/abstract_state.py
      card.h                  # card representation + 52-card deck
    net/
      features.h/.cpp         # InfoSet → 134-dim feature tensor
      advantage_net.h/.cpp    # MLP: 134 → 256 → 256 → 256 → 6 (ReLU, no output activation)
      strategy_net.h/.cpp     # MLP: 134 → 256 → 256 → 256 → 6 (Softmax output)
    cfr/
      traversal.h/.cpp        # external sampling MCCFR querying networks
      reservoir_buffer.h      # fixed-size reservoir sampling buffer
      trainer.h/.cpp          # alternating traversal → buffer → train loop
    bindings/
      bindings.cpp            # pybind11 module definition
  tests/
    BUCK
    test_features.cpp
    test_reservoir_buffer.cpp
    test_abstract_state.cpp
    test_traversal.cpp
  third_party/
    BUCK                      # prebuilt_cxx_library() rules for libtorch + pybind11 + gtest
scripts/
  train_neural.py             # ~30-line Python launcher
tests/
  neural_cfr/
    test_trainer.py
    test_strategy_compat.py
    test_eval_compat.py
```

---

## Feature Encoding (`net/features.h`)

134-dimensional float tensor. No equity bucketing — raw card identity fed directly to the network.

| Component | Encoding | Dims |
|---|---|---|
| Hole cards (2) | 13-dim rank one-hot + 4-dim suit one-hot per card | 34 |
| Board cards (5) | same, zero-padded for missing cards | 85 |
| Street | 4-dim one-hot (preflop/flop/turn/river) | 4 |
| Pot size | scalar, normalized by starting stack | 1 |
| Stack size | scalar, normalized by starting stack | 1 |
| Betting history | raise counts per street (4 streets × 2 players) | 8 |
| Position | scalar (0 or 1) | 1 |

Factored rank/suit encoding (17 dims per card) is more parameter-efficient than 52-dim one-hot
and introduces the correct inductive bias: same-rank cards across suits share rank features.

---

## Network Architecture

Both networks are identical MLPs differing only in output activation and loss:

```
Input: 134-dim feature vector
Hidden: 256 → 256 → 256  (ReLU activations)

AdvNet_p  output: 6 raw values (no activation)  — counterfactual advantage per action
StratNet  output: 6 values + Softmax            — action probability distribution
```

Action vocabulary (6 actions, consistent with existing abstraction):
`fold / check / call / b0.5 / b1.0 / allin`

Training:
- `AdvNet_p`: weighted MSE, weight = iteration `t`, sample from `M_v[p]`
- `StratNet`: weighted cross-entropy, weight = iteration `t`, sample from `M_π`
- Optimizer: Adam, lr = 1e-4
- Batch size: 4096

---

## Buffers

Three reservoir-sampled buffers, all fixed max size (configurable, default 2M entries each):

| Buffer | Populated by | Consumed by |
|---|---|---|
| `M_v[0]` | p=0 traversal | `AdvNet_0` training |
| `M_v[1]` | p=1 traversal | `AdvNet_1` training |
| `M_π` | both traversals | `StratNet` training |

Each entry: `(feature_tensor: float[134], targets: float[6], weight: float)`.

Reservoir sampling: when buffer is full, each new sample replaces an existing entry with
probability `buffer_max_size / n_total_seen`. Memory is bounded regardless of iteration count.

---

## pybind11 Interface (`bindings/bindings.cpp`)

Two classes exposed as `import neural_cfr`:

```python
neural_cfr.Trainer(reservoir_size=2_000_000, batch_size=4096, lr=1e-4)
  .run(iterations: int)            # blocks, logs progress
  .checkpoint(path: str)           # saves networks + buffers
  .load(path: str)                 # restores from checkpoint

neural_cfr.Strategy(checkpoint_path: str)
  .get_average_strategy(info_set) -> dict[str, float]   # drop-in for RegretTable
  .get_action_probs(info_state_str: str) -> dict[str, float]  # for eval_openspiel.py
```

`Strategy` is a drop-in for `RegretTable`: `eval_openspiel.py` and `play.py` require only a
single constructor change.

---

## Build System (Buck2 + libtorch)

libtorch downloaded as a prebuilt tarball from pytorch.org (CPU build for development,
CUDA build optional). pybind11 is header-only.

```python
# third_party/BUCK
prebuilt_cxx_library("libtorch")   # header_dirs + shared_libs pointing at extracted tarball
prebuilt_cxx_library("pybind11")   # header_dirs only, no link step
prebuilt_cxx_library("gtest")      # googletest prebuilt or source

# neural_cfr/BUCK
cxx_library("core")                # all src/ except bindings/, links libtorch
cxx_library("neural_cfr")          # bindings.cpp → neural_cfr.so, preferred_linkage=shared

# neural_cfr/tests/BUCK
cxx_test("test_*")                 # each test file, links //neural_cfr:core + gtest
```

---

## Error Handling

| Scenario | Location | Behavior |
|---|---|---|
| Checkpoint file missing / corrupt | `Trainer::load()` | `std::runtime_error` → Python `RuntimeError` |
| Reservoir buffer OOM | `ReservoirBuffer` constructor | Validate against available RAM, fail fast with size hint |
| Invalid infoset at inference | `Strategy::get_average_strategy()` | Return uniform distribution over legal actions (matches `RegretTable` behavior) |
| Tensor shape mismatch / libtorch error | Anywhere | `c10::Error` propagates through pybind11 as Python `RuntimeError` automatically |

---

## Testing

### C++ unit tests (gtest, link against `:core`)

- `test_features.cpp` — encode known hand (e.g., A♠K♠ preflop), assert shape `[134]`, spot-check rank/suit dims
- `test_reservoir_buffer.cpp` — insert 10× capacity, assert size stays at `max_size`, assert each slot replaced with frequency `~1/max_size` (chi-squared test at p=0.01)
- `test_abstract_state.cpp` — port key assertions from `tests/cfr/test_abstract_state.py`
- `test_traversal.cpp` — run 100 iterations, assert all three buffers non-empty

### Python integration tests (`tests/neural_cfr/`, pytest)

- `test_trainer.py` — run 500 iterations, assert buffers populated, assert checkpoint round-trips cleanly
- `test_strategy_compat.py` — load checkpoint, assert `get_average_strategy()` returns valid distribution summing to 1.0
- `test_eval_compat.py` — swap `RegretTable` for `neural_cfr.Strategy` in minimal OpenSpiel eval, assert no interface errors

### Convergence milestone (manual)

Run 50k iterations. Plot exploitability vs iteration. Expect clear downward trend — absolute
value matters less than direction at this stage.

---

## Relation to Brown et al. 2019

| Aspect | Paper | This implementation |
|---|---|---|
| Traversal | External Sampling MCCFR | Identical |
| Networks | Advantage + Strategy | Identical |
| Buffers | Reservoir sampling | Identical |
| Linear CFR weighting | Yes (weight = t) | Yes |
| Card encoding | 52-dim one-hot per card | 17-dim factored (rank + suit) — more efficient |
| Bet abstraction | Used in NLHE experiments | 6-action abstraction (same pragmatic choice) |
| Player count | N-player | HU only |

---

## Known Gaps / Future Work

- Bet abstraction remains: action space is still 6 discrete actions. True abstraction-free
  bet sizing would require continuous action CFR (different algorithm).
- Single-threaded traversal to start. Parallelism (multiple traversal threads writing to
  shared buffers with a mutex) is a natural follow-on once correctness is established.
- CPU-only initially. CUDA support is a one-flag change in libtorch once the build is stable.
