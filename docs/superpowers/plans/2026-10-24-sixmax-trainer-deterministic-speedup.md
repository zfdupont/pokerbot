# Sixmax Trainer Deterministic Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the sixmax blueprint trainer's allocation and lock overhead without changing a single trained value: a single-threaded run at a fixed seed produces the same table before and after.

**Architecture:** Four local changes in `sixmax/src/`, behind a digest harness built first on the unmodified code. Each later task must reproduce the baseline digests exactly. No key, hash, checkpoint format, RNG call or floating-point operation order changes.

**Tech Stack:** C++17 (`sixmax/src/{blueprint,engine}`), Buck2 (`~/bin/buck2 build //sixmax:sixmax`), pybind11 bindings, pytest via `uv run pytest`.

**Spec:** No separate spec. The evidence and constraints are in Background below. They come from the 2026-10-24 profile of D1 (`sixmax/configs/hu200_c5_d1.toml`), written to `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/results.md` in Task 5.

## Background (the profile this plan acts on)

Profile: macOS `sample`, 20 s, 8 threads, D1 config, 1.5M-iteration run at 5,176 it/s. Busy-time shares:

- malloc/free 38%.
  - `~EngineGameState` 19% of alloc samples.
  - `HandState::settle` + `showdown_order` + `settle_pots` 38%.
  - `HandState` copy-construct, clone, `EngineGameState` ctor 17%.
  - `traverse`'s per-node `std::vector<uint8_t> mask` 10%.
  - `abstract_key` -> `HandState::board()` 5%.
- Mutex waits 23%. 94.5% of these are in `BlueprintTrainer::traverse` (`sixmax/src/blueprint/trainer.cpp:47-96`), which takes a shard lock twice per node and redoes `sh.map[key]` each time.
- Infoset hash table 8.4%.
- Hand evaluator 1.3%. Out of scope.

## Global Constraints

- A single-threaded training run at a fixed seed must produce identical digests before and after every task (Task 1 defines the digest). Multithreaded runs are already order-nondeterministic and are not compared.
- Never change: infoset key bits, `Abstraction::hash()`, checkpoint byte formats (SIXBP001-004), vocab order, the RNG call sequence in `traverse`, or the order of floating-point additions in regret updates and `settle_pots`.
- `kShards` stays 64 and `shard_of` is unchanged. Checkpoint record order follows `export_table()`'s merge of the shards, so changing either would reorder checkpoint bytes.
- The engine supports 2..6 players (`EngineConfig::num_players`). Fixed-capacity buffers use `kMaxPlayers = 6`.
- Public C++ and Python APIs keep their signatures: `sixmax::settle_pots(vector...)`, `HandState::board()`, `HandState::payoffs()`, `EngineGameState::board()`, `GameState::legal_mask(std::vector<uint8_t>&)`.
- `cfr/`, `neural_cfr/` and `sixmax/` do not import each other. This plan touches only `sixmax/` plus one new script and test.
- Brief commit messages. No Co-Authored-By, Claude-Session or "Generated with" lines (user CLAUDE.md).
- Work on a branch from `main` (`perf/trainer-alloc`). `main` already contains the C5 bet-history code.

## Review Focus

- **6-max side pots with ties.** The allocation-free `settle_pots` must match the vector version on 3-6 players with unequal all-ins and split ranks. This is pinned by Task 3's randomized equivalence test plus the existing `tests/sixmax/test_settlement.py`.
- **Fewer than 2 players alive at showdown order.** A fold-win must still return all-zero ranks. Pinned in Task 3.
- **Naive-keyed states with no abstraction attached.** `infoset_key()` must still hash the full action history when `abstraction_ == nullptr`. Pinned in Task 4 (test that the naive key changes after an action).
- **`reroll_hidden` / `set_hole_cards` after the fixed-size deck change.** Search and LBR write into `deck_` slots; writes must land in the same places. Pinned by existing `test_range_tracker.py`, `test_leaf_eval.py` and `test_lbr.py`, which Task 4's verify step runs.
- **Resume.** `import_table` followed by training must still match. Covered because Task 1's digest scenario B trains in two `train()` calls with a save/load in between.

---

### Task 1: Digest harness and benchmark on the unmodified code

**Files:**
- Create: `scripts/trainer_perf_check.py`
- Create: `tests/sixmax/test_trainer_golden.py`

**Interfaces:**
- Produces: `trainer_perf_check.digest(scenario: str, tmp_dir: str) -> str` (hex sha256), `trainer_perf_check.SCENARIOS: dict[str, Callable]` with keys `"hu_bet_history"`, `"sixmax_legacy"`, `"hu_resume"`, and the CLI below. Later tasks run `uv run python scripts/trainer_perf_check.py digest` and compare its output with the baseline.

**Digest definition (order- and platform-layout-independent):**
1. Train, save the checkpoint, then call `sixmax.dump_infosets(path)`.
2. Sort the records by key.
3. sha256 over, for each record, `key`, then `float.hex` of every prob, `float.hex(mass)` and `float.hex(reg)`, joined with `,` and newline-terminated. Prefix the iteration count.

This is bit-exact on the values while ignoring `unordered_map` iteration order.

**Scenarios** (all `num_threads=1`):
- `hu_bet_history`:
  - HU, `EngineConfig(num_players=2, starting_stack=200.0)`, C5's vocab (`load_vocab("sixmax/configs/hu200_c5.toml", "blueprint")`).
  - The synthetic `index % k` bucket map (7/5/3), written with `sixmax._write_bucket_map_for_test` exactly as `tests/sixmax/conftest.py::_write_idx_mod_map` does.
  - `set_pot_layout("bb", [3, 6, 12, 24, 48, 96, 192])`, `set_bet_history("last_raise")`, seed 13, 3,000 iterations.
- `sixmax_legacy`: 6 players, default vocab, `Abstraction(flop_buckets=10, turn_buckets=10, river_buckets=5, equity_rollouts=40, quantile_samples=300, seed=42)` (the `test_blueprint_trainer.py` one), seed 7, 1,500 iterations. It exercises multiway side pots.
- `hu_resume`: as `hu_bet_history`, but 1,500 iterations, save, `load` and resume through the same path `scripts/train_sixmax.py --resume` uses, then 1,500 more.

- [ ] **Step 1: Write `scripts/trainer_perf_check.py`**

  CLI:
  - `digest [--scenario NAME]` prints `NAME sha256` lines for all scenarios, or the one named.
  - `bench --config PATH --iterations N [--threads T]` trains `N` iterations from scratch into a temp checkpoint via `scripts/train_sixmax.py`'s config loading and prints `it/s=<float>` from the trainer's final `[rate]` line.

  Reuse `tests/sixmax/conftest.py`'s module-loading approach (`_force_load` / `SIXMAX_SO_PATH`) so the script and test load the same `.so`.

- [ ] **Step 2: Record the baseline**

  Run: `uv run python scripts/trainer_perf_check.py digest` twice.
  Expected: identical output both times (three lines). Save the output as the baseline in Step 3.

- [ ] **Step 3: Write `tests/sixmax/test_trainer_golden.py`**

```python
import sys
import pytest

GOLDEN = {  # recorded on unmodified main, macOS arm64 (libc++), Task 1
    "hu_bet_history": "<sha256 from Step 2>",
    "sixmax_legacy": "<sha256 from Step 2>",
    "hu_resume": "<sha256 from Step 2>",
}

@pytest.mark.skipif(sys.platform != "darwin",
                    reason="std::uniform_real_distribution differs across C++ standard libraries")
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_trainer_digest_unchanged(name, tmp_path):
    assert digest(name, str(tmp_path)) == GOLDEN[name]
```

  Fill in the three values from Step 2. `digest` is imported from the script.

- [ ] **Step 4: Record the baseline speed**

  Run:
  - `uv run python scripts/trainer_perf_check.py bench --config sixmax/configs/hu200_c5_d1.toml --iterations 1500000`
  - The same with `--config sixmax/configs/hu200_c5.toml`.

  Note both `it/s` values in the commit message body. On a laptop, plug in and run each twice: a 2M probe earlier ran 5x slow for an unknown reason.

- [ ] **Step 5: Run the test and the suite**

  Run: `uv run pytest tests/sixmax/test_trainer_golden.py -v && uv run pytest tests/ -q`
  Expected: 3 passed; suite green.

- [ ] **Step 6: Commit**

```bash
git add scripts/trainer_perf_check.py tests/sixmax/test_trainer_golden.py
git commit -m "test(sixmax): golden trainer digests and speed bench"
```

---

### Task 2: One table lookup per node and a stack mask in `traverse`

**Files:**
- Modify: `sixmax/src/blueprint/trainer.cpp:47-96` (`BlueprintTrainer::traverse`)
- Modify: `sixmax/src/blueprint/mccfr.h`, `sixmax/src/blueprint/mccfr.cpp` (`regret_matched` signature; `MCCFRTrainer` call site)

**Interfaces:**
- Produces: `void regret_matched(const std::vector<double>& regret, const uint8_t* mask, int n, std::array<double, 16>& out)`. Replaces the `const std::vector<uint8_t>&` overload; both trainers pass `mask.data()` or a `std::array` `.data()`.

- [ ] **Step 1: Change `traverse`**

  - **One lookup per node.** Take the first lock, do `sh.map[key]` once and keep `InfosetData* d`. `std::unordered_map` never moves its elements on rehash, so the pointer stays valid. The traverser-node regret update takes the lock again but uses `d` instead of a second lookup.
  - **One lock on opponent nodes.** Regret matching and the `strategy_sum` update go in one critical section. The sampling that follows stays outside the lock and keeps its exact RNG calls.
  - **Mask on the stack.** Fill a `thread_local std::vector<uint8_t>` scratch via `s.legal_mask(scratch)`, then copy it into a local `std::array<uint8_t, kMaxActions> mask` before any recursion. The scratch is reused by the children, so the copy must come first.
  - **Arithmetic and order unchanged.** Every add happens in the same order with the same operands. In particular, `ev` accumulates over `a` ascending as now.

- [ ] **Step 2: Update `regret_matched` and `MCCFRTrainer`'s call** to the pointer signature. The body is unchanged apart from indexing.

- [ ] **Step 3: Build and verify the digests**

  Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_trainer_golden.py tests/sixmax/test_blueprint_trainer.py tests/sixmax/test_mccfr_kuhn.py -v`
  Expected: all pass. The golden digests must be unchanged; a mismatch means an operation order changed. Fix it rather than re-recording.

- [ ] **Step 4: Commit**

```bash
git commit -am "perf(sixmax): one table lookup per traverse node, stack mask"
```

---

### Task 3: Allocation-free settlement and showdown order

**Files:**
- Modify: `sixmax/src/engine/engine.h`, `sixmax/src/engine/engine.cpp:25-62` (`settle_pots`), `:228-270` (`showdown_order`, `settle`)
- Test: `tests/sixmax/test_engine.py` (add one test)

**Interfaces:**
- Produces:
  - `constexpr int kMaxPlayers = 6;` in `engine.h`.
  - `void settle_pots_into(const double* total_bets, const uint8_t* folded, const int* rank_order, int n, double* payout)` in `engine.h`. It writes `n` gross payouts; `payout` must hold `n` values and is zeroed by the function.
  - `std::vector<double> settle_pots(...)` becomes a thin wrapper over it. Same signature and results.
  - `HandState::showdown_order` becomes `void showdown_order(std::array<int, kMaxPlayers>& order) const` (private; only `settle` calls it).
  - `int HandState::board_into(std::array<int, 5>& out) const` (public) returns the revealed count. `board()` is reimplemented on top of it.

- [ ] **Step 1: Write the failing equivalence test** in `tests/sixmax/test_engine.py`

```python
def test_settle_pots_into_matches_reference_on_random_sidepots():
    # 20k random tables, 2-6 players: unequal all-ins, random folds (never all),
    # random dense ranks with ties. The reference is the pre-change algorithm,
    # transcribed in Python, so float results are compared exactly (==).
    ...
    assert sixmax.settle_pots(totals, folded, ranks) == reference(totals, folded, ranks)
```

  The Python reference copies the current C++ loop exactly:
  1. Take the distinct sorted levels with `abs(a - b) < 1e-9` treated as equal.
  2. At each level: amount = sum over i of `max(0, min(total_i, level) - prev)`.
  3. Winners are the best rank among live contributors with `total_i >= level - 1e-9`. If there are none, all live players win.
  4. `payout[w] += amount / len(winners)`, winners in ascending seat order.

  Running it before Step 2 must pass. It pins the current behaviour.

- [ ] **Step 2: Run it.**
  Run: `uv run pytest tests/sixmax/test_engine.py -v -k settle`
  Expected: PASS on the unmodified engine, which shows the reference is faithful.

- [ ] **Step 3: Implement `settle_pots_into`, the `showdown_order` array form, `board_into` and `settle`** with `std::array<..., kMaxPlayers>` and `std::array<double, kMaxPlayers>` buffers.
  - Sort and dedupe `levels` in a fixed array. Use the same `std::sort` + `std::unique` with the same predicate, over `levels.begin()` to `levels.begin() + count`.
  - Keep `winners` as a fixed array plus a count, filled in ascending seat order.
  - `settle` keeps `payoffs_.assign(n, 0.0)`. That vector is the public API and is written once, at terminal.
  - `showdown_order` keeps its fold-win early return (all zeros when fewer than 2 players are alive).

- [ ] **Step 4: Build and verify**

  Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_engine.py tests/sixmax/test_settlement.py tests/sixmax/test_engine_game.py tests/sixmax/test_trainer_golden.py -v`
  Expected: all pass, with the golden digests unchanged.

- [ ] **Step 5: Commit**

```bash
git commit -am "perf(sixmax): allocation-free settlement and showdown order"
```

---

### Task 4: Cheaper state clones and key building

**Files:**
- Modify: `sixmax/src/engine/engine.h:96-108`, `engine.cpp` (ctor at `:67`, `hole_cards`, `board_into`, `reroll_hidden`)
- Modify: `sixmax/src/blueprint/engine_game.h:76`, `engine_game.cpp:77-112` (`apply`), `:114-130` (`abstract_key`)
- Test: `tests/sixmax/test_engine_game.py` (add one test)

**Interfaces:**
- Consumes: `HandState::board_into` and `kMaxPlayers` (Task 3).
- Produces: no new public API. `HandState` members change type:
  - `deck_` becomes `std::array<int, 2 * kMaxPlayers + 5>`. The ctor copies the first `2n+5` cards of its `std::vector<int> deck` argument and keeps the existing `assert(deck.size() >= 2n+5)`.
  - `players_` becomes `std::array<PlayerState, kMaxPlayers>` plus `int n_`, and `num_players()` returns `n_`.
  - `start_stacks_` becomes `std::array<double, kMaxPlayers>`.
  - `payoffs_` stays a `std::vector<double>`. It is empty until terminal, so copying it costs nothing.

- [ ] **Step 1: Write the failing test** in `tests/sixmax/test_engine_game.py`

```python
def test_naive_key_still_tracks_history_without_abstraction():
    # Unattached EngineGameState (no abstraction): infoset_key() is the FNV
    # hash over hole cards, board and action history. Two states that differ
    # only in an earlier action at the same decision point must differ.
    ...
    assert s1.infoset_key() != s2.infoset_key()
```

  Build `s1` and `s2` from the same deal. Apply a call vs a check-back line that reaches the same next actor and street (for example preflop limp-check vs raise-call in HU, compared at the flop's first decision with the same board). Use the binding's `EngineGameState` constructor as the other `test_engine_game.py` tests do.

- [ ] **Step 2: Run it.** Expected: PASS on the current code. It pins naive-key behaviour before `history_` changes.

- [ ] **Step 3: Implement**
  - Convert the `HandState` members as listed in Interfaces. `reroll_hidden` and `set_hole_cards` index the same slots.
  - In `EngineGameState::apply`, push to `history_` only when `abstraction_ == nullptr`. `history_` is read only by the naive `infoset_key()` (`engine_game.cpp:73`), so attached states never need it.
  - In `abstract_key`, call `board_into` into a `std::array<int, 5>`, then `assign` into a `thread_local std::vector<int>` passed to `abs.bucket(hole, board)`. `Abstraction::bucket`'s signature does not change.

- [ ] **Step 4: Build and verify, including the search and LBR users of the deck slots**

  Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/ -q`
  Expected: all pass, including `test_trainer_golden.py`, `test_range_tracker.py`, `test_leaf_eval.py`, `test_lbr.py` and `test_stack_200.py`.

- [ ] **Step 5: Commit**

```bash
git commit -am "perf(sixmax): fixed-capacity hand state, skip naive history when keyed"
```

---

### Task 5: Measure, re-profile, record

**Files:**
- Modify: `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/results.md` (append a section)
- Modify: `.mex/ROUTER.md` and `.mex/context/architecture.md` (GROW: trainer is allocation-light; golden digest test exists)

- [ ] **Step 1: Bench after.**
  Run: `uv run python scripts/trainer_perf_check.py bench --config sixmax/configs/hu200_c5_d1.toml --iterations 1500000`, and the same for `hu200_c5.toml`, each twice and plugged in.
  Expected: it/s above Task 1's baseline. Record both numbers.

- [ ] **Step 2: Re-profile.**
  Run D1 training and `/usr/bin/sample <python pid> 20` (not the `sample` on PATH, which is a Python shim). Bucket by category as in Background.
  Expected: the malloc/free share is well below 38%.

- [ ] **Step 3: Full suite.**
  Run: `uv run pytest tests/ -q`
  Expected: all pass.

- [ ] **Step 4: Write the results section.** Include before/after it/s for both configs, before/after profile shares, and one line on what remains (lock contention on hot preflop shards, macOS mutexes without spinning). Update the `.mex` files.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/results.md .mex/
git commit -m "docs: trainer speedup results"
```

---

## Out of scope (decided)

- Lock-free or racy regret updates: these give up reproducibility, which the gates depend on.
- Changing `kShards` or `shard_of`: these reorder checkpoint bytes.
- Merging `InfosetData`'s two vectors into one allocation: this touches the checkpoint writer, `dump_infosets` and `MCCFRTrainer`. Revisit only if Task 5's profile shows the hash table and regret misses dominating.
- A Linux re-profile on ccx33: optional after Task 5, via `scripts/cloud_benchmark.sh`.
