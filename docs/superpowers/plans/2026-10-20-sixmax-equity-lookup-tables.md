# Six-max Equity Lookup Tables + Bucket Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace noisy 100-rollout postflop bucketing with exact, precomputed equity over suit-canonical hands, served to the trainer and bot as a memory-mapped bucket map, then train and A/B one noise-free HU model.

**Architecture:** A suit-canonical `HandIndexer` gives every (hole, board) a dense per-street index. An offline build computes exact river equity per index and derives turn/flop equity by averaging; a cheap second step turns that feature store into a `uint8` bucket map (percentile or potential-aware k-means). `Abstraction` gains a map-backed kind behind the unchanged `bucket()` call; map-based checkpoints use a new `SIXBP002` format that records the map's content hash.

**Tech Stack:** C++17 in `//sixmax:core` (Buck2), pybind11 bindings in `sixmax/src/bindings/bindings.cpp`, Python 3.10 CLIs and pytest via `uv run`.

**Spec:** `docs/superpowers/specs/2026-10-20-sixmax-equity-lookup-tables-design.md`

## Global Constraints

- Card codes `0..51`, `rank = c / 4`, `suit = c % 4`; tests build cards as `(rank - 2) * 4 + suit`.
- Canonical sizes: flop **1,286,792**, turn **13,960,050**, river **123,156,254**.
- Buckets per street ≤ **255** (8-bit card field in `pack_abstract_key`, unchanged).
- Existing `SIXBP001` checkpoints load and bucket bit-identically; `Abstraction::hash()` for the legacy kind is unchanged.
- `sixmax/` imports nothing from `cfr/`, `neural_cfr/`, `game/poker.py`; hand strength only via `safe_eval` (`common/`), `12 - rank` inversion untouched.
- Tables live under gitignored `sixmax/tables/`; never commit tables or checkpoints.
- Rebuild: `~/bin/buck2 build //sixmax:sixmax`. Tests: `uv run pytest tests/sixmax -q` (conftest discovers the `.so`).
- Equity ties never split across buckets.
- Success: first map-based HU model beats `sixmax/checkpoints/hu_blueprint_20000000.bin` at **t ≥ 2** with `scripts/eval_blueprint_h2h.py` (8 seeds × 20k deals). A miss ends the project; record it, do not continue into other maps.
- Out of scope: bet-size key, six-max retrain, search, DREAM, droplet deployment.

## Review Focus

1. Malformed input to `index()` (duplicate cards, wrong board size, card ≥ 52): expect `std::invalid_argument` / Python `ValueError`, never a garbage index. Test in Task 1.
2. Loading a `SIXBP002` checkpoint whose map file is missing: expect an error naming the hint and every path searched. Test in Task 4.
3. A map whose header indexer version or street sizes do not match this build: expect refusal at open. Test in Task 3.
4. A truncated or corrupted map file: refuse at open (size and content-hash check), never read out of bounds. Test in Task 3.
5. Builder asked for 0 or > 255 buckets, or more buckets than distinct equity values allow: expect a clear error, not an empty bucket. Test in Task 3.

---

### Task 1: Suit-canonical hand indexer

**Files:**
- Create: `sixmax/src/abstraction/hand_index.h`, `sixmax/src/abstraction/hand_index.cpp`
- Modify: `sixmax/BUCK` (add `src/abstraction/hand_index.cpp` to `core` srcs), `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_hand_index.py`

**Interfaces:**
- Produces (C++, namespace `sixmax`):
  - `constexpr uint32_t kHandIndexVersion = 1;`
  - `class HandIndexer { public: static const HandIndexer& instance(); uint64_t size(int street) const; uint64_t index(int street, const std::array<int,2>& hole, const std::vector<int>& board) const; void unindex(int street, uint64_t idx, std::array<int,2>& hole, std::vector<int>& board) const; uint32_t multiplicity(int street, uint64_t idx) const; };` with `street` 1=flop 2=turn 3=river (matches `Abstraction::num_buckets`). `multiplicity` = number of raw (hole, board) deals the class stands for (unordered hole, unordered board).
- Produces (Python): `sixmax.hand_index_size(street) -> int`, `sixmax.hand_index(street, hole, board) -> int`, `sixmax.hand_unindex(street, idx) -> (list[int], list[int])`, `sixmax.hand_multiplicity(street, idx) -> int`, `sixmax.HAND_INDEX_VERSION`.

Implementation: Waugh 2013 ("A Fast and Optimal Hand Isomorphism Algorithm") with rounds {2, 3} / {2, 3, 1} / {2, 3, 1, 1}. Write it, or vendor `kdub0/hand-isomorphism` only after confirming its license permits it (record the license in the file header). `instance()` builds tables once, thread-safe.

- [ ] **Step 1: Write the failing tests**

```python
def test_sizes():
    assert [sixmax.hand_index_size(s) for s in (1, 2, 3)] == [1_286_792, 13_960_050, 123_156_254]

def test_suit_relabel_and_order_invariant():   # As Kd | 7s 7h 2c  vs  suits permuted + reordered
    a = sixmax.hand_index(1, [card(14,3), card(13,1)], [card(7,3), card(7,2), card(2,0)])
    b = sixmax.hand_index(1, [card(13,0), card(14,2)], [card(2,1), card(7,0), card(7,2)])
    assert a == b

def test_roundtrip_random_samples():            # 20k random deals per street
    # unindex(index(x)) re-indexes to the same idx; idx < size
def test_dense_on_flop():                       # every flop idx in range(size) unindexes, re-indexes to itself
def test_multiplicity_sums_to_raw_deals():      # flop only (river too slow for a unit test)
    assert sum(sixmax.hand_multiplicity(1, i) for i in range(1_286_792)) == 1326 * 19600
def test_rejects_bad_input():                   # duplicate card, board of 2 on street 1, card 52 -> ValueError
```

- [ ] **Step 2: Run** `uv run pytest tests/sixmax/test_hand_index.py -q`. Expected: FAIL, `module 'sixmax' has no attribute 'hand_index_size'`.
- [ ] **Step 3: Implement** `HandIndexer` and bindings per the Interfaces block; register the `.cpp` in `sixmax/BUCK`.
- [ ] **Step 4: Rebuild and run** `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_hand_index.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** `git add sixmax/src/abstraction/hand_index.* sixmax/BUCK sixmax/src/bindings/bindings.cpp tests/sixmax/test_hand_index.py && git commit -m "feat(sixmax): suit-canonical hand indexer"`

---

### Task 2: Exact equity feature store

**Files:**
- Create: `sixmax/src/abstraction/equity_table.h`, `sixmax/src/abstraction/equity_table.cpp`, `scripts/build_equity_tables.py`
- Modify: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`, `.gitignore` (add `sixmax/tables/`)
- Test: `tests/sixmax/test_equity_table.py`

**Interfaces:**
- Consumes: `HandIndexer` (Task 1).
- Produces (C++):
  - `double river_equity_exact(const std::array<int,2>& hole, const std::vector<int>& board);` (all non-colliding opponent combos via `safe_eval::rank7`, ties count 0.5)
  - `std::vector<float> build_river_equity(int num_threads);` (indexed by river canonical index)
  - `std::vector<float> derive_equity(int street, const std::vector<float>& next_street, int num_threads);` (street 2 from river, street 1 from turn: mean over every next card not in hole/board of `next_street[index(street+1, hole, board+card)]`)
  - `struct EquityStore { std::array<std::vector<float>,3> eq; uint64_t content_hash; };` with `void save_equity_store(const std::string& path, const EquityStore&)` and `EquityStore load_equity_store(const std::string& path)`; header: magic `SIXEQ001`, `kHandIndexVersion`, the three sizes, content hash; load refuses version/size/hash mismatch.
- Produces (Python): `sixmax.river_equity_exact(hole, board) -> float`, `sixmax.build_equity_store(path, num_threads=0) -> int` (returns content hash; prints per-street progress), `sixmax.equity_store_lookup(path, street, idx) -> float` (test helper; opens read-only).
- `scripts/build_equity_tables.py --out sixmax/tables/equity_v1.bin --threads 0`.

The river build is ~1.2e11 `rank7` calls if done by direct enumeration; per-board sort-and-count with card-removal correction is the faster option (builder's call). Target: hours on 8 cores. Save atomically (`.tmp` + rename, as `save_blueprint` does).

- [ ] **Step 1: Write the failing tests**

```python
def test_river_exact_matches_brute_force():       # 50 random river hands vs a pure-Python loop over all
    # non-colliding opponent combos using sixmax.rank7(...).beats/.ties -> exactly equal (abs < 1e-12)
def test_river_exact_known_values():               # royal flush in hand -> 1.0 ; board royal flush (board plays) -> 0.5
def test_river_exact_close_to_monte_carlo():      # 100 random hands: |exact - hand_equity(h,b,20000,salt)| < 0.02
def test_derived_turn_equals_mean_of_rivers():    # derive on a single index via a helper binding: mean of river_equity_exact over 46 rivers
def test_turn_close_to_monte_carlo():             # 50 random turn hands, tolerance 0.02
def test_store_roundtrip_and_refusals(tmp_path):  # tiny store via a test-only binding writing given arrays; corrupt one byte -> load raises; wrong version -> raises
```

Expose `sixmax.derive_equity_one(street, hole, board) -> float` (exact, computed from `river_equity_exact` recursively) so tests never need the full river table, and `sixmax._write_equity_store_for_test(path, flop, turn, river)` accepting arbitrary-length arrays with a flag that skips the size check.

- [ ] **Step 2: Run** `uv run pytest tests/sixmax/test_equity_table.py -q`. Expected: FAIL, missing attributes.
- [ ] **Step 3: Implement** the Interfaces block and the CLI; add `sixmax/tables/` to `.gitignore`.
- [ ] **Step 4: Rebuild and run** the test file. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "feat(sixmax): exact equity feature store"` (equity_table.*, BUCK, bindings, CLI, test, .gitignore).

---

### Task 3: Bucket maps (builders, file format, mmap reader)

**Files:**
- Create: `sixmax/src/abstraction/bucket_map.h`, `sixmax/src/abstraction/bucket_map.cpp`, `scripts/build_bucket_map.py`
- Modify: `sixmax/BUCK`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_bucket_map.py`

**Interfaces:**
- Consumes: `HandIndexer` (Task 1), `EquityStore` (Task 2).
- Produces (C++):
  - `std::vector<uint8_t> percentile_buckets(const std::vector<float>& eq, const std::vector<uint32_t>& weight, int k);` Equal multiplicity-weighted population as far as ties allow; identical equity → identical bucket; throws `std::invalid_argument` if `k < 1`, `k > 255`, or fewer than `k` distinct values.
  - `std::vector<uint8_t> kmeans_buckets(const std::vector<std::vector<float>>& hists, const std::vector<uint32_t>& weight, int k, uint64_t seed, int iters);` Distance = 1-D EMD between histograms over ordered outcomes (cumulative-sum L1), which is exact for ordered bins; deterministic given `seed`; k-means++ init.
  - `void write_bucket_map(const std::string& path, const std::array<std::vector<uint8_t>,3>& b, const std::string& method, const std::string& params, uint64_t equity_hash);` Header: magic `SIXBM001`, `kHandIndexVersion`, per-street size and bucket count, method, params, equity hash, content hash over the arrays.
  - `class BucketMap { public: static std::shared_ptr<const BucketMap> open(const std::string& path); uint8_t bucket(int street, uint64_t idx) const; int num_buckets(int street) const; uint64_t content_hash() const; const std::string& path() const; };` Read-only `mmap`; `open` refuses wrong magic, version, sizes, file length, or content hash.
  - `void build_bucket_map(const std::string& equity_path, const std::string& out, const std::string& method, std::array<int,3> counts, uint64_t seed, int num_threads);` Methods: `percentile` (each street on its equity), `potential` (river percentile on equity; turn = k-means on the sorted 46 river equities as a histogram over fixed bins; flop = k-means on the distribution over turn buckets from the 47 turn cards, bins ordered by bucket mean equity).
- Produces (Python): `sixmax.build_bucket_map(equity_path, out, method, counts, seed=0, num_threads=0)`, `sixmax.BucketMap.open(path)` with `.bucket(street, idx)`, `.num_buckets(street)`, `.content_hash()`, `sixmax._percentile_buckets(eq, weight, k)`, `sixmax._kmeans_buckets(hists, weight, k, seed, iters)`, `sixmax._write_bucket_map_for_test(path, flop, turn, river, method)` (numpy `uint8` arrays, full sizes required).
- `scripts/build_bucket_map.py --equity sixmax/tables/equity_v1.bin --method percentile --counts 50,50,20 --out sixmax/tables/map_pct_50_50_20.bin`

- [ ] **Step 1: Write the failing tests**

```python
def test_percentile_ties_never_split():           # eq=[.1,.5,.5,.5,.9]*, k=3 -> the three .5 share one bucket
def test_percentile_weighted_equal_population():  # 10k distinct random eqs, random weights, k=50 -> max/min weighted pop < 1.1
def test_percentile_monotone():                   # higher equity never gets a lower bucket
def test_percentile_rejects_bad_k():              # k=0, k=256, k > distinct values -> ValueError
def test_kmeans_deterministic_and_separates():    # two well-separated histogram blobs, k=2 -> clean split; same seed -> same labels
def test_map_roundtrip(tmp_path):                 # write idx % k arrays (numpy, full sizes); open; spot-check buckets and num_buckets
def test_map_refuses_truncated(tmp_path):         # chop last byte -> open raises
def test_map_refuses_corrupt(tmp_path):           # flip one data byte -> open raises (content hash)
def test_map_refuses_wrong_version(tmp_path):     # patch header version field -> open raises
```

- [ ] **Step 2: Run** `uv run pytest tests/sixmax/test_bucket_map.py -q`. Expected: FAIL, missing attributes.
- [ ] **Step 3: Implement** the Interfaces block and the CLI.
- [ ] **Step 4: Rebuild and run** the test file. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "feat(sixmax): bucket maps with percentile and potential-aware builders"`

---

### Task 4: Map-backed `Abstraction` + `SIXBP002` checkpoints

**Files:**
- Modify: `sixmax/src/abstraction/abstraction.{h,cpp}`, `sixmax/src/blueprint/checkpoint.{h,cpp}`, `sixmax/src/bindings/bindings.cpp`, `.mex/AGENTS.md` (Non-Negotiables), `CLAUDE.md` (Hard Invariants)
- Create: `tests/fixtures/legacy_buckets.json`
- Test: `tests/sixmax/test_abstraction_map.py`, `tests/sixmax/test_legacy_buckets_golden.py`

**Interfaces:**
- Consumes: `BucketMap`, `HandIndexer`.
- Produces (C++):
  - `explicit Abstraction(std::shared_ptr<const BucketMap> map);` `bool is_map() const;` `const BucketMap* bucket_map() const;` `bucket()` for the map kind = `map->bucket(street, HandIndexer::instance().index(street, hole, board))` with `street = board.size() - 2`; `num_buckets()` from the map; `hash()` for the map kind mixes a kind tag and `map->content_hash()`. Legacy kind untouched.
  - `save_blueprint` writes magic `SIXBP002` when `abs.is_map()`: same layout as v1 up to `starting_stack`, then `int32 kind=1`, `uint64 map_content_hash`, length-prefixed `map_hint` (the map's basename), then the table as v1.
  - `LoadedBlueprint` gains `int abs_kind; uint64_t map_hash; std::string map_hint;`. `load_blueprint` reads both magics.
  - `std::string resolve_bucket_map(const std::string& ckpt_path, const std::string& hint);` search order: `$SIXMAX_BUCKET_MAP` (full path) if set, `dirname(ckpt_path)/hint`, `$SIXMAX_TABLES_DIR/hint` (default `sixmax/tables/hint`, relative to cwd). Throws `std::runtime_error` listing every path tried.
  - `BlueprintStrategy::load` and the `load_abstraction` binding: for `SIXBP002`, resolve, `BucketMap::open`, and throw if `content_hash() != map_hash`.
- Produces (Python): `sixmax.Abstraction.from_bucket_map(path)`, `.is_map()`.

- [ ] **Step 1: Capture the golden fixture BEFORE touching `abstraction.cpp`.** With the current build, write `tests/fixtures/legacy_buckets.json`: for `sixmax/checkpoints/blueprint.bin` and `hu_blueprint.bin` (via `sixmax.load_abstraction`) record `hash()` and `bucket()` for 300 fixed seeded (hole, board) deals across streets. Commit the generator as `tests/fixtures/make_legacy_buckets.py`.
- [ ] **Step 2: Write the failing tests**

```python
# test_legacy_buckets_golden.py (skips if the checkpoints are absent)
def test_legacy_hash_and_buckets_unchanged():     # every recorded hash and bucket matches

# test_abstraction_map.py  (map = idx % k synthetic, written once per module to tmp_path_factory)
def test_map_abstraction_bucket_equals_map_lookup():
def test_map_abstraction_hash_differs_by_map():
def test_sixbp002_roundtrip(tmp_path):             # train 120 iters HU on the map, save, load via BlueprintStrategy -> same probs
def test_sixbp002_refuses_other_map(tmp_path):     # SIXMAX_BUCKET_MAP points at a different map -> load raises "map hash mismatch"
def test_sixbp002_missing_map_lists_paths(tmp_path):  # move map away -> error mentions hint and each searched path
def test_sixbp001_still_loads():                   # existing small v1 checkpoint from test_blueprint_checkpoint flow loads
```

- [ ] **Step 3: Run** `uv run pytest tests/sixmax/test_abstraction_map.py tests/sixmax/test_legacy_buckets_golden.py -q`. Expected: map tests FAIL (missing `from_bucket_map`); golden test PASS.
- [ ] **Step 4: Implement** the Interfaces block. Add to `.mex/AGENTS.md` Non-Negotiables and `CLAUDE.md` Hard Invariants: "Map-based checkpoints (`SIXBP002`) are bound to their bucket map by content hash; the hand indexer (`kHandIndexVersion`) is part of that contract; never change either without a version bump."
- [ ] **Step 5: Rebuild and run** `uv run pytest tests/sixmax -q`. Expected: all pass, including every pre-existing sixmax test.
- [ ] **Step 6: Commit** `git commit -m "feat(sixmax): map-backed abstraction and SIXBP002 checkpoints"`

---

### Task 5: Trainer config, bot wiring, throughput and memory checks

**Files:**
- Modify: `scripts/train_sixmax.py` (`BUILTIN_DEFAULTS` gains `"bucket_map": ""`; when set, build `sixmax.Abstraction.from_bucket_map(path)` instead of the Monte Carlo abstraction and log the map hash), `sixmax/configs/hu.toml` (commented example only, no behavior change)
- Create: `sixmax/configs/hu_map.toml` (copy of `hu.toml` with `[abstraction] bucket_map = "sixmax/tables/map_pct_50_50_20.bin"`, `iterations = 30000000`, `checkpoint_interval = 5000000`, `checkpoint = "sixmax/checkpoints/hu_map_pct.bin"`, snapshots on)
- Test: `tests/sixmax/test_train_sixmax.py` (extend), `tests/sixmax/test_map_agent_smoke.py`

**Interfaces:**
- Consumes: `Abstraction.from_bucket_map`, `SIXBP002` (Task 4).
- Produces: `train_sixmax.py --config <toml with bucket_map>` writes `SIXBP002`; `--resume` on it works; `SixmaxAgent(path)` plays it with no extra arguments.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_config_accepts_bucket_map(tmp_path):   # TOML [abstraction] bucket_map -> cfg["bucket_map"]
def test_train_with_map_writes_sixbp002(tmp_path):      # 200 iters, num_players=2, synthetic map -> file starts with b"SIXBP002"; resume 100 more works
def test_sixmax_agent_plays_map_checkpoint(tmp_path):   # run_duplicate_match(SixmaxAgent(ckpt) vs itself, 20 deals) == 0.0
```

- [ ] **Step 2: Run** the new tests. Expected: FAIL (`Unknown config key [abstraction].bucket_map`).
- [ ] **Step 3: Implement** the config key and branch in `train_sixmax.py`; create `hu_map.toml`.
- [ ] **Step 4: Run** `uv run pytest tests/ -q`. Expected: full suite passes.
- [ ] **Step 5: Throughput check.** Once a real map exists (Task 6 step 2), run 200k HU iterations twice on this Mac, legacy `hu.toml` vs `hu_map.toml`, same `--num_threads`. Record iters/sec of both in the results doc. Expected: map ≥ legacy. If not, stop and report.
- [ ] **Step 6: Memory check.** Serve the map checkpoint locally (`POKERBOT_CHECKPOINT=sixmax/checkpoints/hu_map_pct.bin uv run python -m web.main --port 8100`) and play 200 hands through the API; record peak RSS of the server process and the map's resident pages (`vmmap <pid>` mapped-file line). Expected: map resident ≤ 140 MB and total RSS well under 2 GB.
- [ ] **Step 7: Commit** `git commit -m "feat(sixmax): train and serve map-based abstractions"`

---

### Task 6: Build the tables, train, A/B, record (operational)

**Files:**
- Modify: `.mex/ROUTER.md`, `docs/superpowers/specs/2026-10-20-sixmax-equity-lookup-tables-design.md` (append a Results section)

- [ ] **Step 1: Build the feature store.** Detached (`nohup caffeinate -is ...`): `uv run python scripts/build_equity_tables.py --out sixmax/tables/equity_v1.bin`. Expected: finishes in hours; prints the content hash. Verify with a 1,000-hand random sample that stored river equity equals `river_equity_exact` exactly.
- [ ] **Step 2: Build the map.** `uv run python scripts/build_bucket_map.py --equity sixmax/tables/equity_v1.bin --method percentile --counts 50,50,20 --out sixmax/tables/map_pct_50_50_20.bin`. Report per-street weighted population max/min ratio. Then run Task 5 steps 5 and 6.
- [ ] **Step 3: Train HU to plateau.** On Hetzner ccx33 via `scripts/cloud_train.sh` (upload the map to the box's `sixmax/tables/`) or locally if the user prefers; `--config sixmax/configs/hu_map.toml`. Snapshots every 5M.
- [ ] **Step 4: Find the plateau.** `eval_blueprint_h2h.py --a <snapshot N> --b <snapshot N-10M>`; plateau = first N where t < 2. Train further (in 5M steps) if 30M has not plateaued.
- [ ] **Step 5: Decision A/B.** `uv run python scripts/eval_blueprint_h2h.py --a sixmax/checkpoints/hu_map_pct_<plateau>.bin --b sixmax/checkpoints/hu_blueprint_20000000.bin`. Pass = `A - B > 0` and `t ≥ 2`.
- [ ] **Step 6: Record.** Results section in the spec (table hashes, build time, throughput, memory, plateau curve, A/B), ROUTER "Current Project State" entry stating pass or miss. On a miss, the project ends here.
- [ ] **Step 7: Commit** `git commit -m "docs(sixmax): equity table build + map-based HU A/B results"`
