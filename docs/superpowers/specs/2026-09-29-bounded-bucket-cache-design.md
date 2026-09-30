# Bounded `bucket()` Cache — Design

**Date:** 2026-09-29
**Status:** Approved (bounded path) — implemented; correctness-verified, cap to be sized by measurement
**Path:** bounded (single private cache in `sixmax/src/abstraction/`; no interface/artifact change)
**Author:** brainstormed with Claude

## Problem

`Abstraction::bucket(hole, board)` memoizes its result in `bucket_cache_` — 64 shards,
each an **unbounded** `unordered_map<uint64_t,int>` (`sixmax/src/abstraction/abstraction.h`).
The memoization was added 2026-07-21 (`9693e6c`) for a ~3× early speedup.

Observed cost on the 2026-09-29 cloud run (`--type cpx42 --iters 25000000`, one chunk,
`checkpoint_interval=0`):

- Container RSS grew **5.6 → 9.3 GiB (~0.8 GiB/h)** over ~9 h; host is 15.6 GiB with
  **no swap** and no container memory limit → real OOM-kill risk (all work unsaved).
- Average throughput fell below **363 iters/s** vs the 518/s benchmark (a 100k-iter
  small-table figure), and kept declining.

Mechanism: the cache accumulates every distinct `(hole, board)` ever seen. Its multi-GB
working set dwarfs L3, so each lookup misses to DRAM — the "3× win" inverts into a memory
and throughput cliff on long runs.

**Hypothesis:** most reuse is *within* a hand (the same `(hole, board)` is queried across
different betting lines), so cross-iteration reuse is low. The unbounded global cache is
therefore largely pure overhead, while a small cache would retain the within-hand benefit.
(Confirmed/refuted during implementation by the size readout + RSS slope in Verification.)

## Goal

Bound the cache's memory and remove the throughput cliff **without changing any output** —
so existing checkpoints stay valid and no retrain is required — and do so behind a seam
that lets the eviction policy be swapped and unit-tested later.

## Non-goals

- Changing the bucketing function, edge construction, or key *layout* (that is the separate
  abstraction work; it invalidates checkpoints and can fail silently — tracked elsewhere).
- A canonical-key redesign (suit-isomorphism) — possible future refinement, deferred.
- Changing the checkpoint format or `AbstractionConfig`.

## Key invariant (why this is free)

`bucket()` is a deterministic pure function of `(hole, board, edges, seed)`; the cache is a
transparent memoizer, and `abstraction.h` states it is **"NOT hashed, serialised, copied, or
moved with the artifact."** Eviction merely forces a recompute that returns the identical
bucket. Therefore any *correct* bound is output-identical ⇒ `Abstraction::hash()` unchanged
⇒ checkpoints unaffected.

**Caveat:** only the *eviction* changes, never the *key*. The cache key deliberately folds
the street in (`fnv_mix(equity_seed(...), s)`); dropping it could, on a 64-bit collision
across boards of different sizes, consult the wrong edge table. The key stays as-is.

## Decisions (pinned)

### D1 — Cap form: per-instance ctor parameter (not in the artifact)
`Abstraction` takes a `size_t cache_cap_per_shard` (default `kDefaultBucketCacheCap`), stored
as a private member — **not** an `AbstractionConfig` field, **not** in `hash()`, **not**
serialized. Rationale: the cap must be tunable to be *sized by measurement*, per-instance
avoids ambient global state, and keeping it out of `AbstractionConfig` avoids the trap that
`hash()`/`save_blueprint` mix fixed field lists (a config field would be silently excluded
and the loaded strategy would quietly get the default). Exposed to Python as an optional
kwarg for the sweep and tests.

### D2 — Eviction policy: clear-shard on full
On insert, if a shard's map exceeds the cap, `clear()` that shard. Rationale: zero per-hit
cost (important — the cache is hit frequently by 8 threads under the shard lock), minimal
code, and the useful working set is small/short-lived so a refined policy buys little.
LRU is explicitly rejected (its move-to-front costs land on the hot path under the lock).
If clear-storms appear in measurement, upgrade to **clock/second-chance** (near-LRU quality,
no per-hit splice) — not LRU.

### D3 — Cache abstraction (policy seam)
Extract the cache into its own unit with the eviction policy behind a swappable interface,
so policies and the bound can be swapped and tested in isolation later:

```
// sixmax/src/abstraction/bucket_cache.h
namespace sixmax {

// How a shard is shrunk when it exceeds the per-shard cap. Called under the
// shard's mutex. New policies (clear / clock / lru) implement this.
class EvictionPolicy {
 public:
  virtual ~EvictionPolicy() = default;
  virtual void on_insert(std::unordered_map<uint64_t, int>& shard) = 0;
  // Optional per-hit hook for future recency-based policies (default no-op).
  virtual void on_hit(uint64_t /*key*/) {}
};

class ClearOnFullPolicy final : public EvictionPolicy {
 public:
  void on_insert(std::unordered_map<uint64_t, int>& shard) override { shard.clear(); }
};

// Sharded, size-bounded, thread-safe memo. Pure accelerator: never affects
// outputs, so its parameters are deliberately absent from any artifact.
class BucketCache {
 public:
  BucketCache(size_t cap_per_shard, std::unique_ptr<EvictionPolicy> policy);
  bool get(uint64_t key, int& out) const;   // may invoke policy->on_hit
  void put(uint64_t key, int value);        // may invoke policy->on_insert
  size_t size() const;                      // diagnostics / tests
 private:
  static constexpr int kShards = 64;
  struct Shard { mutable std::mutex mu; std::unordered_map<uint64_t, int> map; };
  std::array<Shard, kShards> shards_;
  size_t cap_;
  std::unique_ptr<EvictionPolicy> policy_;
};
}  // namespace sixmax
```

- `Abstraction` owns `std::unique_ptr<BucketCache> cache_` (a pointer so `Abstraction`'s
  move ctor simply transfers it; a value member would be non-movable because of the mutexes
  and would need bespoke re-init). `bucket()` stays `const` (pointee is non-const).
- The `EvictionPolicy` seam is virtual but is only invoked on insert/evict (rare on the
  hot path); the plain `unordered_map` lookup itself stays non-virtual.
- **Extension point acknowledged:** a future recency policy (clock/LRU) needs per-entry
  bookkeeping; that would widen `EvictionPolicy` (the `on_hit` hook and/or a policy-owned
  side structure). The seam is designed to absorb that without touching `Abstraction`.

## Files touched

- `sixmax/src/abstraction/bucket_cache.h` (new) — `BucketCache` + `EvictionPolicy` +
  `ClearOnFullPolicy`.
- `sixmax/src/abstraction/abstraction.h` — replace `bucket_cache_` with
  `std::unique_ptr<BucketCache>`; add cap param + default constant; debug size accessor.
- `sixmax/src/abstraction/abstraction.cpp` — construct the cache; route `bucket()` through
  it; fix the move ctor to transfer the pointer.
- `sixmax/src/bindings/bindings.cpp` — optional `cache_cap` kwarg on `Abstraction`; debug
  `bucket_cache_size()`.
- `tests/sixmax/test_abstraction.py` — determinism + bound tests.

No changes to `hash()`, `AbstractionConfig` layout, `save_blueprint`/`load_blueprint`, or
the checkpoint format.

## Testing

- **Determinism:** `bucket()` with a tiny cap (forces eviction) == with a large cap == a
  cache-free reference, over a batch of random `(hole, board)` on flop/turn/river, single-
  and multi-threaded (reuse `tests/sixmax/test_abstraction.py` fixtures).
- **Bound:** after many distinct lookups, `bucket_cache_size()` ≤ `cap × 64`.
- **Eviction seam:** instantiate `BucketCache` with a stub `EvictionPolicy` and assert it is
  invoked on overflow and that `get`/`put` round-trip (C++ or a thin binding-level test).
- **Checkpoint compatibility:** load an existing `sixmax/checkpoints/*.bin`; assert
  `hash()` is unchanged and buckets match a pre-change reference — the guard for the
  no-retrain claim.
- `~/bin/buck2 build //sixmax:sixmax` and `uv run pytest tests/sixmax -q`.

## Verification / measurement

- Add the size readout; reproduce a short run and show RSS **plateaus** instead of climbing.
- Sweep caps (1k / 4k / 16k / 64k per shard ⇒ 64k–4M entries ≈ 3–200 MB) and pick the knee
  where iters/s stops improving; confirm throughput is flat over time rather than decaying.
- If the cap is wrong, revisit with the sweep before merging.

## Risks / open questions

1. **Cap size** — too small ⇒ recompute cost; too large ⇒ the cliff returns. Sized by the
   sweep (D1). Start `kDefaultBucketCacheCap = 4096`.
2. **Eviction policy** — clear-shard chosen (D2); clock/second-chance is the documented
   upgrade if clear-storms appear.
3. **Confirm the hypothesis** — the size readout + short-run RSS slope confirm the cache is
   the (dominant) source of the climb before/while fixing.

## Deferred follow-ups (not this change)

- **Clock/second-chance** eviction policy behind the D3 seam.
- **`thread_local` cache** — since reuse is largely within a traversal/thread, a per-thread
  small cache could remove both the memory growth and the shard-lock cost. Bigger change;
  the D3 seam should make it tractable.
- **Suit-canonical key** — canonicalize `(hole, board)` under the 4! suit relabelings
  (equity is suit-symmetric) to shrink the reachable key set ~24× while staying
  output-identical.
