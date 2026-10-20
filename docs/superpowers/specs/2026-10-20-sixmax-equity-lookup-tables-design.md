# Six-max Equity Lookup Tables + Bucket Maps — Design

Date: 2026-10-20. Status: approved in brainstorming, awaiting plan.

## Why

The HU blueprint plateaued early: `hu_blueprint` 50M beats 10M (+5.4 BB/100,
t=2.86) but is flat against 20M/30M/40M (|t| ≤ 1.3) — 8 seeds × 20k duplicate
deals, `docs/superpowers/specs/2026-10-20-hu-h2h/`. It has only **33,182
infosets**. The model is converged on a coarse abstraction, so iterations no
longer help; resolution does.

The postflop card abstraction is mostly Monte Carlo noise. Buckets come from
`hand_equity(..., equity_rollouts=100)` (±5 pts equity SE) mapped through
percentile edges, while a flop bucket spans ~1.5 pts. Measured against a
2000-rollout reference (800 random hands/street):

| street | buckets | same bucket | within ±1 | mean abs bucket error |
|--------|---------|-------------|-----------|-----------------------|
| flop   | 50      | 16%         | 40%       | 2.6                   |
| turn   | 50      | 16%         | 46%       | 2.2                   |
| river  | 20      | 51%         | 94%       | 0.6                   |

Raising rollouts on the fly is not viable: `Abstraction::bucket()` is the
trainer's dominant cost even at 100 rollouts, and its memo cache (64 shards ×
4096, clear-on-full) cannot hold the billions of distinct turn/river
(hole, board) pairs, so rollout cost is paid on nearly every postflop visit.

The fix is to compute exact equity **once**, offline, over suit-canonical hands,
and turn bucketing into a table lookup. That removes the noise and makes
training cheaper than today.

## What the user wants it to enable

Trying **different bucket counts** and **potential-aware clustering** without
redoing the expensive work. So the design splits into an expensive, build-once
**feature store** and cheap, per-experiment **bucket maps**.

## Goals

1. **Feature store (build once).** Exact equity vs one uniform random opponent
   hand for every suit-canonical (hole, board) on flop, turn, and river.
2. **Bucket-map builder (per experiment, minutes).** Produces hand → bucket id
   per street from the feature store, with at least two methods:
   - `percentile` on equity (today's method, noise-free), any count ≤ 255;
   - `potential` k-means on next-street equity distributions (see below), any
     count ≤ 255.
3. **Map-based abstraction** in the trainer, the deploy bridge, and every other
   `Abstraction` consumer, selected by config.
4. **Artifact safety.** A checkpoint trained on a map records the map's content
   hash; loading against a different map is refused, exactly like the existing
   vocab-hash guard.
5. **First experiment:** retrain HU with a noise-free `percentile` 50/50/20 map
   (same infoset count, ~20M iterations), then `eval_blueprint_h2h.py` vs
   `sixmax/checkpoints/hu_blueprint_20000000.bin`.

## Success (user's criterion)

The first map-based HU model **beats `hu_blueprint_20000000.bin` head-to-head at
t ≥ 2** (`scripts/eval_blueprint_h2h.py`, 8 seeds × 20k deals, default bb=2,
stack=200).

**If it misses (user's decision):** the project stops there. A miss is a valid
result: record it (ROUTER + results doc) and do **not** continue into
`potential` maps, larger bucket counts, or longer runs under this project. The
tooling for those still ships, since it is part of the build.

## Parallel work (not part of this build)

Run the **leak 2 check** alongside the build: an offline self-play
measurement of how much the price faced (`to_call / pot`, pot size) varies
within a single infoset key for `hu_blueprint.bin`. The infoset key carries no
bet-size information, so a 0.33-pot bet and a 1.5-pot overbet can alias. The
result decides whether a bet-size key ("knob B") is the next project; it does
not change this design.

## Non-goals

- Adding bet-size / pot-odds information to the infoset key (separate project,
  "knob B").
- Six-max retraining or changing what production serves.
- Search changes (search is shelved, `enabled=false`).
- Preflop changes (preflop stays the lossless 169-class `preflop_class`).
- DREAM (`sixmax/src/dream/`, `scripts/train_dream.py`): it keeps the legacy
  Monte Carlo abstraction; no map selection, no DREAM checkpoint format change.
- Deploying to the droplet. The memory check runs locally (see Verification).

## Anti-goals (failure even if it works)

- Breaking or silently changing **any existing checkpoint**. `blueprint.bin`,
  `hu_blueprint*.bin`, `checkpoint.bin` (all magic `SIXBP001`, Monte Carlo
  percentile abstraction) must load and bucket bit-identically.
- A deployed bot whose bucket map costs more than a few hundred MB of resident
  memory (droplet: 1 vCPU, 2 GB RAM, 50 GB SSD).
- Training slower than today.

## Constraints and existing code

- Card codes are `0..51`, `rank = c / 4`, `suit = c % 4`
  (`abstraction.cpp:preflop_class`).
- The single bucketing choke point is `Abstraction::bucket(hole, board)`
  (`sixmax/src/abstraction/abstraction.{h,cpp}`). Callers:
  `EngineGameState::abstract_key` (trainer + LBR + policy), the `bucket`
  binding, and `agents/sixmax_agent.py` via `strategy.abstraction().bucket`.
- Checkpoint format (`sixmax/src/blueprint/checkpoint.cpp`, magic `SIXBP001`):
  serializes `AbstractionConfig` fields + per-street edges; `load_blueprint`
  rebuilds the `Abstraction` from them. `train_sixmax.py --resume` uses
  `sixmax.load_abstraction`.
- The infoset key's card field is 8 bits (`pack_abstract_key`), hence ≤ 255
  postflop buckets; unchanged by this project.
- Hard invariants (`.mex/AGENTS.md`): `sixmax/` must not import `cfr/`,
  `neural_cfr/`, or `game/poker.py`; shared C++ only via `common/`; use the
  opaque `safe_eval` API for hand strength (keep the `12 - rank` inversion
  untouched). Never commit checkpoints or large binaries.
- Python env: `.venv` is 3.10 (extension ABI); build with
  `~/bin/buck2 build //sixmax:sixmax`.

## Design

### 1. Suit-canonical indexer

A bijection between (hole, board) on a street and a dense index
`0 .. N_street - 1`, where hands equal up to suit relabeling (and order within
hole / within board) share an index. Expected sizes (Waugh 2013, "A Fast and
Optimal Hand Isomorphism Algorithm"): flop **1,286,792**, turn **13,960,050**,
river **123,156,254**. Needs both directions: `index(hole, board)` for lookups
and `unindex(i)` (one canonical representative) for building tables.

The bucket map and feature store are dense arrays addressed by this index, so
the indexer is part of the artifact contract: changing it invalidates every
table. Lives in `sixmax/src/abstraction/`.

### 2. Feature store: exact river equity, derived turn/flop equity

**Only the river needs real enumeration.** For river hand (hole, board),
equity = (wins + ties/2) / (opponent combos not colliding with the 7 known
cards), enumerating all 990 opponent combos with `safe_eval::rank7`.

**Turn and flop equity are exact averages of river equity.** Equity vs a
uniform random opponent with a uniform random runout equals the mean of the
next street's equity over all next cards, because (next card, opponent hole) is
uniform over the remaining deck either way:
`eq_turn(h, b) = mean over river cards r of eq_river(h, b + r)` (46 cards),
and likewise flop from turn (47 cards). This makes turn/flop exact and cheap
once the river table exists.

Stored as `float32` per canonical index: flop ~5 MB, turn ~56 MB, river
~493 MB. Feature files are **build artifacts, never deployed**.

Potential-aware features (next-street equity histograms) are **not stored**;
they are derived at bucketing time by enumerating next cards through the
indexer (flop: 47 turn lookups per hand; turn: 46 river lookups), which is
cheap. This keeps the store small and lets the histogram definition change per
experiment.

### 3. Bucket maps

A bucket map is, per street, a `uint8` array indexed by canonical index
(flop 1.3 MB, turn 14 MB, river 123 MB; ~140 MB total), plus a header.

Methods for v1:
- **`percentile`**: equal-population buckets by equity rank, computed from the
  exact distribution over all canonical hands **weighted by each class's
  multiplicity** (number of raw hands it stands for), so buckets are as equally
  populated in real deals as ties allow. **Hands with identical equity always
  share a bucket** (edges are equity thresholds, as today); buckets are never
  split by index. On the river, large exact ties make some buckets uneven;
  that is accepted.
- **`potential`**: river stays `percentile` on equity (no future cards). Turn:
  k-means over each hand's histogram of river equity (46 outcomes). Flop:
  k-means over each hand's distribution over **turn buckets** (47 outcomes),
  i.e. potential-aware in the Ganzfried & Sandholm 2014 sense, using earth
  mover's distance (or a defensible approximation). Multiplicity-weighted.

Header carries: magic + format version, method name and parameters, per-street
bucket counts, indexer version, a content hash over the arrays, and the
feature-store hash it was derived from.

### 4. Map-based `Abstraction`

`Abstraction` gains a second kind, backed by a memory-mapped bucket map:
`bucket(hole, board)` = `map[street][index(hole, board)]`. No rollouts, no memo
cache needed. The legacy Monte Carlo percentile kind is unchanged, byte for
byte, including `hash()`.

`num_buckets(street)` comes from the map header. `hash()` for the map kind
mixes the map's content hash (so the vocab/abstraction guard machinery treats a
different map as a different abstraction).

Memory-mapping (read-only, shared) means each process touches only the pages it
uses and processes on one host share them — this is what keeps the droplet
within budget.

### 5. Checkpoints

New magic **`SIXBP002`** for map-based checkpoints. It stores the abstraction
kind, the map's content hash, and a filename hint for the map, plus everything
v1 stores that still applies. `SIXBP001` files load exactly as today.

Loading a `SIXBP002` checkpoint requires the map file and **refuses** if the
map's content hash differs. `SixmaxAgent(path)` (used by
`eval_blueprint_h2h.py`) takes no config, so the map is found from the
checkpoint's filename hint, resolved first relative to the checkpoint's
directory, then under `sixmax/tables/`; config may override. Same for
`--resume` in `train_sixmax.py`.

### 6. Config and tooling

- `[abstraction]` gains a way to select a bucket map (e.g. `bucket_map =
  "<path>"`); when absent, today's Monte Carlo path is used, so all existing
  configs behave identically.
- Build CLIs (Python wrappers over multithreaded C++ bindings, matching the
  repo's pattern): build the feature store; build a bucket map from it with
  method + counts.
- Table outputs go under a gitignored directory (e.g. `sixmax/tables/`). Add the
  `.gitignore` entry.
- Cloud training needs the map on the box: either upload it (~140 MB) or build
  it there. The trainer Docker image itself should not embed tables.

## Verification the build must provide

- Indexer: round-trip `unindex(index(x))` is canonical-equivalent to `x`;
  suit-relabeled / reordered hands map to the same index; counts match the
  three numbers above; index is dense.
- Equity: river table matches brute force on random samples exactly; turn/flop
  derived equity matches `hand_equity` with many rollouts within MC tolerance.
- Percentile map: no equity value appears in two buckets; multiplicity-weighted
  populations are equal within rounding wherever ties do not force otherwise
  (report the max/min population ratio per street).
- Legacy path: every existing `SIXBP001` checkpoint in `sixmax/checkpoints/`
  loads, and a fixed sample of `bucket()` outputs is identical before and after
  (golden test).
- `SIXBP002`: save → load round-trip; loading with a different map is refused
  with a clear error.
- Throughput: map-based HU training iters/sec ≥ legacy on the same machine.
- Deploy (local only): `SixmaxAgent` on a map-based checkpoint works end to
  end (local web-service smoke), and the map's resident memory over a session
  is measured and reported against the droplet's 2 GB.

## Decided by the user

- Purpose: enable trying different bucket counts and potential-aware
  clustering without rebuilding.
- Success: first map-based HU model beats `hu_blueprint_20000000.bin` at t ≥ 2.
- Deployment target: the existing droplet (1 vCPU, 2 GB RAM, 50 GB SSD).

## Calls made in design (check these)

- Store exact equity only; derive turn/flop by averaging and derive histograms
  at bucketing time (instead of storing histograms).
- Memory-map the bucket map; never ship the feature store.
- First experiment is noise-free `percentile` 50/50/20 (isolates the noise
  fix). It trains to its **own plateau** (snapshots every 5M; plateau = latest
  snapshot no longer beats the one 10M earlier at t ≥ 2), expected ~20M since
  the infoset count is unchanged, and its plateau snapshot is what faces
  `hu_blueprint_20000000.bin`.
- Equity ties never split across buckets; uneven river buckets accepted.
- Map lookup for `SixmaxAgent`: checkpoint-relative, then `sixmax/tables/`.
- DREAM and droplet deployment out of scope; the memory check is local.
- New checkpoint magic `SIXBP002`; `SIXBP001` untouched.
- Bet-size key, six-max retrain, and search are out of scope.

## Left to the builder

- Indexer implementation (write one, or vendor one with a compatible license).
- River equity algorithm (direct 990-combo enumeration vs per-board
  sort-and-count with card-removal correction) and parallelism, as long as the
  river build finishes in hours on 8 cores.
- File layout/endianness of tables and headers, hash function choice.
- Histogram binning and the EMD approximation for `potential`; k-means init and
  iteration count (deterministic given a seed).
- Exact config key names and CLI names.
- Whether `Abstraction` becomes a variant, a strategy object, or two classes
  behind one interface, as long as callers keep calling `bucket()`.
