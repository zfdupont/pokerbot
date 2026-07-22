# Six-max Blueprint-vs-Baseline Eval Harness

**Date:** 2026-07-21
**Status:** Approved (design)
**Owner:** zfdupont

## Problem

The six-max blueprint pipeline is proven end-to-end, but the only strength
signal so far is the HU-mode sanity eval (`scripts/eval_hu_sanity.py`), which
pits the blueprint against frozen *heads-up specialists* in a 2-player game.
That is the wrong benchmark: it is neither 6-max nor a stable yardstick, and it
told us the 1M-iter blueprint is badly undertrained (−226 BB/100 vs tabular)
without telling us whether *more* training helps.

We need a real **6-max** benchmark against a **fixed, stable baseline** so that
"is training helping?" becomes a measurable curve: BB/100 per checkpoint, with
enough statistical resolution to distinguish a real gain from variance.

`PotOddsAgent` (`agents/potodds_agent.py`) was built for exactly this — a fixed,
non-adaptive reference opponent. This harness sits the blueprint against five of
them at a full-ring 6-max table, seat-rotated over identical decks, and sweeps a
series of checkpoints to draw the curve.

## Goals

- Blueprint in one seat vs 5 baseline agents (default `PotOddsAgent`) at a
  6-player table, in the live `game/poker.py` engine.
- Seat-rotated duplicate deals for variance reduction.
- Report **BB/100 with a standard error / 95% CI** per checkpoint, computed from
  independent per-deck samples.
- Sweep a series of checkpoints → a printed (and optional CSV) curve.
- Pluggable villain (default `potodds`), reusing existing live agents.

## Non-Goals (YAGNI)

- Button rotation across decks — seat rotation already gives perfect positional
  balance (hero covers all 6 positions per deck).
- A pure-C++ engine path — `PotOddsAgent` is a live-engine agent and cannot run
  inside `sixmax.EngineGameState`.
- Per-seat BB/100 breakdown, plotting (CSV output is sufficient).

## Why the live engine (not the C++ sixmax engine)

`PotOddsAgent` reads `game_state.community_cards / pot / current_bet` and returns
live `Action` enums — it only exists inside `game/poker.py`. So the harness runs
in the Python engine and drives the blueprint through the `SixmaxAgent` deploy
bridge (`agents/sixmax_agent.py`), exactly as `eval_hu_sanity.py` does. This is a
6-player generalization of that HU harness.

## Architecture

New script: `scripts/eval_sixmax_baseline.py`. Core functions take **already-built
agent objects** so they are unit-testable without buck2 or a checkpoint; only
`main()` constructs a `SixmaxAgent` and touches the extension.

### 1. Play unit — one deck, mirrored across seats

```
play_deck(hero_agent, villain_agent, seed, n, small_blind, stack) -> float
```

Returns the hero's total chip delta over the deck's `n` mirrored hands. For
`hero_seat in range(n)`:

1. `random.seed(seed)` — the deck shuffle is the only consumer of global
   `random` (agents use a separate `random.Random()` stream), so every rotation
   deals a byte-identical deck.
2. Build `n` fresh `Player(name, stack, agent)` — hero in `hero_seat`, villain
   agent in the rest.
3. `PokerGame(players, small_blind=small_blind).play_hand()`.
4. Accumulate `players[hero_seat].stack - stack`.

Button stays 0 (a fresh `PokerGame`/`GameState` each hand resets `button_pos`),
so across the `n` rotations the hero occupies all `n` positions relative to a
fixed button exactly once on the same deck.

Agent instances are built once per checkpoint and reused across all decks and
seats (stateless; `SixmaxAgent` holds only a loaded strategy + its own RNG). A
single shared villain instance serves all `n-1` villain seats — `PotOddsAgent`
is stateless and reads only `(player, game_state)`.

### 2. Match runner — decks are the independent samples

```
run_match(hero_agent, villain_agent, hands, seed, n, small_blind, stack)
    -> (bb100, stderr, n_hands)
```

For deck `h in range(hands)`:
- `s = seed * 1_000_003 + h`
- `x_h = play_deck(hero_agent, villain_agent, s, n, small_blind, stack)
         / (n * big_blind)`  — mean BB/hand for that deck (big_blind = 2·small_blind)

Then over the collected `x_h`:
- `bb100 = 100 * mean(x_h)`
- `stderr = 100 * std(x_h, ddof=1) / sqrt(hands)`
- `n_hands = hands * n`

Decks are independent (distinct seeds); within-deck seat correlation is absorbed
into the block `x_h`, so this SE is honest. The CLI prints `±1.96·stderr` as the
95% CI. **This block-statistics computation is the `TODO(human)` contribution
during implementation** (the independent unit, `ddof`, and CI multiplier are a
real judgment call).

### 3. Curve driver

Accept multiple checkpoint paths: `--checkpoints p1 p2 …` (glob-friendly, e.g.
`blueprint_*.bin`, matching `train_sixmax.py`'s `snapshots` naming
`<base>_<iters>.bin`) or a single `--checkpoint`. For each path:

1. Parse iteration count from the filename (last integer group; e.g.
   `blueprint_1000000.bin` → 1000000, `checkpoint_09040000` → 9040000). If none,
   fall back to input order.
2. Build one `SixmaxAgent(path, config_toml=config)`.
3. `run_match(...)`, collect `(iters, bb100, stderr, n_hands)`.

Print rows sorted ascending by iters:

```
     iters      BB/100        95% CI      hands
   1000000     -180.42      ±  12.31      3000
```

Optional `--csv PATH` writes `iters,bb100,stderr,n_hands`.

### 4. Villain registry

`VILLAINS = {name: factory}` mapping `--villain` to a zero-arg factory:
- `potodds` → `PotOddsAgent()` (default)
- `simple` → `SimpleAgent()`
- `position` → `PositionAgent()`
- `hand_strength` → `HandStrengthAgent()`
- `random` → a trivial uniform-legal live agent (or reuse an existing simple one)

Unknown names raise a clear `SystemExit` listing valid choices.

### 5. CLI

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoints` / `--checkpoint` | (required) | one or more blueprint `.bin` paths |
| `--villain` | `potodds` | baseline opponent name |
| `--hands` | `500` | number of decks (each played `n` times) |
| `--seed` | `1` | base seed |
| `--players` | `6` | table size `n` |
| `--bb` | `2` | big blind (small_blind = bb // 2) |
| `--stack` | `100 * bb` | starting stack (100 BB by default) |
| `--config` | `sixmax/configs/default.toml` | vocab/config TOML |
| `--csv` | none | optional CSV output path |

Chip frame: `bb=2, stack=200` → the `SixmaxAgent` rescales by `game_state.big_blind`
to the 100BB/1BB training frame, satisfying the boundary-rescale invariant.

## Boundaries / invariants respected

- The script bridges to the blueprint only through `agents/sixmax_agent.py`; it
  never imports `sixmax/` internals. Bridging stays in `scripts/` + `agents/`.
- BB/100 normalization divides by the table big blind (chip-scale invariant).
- Illegal-action masking is handled inside `SixmaxAgent` (unchanged).

## Testing

`tests/test_eval_sixmax_baseline.py` (no buck2 / checkpoint needed):

1. **Seat-rotation determinism** — with a stub agent that records its hole cards,
   assert that re-seeding produces an identical deck across rotations, and that
   the hero sees each of the `n` dealt hands exactly once per deck.
2. **Block statistics** — feed synthetic per-deck deltas into the stats function
   and assert `bb100` and `stderr` match a hand-computed reference (including
   `ddof=1` and the BB normalization).
3. **Villain registry** — known names build; unknown names raise with a helpful
   message.
4. **Smoke** (optional, marked slow / skipped if no checkpoint) — a few decks
   with a real `SixmaxAgent` if `best_checkpoint.bin` exists, asserting a finite
   BB/100 and `n_hands == hands * n`.

## Verify checklist (from `.mex/context/conventions.md`, run at VERIFY)

- `uv run pytest tests/` green (new tests included).
- `uv run python scripts/eval_sixmax_baseline.py --checkpoint sixmax/checkpoints/best_checkpoint.bin --hands 50`
  runs and prints a finite BB/100 ± CI.
- No import of `sixmax/` internals from the script (grep).
- No absolute-chip constants; BB normalization present.

## Scaffold growth (GROW)

- Update `.mex/ROUTER.md` "Current Project State": a real 6-max baseline benchmark
  now exists (closes the "no 6-max opponent baseline" gap).
- Add a `patterns/` entry or extend `eval-checkpoint.md` for the 6-max-baseline
  eval path.
