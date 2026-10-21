# HU 200 BB Abstraction Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implementers run as pi + DeepSeek Flash per `.mex/patterns/sdd-pi-implementer.md`.

**Goal:** Build everything the roadmap's experiments need: a frozen Slumbot evaluator, 200 BB training, config-defined pot buckets bound into a new SIXBP003 checkpoint, a cloud runner that ships one bucket map per arm, and a plateau/cap tool. Then run Phases 2, 3 and 5 from the runbook.

**Architecture:**
- Pot buckets become a `PotLayout` owned by `Abstraction`, so the trainer (`EngineGameState::abstract_key`), `BlueprintStrategy` and the Python bridge all key through one function.
- The legacy layout keeps today's bits and hash exactly. A non-legacy layout moves the pot index to key bits 26-29 and is persisted only in SIXBP003.
- The evaluator, cloud runner and plateau tool are host-side scripts.

**Tech Stack:** C++17 + pybind11 (`//sixmax:sixmax` via `~/bin/buck2`), Python 3.10 (`.venv`, `uv run`), pytest, bash, hcloud, Docker image `ghcr.io/zfdupont/pokerbot-trainer:latest`.

**Spec:** `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap-design.md` (evidence log and spike beside it in `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/`)

**Scope:**
- This plan implements spec Phase 0, Phase 1 (1a-1d) and the Phase 2 code (key change and versioning).
- Phases 2, 3 and 5 are operations, in the Runbook at the end.
- **Phase 4 (B) is a separate plan written after Gate 3.** The spec leaves its field bucketing open, to be informed by the phase results.

## Global Constraints

- Slumbot: strictly one request at a time, never parallel processes, even hand counts. Optional login only from `SLUMBOT_USERNAME` / `SLUMBOT_PASSWORD`, never logged or committed.
- No new Python dependencies. HTTP uses the stdlib `urllib`.
- Never commit `sixmax/tables/`, `sixmax/checkpoints/`, `.env`, or credentials.
- Legacy keys and hashes stay byte-identical. SIXBP001/002 files load as the legacy pot layout (pot at bits 18-19, edges 7/15/40), implied by the magic.
- A checkpoint trained under one pot layout refuses to load or resume under another, with an error containing "pot layout mismatch".
- Non-legacy pot layouts require a bucket map (SIXBP003 extends SIXBP002).
- Pot field: up to 16 buckets (at most 15 edges), key bits 26-29, bits 18-19 zero under a non-legacy layout.
- `pot_mode` is `"bb"` (edges are BB upper bounds on pot) or `"spr"` (edges are upper bounds on effective stack behind / pot, at the decision). Edges must be strictly increasing and > 0.
- Effective stack behind at a decision = min(acting player's stack behind, max stack behind among other non-folded players).
- Build: `~/bin/buck2 build //sixmax:sixmax`. Full suite: `uv run pytest tests/ -q` before each commit.
- Cloud spend: ask before exceeding $40 total, or before any arm exceeds 2x its benchmark-based estimate.
- Commit messages brief, no AI attribution lines.

## Review Focus

1. **A Slumbot error mid-hand** (`error_msg`, HTTP failure after retries) must not log a half-played hand. The hand is dropped and the run continues or stops cleanly. Test in Task 1.
2. **A resumed evaluator log with an odd hand count, or an odd `--hands`,** must be refused with a clear message, never silently skewed toward one seat. Test in Task 1.
3. **SPR mode when the opponent is all-in** (effective stack 0) gives bucket 0 with no division error. Test in Task 3.
4. **A relative `--bucket-map` path** passed to `cloud_train.sh` must resolve to its basename on the box and in `SIXMAX_BUCKET_MAP`. Test in Task 5.
5. **The plateau tool given irregular snapshot spacing** uses the newest snapshot at or below N-lag, and errors if none exists. Test in Task 6.

---

### Task 0: Close out the equity-lookup-tables branch (controller, not dispatched)

- [ ] In `.superpowers/sdd/2026-10-20-sixmax-equity-lookup-tables/progress.md`, append: `Ruling: Task 6 superseded by docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap-design.md (200 BB, Slumbot gate); no 100 BB hu_map training`. Mark Task 6 complete with that note.
- [ ] Run that plan's final whole-branch review (most capable model), one fix wave, and the scoped re-review.
- [ ] Finish the branch via `superpowers:finishing-a-development-branch`. **Ask the owner before merging to `main`.**
- [ ] Create `feature/hu200-roadmap` from the updated `main`. All tasks below run there.

### Task 1: Slumbot evaluator

**Files:**
- Create: `scripts/eval_slumbot.py` (port of `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/slumbot_spike.py`)
- Test: `tests/scripts/test_eval_slumbot.py`

**Interfaces:**
- Produces:
  - `parse_action(action: str, street: int) -> tuple[list[int], list[int], list[int]]`. Returns (prior-street totals, this-street commitments, raises per street), each chip list indexed by Slumbot pos (0 = BB, 1 = SB).
  - `build_state(resp: dict) -> tuple[SimpleNamespace, SimpleNamespace, bool]`. Returns (game_state, hero, facing).
  - `to_incr(action, amount, hero, facing) -> str`.
  - `play_hand(post: Callable[[str, dict], dict], agent, token: str | None) -> tuple[str | None, dict | None, list[str]]`. The middle item is `None` when the hand was aborted.
  - `summarize(rows: list[dict]) -> dict` with keys `hands, raw_bb100, raw_se, baseline_bb100, baseline_se, sb_baseline_bb100, bb_baseline_bb100`.
  - `breakdown(rows: list[dict]) -> dict` with keys `size_bins` (`"<10"`, `"10-40"`, `"40-100"`, `"100+"` → `{n, baseline_bb}`), `threebet_pots`, `showdown`, `fold`.
  - CLI: `--checkpoint`, `--config` (default `sixmax/configs/hu.toml`), `--hands N` (total, must be even), `--log PATH` (JSONL; resumes by continuing from the existing line count), `--summary` (print summary and breakdown of `--log` and exit).
- Behaviour is frozen to the spike: same state construction, same outbound translation, same fallback to `c`/`k` when Slumbot rejects a bet size. Log rows keep the spike's keys (`hand, pos, action, ours, hole, board, bot_hole, winnings, baseline`).

- [ ] **Step 1: Write the failing tests**

```python
from scripts.eval_slumbot import parse_action, build_state, to_incr, play_hand, summarize, breakdown

def test_parse_preflop_unopened():
    assert parse_action("", 0) == ([0, 0], [100, 50], [0, 0, 0, 0])

def test_parse_flop_bet_after_check():
    assert parse_action("b200c/kb400", 1) == ([200, 200], [0, 400], [1, 1, 0, 0])

def test_parse_three_bet():
    assert parse_action("b300b900", 0) == ([0, 0], [900, 300], [2, 0, 0, 0])

def test_parse_new_street_and_check_through():
    assert parse_action("b200c/", 1) == ([200, 200], [0, 0], [1, 0, 0, 0])
    assert parse_action("b200c/kk/kk/", 3) == ([200, 200], [0, 0], [1, 0, 0, 0])

def test_parse_turn_reraise():
    assert parse_action("b200c/kk/b300b900", 2) == ([200, 200], [300, 900], [1, 0, 2, 0])

def test_parse_preflop_shove():
    assert parse_action("b20000", 0) == ([0, 0], [100, 20000], [1, 0, 0, 0])

def test_build_state_bb_facing_open():
    gs, hero, facing = build_state({"action": "b250", "client_pos": 0,
                                    "hole_cards": ["Ac", "Kd"], "board": []})
    assert facing and gs.pot == 350 and gs.current_bet == 250
    assert gs.players[1] is hero and hero.current_bet == 100 and hero.stack == 19900
    assert gs.raises_per_street == [1, 0, 0, 0] and gs.button_pos == 0

def test_to_incr_maps_and_caps():
    hero = type("H", (), {"current_bet": 100, "stack": 19900})()
    from models.enums import Action
    assert to_incr(Action.FOLD, None, hero, facing=False) == "k"
    assert to_incr(Action.CALL, None, hero, facing=True) == "c"
    assert to_incr(Action.BET, 50000, hero, facing=True) == "b20000"
```

Plus, with a fake `post`:
- `test_play_hand_slumbot_folds_first`: `new_hand` returns a terminal response with `winnings: 50`. Expect `play_hand` to return that response and `[]`.
- `test_play_hand_error_mid_hand_aborts`: `act` returns `{"error_msg": "x"}` for a check/call. Expect the middle item to be `None`, and no row is written by the CLI loop.
- `test_bet_rejected_falls_back`: the first `act` with `b...` returns `error_msg`. Expect a retry with `c` when facing, and `"fallback:c"` recorded in `ours`.
- `test_odd_hands_refused`: `main(["--hands", "3", ...])` raises `SystemExit` with a message containing "even".
- `test_resume_odd_log_refused`: a log with 3 rows raises `SystemExit` containing "even".
- `test_summarize_known_rows`: rows with winnings `[100, -300]` and baseline `[200, -200]` give `raw_bb100 == -100.0`, `baseline_bb100 == 0.0`, `hands == 2`.
- `test_breakdown_bins`: a row with `winnings` 15000 lands in `"100+"`, and a row with action `b300b900c/...` counts in `threebet_pots`.

- [ ] **Step 2:** `uv run pytest tests/scripts/test_eval_slumbot.py -q`. Expect FAIL (module missing).
- [ ] **Step 3:** Implement `scripts/eval_slumbot.py` by porting the spike.
  - `post` is a module function built on `urllib.request` with the spike's retry/backoff, injected into `play_hand`.
  - Login, only when both env vars are set: POST `/slumbot/api/login` and use the returned token. Never print it.
  - The loop appends one JSONL row per finished hand, flushes it, and prints a summary line every 50 hands.
- [ ] **Step 4:** Run the tests again. Expect PASS. Then `uv run python scripts/eval_slumbot.py --summary --log docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/slumbot_hu50m_3000.jsonl` must print `baseline_bb100` -118.3 and size bin `"100+"` n=34.
- [ ] **Step 5:** Commit: `git add scripts/eval_slumbot.py tests/scripts/test_eval_slumbot.py && git commit -m "feat(eval): Slumbot evaluator"`

### Task 2: 200 BB training config and stack audit

**Files:**
- Create: `sixmax/configs/hu200.toml`
- Test: `tests/sixmax/test_stack_200.py`
- Modify: `.mex/AGENTS.md` (Non-Negotiables) and `CLAUDE.md` (Hard Invariants), the stack invariant line
- Modify: any file the audit finds with a hard-coded 100 BB stack

**Interfaces:**
- Produces: `sixmax/configs/hu200.toml`. It is `hu.toml` with:
  - `[abstraction] bucket_map = "sixmax/tables/map_pct_50_50_20.bin"`;
  - `[train.blueprint] starting_stack = 200.0`, `iterations = 60000000`, `checkpoint_interval = 5000000`, `snapshots = true`, `checkpoint = "sixmax/checkpoints/hu200_c0.bin"`.
  - `[actions.*]` blocks unchanged (same vocab hash).
  - This is arm C0.

- [ ] **Step 1: Write the failing tests** (synthetic map via the session fixture in `tests/sixmax/conftest.py`):
  - `test_engine_200bb_stacks`: `EngineConfig(num_players=2, starting_stack=200.0)` dealt hand → SB stack behind 199.5, BB 199.0, and the vocab all-in target from the SB preflop == 200.0.
  - `test_checkpoint_records_200`: train 120 iterations at `starting_stack=200.0` with the synthetic map, save, load → `meta.starting_stack == 200.0`.
  - `test_agent_allin_at_200bb`: build a `SixmaxAgent` on that checkpoint. In a live `PokerGame` HU hand with `stack=400, small_blind=1`, a forced all-in translation returns `Action.BET, 400`.
  - `test_h2h_runs_at_200bb`: `run_duplicate_match(..., hands=2, seed=1, bb=2, stack=400)` returns a float.
- [ ] **Step 2:** Run the tests. Expect FAIL until `hu200.toml` exists, or wherever the audit finds a hard-coded 100.
- [ ] **Step 3: Audit and fix.** Grep `sixmax/src`, `agents/sixmax_agent.py`, `scripts/train_sixmax.py`, `scripts/eval_blueprint_h2h.py` and `scripts/eval_hu_sanity.py` for stack assumptions. Fix any that ignore config. Create `hu200.toml`.
- [ ] **Step 4: Replace the invariant line** in both files with exactly:
  `- Chip values crossing an engine boundary must be rescaled to the big_blind=1 training frame. `cfr/`/`neural_cfr/` train at starting_stack=100; `sixmax/` checkpoints record their own starting_stack (read it from the checkpoint or its config, never assume 100).`
- [ ] **Step 5:** Run the tests, then the full suite. Expect PASS. Commit: `feat(sixmax): 200 BB HU config`.

### Task 3: PotLayout and the key

**Files:**
- Modify: `sixmax/src/abstraction/abstract_key.{h,cpp}`, `sixmax/src/abstraction/abstraction.{h,cpp}`, `sixmax/src/blueprint/engine_game.cpp`, `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_pot_layout.py`

**Interfaces:**
- Produces (C++, in `abstract_key.h`):
  ```cpp
  enum class PotMode : int32_t { BB = 0, SPR = 1 };
  struct PotLayout {
      PotMode mode = PotMode::BB;
      std::vector<double> edges = {7.0, 15.0, 40.0};
      bool is_legacy() const;            // mode==BB && edges=={7,15,40}
      int num_buckets() const;           // edges.size()+1
  };
  void validate_pot_layout(const PotLayout&);  // throws std::invalid_argument containing "pot_edges"
  int pot_bucket(const PotLayout&, double pot_bb, double eff_stack_bb);
  uint64_t pack_abstract_key(int card, int street, const std::array<uint8_t,4>& raises,
                             int pot_idx, bool legacy_layout, int live, int after);
  ```
- Behaviour:
  - `pot_bucket` takes value = pot_bb (BB mode) or eff_stack_bb / pot_bb (SPR mode). It returns the first i with value <= edges[i], else edges.size().
  - Legacy `pack_abstract_key` puts pot_idx at bits 18-19. Non-legacy puts it at bits 26-29 and leaves bits 18-19 zero.
  - The existing `pot_bucket(double)` and the pot_bb-taking `pack_abstract_key` stay as legacy wrappers. Their Python bindings are unchanged.
- `Abstraction` gains:
  - `void set_pot_layout(PotLayout)` (validates);
  - `const PotLayout& pot_layout() const` (default legacy);
  - `uint64_t pack_key(int card, int street, const std::array<uint8_t,4>& raises, double pot_bb, double eff_stack_bb, int live, int after) const`.
- `hash()` is unchanged under the legacy layout. Otherwise it folds `"pot1"`, the mode and each edge's bits into the existing hash via `fnv_mix`.
- `EngineGameState::abstract_key` calls `abs.pack_key(...)` with eff_stack_bb computed per the Global Constraints, using `PlayerState::stack`.
- Python:
  - `Abstraction.set_pot_layout(mode: str, edges: list[float])` with mode `"bb"`/`"spr"`, else `ValueError`;
  - `Abstraction.pot_layout() -> tuple[str, list[float]]`;
  - `Abstraction.pack_key(card, street, raises, pot_bb, eff_stack_bb, live, after) -> int`.

- [ ] **Step 1: Write the failing tests:**
  - `test_legacy_pack_key_matches_old`: for pots `[1, 7, 7.01, 15, 15.01, 40, 40.01, 500]` × cards `[0, 37, 168]` × streets 0-3, `abs.pack_key(...)` == `sixmax.pack_abstract_key(card, street, raises, pot, live, after)` on a default-layout map abstraction.
  - `test_legacy_hash_unchanged`: `hash()` is the same before and after `set_pot_layout("bb", [7, 15, 40])`.
  - `test_c1_bins_and_bits`: with edges `[4, 8, 16, 40, 100]`, pots 4→0, 4.01→1, 100→4, 100.01→5. `(key >> 18) & 3 == 0` and `(key >> 26) & 15 == idx`.
  - `test_spr_bins`: edges `[0.5, 1, 2, 4, 8]`, pot 10, eff 15 → idx 2.
  - `test_spr_allin_opponent`: eff 0 → idx 0 (Review Focus 3).
  - `test_validation`: `[]`, `[5, 5]`, `[-1, 2]`, 16 edges, and mode `"x"` each raise `ValueError`, with "pot_edges" in the message for the edge cases.
  - `test_layout_hashes_distinct`: legacy, C1 `[4, 8, 16, 40, 100]`, C2 `[3, 6, 12, 24, 48, 96, 192]` and SPR all give distinct `hash()`.
  - `test_pack_key_reproduces_engine_key_c1` and `_spr`: follow `tests/sixmax/test_pack_abstract_key.py::test_pack_reproduces_engine_key`.
    - Build a HU `EngineGameState` at `starting_stack=200.0` on a map abstraction with the C1 (resp. SPR) layout, and apply a few actions to reach a postflop decision.
    - `s.abstract_key(abs)` == `abs.pack_key(...)` with the same inputs, where eff_stack_bb = min of the two stacks behind.
  - The existing `tests/sixmax/test_legacy_buckets_golden.py` and `test_pack_abstract_key.py` still pass.
- [ ] **Step 2:** Build, run the tests. Expect FAIL.
- [ ] **Step 3:** Implement as specified. Build.
- [ ] **Step 4:** Run the tests and the full suite. Expect PASS. Commit: `feat(sixmax): config-defined pot layout in the infoset key`.

### Task 4: SIXBP003 checkpoints and consumers

**Files:**
- Modify: `sixmax/src/blueprint/checkpoint.{h,cpp}`, `sixmax/src/bindings/bindings.cpp` (`resume_blueprint`, `load_abstraction`), `agents/sixmax_agent.py`, `scripts/train_sixmax.py`
- Create: `sixmax/configs/hu200_c1.toml`, `hu200_c2.toml`, `hu200_c3.toml`
- Test: `tests/sixmax/test_sixbp003.py`

**Interfaces:**
- Consumes: Task 3 `PotLayout`, `Abstraction::set_pot_layout/pot_layout`.
- Produces:
  - `LoadedBlueprint::pot_layout` (`PotLayout`, legacy default).
  - **SIXBP003 layout** = the SIXBP002 layout through the map hint, then `int32 mode`, `uint32 n` (≤ 15, else error "pot layout too long"), `n × double` edges, then the table.
  - `save_blueprint` writes SIXBP003 iff `!abs.pot_layout().is_legacy()`, and throws "non-legacy pot layout requires a bucket map" for a legacy-kind abstraction. Otherwise bytes are unchanged.
  - `load_abstraction_for` applies the stored layout.
  - `resume_blueprint` refuses a layout difference with "pot layout mismatch".
  - `SixmaxDeployStrategy.decide(..., eff_stack_bb, ...)` keys via `self._strategy.abstraction().pack_key(...)`. `SixmaxAgent.get_action` computes `eff_stack_bb` from live `Player.stack` values / bb.
  - `train_sixmax.py`:
    - TOML `[abstraction]` keys `pot_mode` (default `"bb"`) and `pot_edges` (default `[7.0, 15.0, 40.0]`), applied after building the map abstraction;
    - a non-default layout without `bucket_map` exits with a message containing "pot_edges requires bucket_map";
    - CLI `--bucket-map PATH` overrides the TOML;
    - logs `pot layout <mode> <edges>`.
  - Arm configs: `hu200.toml` with `pot_edges` C1 `[4, 8, 16, 40, 100]` / C2 `[3, 6, 12, 24, 48, 96, 192]` / C3 `pot_mode = "spr"`, `[0.5, 1, 2, 4, 8]`, and checkpoints `hu200_c1.bin` / `_c2` / `_c3`.

- [ ] **Step 1: Write the failing tests** (synthetic map, 120 iterations):
  - `test_c1_writes_sixbp003`: the file starts with `b"SIXBP003"`, `sixmax.load_abstraction(path).pot_layout() == ("bb", [4, 8, 16, 40, 100])`, and every `sixmax.dump_infosets(path)` key has `(k >> 18) & 3 == 0`, with at least one postflop key where `(k >> 26) & 15 > 0`.
  - `test_legacy_layout_still_sixbp002`: a default-layout map run writes `b"SIXBP002"`.
  - `test_resume_refuses_layout_mismatch`: resume a C1 checkpoint with a legacy-layout abstraction, then with C2 edges. Both raise, with "pot layout mismatch". Happy-path resume with the same layout trains 10 more iterations.
  - `test_nonlegacy_without_map_refused`: a legacy-kind abstraction with C1 layout raises "requires a bucket map" on save.
  - `test_corrupt_layout_field`: truncate inside the layout → error with "truncated". Rewrite n=16 → error with "pot layout too long".
  - `test_agent_plays_c1_checkpoint`: `SixmaxAgent` on the C1 checkpoint plays `run_duplicate_match(..., hands=10, seed=1, bb=2, stack=400)` against itself without error. Its keys are queryable: on at least one postflop decision `strategy.probs(key)` is non-empty. Use a mock spy on `SixmaxDeployStrategy.decide` or its return path.
  - `test_train_cli_pot_edges_requires_map`: `train_sixmax.py` with `pot_edges` and no `bucket_map` exits with "pot_edges requires bucket_map".
- [ ] **Step 2:** Build, run the tests. Expect FAIL.
- [ ] **Step 3:** Implement. Build.
- [ ] **Step 4:** Run the tests and the full suite. Expect PASS. Update the AGENTS.md/CLAUDE.md map invariant line to read: `Map-based checkpoints (SIXBP002/SIXBP003) are bound to their bucket map by content hash, and SIXBP003 also to its pot layout; ...` (rest unchanged).
- [ ] **Step 5:** Commit: `feat(sixmax): SIXBP003 checkpoints bound to pot layout`.

### Task 5: Cloud arm runner

**Files:**
- Modify: `scripts/cloud_train.sh`, `scripts/train_sixmax.py` (rate log)
- Test: `tests/scripts/test_cloud_train_dry_run.py`, `tests/scripts/test_train_rate_log.py`

**Interfaces:**
- Produces:
  - `cloud_train.sh --bucket-map PATH`:
    - the source rsync gains `--exclude='sixmax/tables/'`;
    - PATH is uploaded to `/tmp/pokerbot/tables/<basename>` and mounted `-v /tmp/pokerbot/tables:/pokerbot/tables:ro`;
    - the container gets `-e SIXMAX_BUCKET_MAP=/pokerbot/tables/<basename>`, and the trainer gets `--bucket-map /pokerbot/tables/<basename>`;
    - if `--config`'s TOML sets `bucket_map` and `--bucket-map` is absent, it exits 2 with "config needs a bucket map; pass --bucket-map".
  - `cloud_train.sh --dry-run` prints the rsync excludes, upload target and `docker run` command, then exits 0 before sourcing `.env` or calling `hcloud`.
  - `train_sixmax.py` logs `[rate] iters=<N> it/s=<cumulative> window_it/s=<since last save>` at every checkpoint save (`N` = the trainer's TOTAL iterations, so a resumed run's log stays monotone; both rates cover this run only, and the window excludes selection).

- [ ] **Step 1: Write the failing tests:**
  - `test_dry_run_with_map`: `bash scripts/cloud_train.sh --dry-run --config sixmax/configs/hu200.toml --bucket-map sixmax/tables/map_pct_50_50_20.bin --checkpoint-name hu200_c0.bin` prints `SIXMAX_BUCKET_MAP=/pokerbot/tables/map_pct_50_50_20.bin`, `--exclude='sixmax/tables/'` and `--bucket-map /pokerbot/tables/map_pct_50_50_20.bin`.
  - `test_dry_run_relative_map_path`: `--bucket-map ./sixmax/tables/../tables/map_x.bin` → `/pokerbot/tables/map_x.bin` (Review Focus 4). The dry run does not require the file to exist.
  - `test_missing_map_refused`: the same command without `--bucket-map` exits 2 with "pass --bucket-map".
  - `test_rate_lines`: train with `checkpoint_interval=50`, `iterations=150` and the synthetic map → exactly 3 `[rate]` lines in the log, each with numeric `it/s`.
- [ ] **Step 2:** Run the tests. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and the full suite. Expect PASS. Add a line about `--bucket-map` and `--dry-run` to `.mex/context/setup.md` next to the existing `cloud_train.sh` entry.
- [ ] **Step 5:** Commit: `feat(cloud): ship one bucket map per arm; log rate`.

### Task 6: Plateau and cap tool

**Files:**
- Modify: `scripts/eval_blueprint_h2h.py` (extract a function)
- Create: `scripts/arm_plateau.py`
- Test: `tests/scripts/test_arm_plateau.py`

**Interfaces:**
- Produces:
  - `eval_blueprint_h2h.run_h2h(a: str, b: str, hands: int, seeds: list[int], bb: int, stack: int) -> dict` with keys `mean, sd, se, t, per_seed`. `main` uses it; CLI output unchanged.
  - `arm_plateau.snapshot_iters(path: str) -> int`: parses the trailing `_<digits>.bin`.
  - `arm_plateau.pick_pair(paths: list[str], lag: int) -> tuple[str, str]`: (newest, newest with iters <= newest - lag). `ValueError` "no snapshot at or below" if none.
  - `arm_plateau.iteration_cap(c0_plateau: int, c0_infosets: int, arm_infosets: int) -> int`: `clamp(round(c0_plateau * arm_infosets / c0_infosets), 20_000_000, 150_000_000)`.
  - `arm_plateau.converged(stats: dict) -> bool`: `not (stats["mean"] > 0 and stats["t"] >= 2)`.
  - CLI: `arm_plateau.py --snapshots GLOB --lag 10000000 --hands H --seeds 1,...,8` (defaults `--bb 2 --stack 400`, i.e. 200 BB) prints both iters, h2h stats and `VERDICT: converged|training`. `--infosets PATH` prints `BlueprintStrategy` queryable infoset count. `--cap C0_PLATEAU C0_INFOSETS ARM_INFOSETS` prints the cap.

- [ ] **Step 1: Write the failing tests:**
  - `test_snapshot_iters`: `"x/hu200_c1_15000000.bin"` → 15000000.
  - `test_pick_pair_regular`: snapshots at 5, 10, 15, 20M with lag 10M → (20M, 10M).
  - `test_pick_pair_irregular`: snapshots at 5, 12, 21M with lag 10M → (21M, 5M) (Review Focus 5).
  - `test_pick_pair_none`: snapshots at 5, 10M with lag 10M → `ValueError`.
  - `test_cap_worked_example`: `iteration_cap(25_000_000, 40_000, 120_000) == 75_000_000`. The clamps hold: `iteration_cap(25_000_000, 40_000, 10_000) == 20_000_000` and `iteration_cap(60_000_000, 40_000, 400_000) == 150_000_000`.
  - `test_converged_rule`: `{mean: 3, t: 2.5}` → False; `{mean: 3, t: 1.2}` → True; `{mean: -4, t: -3}` → True.
  - `test_run_h2h_self_is_zero`: `run_h2h(ckpt, ckpt, hands=4, seeds=[1], bb=2, stack=400)["mean"] == 0.0`, using the Task 4 test checkpoint built in the test.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and the full suite. Expect PASS. Commit: `feat(eval): arm plateau and iteration cap tool`.

After Task 6: the final whole-branch review per SDD, then finish the branch. **Ask the owner before merging to `main`**, since the CI image must contain this code before any arm runs.

---

## Runbook (operations after merge; controller with the owner)

Record every result in `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap/results.md`: date, arm, iterations, infosets, it/s, head-to-head stats, Slumbot score, cost.

1. **Evaluator freeze check.** Run `scripts/eval_slumbot.py --checkpoint sixmax/checkpoints/hu_blueprint_50000000.bin --hands 3000 --log <new log>`. It must agree with -118.3 within 2 combined SE (±59). If not, fix the evaluator before anything else.
2. **Benchmarks.**
   - Local: `train_sixmax.py --config sixmax/configs/hu200.toml --iterations 600000 --checkpoint /tmp/bench.bin --checkpoint-interval 200000`. Read the `[rate]` lines.
   - Then ask the owner and run one ccx33 for 10 minutes (about $0.04): `cloud_train.sh --type ccx33 --config sixmax/configs/hu200.toml --bucket-map sixmax/tables/map_pct_50_50_20.bin --iters 600000 --checkpoint-name bench_hu200.bin --max-hours 1`.
   - Write per-arm time and cost estimates into results.md.
3. **C0.**
   - Train `hu200.toml` on ccx33 with snapshots every 5M.
   - After each sync from 15M on, run `arm_plateau.py`. Stop at the first `converged`, or at 60M.
   - Record the plateau and infoset count.
   - **Budget checkpoint:** re-estimate the remaining schedule. If it exceeds $40, cut C3, then D2, then the interaction arm, and tell the owner.
4. **C1, C2, C3, one at a time.**
   - Probe each arm's infosets with a 2M-iteration run.
   - Cap = `arm_plateau.py --cap`.
   - Train to plateau or cap.
5. **Gate 2** (spec "Gate policy"):
   - head-to-head of each arm vs C0 at its plateau or cap, `|t| >= 2`:
     `eval_blueprint_h2h.py --a ARM --b C0 --hands 20000 --seeds 1,...,8 --bb 2 --stack 400`.
     **Always pass `--bb 2 --stack 400` for a 200 BB arm** (stack is in chips: 400 chips at bb 2 = 200 BB); the script's own default is the 100 BB frame, while `arm_plateau.py` already defaults to 200 BB;
   - the winner (or C0) to Slumbot at 13k hands;
   - apply the pass/block rules;
   - also check the Phase 2 extra stop (at least 2 SE better than -118).
6. **D maps** (local, from `equity_v1.bin`):
   - `scripts/build_bucket_map.py --method percentile --counts 100,100,50` and `--counts 200,200,100`;
   - then `--method potential` at the better count.
   - Each arm reuses the winning C config with `--bucket-map` and a new `--checkpoint-name`.
7. **D1-D3 and the interaction arm** (runner-up C + winning D) are trained like C arms, then **Gate 3.**
8. **Write the Phase 4 (B) plan** from the Gate 3 evidence, using the leak-2 aliasing measurement (`scripts/diagnose_key_aliasing.py`).
9. **Phase 5:**
   - the best checkpoint at 50k Slumbot hands, sequential (about 14 h);
   - pass at >= -20 BB/100;
   - if short by under 1 SE, an AIVAT-style reducer comes next.
