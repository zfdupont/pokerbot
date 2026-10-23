# C5 Bet-History Key Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in infoset-key field holding the size class of the last raise on each street, so the sixmax blueprint can tell bet sizes apart, and make the deployed agent build that field from the same engine code the trainer uses.

**Architecture:** The field lives in the C++ key packer (bits 30-41) behind `Abstraction::set_bet_history`. `EngineGameState` records each street's last-raise class as it applies actions, so training keys get it for free. Checkpoints carry the setting in a new `SIXBP004` magic. The deployed agent replays the live hand into an `EngineGameState` (the existing search replay) and passes the replayed classes to `pack_key`.

**Tech Stack:** C++17 (Buck2 `//sixmax:sixmax`), pybind11 bindings, Python 3.10, pytest.

**Spec:** `docs/superpowers/specs/2026-10-21-hu-200bb-abstraction-roadmap-design.md`, section "Phase 4a: C5, last raise size per street".

## Global Constraints

- Off mode (the default) changes nothing: every key, abstraction hash and checkpoint byte of existing SIXBP001/002/003 artifacts stays identical.
- Key layout: last-raise class of street s (0 preflop .. 3 river) at bits `30 + 3*s` .. `32 + 3*s`; values 0 = no raise, 1-3 = rank (smallest first) among the vocabulary's sizes of the same unit, 4 = the raise left the raiser all-in.
- Config: `[abstraction] bet_history = "off" | "last_raise"`, default `"off"`.
- `SIXBP004` = the SIXBP003 layout (magic, meta, kind, map hash, map hint, pot mode, edge count, edges) followed by `int32` bet-history mode (0 = off, 1 = last_raise), then the table. Written only when the field is on; requires a bucket map.
- Hard invariants (CLAUDE.md): `cfr/`, `neural_cfr/`, `sixmax/` never import `game/poker.py` or each other; never reorder or filter the vocab; checkpoints are bound to their map, pot layout and hand indexer; never commit checkpoints, `sixmax/tables/` or secrets.
- Commits: brief messages, no Co-Authored-By, no Claude-Session, no "Generated with" lines.
- Build: `~/bin/buck2 build //sixmax:sixmax`. Tests: `uv run pytest tests/`.

## Review Focus

1. A raise that the engine caps at the raiser's stack (a Bet whose target exceeds the stack) must read class 4 on both the trainer and deploy sides; otherwise trainer and deploy keys diverge on short-stack all-ins. Pinned in Task 1 (`test_capped_bet_is_all_in_class`) and Task 3 (parity test).
2. Street transitions: earlier streets' classes persist and the new street starts at 0. Task 1 (`test_engine_tracks_last_raise_per_street`).
3. Existing checkpoints and off-mode keys are untouched. Task 1 (`test_off_mode_ignores_last_raise`) and Task 2 (`test_off_checkpoint_keeps_sixbp003`).
4. Off-menu opponent bets (Slumbot's 7.5 BB 3-bet) translate through the replay; an unreplayable line plays the safe action rather than a wrong key. Task 3 (`test_replay_desync_plays_safe`), Task 4.
5. A resume or load whose bet-history setting differs from the config is refused, not silently trained across two key spaces. Task 2 (`test_resume_bet_history_mismatch_refused`).

---

### Task 1: Key field, vocab raise class, engine tracking (C++ + bindings)

**Files:**
- Modify: `sixmax/src/abstraction/abstract_key.h`, `sixmax/src/abstraction/abstract_key.cpp` (pack_abstract_key gains `last_raise`; `BetHistory` enum)
- Modify: `sixmax/src/abstraction/abstraction.h`, `sixmax/src/abstraction/abstraction.cpp` (setting, pack_key, hash)
- Modify: `sixmax/src/vocab/vocab.h`, `sixmax/src/vocab/vocab.cpp` (`raise_class`)
- Modify: `sixmax/src/blueprint/engine_game.h`, `sixmax/src/blueprint/engine_game.cpp` (track classes, use in `abstract_key`)
- Modify: `sixmax/src/bindings/bindings.cpp`
- Test: `tests/sixmax/test_bet_history_key.py`

**Interfaces:**
- Produces (C++): `enum class BetHistory : int32_t { Off = 0, LastRaise = 1 };` in `abstract_key.h`. `uint64_t pack_abstract_key(int card, int street, const std::array<uint8_t,4>& raises, int pot_idx, bool legacy_layout, int live, int after, const std::array<uint8_t,4>& last_raise = {})`. `void Abstraction::set_bet_history(BetHistory)`, `BetHistory Abstraction::bet_history() const`, `uint64_t Abstraction::pack_key(..., int after, const std::array<uint8_t,4>& last_raise = {}) const`. `int ActionVocab::raise_class(int idx) const`. `std::array<uint8_t,4> EngineGameState::last_raise_classes() const`.
- Produces (Python): `Abstraction.set_bet_history(mode: str)` with `"off"`/`"last_raise"`, `Abstraction.bet_history() -> str`, `Abstraction.pack_key(card, street, raises, pot_bb, eff_stack_bb, live, after, last_raise=None)` (None = zeros), `ActionVocab.raise_class(idx) -> int`, `EngineGameState.last_raise_classes() -> list[int]` (length 4).

- [ ] **Step 1: Write the failing tests** in `tests/sixmax/test_bet_history_key.py`. Use the `synthetic_bucket_map` fixture and `hu200_c4.toml`'s blueprint vocab (indices: 0 Fold, 1 Check, 2 Call, 3/4/5 Bet 2.0/3.5/5.0 BB, 6/7/8 Bet 0.33/0.75/1.5 pot, 9 AllIn). Construct `EngineGameState` the way `tests/sixmax/test_abstract_key.py` does.

```python
def test_raise_class_ranks_within_unit(c4_vocab):
    assert [c4_vocab.raise_class(i) for i in range(10)] == [0, 0, 0, 1, 2, 3, 1, 2, 3, 4]

def test_raise_class_refuses_four_sizes_in_a_unit(tmp_path):
    vocab = _vocab_with_pot_sizes(tmp_path, [0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="raise_class"):
        vocab.raise_class(6)

def test_off_mode_ignores_last_raise(abs_c4_layout):
    base = abs_c4_layout.pack_key(7, 2, [1, 1, 0, 0], 20.0, 180.0, 1, 0)
    assert abs_c4_layout.pack_key(7, 2, [1, 1, 0, 0], 20.0, 180.0, 1, 0,
                                  last_raise=[1, 2, 3, 4]) == base
    assert (base >> 30) & 0xFFF == 0

def test_last_raise_bits(abs_c4_layout):
    off = abs_c4_layout.pack_key(7, 3, [1, 1, 1, 1], 60.0, 140.0, 1, 0)
    abs_c4_layout.set_bet_history("last_raise")
    on = abs_c4_layout.pack_key(7, 3, [1, 1, 1, 1], 60.0, 140.0, 1, 0,
                                last_raise=[1, 2, 3, 4])
    assert (on >> 30) & 0xFFF == 1 | (2 << 3) | (3 << 6) | (4 << 9)
    assert on & ((1 << 30) - 1) == off

def test_hash_tracks_bet_history(abs_c4_layout):
    h_off = abs_c4_layout.hash()
    abs_c4_layout.set_bet_history("last_raise")
    assert abs_c4_layout.hash() != h_off
    abs_c4_layout.set_bet_history("off")
    assert abs_c4_layout.hash() == h_off

def test_unknown_bet_history_refused(abs_c4_layout):
    with pytest.raises(ValueError, match="bet_history"):
        abs_c4_layout.set_bet_history("full")

def test_engine_tracks_last_raise_per_street(c4_vocab):
    # Open 3.5 BB (idx 4), call; flop: check, bet 0.33 (6), raise 1.5 (8), call.
    st = _hu_state(c4_vocab)
    for a in (4, 2, 1, 6, 8, 2):
        st.apply(a)
    assert st.last_raise_classes() == [2, 3, 0, 0]   # turn: flop's last raise kept

def test_capped_bet_is_all_in_class(c4_vocab):
    # 20 BB stacks: a 1.5-pot raise over a 5 BB open targets 26 BB > 20 BB.
    # First check st.legal_mask()[8]: if the engine allows it (and caps the
    # raise at the stack), assert class 4 after apply(8). If it masks it, the
    # capped case cannot arise in training; assert instead that the legal
    # AllIn (9) gives class 4, and keep the all-in-by-state rule anyway.
    st = _hu_state(c4_vocab, stack_bb=20.0)
    st.apply(5)
    st.apply(8 if st.legal_mask()[8] else 9)
    assert st.last_raise_classes()[0] == 4

def test_engine_abstract_key_includes_classes_only_when_on(c4_vocab, abs_c4_layout):
    st = _hu_state(c4_vocab)
    for a in (4, 2, 1, 7):
        st.apply(a)
    off_key = st.abstract_key(abs_c4_layout)
    abs_c4_layout.set_bet_history("last_raise")
    on_key = st.abstract_key(abs_c4_layout)
    assert (off_key >> 30) & 0xFFF == 0
    assert (on_key >> 30) & 0xFFF == 2 | (2 << 3)
```

Fixtures to add in the test file: `c4_vocab` (blueprint vocab of `sixmax/configs/hu200_c4.toml`), `abs_c4_layout` (`Abstraction.from_bucket_map(synthetic_bucket_map)` with `set_pot_layout("bb", [3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0])`), `_hu_state(vocab, stack_bb=200.0)` (HU, button 0, fixed deck), `_vocab_with_pot_sizes(tmp_path, sizes)` (a tmp TOML with `hu200_c4.toml`'s opens and the given pot sizes).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/sixmax/test_bet_history_key.py -v`
Expected: FAIL (`AttributeError`: no `raise_class` / `set_bet_history` / `last_raise_classes`).

- [ ] **Step 3: Implement.**
- `ActionVocab::raise_class`: 0 for Fold/Check/Call; 4 for AllIn; for Bet, 1 + the index's rank among Bet entries with the same `SizeUnit`, in vocab order (the vocab lists sizes ascending). Throw `std::invalid_argument("raise_class: ...")` if that unit has more than 3 Bet entries.
- `EngineGameState`: a `std::array<uint8_t,4> last_raise_{}` member, reset with the hand. In `apply`, for Bet/AllIn, after the engine applies the raise, set `last_raise_[street] = 4` if the acting player is now all-in, else `vocab_->raise_class(action)`. Then `abstract_key` passes `last_raise_` to `abs.pack_key` when `abs.bet_history() == BetHistory::LastRaise`, else zeros.
- `Abstraction::hash`: when the field is on, mix the tag `0x62687431ull` ("bht1") and the mode after the pot-layout mix. Off leaves the hash unchanged.
- `pack_abstract_key`: OR in `last_raise[s] << (30 + 3*s)`, each masked to 3 bits. `Abstraction::pack_key` passes zeros when the field is off, whatever the caller sent.
- Bindings: the Python names in Interfaces; `set_bet_history` raises `ValueError` containing "bet_history" for unknown strings.

- [ ] **Step 4: Build and run**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_bet_history_key.py tests/sixmax/test_abstract_key.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite**, then commit `feat(sixmax): opt-in last-raise class per street in the infoset key`.

Run: `uv run pytest tests/ -q --deselect tests/scripts/test_eval_mixed_table.py::test_smoke_50_hands`
Expected: all pass (no change in off mode).

---

### Task 2: SIXBP004 checkpoints, training config, C5 config

**Files:**
- Modify: `sixmax/src/blueprint/checkpoint.h`, `sixmax/src/blueprint/checkpoint.cpp` (`LoadedBlueprint.bet_history`; SIXBP004 save/load; `load_abstraction_for` applies it)
- Modify: `scripts/train_sixmax.py` (`bet_history` config key, set on fresh abstractions, resume check)
- Create: `sixmax/configs/hu200_c5.toml`
- Test: `tests/sixmax/test_sixbp004.py`, `tests/scripts/test_train_sixmax_bet_history.py`

**Interfaces:**
- Consumes: Task 1's `Abstraction.set_bet_history` / `bet_history()`.
- Produces: `sixmax.load_abstraction(path)` and `BlueprintStrategy.load` restore the setting. `train_sixmax.resolve_config` returns `cfg["bet_history"]` (default `"off"`).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/sixmax/test_sixbp004.py (tiny map-backed HU trainer, as tests/scripts/test_arm_plateau.py's tiny_ckpt)
def test_sixbp004_round_trip(tmp_path, ...):
    path = _train_tiny(tmp_path, bet_history="last_raise")
    assert open(path, "rb").read(8) == b"SIXBP004"
    abs_ = sixmax.load_abstraction(path)
    assert abs_.bet_history() == "last_raise"
    assert abs_.hash() == _trained_abstraction_hash   # same hash as at save
    assert sixmax.BlueprintStrategy.load(path, vocab).num_infosets() > 0

def test_off_checkpoint_keeps_sixbp003(tmp_path, ...):
    path = _train_tiny(tmp_path, bet_history="off")
    assert open(path, "rb").read(8) == b"SIXBP003"
    assert sixmax.load_abstraction(path).bet_history() == "off"

def test_bet_history_requires_bucket_map(tmp_path):
    abs_ = sixmax.Abstraction(flop_buckets=3, turn_buckets=3, river_buckets=3,
                              equity_rollouts=8, quantile_samples=30, seed=1)
    abs_.set_bet_history("last_raise")
    with pytest.raises(Exception, match="bet_history requires a bucket map"):
        _save_tiny(tmp_path, abs_)

# tests/scripts/test_train_sixmax_bet_history.py
def test_config_default_is_off():           # resolve_config on hu200_c4.toml -> "off"
def test_c5_config_reads_last_raise():      # hu200_c5.toml -> "last_raise"; same vocab hash as hu200_c4.toml
def test_resume_bet_history_mismatch_refused(tmp_path):
    # train 20 iters with bet_history on, resume with hu200_c4.toml (off):
    # SystemExit whose message contains "bet history mismatch"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/sixmax/test_sixbp004.py tests/scripts/test_train_sixmax_bet_history.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement.**
- Save: when the field is on, require `abs.is_map()` (else `std::invalid_argument("blueprint checkpoint: bet_history requires a bucket map")`). Write magic `SIXBP004`, then the SIXBP003 fields (the pot layout is always written in v4, legacy or not), then `int32` mode.
- Load: accept `SIXBP004` and read it as v3 plus the mode. Reject unknown modes with `runtime_error("blueprint checkpoint: unknown bet history ...")`.
- `load_abstraction_for`: call `set_bet_history(loaded.bet_history)`.
- `train_sixmax.py`: add `"bet_history": "off"` to defaults, read `[abstraction].bet_history`, and require a bucket map when on, mirroring the existing pot_edges check. On a fresh map abstraction call `abstraction.set_bet_history(cfg["bet_history"])`. On resume, compare `abstraction.bet_history()` with the config and `SystemExit("bet history mismatch: ...")` on a difference.
- `hu200_c5.toml`: `hu200_c4.toml` with `arm C5` in the header, `bet_history = "last_raise"` under `[abstraction]`, and `checkpoint = "sixmax/checkpoints/hu200_c5.bin"`. The `[actions.*]` blocks stay byte-identical to C4's.

- [ ] **Step 4: Run the tests and the existing checkpoint tests**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/sixmax/test_sixbp004.py tests/scripts/test_train_sixmax_bet_history.py tests/sixmax/test_sixbp003.py tests/sixmax/test_blueprint_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite**, then commit `feat(sixmax): SIXBP004 checkpoints and bet_history config; C5 config`.

---

### Task 3: Deployed agent builds last-raise classes from a hand replay

**Files:**
- Modify: `agents/sixmax_agent.py`
- Test: `tests/agents/test_deploy_bet_history.py`

**Interfaces:**
- Consumes: Task 1's `pack_key(..., last_raise=)` and `EngineGameState.last_raise_classes()`; Task 2's checkpoints that load with the field on.
- Produces: `SixmaxDeployStrategy.decide(..., last_raise=None)`; `SixmaxDeployStrategy.bet_history -> str`; `SixmaxAgent._last_raise_classes(game_state, hole, board) -> list[int]`; a new lookup outcome `"desync"` in `SixmaxDeployStrategy.lookups`.

- [ ] **Step 1: Write the failing tests** (tiny HU 200 BB checkpoint trained with `hu200_c5.toml`'s vocab and `bet_history="last_raise"`, as in Task 2):

```python
def test_decide_requires_last_raise_when_on(c5_deploy):
    with pytest.raises(ValueError, match="last_raise"):
        c5_deploy.decide(**_kwargs())                  # no last_raise

def test_agent_key_matches_engine_key(c5_ckpt):
    # Play 300 hands SixmaxAgent vs SixmaxAgent in game.poker.PokerGame (seeded).
    # Hook SixmaxDeployStrategy.decide to rebuild the key it looks up, and at
    # the same decision replay the hand into an EngineGameState (the agent's
    # own replay helper) and take eng.abstract_key(abstraction).
    # Every decision: the two keys are equal. At least one decision has a
    # nonzero class on an earlier street.

def test_replay_desync_plays_safe(c5_agent, monkeypatch):
    # Make the replay raise ValueError("replay desync ..."): get_action returns
    # the safe action (check/call/fold, never a bet) and lookups["desync"] == 1.

def test_legacy_checkpoint_ignores_history(c4_tiny_ckpt):
    # A bet_history="off" checkpoint plays exactly as before: decide without
    # last_raise works, and SixmaxAgent never calls the replay.
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/agents/test_deploy_bet_history.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement.**
- `SixmaxDeployStrategy`:
  - `bet_history` property, read from `abstraction().bet_history()`.
  - `decide` raises `ValueError("last_raise is required for bet_history checkpoints")` when the field is on and `last_raise` is None.
  - `last_raise` is passed to every `pack_key` call, including the nearest-bucket retries.
  - `lookups` gains `"desync": 0`.
- `SixmaxAgent`:
  - `_replay_to_now(game_state, hole, board)` applies the current-street prefix that `_replay_to_street_root` returns, checking legality the same way, and returns the engine state.
  - `_last_raise_classes` returns its `last_raise_classes()`.
  - In `get_action`, when `self._deploy.bet_history != "off"`: compute the classes. On `ValueError` from the replay, count `"desync"` and return the deploy strategy's safe action. Otherwise pass `last_raise=` to `decide`.
  - The replay is HU only, as today. When the field is on and `len(players) != 2`, raise `ValueError("bet_history is heads-up only")`.

- [ ] **Step 4: Run**

Run: `~/bin/buck2 build //sixmax:sixmax && uv run pytest tests/agents/ -v`
Expected: PASS.

- [ ] **Step 5: Full suite**, then commit `feat(agent): last-raise classes from the hand replay for bet_history checkpoints`.

---

### Task 4: Slumbot stand-in state carries the action history

**Files:**
- Modify: `scripts/eval_slumbot.py` (`build_state`)
- Test: `tests/scripts/test_eval_slumbot.py`

**Interfaces:**
- Consumes: Task 3's `SixmaxAgent._last_raise_classes`.
- Produces: `build_state(resp)` namespaces with `action_history` (list of `(player_index, Action, amount_chips_or_None, round)` in `players` order, as `game/poker.py` records it) and per-player `total_contributed` (chips put in before the current street plus the current street's bet).

- [ ] **Step 1: Write the failing tests**

```python
def test_build_state_records_action_history():
    gs, hero, _ = build_state({"action": "b200c/kb300", "board": ["Ah", "7c", "2d"],
                               "client_pos": 0, "hole_cards": ["Kd", "Qs"]})
    # Slumbot pos 1 (SB) is engine seat 0; pos 0 (BB) is seat 1.
    assert [(p, a.name, amt, r) for p, a, amt, r in gs.action_history] == [
        (0, "RAISE", 200, 0), (1, "CALL", None, 0), (1, "CHECK", None, 1), (0, "BET", 300, 1)]
    assert [p.total_contributed for p in gs.players] == [500, 200]

def test_slumbot_state_replays_to_last_raise_classes(c5_agent):
    gs, hero, _ = build_state({"action": "b200c/kb300", "board": ["Ah", "7c", "2d"],
                               "client_pos": 0, "hole_cards": ["Kd", "Qs"]})
    c5_agent._hero = hero
    # 2 BB open = class 1; 3 BB into a 4 BB pot = 0.75 pot = class 2.
    assert c5_agent._last_raise_classes(gs, _ints(["Kd", "Qs"]), _ints(["Ah", "7c", "2d"])) == [1, 2, 0, 0]
```

Use the amount convention `game/poker.py` records: a BET/RAISE amount is the raise-to total for the street in chips. Match whatever `_action_to_vocab` expects, and check it against `game/poker.py` before writing the expected tuple.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/scripts/test_eval_slumbot.py -v -k "action_history or last_raise"`
Expected: FAIL.

- [ ] **Step 3: Implement** in `build_state`: walk the tokens with `TOKEN_RE` the way `parse_action` does, emitting one `action_history` entry per token. `b` is RAISE when it raises a bet and BET when nothing is to call; `c` is CALL (CHECK when nothing is to call); `k` is CHECK; `f` is FOLD. Set `total_contributed` per player.

- [ ] **Step 4: Run**

Run: `uv run pytest tests/scripts/test_eval_slumbot.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite**, then commit `feat(eval): Slumbot stand-in state carries action history`.

---

## After the tasks (controller, not implementers)

1. Probe C5 locally: 2M iterations with `hu200_c5.toml`, then `arm_plateau.py --infosets`. Cap = `arm_plateau.py --cap 20000000 33572 <infosets> --phase 3` (the field can multiply infosets). Ask the owner before cloud spend.
2. Update `.mex/context/architecture.md` (key layout) and `.mex/context/decisions.md`.
