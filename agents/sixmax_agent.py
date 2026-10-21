"""Deployment bridge for the six-max blueprint.

SixmaxDeployStrategy is host-agnostic: it takes public state in big-blind units
and returns a chosen vocab index plus a raise-to (also in BB). Hosts (the live
engine via SixmaxAgent, the openpoker connector via HandTracker) compute the
legal mask and the (live, after) counts, and translate BB back to table chips.

Key reconstruction routes through the checkpoint abstraction's pack_key so it
is byte-identical to the trainer's EngineGameState::abstract_key, including a
non-legacy pot layout. Bet legality and translation route through the C++
ActionVocab (target_bb), the single vocab translation layer.
"""
import importlib.util
import os
import random
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_sixmax_extension():
    """Force-load the sixmax C++ extension as sys.modules['sixmax'].

    The repo-root `sixmax/` directory is a namespace package that shadows the
    Buck2-built `sixmax.so`, so a bare `import sixmax` from a plain script gets
    the Python package (no C++ symbols). Under pytest the conftests already
    force-load the .so, so this is a no-op there (guarded on a C++-only
    attribute); for scripts (eval_hu_sanity, openpoker_bot) it builds and loads
    the extension the same way train_sixmax.py does."""
    mod = sys.modules.get("sixmax")
    if mod is not None and hasattr(mod, "SizeUnit"):
        return  # real extension already loaded (e.g. by a test conftest)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = None
    for line in result.stdout.splitlines():
        if "sixmax.so" in line:
            so_dir = os.path.join(_ROOT, os.path.dirname(line.split()[-1]))
            break
    if so_dir is None:
        raise RuntimeError("Could not locate sixmax.so in buck2 output")
    so_path = os.path.join(so_dir, "sixmax.so")
    spec = importlib.util.spec_from_file_location("sixmax", so_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sixmax"] = m  # register before exec so circular refs resolve
    spec.loader.exec_module(m)


def _ensure_sixmax_dream_extension():
    """Lazily force-load sixmax_dream (keeps blueprint deploy path torch-free)."""
    mod = sys.modules.get("sixmax_dream")
    if mod is not None and hasattr(mod, "DreamTrainer"):
        return mod
    _ensure_sixmax_extension()  # sixmax first (cross-module types)
    buck2 = os.path.expanduser("~/bin/buck2")
    result = subprocess.run([buck2, "build", "//sixmax:sixmax_dream", "--show-output"],
                            capture_output=True, text=True, cwd=_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Buck2 build failed:\n{result.stderr}")
    so_dir = next((os.path.join(_ROOT, os.path.dirname(l.split()[-1]))
                   for l in result.stdout.splitlines() if "sixmax_dream.so" in l), None)
    if so_dir is None:
        raise RuntimeError("Could not locate sixmax_dream.so in buck2 output")
    so_path = os.path.join(so_dir, "sixmax_dream.so")
    spec = importlib.util.spec_from_file_location("sixmax_dream", so_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sixmax_dream"] = m
    spec.loader.exec_module(m)
    return m


_ensure_sixmax_extension()
import sixmax  # noqa: E402  (must follow _ensure_sixmax_extension)

from agents.base_agent import PokerAgent  # noqa: E402
from models.enums import Action, Suit  # noqa: E402


def _load_vocab(config_toml, section):
    spec = importlib.util.spec_from_file_location(
        "vocab_config", os.path.join(_ROOT, "sixmax", "vocab_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vocab(config_toml, section)


def canonical_live_after(n, button, hero, folded, all_in, street):
    """(live, after) per the pinned canonical action order.

    Mirrors EngineGameState::abstract_key exactly. Positions are table indices
    0..n-1; folded/all_in are length-n bool lists indexed the same way."""
    start = (button if n == 2 else (button + 3) % n) if street == 0 \
        else (button + 1) % n

    def order(seat):
        return (seat - start + n) % n

    live = after = 0
    for s in range(n):
        if s == hero or folded[s]:
            continue
        live += 1
        if not all_in[s] and order(s) > order(hero):
            after += 1
    return live, after


_LEGACY_POT_EDGES = [7.0, 15.0, 40.0]


def _is_legacy_pot_layout(layout):
    """True for the historical BB 7/15/40 layout (the only legacy one)."""
    mode, edges = layout
    return mode == "bb" and [float(e) for e in edges] == _LEGACY_POT_EDGES


class SixmaxDeployStrategy:
    def __init__(self, strategy, vocab):
        self._strategy = strategy
        self._vocab = vocab

    @classmethod
    def load(cls, path, config_toml, section="blueprint"):
        vocab = _load_vocab(config_toml, section)
        strategy = sixmax.BlueprintStrategy.load(path, vocab)
        return cls(strategy, vocab)

    @property
    def num_players(self):
        return self._strategy.num_players()

    @property
    def num_infosets(self):
        """Queryable infosets in the loaded checkpoint (unseen keys dropped)."""
        return self._strategy.num_infosets()

    def _card_bucket(self, hole, board, street):
        if street == 0:
            return sixmax.preflop_class(hole)
        return self._strategy.abstraction().bucket(hole, board)

    def decide(self, *, hole, board, street, raises_per_street, pot_bb,
               current_bet_bb, to_call_bb, stack_bb, live, after, legal, rng,
               eff_stack_bb=None):
        abstraction = self._strategy.abstraction()
        if eff_stack_bb is None:
            if not _is_legacy_pot_layout(abstraction.pot_layout()):
                raise ValueError(
                    "eff_stack_bb is required for non-legacy pot layouts")
            eff_stack_bb = stack_bb
        card = self._card_bucket(hole, board, street)
        key = abstraction.pack_key(card, street, list(raises_per_street),
                                   pot_bb, eff_stack_bb, live, after)
        probs = self._strategy.probs(key)  # [] if unseen

        legal_idx = [i for i, m in enumerate(legal) if m]
        weights = [probs[i] if i < len(probs) else 0.0 for i in legal_idx]
        total = sum(weights)
        if total <= 0.0:
            idx = rng.choice(legal_idx)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = legal_idx[-1]
            for i, w in zip(legal_idx, weights):
                acc += w
                if r <= acc:
                    idx = i
                    break

        action = self._vocab.at(idx)
        if action.type in (sixmax.ActionType.Bet, sixmax.ActionType.AllIn):
            ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                    to_call=to_call_bb, stack=stack_bb)
            raise_to = self._vocab.target_bb(idx, ctx)
        else:
            raise_to = 0.0
        return idx, raise_to


_SUIT_TO_IDX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_DEFAULT_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")
_RAISE_CAP = 3  # the sixmax abstraction's per-street raise cap (2 bits)


def _card_to_int(card):
    return (card.rank - 2) * 4 + _SUIT_TO_IDX[card.suit]


class DreamDeployStrategy:
    """Deployment wrapper for sixmax_dream.DreamStrategy.

    Provides the same decide() interface as SixmaxDeployStrategy so that
    SixmaxHandTracker (openpoker_bot.py) can route to it unchanged.

    sixmax_dream (libtorch) is loaded LAZILY — only when a DREAM checkpoint is
    actually deployed — so importing this module never pulls in libtorch and the
    blueprint deploy path (SixmaxDeployStrategy) remains torch-free.

    The neural net reads an EngineGameState feature vector; we reconstruct
    the full n_players state with real per-seat stacks so that seats 2-5
    features (stacks, street_bets, live_mask) are on-distribution.
    """

    def __init__(self, path, config_toml, section="blueprint"):
        vocab = _load_vocab(config_toml, section)
        sixmax_dream = _ensure_sixmax_dream_extension()
        strategy = sixmax_dream.DreamStrategy.load(path, "cpu", vocab)
        self._strategy = strategy
        self._vocab = vocab

    @classmethod
    def load(cls, path, config_toml, section="blueprint"):
        return cls(path, config_toml, section)

    @property
    def num_players(self):
        return 6  # DreamStrategy is a 2-6 player generalist

    def decide(self, *, hole, board, street, raises_per_street, pot_bb,
               current_bet_bb, to_call_bb, stack_bb, live, after, legal, rng,
               n_players=None, seat_stacks_bb=None):
        """Sample an action from the DreamStrategy network.

        Constructs an n_players EngineGameState with the hero at seat 0 and
        real per-seat stacks when available, so all feature dimensions are
        on-distribution.  Falls back to a 2-player reconstruction when
        n_players is not supplied (e.g. unit tests or legacy callers).

        Args:
            n_players:       Actual number of active players at the table.
                             When None, defaults to 2 (HU fallback).
            seat_stacks_bb:  List of per-seat stacks in BB (length n_players),
                             with hero at index 0.  When None, all opponents
                             are assigned stack_bb as an approximation.
        """
        if n_players is None:
            n_players = 2

        if seat_stacks_bb is None:
            # Approximate: give every seat the hero's current stack
            seat_stacks_bb = [stack_bb] * n_players

        # Build a full 52-card deck placeholder, then overwrite the hero's
        # cards (deck[0], deck[1]) and board (deck[2*n_players..2*n_players+4]).
        used = set(hole) | set(board)
        placeholder = iter(c for c in range(52) if c not in used)
        deck = list(range(52))

        # Hero at seat 0
        deck[0] = hole[0]
        deck[1] = hole[1]
        # Opponents at seats 1..n_players-1 — distinct placeholder cards
        for opp in range(1, n_players):
            deck[2 * opp]     = next(placeholder)
            deck[2 * opp + 1] = next(placeholder)
        # Board: deck[2*n_players .. 2*n_players+4]
        board_start = 2 * n_players
        for j, c in enumerate(board):
            deck[board_start + j] = c
        # Fill remaining board positions with placeholders if board is partial
        for j in range(len(board), 5):
            deck[board_start + j] = next(placeholder)

        cfg = sixmax.EngineConfig(num_players=n_players)
        cfg.starting_stack = stack_bb  # feature scaling: training frame is BB=1
        # Pass per-seat stacks so the encoder sees real chip counts for all seats
        stacks = [float(s) for s in seat_stacks_bb]
        state = sixmax.EngineGameState(cfg, 0, deck, self._vocab, stacks)
        probs = self._strategy.get_probs(state)

        legal_idx = [i for i, m in enumerate(legal) if m]
        weights = [probs[i] if i < len(probs) else 0.0 for i in legal_idx]
        total = sum(weights)
        if total <= 0.0:
            idx = rng.choice(legal_idx)
        else:
            r = rng.random() * total
            acc = 0.0
            idx = legal_idx[-1]
            for i, w in zip(legal_idx, weights):
                acc += w
                if r <= acc:
                    idx = i
                    break

        action = self._vocab.at(idx)
        if action.type in (sixmax.ActionType.Bet, sixmax.ActionType.AllIn):
            ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                    to_call=to_call_bb, stack=stack_bb)
            raise_to = self._vocab.target_bb(idx, ctx)
        else:
            raise_to = 0.0
        return idx, raise_to


_SEARCH_BUILTIN = {
    "enabled": False,
    "budget_ms": 5000.0,
    "max_nodes": 20000,
    "max_iters": 2000,
    "leaf_samples": 256,
    "seed": 1,
    "leaf_threads": 0,
    "narrow_preflop": True,
    "prune_keep_mass": 0.99,
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _merge_search_config(config_toml, section="search"):
    """Resolve the [search] block (all keys) with builtin < TOML < env
    precedence. Unknown TOML keys are rejected, mirroring
    scripts/train_sixmax.py::resolve_config. Env keys are SIXMAX_SEARCH_<KEY>."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        import tomli as tomllib
    cfg = dict(_SEARCH_BUILTIN)
    if os.path.exists(config_toml):
        with open(config_toml, "rb") as f:
            data = tomllib.load(f)
        for key, value in data.get(section, {}).items():
            if key not in cfg:
                raise KeyError(
                    f"Unknown config key [{section}].{key} in {config_toml}")
            cfg[key] = value
    for key in list(cfg):
        env = os.environ.get("SIXMAX_SEARCH_" + key.upper())
        if env is not None:
            cfg[key] = env
    return cfg


def _load_search_config(config_toml, section="search"):
    """Merge the [search] block into a sixmax.SearchConfig (ignoring the
    narrowing knobs, which the agent reads via _load_narrow_config)."""
    cfg = _merge_search_config(config_toml, section)
    out = sixmax.SearchConfig()
    out.enabled = _as_bool(cfg["enabled"])
    out.budget_ms = float(cfg["budget_ms"])
    out.max_nodes = int(cfg["max_nodes"])
    out.max_iters = int(cfg["max_iters"])
    out.leaf_samples = int(cfg["leaf_samples"])
    out.seed = int(cfg["seed"])
    out.leaf_threads = int(cfg["leaf_threads"])
    return out


def _load_narrow_config(config_toml, section="search"):
    """Python-side narrowing knobs, same precedence as _load_search_config."""
    cfg = _merge_search_config(config_toml, section)
    return {
        "narrow_preflop": _as_bool(cfg["narrow_preflop"]),
        "prune_keep_mass": float(cfg["prune_keep_mass"]),
    }


class SearchStrategy:
    """Thin host-side wrapper: translation and fallback live in C++."""

    def __init__(self, checkpoint, config_toml, section="blueprint"):
        vocab = _load_vocab(config_toml, "search")
        blueprint_vocab = _load_vocab(config_toml, section)
        if vocab.hash() != blueprint_vocab.hash():
            raise ValueError(
                "search vocab must match the blueprint vocab "
                f"(search={vocab.hash()} blueprint={blueprint_vocab.hash()})")
        bp = sixmax.BlueprintStrategy.load(checkpoint, vocab)
        if not _is_legacy_pot_layout(bp.abstraction().pot_layout()):
            raise RuntimeError(
                "search does not support non-legacy pot layouts")
        self._cfg = _load_search_config(config_toml)
        self._impl = sixmax.SearchStrategy(
            bp, vocab, sixmax.EngineConfig(num_players=2), self._cfg)
        self._vocab = vocab

    @property
    def enabled(self):
        return bool(self._cfg.enabled)

    def probs(self, *args, **kwargs):
        return list(self._impl.probs(*args, **kwargs))


class SixmaxAgent(PokerAgent):
    """Live-engine adapter over SixmaxDeployStrategy. Computes the legal mask,
    (live, after), and BB rescale from GameState; translates the blueprint's
    raise-to (BB) into an engine (Action, chips).

    game_state.raises_per_street is now faithful (Step 1 uncapped the engine),
    so we clamp it to the abstraction's cap (3) here at consumption — the same
    pattern neural_cfr uses (it clamps to 2). No observer plumbing needed."""

    def __init__(self, checkpoint_path, config_toml=_DEFAULT_TOML, search_toml=None):
        self._deploy = SixmaxDeployStrategy.load(checkpoint_path, config_toml)
        self._rng = random.Random()
        self._search = (SearchStrategy(checkpoint_path, search_toml)
                        if search_toml is not None else None)
        self._narrow = (_load_narrow_config(search_toml)
                        if search_toml is not None else None)
        self._hero = None

    @classmethod
    def from_strategy(cls, strategy, config_toml=_DEFAULT_TOML):
        """Build an agent over an already-loaded strategy (shares the loaded
        checkpoint; each agent gets its own decision RNG)."""
        agent = cls.__new__(cls)
        agent._deploy = strategy
        agent._rng = random.Random()
        agent._search = None
        agent._narrow = None
        agent._hero = None
        return agent

    def get_action(self, player, game_state):
        self._hero = player
        bb = game_state.big_blind
        players = game_state.players
        n = len(players)
        hero = players.index(player)
        street = game_state.betting_round

        hole = [_card_to_int(c) for c in player.hole_cards]
        board = [_card_to_int(c) for c in game_state.community_cards]

        folded = [not p.is_active for p in players]
        all_in = [p.is_all_in for p in players]
        live, after = canonical_live_after(n, game_state.button_pos, hero,
                                           folded, all_in, street)

        to_call_chips = game_state.current_bet - player.current_bet
        pot_bb = game_state.pot / bb
        current_bet_bb = game_state.current_bet / bb
        to_call_bb = to_call_chips / bb
        stack_bb = (player.current_bet + player.stack) / bb  # all-in target
        other_stacks = [p.stack for p in players
                        if p is not player and p.is_active]
        eff_stack_bb = min(player.stack, max(other_stacks, default=0)) / bb

        legal = self._legal_mask(game_state, player, to_call_chips,
                                 current_bet_bb, pot_bb, to_call_bb, stack_bb)

        if self._search is not None and self._search.enabled and street != 0:
            # Reconstruct the street root from the real action line and solve the
            # subgame against the preflop-narrowed, pruned opponent range. Any
            # failure here leaves p empty -> fall through to the blueprint, so
            # search is "never worse than the blueprint by construction".
            try:
                engine_state, prefix = self._replay_to_street_root(
                    game_state, hole, board)
                opp_prior = []
                if self._narrow and self._narrow["narrow_preflop"]:
                    raw = self._opponent_prior(game_state, hole, board)
                    opp_prior = self._prune_prior(
                        raw, self._narrow["prune_keep_mass"])
                p = self._search.probs(engine_state, 0, prefix, opp_prior=opp_prior)
            except Exception:
                p = []
            if p:
                idx = self._sample_search(p, legal)
                if idx is not None:
                    action = self._deploy._vocab.at(idx)
                    if action.type in (sixmax.ActionType.Bet,
                                       sixmax.ActionType.AllIn):
                        ctx = sixmax.BetContext(pot=pot_bb,
                                                current_bet=current_bet_bb,
                                                to_call=to_call_bb,
                                                stack=stack_bb)
                        raise_to_bb = self._deploy._vocab.target_bb(idx, ctx)
                    else:
                        raise_to_bb = 0.0
                    return self._translate(idx, to_call_chips, raise_to_bb, bb,
                                           player)

        raises = [min(r, _RAISE_CAP) for r in game_state.raises_per_street]
        idx, raise_to_bb = self._deploy.decide(
            hole=hole, board=board, street=street,
            raises_per_street=raises, pot_bb=pot_bb,
            current_bet_bb=current_bet_bb, to_call_bb=to_call_bb,
            stack_bb=stack_bb, eff_stack_bb=eff_stack_bb, live=live,
            after=after, legal=legal, rng=self._rng)
        return self._translate(idx, to_call_chips, raise_to_bb, bb, player)

    def _action_to_vocab(self, state, action, amount, bb):
        """Map a live (Action, amount) at EngineGameState `state` to a vocab
        index. FOLD/CHECK/CALL are the fixed indices 0/1/2; BET/RAISE quantise
        the live raise-to (in bb) to the nearest vocab entry. u=0.0 makes the
        pseudo-harmonic pick deterministic, which replay requires."""
        from models.enums import Action
        if action == Action.FOLD:
            return 0
        if action == Action.CHECK:
            return 1
        if action == Action.CALL:
            return 2
        ctx = state.bet_context()
        return self._deploy._vocab.nearest((amount or 0) / bb, ctx, 0.0)

    def _replay_to_street_root(self, game_state, hole, board):
        """Faithful EngineGameState at the START of the current betting round,
        plus the current street's observed_actions (vocab indices).

        Hero is engine seat 0; the engine button is derived from the live button
        so preflop action order matches. Starting stacks are each seat's pre-hand
        stack (stack + total_contributed) in the engine's BB=1 frame, so the
        replayed pot equals the live pot. HU only (seat 1 is the sole opponent)."""
        import sixmax
        players = game_state.players
        n = len(players)
        bb = game_state.big_blind
        street = game_state.betting_round

        hero_idx = players.index(self._hero)
        seat_live = [hero_idx] + [i for i in range(n) if i != hero_idx]

        used = set(hole) | set(board)
        fillers = (c for c in range(52) if c not in used)
        deck = list(range(52))
        deck[0], deck[1] = hole[0], hole[1]            # hero = engine seat 0
        for i in range(1, n):
            deck[2 * i] = next(fillers)
            deck[2 * i + 1] = next(fillers)
        for j, c in enumerate(board):
            deck[2 * n + j] = c
        for j in range(len(board), 5):
            deck[2 * n + j] = next(fillers)

        cfg = sixmax.EngineConfig(num_players=n)
        # Per-seat pre-hand stacks in BB (engine frame is BB=1), engine-seat order.
        stacks = [float(players[li].stack + players[li].total_contributed) / bb
                  for li in seat_live]
        button = (game_state.button_pos - hero_idx) % n
        eng = sixmax.EngineGameState(cfg, button, deck, self._deploy._vocab, stacks)

        prefix = []
        for (_pidx, action, amount, rnd) in game_state.action_history:
            if eng.is_terminal():
                break
            idx = self._action_to_vocab(eng, action, amount, bb)
            if rnd < street:
                # A faithful replay only applies actions legal at the engine
                # state; an illegal index means the live line desynced from the
                # engine (e.g. an unrepresentable quantized raise), so bail and
                # let get_action fall back to the blueprint rather than misread.
                if not eng.legal_mask()[idx]:
                    raise ValueError("replay desync: action illegal at engine state")
                eng.apply(idx)              # advance toward the street root
            elif rnd == street:
                prefix.append(idx)          # in-street prefix: collected, not applied
        return eng, prefix

    def _opponent_prior(self, game_state, hole, board):
        """Opponent range prior over the 1326 combos from the preflop+earlier
        line: for each combo unblocked by `hole`/`board`, the product of the
        blueprint's probability of each observed opponent action with that combo.
        Blocked combos are exactly 0.0. No opponent pre-street decision -> uniform
        over unblocked. Returns a normalised length-1326 list."""
        import sixmax
        hero_idx = game_state.players.index(self._hero)
        street = game_state.betting_round
        opp_decisions = any(
            rnd < street and pidx != hero_idx
            for (pidx, _a, _amt, rnd) in game_state.action_history)
        blocked = set(hole) | set(board)
        prior = [0.0] * 1326
        for c0 in range(52):
            if c0 in blocked:
                continue
            for c1 in range(c0 + 1, 52):
                if c1 in blocked:
                    continue
                idx = sixmax.RangeTracker.combo_index(c0, c1)
                prior[idx] = (self._combo_likelihood(game_state, hole, board, (c0, c1))
                              if opp_decisions else 1.0)
        total = sum(prior)
        if total <= 0.0:
            # No information: uniform over unblocked combos.
            for c0 in range(52):
                if c0 in blocked:
                    continue
                for c1 in range(c0 + 1, 52):
                    if c1 in blocked:
                        continue
                    prior[sixmax.RangeTracker.combo_index(c0, c1)] = 1.0
            total = sum(prior)
        return [p / total for p in prior]

    def _combo_likelihood(self, game_state, hole, board, opp_combo):
        """Product over the opponent's pre-current-street decisions of the
        blueprint P(observed action | opponent holds opp_combo), replayed into a
        per-combo engine state so probs_for buckets that combo."""
        import sixmax
        players = game_state.players
        n = len(players)
        bb = game_state.big_blind
        street = game_state.betting_round
        hero_idx = players.index(self._hero)
        seat_live = [hero_idx] + [i for i in range(n) if i != hero_idx]

        used = set(hole) | set(board) | set(opp_combo)
        fillers = (c for c in range(52) if c not in used)
        deck = list(range(52))
        deck[0], deck[1] = hole[0], hole[1]            # hero = engine seat 0
        deck[2], deck[3] = opp_combo[0], opp_combo[1]  # opponent = engine seat 1
        for i in range(2, n):
            deck[2 * i] = next(fillers)
            deck[2 * i + 1] = next(fillers)
        for j, c in enumerate(board):
            deck[2 * n + j] = c
        for j in range(len(board), 5):
            deck[2 * n + j] = next(fillers)

        cfg = sixmax.EngineConfig(num_players=n)
        stacks = [float(players[li].stack + players[li].total_contributed) / bb
                  for li in seat_live]
        button = (game_state.button_pos - hero_idx) % n
        eng = sixmax.EngineGameState(cfg, button, deck, self._deploy._vocab, stacks)

        like = 1.0
        for (pidx, action, amount, rnd) in game_state.action_history:
            if rnd >= street or eng.is_terminal():
                break
            idx = self._action_to_vocab(eng, action, amount, bb)
            if not eng.legal_mask()[idx]:
                # Desync (as in _replay_to_street_root): abandon this combo's
                # likelihood; the caller's prior still normalises over the rest.
                raise ValueError("likelihood replay desync: illegal action")
            if pidx != hero_idx:
                probs = self._deploy._strategy.probs_for(eng)
                like *= probs[idx] if idx < len(probs) else 0.0
            eng.apply(idx)
        return like

    @staticmethod
    def _prune_prior(prior, keep_mass):
        """Zero every combo outside the smallest set whose cumulative mass
        reaches `keep_mass`, then renormalise. keep_mass >= 1.0 keeps all."""
        order = sorted(range(len(prior)), key=lambda i: prior[i], reverse=True)
        out = [0.0] * len(prior)
        acc = 0.0
        for i in order:
            if prior[i] <= 0.0:
                break
            out[i] = prior[i]
            acc += prior[i]
            if acc >= keep_mass:
                break
        total = sum(out)
        return [v / total for v in out] if total > 0 else list(prior)

    def _reconstruct_search_state(self, game_state, hole, board):
        """Best-effort EngineGameState at the current postflop decision.

        Same deck-reconstruction technique as DreamDeployStrategy; the hero is
        engine seat 0. We advance a fresh hand to the live street and then to the
        hero's turn with checks. Chip amounts are approximate (a canonical
        check-down pot), which v1 search tolerates."""
        players = game_state.players
        n = len(players)
        used = set(hole) | set(board)
        placeholders = (c for c in range(52) if c not in used)
        deck = list(range(52))
        deck[0], deck[1] = hole[0], hole[1]
        for i in range(1, n):
            deck[2 * i] = next(placeholders)
            deck[2 * i + 1] = next(placeholders)
        for j, c in enumerate(board):
            deck[2 * n + j] = c
        for j in range(len(board), 5):
            deck[2 * n + j] = next(placeholders)
        cfg = sixmax.EngineConfig(num_players=n)
        stacks = [float(p.stack + p.current_bet) for p in players]
        eng = sixmax.EngineGameState(cfg, 0, deck, self._deploy._vocab, stacks)
        guard = 0
        # Repeated CALL is CheckCall when not facing a bet; it advances a street
        # once every player has acted.
        while (eng.street() < game_state.betting_round and not eng.is_terminal()
               and guard < 256):
            eng.apply(2)   # CALL
            guard += 1
        while (not eng.is_terminal() and eng.current_player() != 0
               and guard < 512):
            eng.apply(1)   # CHECK
            guard += 1
        return eng

    def _sample_search(self, probs, legal):
        """Sample a vocab index over the live-legal support of `probs`."""
        idxs = [i for i, m in enumerate(legal)
                if m and i < len(probs) and probs[i] > 0.0]
        if not idxs:
            return None
        total = sum(probs[i] for i in idxs)
        r = self._rng.random() * total
        acc = 0.0
        for i in idxs:
            acc += probs[i]
            if r <= acc:
                return i
        return idxs[-1]

    def _legal_mask(self, game_state, player, to_call_chips, current_bet_bb,
                    pot_bb, to_call_bb, stack_bb):
        """Mirror EngineGameState::legal_mask on live state. Bet entries are
        legal when their vocab target lands in [min_raise, all_in) BB."""
        vocab = self._deploy._vocab
        facing = to_call_chips > 1e-9
        can_raise = player.stack > to_call_chips + 1e-9
        street = game_state.betting_round
        unopened_preflop = street == 0 and current_bet_bb <= 1.0 + 1e-9
        min_raise_to_bb = (game_state.current_bet + max(
            game_state.big_blind, to_call_chips)) / game_state.big_blind
        ctx = sixmax.BetContext(pot=pot_bb, current_bet=current_bet_bb,
                                to_call=to_call_bb, stack=stack_bb)
        mask = [0] * vocab.size()
        for i in range(vocab.size()):
            a = vocab.at(i)
            if a.type == sixmax.ActionType.Fold:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Check:
                mask[i] = 0 if facing else 1
            elif a.type == sixmax.ActionType.Call:
                mask[i] = 1 if facing else 0
            elif a.type == sixmax.ActionType.Bet:
                bb_unit = a.unit == sixmax.SizeUnit.BB
                if not can_raise or bb_unit != unopened_preflop:
                    continue
                t = vocab.target_bb(i, ctx)
                mask[i] = 1 if (t >= min_raise_to_bb - 1e-9
                                and t < stack_bb - 1e-9) else 0
            elif a.type == sixmax.ActionType.AllIn:
                mask[i] = 1 if can_raise else 0
        return mask

    def _translate(self, idx, to_call_chips, raise_to_bb, bb, player):
        a = self._deploy._vocab.at(idx)
        if a.type == sixmax.ActionType.Fold:
            return Action.FOLD, None
        if a.type == sixmax.ActionType.Check:
            return Action.CHECK, None
        if a.type == sixmax.ActionType.Call:
            return Action.CALL, None
        # Bet / AllIn: game/poker.py:139 treats Action.BET amount as raise-TO
        # (the total street commitment; additional = amount - player.current_bet).
        # This matches CFRAgent._translate, whose all-in returns
        # stack + current_bet and whose bets cap at stack + current_bet.
        max_to = int(round(player.current_bet + player.stack))  # all-in raise-to
        if a.type == sixmax.ActionType.AllIn:
            return Action.BET, max_to
        raise_to = int(round(raise_to_bb * bb))
        return Action.BET, max(0, min(raise_to, max_to))
