"""
Kuhn poker convergence gate for DREAM with stochastic external-sampling MCCFR.

Validates the same algorithm now implemented in DreamTrainer.cpp:
  - Updating player's node: enumerate ALL legal actions, recurse each, store
    instantaneous regrets adv_target[a] = v(a) - E_sigma[v] (unbiased).
  - Opponent node: sample one action from sigma (outcome-sampling style).
  - Two traversals per hand (alternating updating_player 0/1).
  - Advantage net reinitialised from scratch each cycle (DREAM protocol).
  - Strategy net accumulates across all cycles.

Uses 9-dim feature encoding (card one-hot + history one-hot + player one-hot).
Trains for 50K external-sampling MCCFR traversals (25K per player).
Expected strategy-net P0 EV ≈ -1/18 ≈ -0.0556 (tolerance ±0.02).

Standard Kuhn poker: 3 cards (J=0, Q=1, K=2), ante=1, bet=1.
Terminal histories (actions: 0=check/fold, 1=bet/call):
  [0, 0]       → check-check showdown          (win ±1)
  [0, 1, 0]    → check-bet-fold (p0 folds)     (p0 loses ante)
  [0, 1, 1]    → check-bet-call showdown        (win ±2)
  [1, 0]       → bet-fold (p1 folds)            (p0 wins ante)
  [1, 1]       → bet-call showdown              (win ±2)

History codes matching sixmax.kuhn_infoset_key convention:
  ROOT=0, AFTER_CHECK=1, AFTER_BET=2, AFTER_CHECK_BET=3

Implementation:
- External-sampling MCCFR: full tree traversal for the updating player,
  sampled single action for the opponent.  Stores instantaneous regrets
  (v(a) - E_sigma[v]) in the advantage reservoir — no baseline tricks needed.
- Pure Python torch.nn.Sequential MLP.  sixmax.DreamMLP uses a separate
  libtorch instance and its forward() cannot receive Python torch.Tensors
  across the pybind11 ABI boundary.
- Pure Python reservoir (Algorithm A-Res).  C++ reservoir thread-safety is
  tested in test_dream_reservoir.py.
- Advantage net reinitialised from scratch every 500 traversals (DREAM).
- Strategy net NOT reinitialised — accumulates across all retraining cycles.
"""

import random

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KUHN_FEATURE_DIM = 9   # 3 card one-hot + 4 history one-hot + 2 player one-hot
NUM_ACTIONS = 2         # 0 = check/fold, 1 = bet/call

J, Q, K = 0, 1, 2
ROOT, AFTER_CHECK, AFTER_BET, AFTER_CHECK_BET = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_kuhn(card: int, history_code: int, player: int) -> torch.Tensor:
    """9-dim feature: 3 card one-hot + 4 history one-hot + 2 player one-hot."""
    t = torch.zeros(KUHN_FEATURE_DIM)
    t[card] = 1.0
    t[3 + history_code] = 1.0
    t[7 + player] = 1.0
    return t


def history_to_code(history: list) -> int:
    """Convert action sequence to info-state code 0-3.

    Matches the sixmax.kuhn_infoset_key convention.
    """
    if not history:
        return ROOT
    if history == [0]:
        return AFTER_CHECK
    if history == [1]:
        return AFTER_BET
    if history == [0, 1]:
        return AFTER_CHECK_BET
    raise ValueError(f"unexpected history {history}")


# ---------------------------------------------------------------------------
# Kuhn poker game logic
# ---------------------------------------------------------------------------

def kuhn_is_terminal(history: list) -> bool:
    return tuple(history) in (
        (0, 0),       # check-check showdown
        (0, 1, 0),    # check-bet-fold
        (0, 1, 1),    # check-bet-call showdown
        (1, 0),       # bet-fold
        (1, 1),       # bet-call showdown
    )


def kuhn_utility(cards: list, history: list) -> list:
    """Returns [u0, u1] for a Kuhn terminal state. Ante=1, bet=1."""
    h = tuple(history)
    c0, c1 = cards
    winner = 0 if c0 > c1 else 1

    if h == (0, 0):
        return [1, -1] if winner == 0 else [-1, 1]
    if h == (0, 1, 0):
        return [-1, 1]    # p0 folded to p1's bet
    if h == (0, 1, 1):
        return [2, -2] if winner == 0 else [-2, 2]
    if h == (1, 0):
        return [1, -1]    # p1 folded to p0's bet
    if h == (1, 1):
        return [2, -2] if winner == 0 else [-2, 2]
    raise ValueError(f"unexpected terminal history {h}")


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def regret_match(advantages: list) -> list:
    """Regret matching over 2 legal actions."""
    pos = [max(0.0, a) for a in advantages]
    total = sum(pos)
    if total < 1e-9:
        return [0.5, 0.5]
    return [p / total for p in pos]


def eps_greedy(sigma: list, eps: float) -> list:
    """Epsilon-greedy exploration (uniform over both actions)."""
    n = len(sigma)
    u = eps / n
    return [(1.0 - eps) * s + u for s in sigma]


# ---------------------------------------------------------------------------
# Pure-Python reservoir (weighted reservoir sampling, Algorithm A-Res)
# ---------------------------------------------------------------------------

class PythonReservoir:
    """Weighted reservoir using Algorithm A-Res (Efraimidis & Spirakis 2006).

    Higher-weight items are exponentially more likely to survive; capacity is
    hard-capped.  Newer items have linearly higher weights (w = iteration t),
    so the reservoir approximates the time-weighted average training signal.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self._cap = capacity
        self._rng = random.Random(seed)
        self._items: list = []

    def add(self, feat: torch.Tensor, tgt: torch.Tensor, weight: float):
        if weight <= 0:
            return
        key = self._rng.random() ** (1.0 / weight)
        if len(self._items) < self._cap:
            self._items.append((key, feat, tgt, weight))
        else:
            min_idx = min(range(len(self._items)), key=lambda i: self._items[i][0])
            if key > self._items[min_idx][0]:
                self._items[min_idx] = (key, feat, tgt, weight)

    def sample_batch(self, n: int):
        """Return (feat_batch, tgt_batch, weight_batch) as torch tensors."""
        n = min(n, len(self._items))
        chosen = self._rng.sample(self._items, n)
        feats = torch.stack([c[1] for c in chosen])
        tgts = torch.stack([c[2] for c in chosen])
        weights = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        return feats, tgts, weights

    def size(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# MLP factory
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# External-sampling MCCFR traversal
# ---------------------------------------------------------------------------

def traverse(
    cards: list,
    history: list,
    updating_player: int,
    adv_net: nn.Module,
    rng: random.Random,
    M_v: PythonReservoir,
    M_pi: PythonReservoir,
    t: int,
    eps: float,
) -> float:
    """External-sampling MCCFR traversal (DREAM variant).

    For the updating player: visit ALL actions and compute exact instantaneous
    regrets (counterfactual value of each action minus expected value).
    For the opponent: sample a single action from the epsilon-greedy policy.

    Returns the expected value for the updating player at this node.
    """
    if kuhn_is_terminal(history):
        return float(kuhn_utility(cards, history)[updating_player])

    player = len(history) % 2
    hcode = history_to_code(history)
    feat = encode_kuhn(cards[player], hcode, player)

    with torch.no_grad():
        raw_adv = adv_net(feat.unsqueeze(0)).squeeze(0).tolist()

    sigma = regret_match(raw_adv)
    sigma_eps = eps_greedy(sigma, eps)

    if player == updating_player:
        # Full traversal for updating player: compute value for each action
        action_values = [
            traverse(cards, history + [a], updating_player, adv_net, rng,
                     M_v, M_pi, t, eps)
            for a in range(NUM_ACTIONS)
        ]
        expected_v = sum(sigma[a] * action_values[a] for a in range(NUM_ACTIONS))

        # Instantaneous regret = counterfactual value of action - expected value.
        # These are the advantage targets for the advantage net.
        adv_target = torch.tensor(
            [action_values[a] - expected_v for a in range(NUM_ACTIONS)],
            dtype=torch.float32,
        )
        M_v.add(feat, adv_target, float(t))
        M_pi.add(feat, torch.tensor(sigma_eps, dtype=torch.float32), float(t))
        return expected_v

    else:
        # Opponent: sample single action (external sampling)
        a_opp = rng.choices(range(NUM_ACTIONS), weights=sigma_eps)[0]
        M_pi.add(feat, torch.tensor(sigma_eps, dtype=torch.float32), float(t))
        return traverse(cards, history + [a_opp], updating_player, adv_net, rng,
                        M_v, M_pi, t, eps)


# ---------------------------------------------------------------------------
# Main convergence test
# ---------------------------------------------------------------------------

def test_dream_kuhn_convergence():
    """
    50K external-sampling MCCFR traversals (alternating players) on Kuhn poker.

    Advantage net reinitialised every 500 traversals (DREAM protocol).
    Strategy net accumulates.  After 50K traversals, the strategy net's policy
    is evaluated by exact enumeration of all 6 Kuhn deals.

    Gate: |ev_p0 - (-1/18)| < 0.02
    """
    torch.manual_seed(0)
    rng = random.Random(0)

    HIDDEN = 32
    TRAIN_INTERVAL = 500
    BATCH = 256
    SGD_STEPS = 500
    LR = 1e-3
    EPS = 0.06
    N_ITER = 50_000
    RESERVOIR_CAP = 200_000

    adv_net = make_mlp(KUHN_FEATURE_DIM, HIDDEN, NUM_ACTIONS)
    strat_net = make_mlp(KUHN_FEATURE_DIM, HIDDEN, NUM_ACTIONS)

    M_v = PythonReservoir(RESERVOIR_CAP, seed=42)
    M_pi = PythonReservoir(RESERVOIR_CAP, seed=43)

    for t in range(1, N_ITER + 1):
        cards = rng.sample([J, Q, K], 2)
        updating_player = (t - 1) % 2   # alternate P0 / P1 each traversal

        traverse(cards, [], updating_player, adv_net, rng, M_v, M_pi, t, EPS)

        if t % TRAIN_INTERVAL == 0 and M_v.size() >= BATCH:
            # Advantage net: reinitialise from scratch (DREAM protocol)
            adv_net = make_mlp(KUHN_FEATURE_DIM, HIDDEN, NUM_ACTIONS)
            opt_adv = torch.optim.Adam(adv_net.parameters(), lr=LR)
            for _ in range(SGD_STEPS):
                feat_b, tgt_b, w_b = M_v.sample_batch(BATCH)
                opt_adv.zero_grad()
                pred = adv_net(feat_b)
                loss = ((pred - tgt_b).pow(2) * w_b.unsqueeze(1)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(adv_net.parameters(), 1.0)
                opt_adv.step()

            # Strategy net: NOT reinitialised — accumulates across all cycles
            opt_strat = torch.optim.Adam(strat_net.parameters(), lr=LR)
            for _ in range(SGD_STEPS):
                feat_b, tgt_b, w_b = M_pi.sample_batch(BATCH)
                opt_strat.zero_grad()
                log_probs = torch.log_softmax(strat_net(feat_b), dim=1)
                loss = -(tgt_b * log_probs * w_b.unsqueeze(1)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(strat_net.parameters(), 1.0)
                opt_strat.step()

    # ---------------------------------------------------------------------------
    # Evaluate: compute exact P0 EV under strategy net's policy by enumeration
    # ---------------------------------------------------------------------------
    def strat_probs(card: int, hcode: int, player: int) -> list:
        feat = encode_kuhn(card, hcode, player)
        with torch.no_grad():
            logits = strat_net(feat.unsqueeze(0)).squeeze(0)
            return torch.softmax(logits, dim=0).tolist()

    total_ev = 0.0
    for c0 in range(3):
        for c1 in range(3):
            if c0 == c1:
                continue

            s0_root = strat_probs(c0, ROOT, 0)
            ev = 0.0

            # P0 checks (action 0)
            p_check = s0_root[0]
            s1_check = strat_probs(c1, AFTER_CHECK, 1)
            # check-check
            ev += p_check * s1_check[0] * kuhn_utility([c0, c1], [0, 0])[0]
            # check-bet: P0 acts again
            p_cb = p_check * s1_check[1]
            s0_cbr = strat_probs(c0, AFTER_CHECK_BET, 0)
            ev += p_cb * s0_cbr[0] * kuhn_utility([c0, c1], [0, 1, 0])[0]
            ev += p_cb * s0_cbr[1] * kuhn_utility([c0, c1], [0, 1, 1])[0]

            # P0 bets (action 1)
            p_bet = s0_root[1]
            s1_bet = strat_probs(c1, AFTER_BET, 1)
            ev += p_bet * s1_bet[0] * kuhn_utility([c0, c1], [1, 0])[0]
            ev += p_bet * s1_bet[1] * kuhn_utility([c0, c1], [1, 1])[0]

            total_ev += ev

    ev_p0 = total_ev / 6.0  # average over 6 equally-likely ordered deals

    nash_value = -1.0 / 18.0   # closed-form Nash value for P0 ≈ -0.0556
    tolerance = 0.02
    assert abs(ev_p0 - nash_value) < tolerance, (
        f"Kuhn P0 EV = {ev_p0:.4f}, expected Nash {nash_value:.4f} ± {tolerance}. "
        f"Check traversal logic, IS weight formula, or retraining loop."
    )
