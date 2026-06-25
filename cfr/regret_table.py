import pickle
from typing import Dict, List
import numpy as np
from cfr.info_set import InfoSet

ALL_ACTIONS = ["fold", "check", "call", "b0.5", "b1.0", "allin"]
_ACTION_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}
N_ACTIONS = len(ALL_ACTIONS)


class RegretTable:
    def __init__(self):
        self.regrets: Dict[InfoSet, np.ndarray] = {}
        self.strategy: Dict[InfoSet, np.ndarray] = {}

    def _ensure(self, infoset: InfoSet) -> None:
        if infoset not in self.regrets:
            self.regrets[infoset] = np.zeros(N_ACTIONS)
        if infoset not in self.strategy:
            self.strategy[infoset] = np.zeros(N_ACTIONS)

    def _mask(self, actions: List[str]) -> np.ndarray:
        mask = np.zeros(N_ACTIONS)
        for a in actions:
            mask[_ACTION_IDX[a]] = 1.0
        return mask

    def get_strategy(self, infoset: InfoSet, actions: List[str]) -> np.ndarray:
        """Current strategy via regret matching over legal actions."""
        self._ensure(infoset)
        mask = self._mask(actions)
        pos = np.maximum(self.regrets[infoset], 0.0) * mask
        total = pos.sum()
        if total > 0:
            probs_full = pos / total
        else:
            probs_full = mask / mask.sum()
        return probs_full[mask.astype(bool)]

    def update_regrets(
        self,
        infoset: InfoSet,
        action_values: Dict[str, float],
        node_value: float,
        actions: List[str],
    ) -> None:
        self._ensure(infoset)
        for a in actions:
            self.regrets[infoset][_ACTION_IDX[a]] += action_values[a] - node_value

    def accumulate_strategy(
        self, infoset: InfoSet, strategy: np.ndarray, actions: List[str]
    ) -> None:
        self._ensure(infoset)
        for prob, a in zip(strategy, actions):
            self.strategy[infoset][_ACTION_IDX[a]] += prob

    def get_average_strategy(self, infoset: InfoSet, actions: List[str]) -> np.ndarray:
        """Average strategy over legal actions — converges to Nash."""
        self._ensure(infoset)
        mask = self._mask(actions)
        masked = self.strategy[infoset] * mask
        total = masked.sum()
        if total > 0:
            probs_full = masked / total
        else:
            probs_full = mask / mask.sum()
        return probs_full[mask.astype(bool)]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"regrets": self.regrets, "strategy": self.strategy}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regrets = data["regrets"]
        self.strategy = data["strategy"]
