import pickle
from typing import Dict, List
import numpy as np
from cfr.info_set import InfoSet


class RegretTable:
    def __init__(self):
        self.regrets: Dict[InfoSet, np.ndarray] = {}
        self.strategy: Dict[InfoSet, np.ndarray] = {}

    def _ensure(self, infoset: InfoSet, actions: List[str]) -> None:
        n = len(actions)
        if infoset not in self.regrets:
            self.regrets[infoset] = np.zeros(n)
        if infoset not in self.strategy:
            self.strategy[infoset] = np.zeros(n)

    def get_strategy(self, infoset: InfoSet, actions: List[str]) -> np.ndarray:
        """Current strategy via regret matching. Returns probability array."""
        self._ensure(infoset, actions)
        pos = np.maximum(self.regrets[infoset], 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(len(actions)) / len(actions)

    def update_regrets(
        self,
        infoset: InfoSet,
        action_values: Dict[str, float],
        node_value: float,
        actions: List[str],
    ) -> None:
        self._ensure(infoset, actions)
        for idx, a in enumerate(actions):
            self.regrets[infoset][idx] += action_values[a] - node_value

    def accumulate_strategy(
        self, infoset: InfoSet, strategy: np.ndarray, actions: List[str]
    ) -> None:
        self._ensure(infoset, actions)
        self.strategy[infoset] += strategy

    def get_average_strategy(self, infoset: InfoSet, actions: List[str]) -> np.ndarray:
        """Average strategy — what converges to Nash."""
        self._ensure(infoset, actions)
        total = self.strategy[infoset].sum()
        if total > 0:
            return self.strategy[infoset] / total
        return np.ones(len(actions)) / len(actions)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"regrets": self.regrets, "strategy": self.strategy}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regrets = data["regrets"]
        self.strategy = data["strategy"]
