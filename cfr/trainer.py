import os
import glob
import time
from typing import Optional

from tqdm import tqdm

from cfr.regret_table import RegretTable
from cfr.mccfr import external_sample, compute_exploitability
from cfr.abstract_state import deal_heads_up


class Trainer:
    def __init__(self, table: Optional[RegretTable] = None):
        self.table = table or RegretTable()
        self.iterations_done = 0

    def train(
        self,
        num_iterations: int,
        checkpoint_interval: int = 100_000,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        pbar = tqdm(total=num_iterations, unit="iter", desc="MCCFR")
        start = time.time()

        for i in range(num_iterations):
            state = deal_heads_up()
            external_sample(state, traversing_player=0, table=self.table)
            external_sample(state, traversing_player=1, table=self.table)
            self.iterations_done += 1

            if checkpoint_interval and (self.iterations_done % checkpoint_interval == 0):
                expl = compute_exploitability(self.table, num_samples=1_000)
                elapsed = time.time() - start
                pbar.set_postfix(
                    expl_mbbh=f"{expl:.1f}",
                    elapsed=f"{elapsed:.0f}s",
                    infosets=len(self.table.regrets),
                )
                if checkpoint_dir:
                    path = os.path.join(
                        checkpoint_dir, f"checkpoint_{self.iterations_done:08d}.pkl"
                    )
                    self.table.save(path)

            pbar.update(1)

        pbar.close()

    def load_latest(self, checkpoint_dir: str) -> None:
        files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
        self.table.load(files[-1])
