import os
import re
import sys
import glob
import time
import random
import multiprocessing as mp
from typing import Optional

from tqdm import tqdm

from cfr.regret_table import RegretTable
from cfr.mccfr import external_sample, compute_exploitability
from cfr.abstract_state import deal_heads_up
try:
    from cfr.mccfr_cy import external_sample_cy as _external_sample
except ImportError:
    _external_sample = external_sample
from cfr.abstraction import prewarm_preflop_buckets


# ---------- multiprocessing worker (top-level so pickle can find it) ----------

def _worker_init():
    """Called once per worker process at pool startup."""
    prewarm_preflop_buckets()


def _worker_chunk(args):
    """Run n_iters of MCCFR in this worker and return the local RegretTable."""
    n_iters, seed = args
    random.seed(seed)
    rt = RegretTable()
    for _ in range(n_iters):
        state = deal_heads_up()
        _external_sample(state, traversing_player=0, table=rt)
        _external_sample(state, traversing_player=1, table=rt)
    return rt

_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pkl$")


def _parse_iterations(path: str) -> int:
    m = _CHECKPOINT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0


class Trainer:
    def __init__(self, table: Optional[RegretTable] = None):
        self.table = table or RegretTable()
        self.iterations_done = 0

    def load_checkpoint(self, path: str) -> None:
        self.table.load(path)
        self.iterations_done = _parse_iterations(path)

    def train(
        self,
        num_iterations: int,
        checkpoint_interval: int = 10_000,
        checkpoint_dir: Optional[str] = None,
        prewarm: bool = True,
    ) -> None:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if prewarm:
            print("Pre-warming preflop bucket cache (1326 combos)...", end=" ", flush=True)
            t0 = time.time()
            n = prewarm_preflop_buckets()
            print(f"done in {time.time()-t0:.1f}s  ({n} entries)")

        pbar = tqdm(
            total=self.iterations_done + num_iterations,
            initial=self.iterations_done,
            unit="iter",
            desc="MCCFR",
        )
        start = time.time()

        _DISPLAY_INTERVAL = 1_000

        for i in range(num_iterations):
            state = deal_heads_up()
            _external_sample(state, traversing_player=0, table=self.table)
            _external_sample(state, traversing_player=1, table=self.table)
            self.iterations_done += 1

            if self.iterations_done % _DISPLAY_INTERVAL == 0:
                pbar.set_postfix(infosets=len(self.table.regrets))

            if checkpoint_interval and (self.iterations_done % checkpoint_interval == 0):
                expl = compute_exploitability(self.table, num_samples=1_000, show_progress=True)
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

    def parallel_train(
        self,
        num_iterations: int,
        n_workers: int = 0,
        chunk_size: int = 20_000,
        checkpoint_interval: int = 100_000,
        checkpoint_dir: Optional[str] = None,
        prewarm: bool = True,
    ) -> None:
        """Train using N parallel worker processes.

        On macOS/Linux uses 'fork' so workers inherit the parent's warm
        lru_cache and the 255MB RANK7 table via OS copy-on-write — no per-worker
        disk I/O. Each round: N workers each run chunk_size iterations
        independently, return their local RegretTable, which is merged into
        self.table. Workers stay alive across rounds.
        """
        n_workers = n_workers or mp.cpu_count()
        iters_per_round = chunk_size * n_workers

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Prewarm once in the main process; fork inherits the warm cache.
        if prewarm:
            print("Pre-warming preflop bucket cache...", end=" ", flush=True)
            t0 = time.time()
            prewarm_preflop_buckets()
            print(f"done in {time.time()-t0:.1f}s")

            # Postflop warmup: run training iterations single-threaded so workers
            # inherit a warm postflop lru_cache via fork (zero-copy CoW).
            # ~10k iters reaches ~94% hit rate; workers start hot instead of cold.
            _N_POSTFLOP_WARMUP = 10_000
            print(f"Pre-warming postflop bucket cache ({_N_POSTFLOP_WARMUP:,} iters)...",
                  end=" ", flush=True)
            t0 = time.time()
            _rt_warm = RegretTable()
            for _ in range(_N_POSTFLOP_WARMUP):
                _s = deal_heads_up()
                _external_sample(_s, 0, _rt_warm)
                _external_sample(_s, 1, _rt_warm)
            del _rt_warm
            print(f"done in {time.time()-t0:.1f}s")

        use_fork = sys.platform != 'win32'
        ctx = mp.get_context('fork' if use_fork else 'spawn')
        init_fn = None if use_fork else _worker_init

        print(f"Parallel training: {n_workers} workers × {chunk_size:,} iters/chunk "
              f"= {iters_per_round:,} iters/round  [{'fork' if use_fork else 'spawn'}]")

        n_rounds = (num_iterations + iters_per_round - 1) // iters_per_round
        start = time.time()

        with ctx.Pool(n_workers, initializer=init_fn) as pool:
            pbar = tqdm(total=num_iterations, unit="iter", desc="Parallel MCCFR")
            iters_remaining = num_iterations
            for _ in range(n_rounds):
                this_chunk = min(chunk_size, (iters_remaining + n_workers - 1) // n_workers)
                args = [(this_chunk, random.randint(0, 2**31)) for _ in range(n_workers)]
                tables = pool.map(_worker_chunk, args)

                actual_iters = this_chunk * n_workers
                prev_done = self.iterations_done
                for t in tables:
                    self.table.merge(t)
                self.iterations_done += actual_iters
                iters_remaining -= actual_iters
                pbar.update(actual_iters)
                pbar.set_postfix(infosets=len(self.table.regrets))

                # Fire checkpoint whenever we cross a boundary, not just land on one.
                # iters_per_round rarely divides checkpoint_interval exactly.
                if checkpoint_interval and (
                    self.iterations_done // checkpoint_interval
                    > prev_done // checkpoint_interval
                ):
                    expl = compute_exploitability(self.table, num_samples=1_000, show_progress=False)
                    elapsed = time.time() - start
                    pbar.set_postfix(
                        expl_mbbh=f"{expl:.1f}",
                        elapsed=f"{elapsed:.0f}s",
                        infosets=len(self.table.regrets),
                    )
                    if checkpoint_dir:
                        path = os.path.join(checkpoint_dir, f"checkpoint_{self.iterations_done:08d}.pkl")
                        self.table.save(path)
            pbar.close()

    def load_latest(self, checkpoint_dir: str) -> None:
        files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
        self.load_checkpoint(files[-1])
