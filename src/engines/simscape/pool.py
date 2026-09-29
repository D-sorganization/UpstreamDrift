"""Pool of :class:`SimscapeAdapter` instances for parallel inference.

This module implements :class:`SimscapeAdapterPool`, the multi-engine
extension of the singleton MATLAB engine that ``_engine_pool.py``
provides for issue #4007. Each worker thread owns its own adapter and
its own MATLAB Engine, so N candidate-fit simulations can run truly in
parallel — the MATLAB Engine API for Python releases the GIL while a
``sim()`` is in flight.

Concurrency model
-----------------
* Threads, not processes. A ``matlab.engine.MatlabEngine`` handle is
  *not* picklable, so a :class:`concurrent.futures.ProcessPoolExecutor`
  cannot ferry tasks to a worker. ``ThreadPoolExecutor`` works because
  the engine call is native code that releases the GIL.
* Each worker thread owns exactly one :class:`SimscapeAdapter` (lazily
  started on first dispatch). Adapters are pinned to their thread and
  never migrate.
* The pool itself is safe to share across threads at the call-site
  (``simulate_batch`` / ``map`` are reentrant only across distinct
  pools — nesting a pool call inside another worker is **not**
  supported and would deadlock).

Sizing
------
* ``pool_size`` is bounded above by the host's MATLAB licence count.
  We do not check this at construction; an over-large pool will simply
  fail when a worker tries to start its engine and ``Simscape``
  refuses the licence.

Cache
-----
* Each worker has its own in-process LRU cache. Cross-worker caching
  is an explicit non-goal for v1 (see issue #4008 for details).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape._pool_worker import _PoolWorker
from src.engines.simscape.adapter import SimscapeAdapter
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import TracebackType

logger = get_logger(__name__)

__all__ = [
    "PoolConfig",
    "SimscapeAdapterPool",
]

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PoolConfig:
    """Tunable parameters for :class:`SimscapeAdapterPool`.

    Args:
        pool_size: Number of worker threads / MATLAB engines. Bounded
            above by the host's MATLAB licence count.
        cache_capacity_per_worker: LRU capacity passed into each
            worker's :class:`SimscapeAdapter`. ``0`` disables caching.
        startup_timeout_s: Wall-clock budget for each engine startup.
    """

    pool_size: int = 4
    cache_capacity_per_worker: int = 16
    startup_timeout_s: float = 90.0


class SimscapeAdapterPool:
    """Thread-pool of :class:`SimscapeAdapter` instances.

    Designed for batched inference workloads (dataset generator,
    MultiStart driven from Python, surrogate validation across N
    candidate fits).

    Usage::

        cfg = PoolConfig(pool_size=4)
        with SimscapeAdapterPool("GolfSwing3D_Kinetic.slx", cfg) as pool:
            outputs = pool.simulate_batch(coeffs_batch)

    The pool is process-safe but **not** nested-thread-safe: invoking
    :meth:`simulate_batch` from inside another worker thread of the
    same pool would deadlock and is unsupported.

    Args:
        model_path: Path to the ``.slx`` model that every worker loads.
        cfg: Pool configuration. Defaults to :class:`PoolConfig`.
        adapter_factory: Optional callable yielding a fresh adapter;
            primarily for testing with mocked adapters.

    Raises:
        TypeError: If ``model_path`` has the wrong type.
        ValueError: If ``model_path`` does not end in ``.slx`` or
            ``cfg.pool_size`` is not a positive int.
    """

    @precondition(
        lambda self, model_path, cfg=None, adapter_factory=None: (
            isinstance(model_path, (str, Path)) and str(model_path) != ""
        ),
        "model_path must be a non-empty path-like",
    )
    def __init__(
        self,
        model_path: Path | str,
        cfg: PoolConfig | None = None,
        adapter_factory: type[SimscapeAdapter] | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else PoolConfig()
        if not isinstance(cfg.pool_size, int) or cfg.pool_size < 1:
            raise ValueError(
                f"PoolConfig.pool_size must be a positive int; got {cfg.pool_size!r}"
            )
        if not isinstance(cfg.cache_capacity_per_worker, int) or (
            cfg.cache_capacity_per_worker < 0
        ):
            raise ValueError(
                "PoolConfig.cache_capacity_per_worker must be a non-negative int"
            )
        path = Path(model_path)
        if path.suffix.lower() != ".slx":
            raise ValueError(f"model_path must end in .slx; got '{path.suffix}'")

        self._model_path: str = str(path)
        self._cfg: PoolConfig = cfg
        self._factory: type[SimscapeAdapter] | None = adapter_factory
        self._workers: list[_PoolWorker] = [
            _PoolWorker(
                model_path=self._model_path,
                cache_capacity=cfg.cache_capacity_per_worker,
                startup_timeout_s=cfg.startup_timeout_s,
                adapter_factory=adapter_factory,
            )
            for _ in range(cfg.pool_size)
        ]
        # Round-robin assignment of workers to spawned threads. We use a
        # thread-local so each pool-thread always picks the same worker.
        self._tls: local = local()
        self._next_worker_index: int = 0
        self._assign_lock: Lock = Lock()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=cfg.pool_size,
            thread_name_prefix="simscape-pool",
        )
        self._closed: bool = False
        logger.info(
            "SimscapeAdapterPool created (pool_size=%d, model=%s)",
            cfg.pool_size,
            path.name,
        )

    @property
    def pool_size(self) -> int:
        """Number of worker threads / engines."""
        return self._cfg.pool_size

    def __enter__(self) -> SimscapeAdapterPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def simulate_batch(self, coeffs_batch: np.ndarray) -> list[SimscapeOutput]:
        """Run one simulation per row of ``coeffs_batch`` in parallel.

        Args:
            coeffs_batch: ``(B, n_joints*7)`` float array. Each row is
                forwarded to :meth:`SimscapeAdapter.simulate_with_coefficients`.

        Returns:
            List of :class:`SimscapeOutput`, one per input row, in the
            input row order.

        Raises:
            ValueError: If ``coeffs_batch`` is not a 2-D array whose
                column count is a multiple of 7.
            RuntimeError: If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        if not isinstance(coeffs_batch, np.ndarray):
            raise TypeError(
                f"coeffs_batch must be np.ndarray; got {type(coeffs_batch).__name__}"
            )
        if coeffs_batch.ndim != 2 or coeffs_batch.shape[1] == 0:
            raise ValueError(
                f"coeffs_batch must be 2-D with non-zero width; got {coeffs_batch.shape}"
            )
        if coeffs_batch.shape[1] % 7 != 0:
            raise ValueError(
                "coeffs_batch.shape[1] must be a multiple of 7 "
                f"(n_joints*7); got {coeffs_batch.shape[1]}"
            )

        rows = [np.ascontiguousarray(row, dtype=np.float64) for row in coeffs_batch]
        return self.map(
            lambda adapter, coeffs: adapter.simulate_with_coefficients(coeffs),
            rows,
        )

    def map(
        self,
        fn: Callable[[SimscapeAdapter, T], R],
        items: list[T],
    ) -> list[R]:
        """General-purpose worker dispatch preserving input order.

        Args:
            fn: Callable invoked as ``fn(adapter, item)`` on a worker
                thread. Must be thread-safe with respect to its closure.
            items: Inputs to dispatch.

        Returns:
            Results in the same order as ``items``.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        if not items:
            return []

        futures: list[Future[R]] = [
            self._executor.submit(self._invoke, fn, item) for item in items
        ]
        # Preserve input order by iterating futures in submission order.
        results: list[R] = []
        for fut in futures:
            results.append(fut.result())
        return results

    def close(self) -> None:
        """Quit every worker engine and shut down the executor.

        Idempotent. After ``close`` returns, subsequent
        :meth:`simulate_batch` / :meth:`map` calls raise
        :class:`RuntimeError`.
        """
        if self._closed:
            return
        self._closed = True
        # Stop accepting new work first.
        self._executor.shutdown(wait=True, cancel_futures=True)
        # Then quit every worker (engine.quit() runs on the calling
        # thread; we accept that it serialises here because shutdown is
        # not on the hot path).
        for worker in self._workers:
            worker.close()
        logger.info("SimscapeAdapterPool closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke(
        self,
        fn: Callable[[SimscapeAdapter, T], R],
        item: T,
    ) -> R:
        """Resolve the thread-local worker and dispatch ``fn``.

        Each pool thread caches its assigned worker on a
        :class:`threading.local` so the same engine handles every task
        landing on that thread.
        """
        worker = self._worker_for_current_thread()
        return fn(worker.adapter, item)

    def _worker_for_current_thread(self) -> _PoolWorker:
        """Return (and lazily assign) the worker pinned to this thread."""
        worker = getattr(self._tls, "worker", None)
        if worker is not None:
            return worker  # type: ignore[no-any-return]
        with self._assign_lock:
            idx = self._next_worker_index
            self._next_worker_index += 1
            if idx >= len(self._workers):  # pragma: no cover - defensive
                idx = idx % len(self._workers)
        worker = self._workers[idx]
        self._tls.worker = worker
        return worker
