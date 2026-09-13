"""Optional persistent execution for independent native shooting windows.

The caller supplies the existing evaluation contract as a module-level callable.
Requests are immutable bytes containing trusted internal transport only; this
module neither serializes nor deserializes them. Never deserialize untrusted
pickle requests. Results retain the caller's type and submission order. There
is no solver, cache, dynamics, or residual assembly in this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing
import os
from typing import Generic, TypeVar

Result = TypeVar("Result")


class WindowEvaluationError(RuntimeError):
    """An indexed window failed; the original exception is chained as cause."""

    def __init__(self, window_index: int) -> None:
        self.window_index = window_index
        super().__init__(f"Native window {window_index} evaluation failed")


class NativeWindowExecutor(Generic[Result]):
    """Evaluate batches sequentially or with two persistent spawned workers.

    workers=0 is the explicit sequential fallback. workers=2 requires the caller
    to set BLAS/OMP thread limits before construction and preserve them while
    workers start. The callable and results must support multiprocessing pickle
    transport. Use from the application's guarded main entry point.

    Own this object in one coordinating thread. A failed batch closes the pool,
    cancels queued futures, and waits for running evaluations to finish before
    raising. There is deliberately no unsafe forced termination or silent retry;
    a hung evaluator requires external process supervision. close is idempotent.
    """

    def __init__(
        self, evaluate: Callable[[bytes], Result], *, workers: int = 0
    ) -> None:
        if type(workers) is not int or workers not in (0, 2):
            raise ValueError("workers must be 0 (sequential) or 2")
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        if workers:
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
                if os.environ.get(name) != "1":
                    raise ValueError(f"{name}=1 is required before starting workers")
        self._evaluate = evaluate
        self._closed = False
        self._pool = (
            ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            if workers
            else None
        )

    def __enter__(self) -> NativeWindowExecutor[Result]:
        if self._closed:
            raise RuntimeError("Native window executor is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Wait for running work and release all owned worker processes."""
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)

    def evaluate(self, requests: Sequence[bytes]) -> tuple[Result, ...]:
        """Return a complete ordered batch or raise without partial results."""
        if self._closed:
            raise RuntimeError("Native window executor is closed")
        batch = tuple(requests)
        if any(type(request) is not bytes for request in batch):
            raise TypeError("Window requests must be immutable bytes")
        index = 0
        futures: dict[Future[Result], int] = {}
        try:
            if self._pool is None:
                results = []
                for index, request in enumerate(batch):  # noqa: B007 - retained for failure index
                    results.append(self._evaluate(request))
                return tuple(results)
            for index, request in enumerate(batch):
                futures[self._pool.submit(self._evaluate, request)] = index
            ordered = {}
            for future in as_completed(futures):
                index = futures[future]
                ordered[index] = future.result()
            return tuple(ordered[index] for index in range(len(batch)))
        except BaseException as error:
            for future in futures:
                future.cancel()
            self.close()
            if isinstance(error, Exception):
                raise WindowEvaluationError(index) from error
            raise
