"""Per-thread worker that owns one :class:`SimscapeAdapter` instance.

Each worker is bound to exactly one Python thread inside the
:class:`~concurrent.futures.ThreadPoolExecutor` driving
:class:`src.engines.simscape.pool.SimscapeAdapterPool`. The MATLAB
Engine API for Python releases the GIL while a MATLAB call is in
flight, so threads (rather than subprocesses) are sufficient to exploit
N independent engines without paying the pickling tax that processes
would incur — the engine handle itself is not picklable.

Lifecycle:
    * The worker is constructed on the dispatching thread but does NOT
      eagerly start its MATLAB engine; the first :meth:`ensure_started`
      call (issued from inside the worker thread) runs
      :meth:`SimscapeAdapter.load_from_path`.
    * :meth:`close` is idempotent and quits the underlying adapter.

This module is import-time safe on hosts without MATLAB; the heavy
imports happen inside the adapter only when a real call lands.
"""

from __future__ import annotations

from src.engines.simscape.adapter import SimscapeAdapter
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_PoolWorker"]


class _PoolWorker:
    """Owns one :class:`SimscapeAdapter` for a single pool thread.

    Not part of the public API. The pool injects a worker factory in
    tests so the offline test-suite can substitute a stub adapter.

    Args:
        model_path: Absolute path to the .slx file to load.
        cache_capacity: Per-worker LRU cache capacity (``0`` disables).
        startup_timeout_s: Engine-startup wall-clock budget.
        adapter_factory: Optional callable returning a fresh adapter.
            Defaults to :class:`SimscapeAdapter`. Tests pass a mock.
    """

    def __init__(
        self,
        model_path: str,
        cache_capacity: int,
        startup_timeout_s: float,
        adapter_factory: type[SimscapeAdapter] | None = None,
    ) -> None:
        self._model_path: str = str(model_path)
        self._cache_capacity: int = int(cache_capacity)
        self._startup_timeout_s: float = float(startup_timeout_s)
        self._factory: type[SimscapeAdapter] = adapter_factory or SimscapeAdapter
        self._adapter: SimscapeAdapter | None = None
        self._closed: bool = False

    @property
    def adapter(self) -> SimscapeAdapter:
        """Return the underlying adapter, lazily starting on first use.

        Raises:
            RuntimeError: If invoked after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError("pool worker has been closed")
        if self._adapter is None:
            self._adapter = self._factory(
                cache_max_entries=self._cache_capacity,
                startup_timeout_s=self._startup_timeout_s,
            )
            self._adapter.load_from_path(self._model_path)
            logger.debug("pool worker started adapter for %s", self._model_path)
        return self._adapter

    def close(self) -> None:
        """Close the adapter (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._adapter is not None:
            try:
                self._adapter.close()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.exception("pool worker close failed (ignored)")
            finally:
                self._adapter = None
