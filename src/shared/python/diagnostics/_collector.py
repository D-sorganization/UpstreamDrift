"""DiagnosticsCollector — collect a DiagnosticsSnapshot on demand.

The collector attempts to gather system metrics via ``psutil`` (optional
dependency) and GPU metrics via ``GPUtil`` (optional dependency).  When
a dependency is absent the corresponding metric falls back to ``-1`` /
``-1.0`` so callers can always rely on a valid ``DiagnosticsSnapshot``.

Design-by-Contract invariants
------------------------------
- ``collect()`` postcondition: always returns a ``DiagnosticsSnapshot``.
- ``active_simulations`` callback (if provided) must be callable and must
  return an ``int``.

Law of Demeter
--------------
``DiagnosticsCollector`` delegates system metric gathering to private
helper functions (``_collect_system_metrics``, ``_collect_simulation_metrics``).
It never reaches through ``psutil`` sub-objects more than one level.

Implements part of Epic #5698.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.shared.python.contracts import ensure, require
from src.shared.python.diagnostics._snapshot import (
    DiagnosticsSnapshot,
    SimulationMetrics,
    SystemMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers — LOD boundary
# ---------------------------------------------------------------------------


def _try_import_psutil() -> Any:
    """Return the ``psutil`` module or ``None`` if not installed."""
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil
    except ImportError:
        return None


def _try_import_gputil() -> Any:
    """Return the ``GPUtil`` module or ``None`` if not installed."""
    try:
        import GPUtil  # type: ignore[import-not-found]

        return GPUtil
    except ImportError:
        return None


def _collect_system_metrics(psutil_mod: Any, gputil_mod: Any) -> SystemMetrics:
    """Gather system resource metrics, returning safe defaults on failure."""
    cpu_percent: float = -1.0
    memory_used_mb: float = -1.0
    memory_total_mb: float = -1.0
    memory_percent: float = -1.0
    gpu_percent: float = -1.0
    open_file_handles: int = -1

    if psutil_mod is not None:
        try:
            cpu_percent = float(psutil_mod.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001
            logger.debug("diagnostics: cpu_percent unavailable")

        try:
            vm = psutil_mod.virtual_memory()
            memory_used_mb = vm.used / (1024 * 1024)
            memory_total_mb = vm.total / (1024 * 1024)
            memory_percent = float(vm.percent)
        except Exception:  # noqa: BLE001
            logger.debug("diagnostics: virtual_memory unavailable")

        try:
            proc = psutil_mod.Process()
            open_file_handles = proc.num_fds()
        except AttributeError:
            # Windows: num_fds() is not available; use num_handles() instead
            try:
                proc = psutil_mod.Process()
                open_file_handles = proc.num_handles()
            except Exception:  # noqa: BLE001
                logger.debug("diagnostics: open_file_handles unavailable")
        except Exception:  # noqa: BLE001
            logger.debug("diagnostics: open_file_handles unavailable")

    if gputil_mod is not None:
        try:
            gpus = gputil_mod.getGPUs()
            if gpus:
                gpu_percent = float(gpus[0].load * 100)
        except Exception:  # noqa: BLE001
            logger.debug("diagnostics: GPU metrics unavailable")

    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        memory_percent=memory_percent,
        gpu_percent=gpu_percent,
        open_file_handles=open_file_handles,
    )


def _collect_simulation_metrics(
    active_simulations_fn: Callable[[], int] | None,
) -> SimulationMetrics:
    """Gather simulation subsystem metrics."""
    active: int = -1

    if active_simulations_fn is not None:
        try:
            active = int(active_simulations_fn())
        except Exception:  # noqa: BLE001
            logger.debug("diagnostics: active_simulations_fn raised")

    return SimulationMetrics(active_simulations=active)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DiagnosticsCollector:
    """Collect a point-in-time ``DiagnosticsSnapshot``.

    Args:
        active_simulations_fn: Optional zero-argument callable that returns
            the number of currently active simulations.  When ``None`` the
            ``active_simulations`` field in the snapshot will be ``-1``.

    Examples::

        collector = DiagnosticsCollector()
        snapshot = collector.collect()
        print(snapshot.system_metrics.cpu_percent)
    """

    def __init__(
        self,
        active_simulations_fn: Callable[[], int] | None = None,
    ) -> None:
        if active_simulations_fn is not None:
            require(
                callable(active_simulations_fn),
                "active_simulations_fn must be callable",
                active_simulations_fn,
            )
        self._active_simulations_fn = active_simulations_fn
        self._psutil = _try_import_psutil()
        self._gputil = _try_import_gputil()
        logger.debug(
            "diagnostics_collector_created psutil=%s gputil=%s",
            self._psutil is not None,
            self._gputil is not None,
        )

    def collect(self) -> DiagnosticsSnapshot:
        """Capture and return a diagnostics snapshot.

        Postcondition:
            Returns a ``DiagnosticsSnapshot`` instance.

        Returns:
            Immutable snapshot of current system and simulation state.
        """
        timestamp = datetime.now(tz=timezone.utc)
        system_metrics = _collect_system_metrics(self._psutil, self._gputil)
        simulation_metrics = _collect_simulation_metrics(self._active_simulations_fn)

        snapshot = DiagnosticsSnapshot(
            timestamp=timestamp,
            system_metrics=system_metrics,
            simulation_metrics=simulation_metrics,
        )
        ensure(
            isinstance(snapshot, DiagnosticsSnapshot),
            "collect postcondition: must return DiagnosticsSnapshot",
        )
        logger.debug(
            "diagnostics_snapshot_collected ts=%s cpu=%.1f%%",
            snapshot.timestamp.isoformat(),
            snapshot.system_metrics.cpu_percent,
        )
        return snapshot
