"""Process-singleton accessor for the MATLAB Engine.

This module owns *the* shared :class:`matlab.engine.MatlabEngine`
instance for the current Python process. Starting MATLAB is expensive
(10-30s wall-clock) so we amortise the cost across every adapter
instance in the same process.

Concurrency model
-----------------
**Single MATLAB engine per process.** A true engine *pool* (multiple
engines, work-stealing) is tracked in issue #4008 / #039. Until then,
adapters serialise their calls behind the engine reference returned by
:func:`get_shared_engine`.

Import safety
-------------
``import matlab.engine`` is *not* attempted at module import time. The
import happens lazily inside :func:`get_shared_engine` so that this
module is safe to import on hosts without a MATLAB licence or without
the ``matlabengine`` Python package installed. When the import fails
or the start times out, we raise the appropriate
:mod:`src.engines.simscape._errors` subclass.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from src.engines.simscape._errors import (
    SimscapeEngineStartupError,
    SimscapeNotInstalledError,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_shared_engine",
    "is_matlab_available",
    "shutdown_shared_engine",
]


_engine: Any | None = None
"""Module-level singleton handle. ``None`` until first successful start."""

_lock: threading.Lock = threading.Lock()
"""Guards lazy initialisation of :data:`_engine`."""


def is_matlab_available() -> bool:
    """Return ``True`` iff ``matlab.engine`` can be imported.

    Honours the ``UD_SIMSCAPE_FORCE_NO_MATLAB`` environment variable as
    a test escape hatch: when set to ``"1"`` we report unavailable even
    if ``matlabengine`` is installed.
    """
    if os.environ.get("UD_SIMSCAPE_FORCE_NO_MATLAB") == "1":
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("matlab.engine") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def get_shared_engine(*, startup_timeout_s: float = 60.0) -> Any:
    """Return the process-wide shared MATLAB engine, starting it if needed.

    Args:
        startup_timeout_s: Wall-clock deadline for ``start_matlab``.
            Forwarded to ``matlab.engine.start_matlab`` via the
            ``-startupOptions`` mechanism is not portable; we instead
            use a watchdog thread (see implementation).

    Returns:
        The shared ``matlab.engine.MatlabEngine`` instance.

    Raises:
        SimscapeNotInstalledError: If ``matlab.engine`` cannot be
            imported on this host.
        SimscapeEngineStartupError: If the engine fails to start within
            the timeout, or if MATLAB raises a license error during
            startup.
    """
    global _engine
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine

        if not is_matlab_available():
            raise SimscapeNotInstalledError(
                "matlab.engine is not importable; cannot start engine"
            )

        try:
            import matlab.engine  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - guarded by is_matlab_available
            raise SimscapeNotInstalledError(str(exc)) from exc

        # Watchdog thread approach: start_matlab has no timeout kwarg, so
        # we run it on a worker thread and wait. If it doesn't return in
        # time we raise; we cannot truly cancel start_matlab but the
        # process-level timeout still gives the caller a fast failure.
        result: dict[str, Any] = {}

        def _start() -> None:
            try:
                result["engine"] = matlab.engine.start_matlab()
            except Exception as exc_inner:  # noqa: BLE001 - propagated below
                result["error"] = exc_inner

        worker = threading.Thread(target=_start, name="matlab-engine-start")
        worker.daemon = True
        worker.start()
        worker.join(timeout=startup_timeout_s)

        if worker.is_alive():
            raise SimscapeEngineStartupError(
                f"matlab.engine.start_matlab did not return within "
                f"{startup_timeout_s:.1f}s"
            )
        if "error" in result:
            err = result["error"]
            err_id = getattr(err, "MatlabError", "") or ""
            if "license" in str(err).lower() or err_id.startswith("MATLAB:license:"):
                raise SimscapeEngineStartupError(
                    f"MATLAB license error: {err}",
                    matlab_error_id=err_id,
                ) from err
            raise SimscapeEngineStartupError(
                f"matlab.engine.start_matlab failed: {err}",
                matlab_error_id=err_id,
            ) from err

        _engine = result["engine"]
        logger.info("Shared MATLAB engine started")
        return _engine


def shutdown_shared_engine() -> None:
    """Quit the shared engine if one is running. Idempotent.

    After this returns, the next :func:`get_shared_engine` call will
    start a fresh engine.
    """
    global _engine
    with _lock:
        if _engine is None:
            return
        try:
            _engine.quit()
        except Exception:  # noqa: BLE001 - shutdown best-effort
            logger.exception("Error quitting shared MATLAB engine (ignored)")
        finally:
            _engine = None
            logger.info("Shared MATLAB engine shut down")
