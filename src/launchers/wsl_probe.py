"""Non-blocking WSL availability probing for the launcher (#8903).

The WSL-mode checkbox handler used to call ``subprocess.run(["wsl", ...])``
directly on the GUI thread, freezing the window for up to 5 s on hosts with a
cold or wedged WSL service. This module provides:

- :func:`probe_wsl_available` — the blocking probe itself (call off the GUI
  thread only);
- :class:`WslAvailabilityWorker` — a ``QThread`` that runs the probe and
  delivers the result via a signal (mirrors
  ``settings_runtime.RuntimeDependencyCheckWorker``);
- a process-lifetime result cache — WSL availability does not change while
  the app runs, so later toggles are instant.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from src.launchers.launcher_constants import CREATE_NO_WINDOW
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_WSL_PROBE_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class WslProbeResult:
    """Outcome of a WSL availability probe.

    Attributes:
        available: True when WSL responded and an Ubuntu distro is registered.
        detail: Human-readable reason when unavailable (empty when available).
    """

    available: bool
    detail: str = ""


_cached_result: WslProbeResult | None = None


def cached_wsl_result() -> WslProbeResult | None:
    """Return the process-lifetime cached probe result, if any."""
    return _cached_result


def store_wsl_result(result: WslProbeResult) -> None:
    """Cache a probe result for the rest of the process lifetime."""
    if result is None:
        raise ValueError("result must be provided")
    global _cached_result
    _cached_result = result


def reset_wsl_probe_cache() -> None:
    """Clear the cached result (test isolation)."""
    global _cached_result
    _cached_result = None


def probe_wsl_available(
    timeout: float = _WSL_PROBE_TIMEOUT_SEC,
) -> WslProbeResult:
    """Blocking WSL probe. MUST run off the GUI thread (#8903).

    Runs ``wsl --list --quiet`` and checks for a registered Ubuntu distro.

    Returns:
        A :class:`WslProbeResult`; never raises for expected failure modes
        (missing wsl.exe, timeout, decode issues).
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return WslProbeResult(available=False, detail=str(exc))

    try:
        output = result.stdout.decode("utf-16-le")
    except UnicodeError:
        output = result.stdout.decode("utf-8", errors="ignore")

    if result.returncode != 0 or "Ubuntu" not in output:
        return WslProbeResult(available=False, detail="Ubuntu not found in WSL")
    return WslProbeResult(available=True)


class WslAvailabilityWorker(QThread):
    """Run the WSL probe away from the GUI thread and signal the result.

    The ``result_ready`` signal carries a :class:`WslProbeResult` and is
    delivered on the GUI thread via the usual queued-connection mechanics.
    """

    result_ready = pyqtSignal(object)

    def __init__(
        self,
        probe: Callable[[], WslProbeResult] | None = None,
        parent: object | None = None,
    ) -> None:
        """Create the worker.

        Args:
            probe: Probe callable (injectable for tests). ``None`` selects the
                module-level :func:`probe_wsl_available` at run time, so test
                monkeypatching of the module attribute is honoured.
            parent: Optional QObject parent.
        """
        super().__init__(parent)  # type: ignore[arg-type]
        self._probe = probe

    def run(self) -> None:
        probe = self._probe if self._probe is not None else probe_wsl_available
        try:
            result = probe()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("WSL probe failed unexpectedly")
            result = WslProbeResult(available=False, detail=str(exc))
        self.result_ready.emit(result)
