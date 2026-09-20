"""Probe physics engine availability across WSL host boundaries (MV-03, #10479).

Prevents native SDK availability on WSL (e.g. Pinocchio, Coal, Drake) from being
misrepresented as completely absent by Windows-only import probes.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "WslEngineReport",
    "get_wsl_engine_status",
    "is_wsl_available",
    "probe_wsl_engine",
]

DEFAULT_WSL_DISTRO = "Ubuntu-24.04"


@dataclass(frozen=True)
class WslEngineReport:
    """Diagnostic report for an engine installed inside a WSL environment."""

    engine_name: str
    available_in_wsl: bool
    distro: str | None
    version: str | None
    diagnostic_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "available_in_wsl": self.available_in_wsl,
            "distro": self.distro,
            "version": self.version,
            "diagnostic_message": self.diagnostic_message,
        }


def is_wsl_available() -> bool:
    """Check if the WSL executable is reachable on this host."""
    return shutil.which("wsl") is not None or shutil.which("wsl.exe") is not None


def _format_probe_command(engine: str) -> str:
    """Format safe inline python snippet to extract engine version."""
    pkg = "pydrake" if engine == "drake" else engine
    return f"import {pkg}; print(getattr({pkg}, '__version__', 'unknown'))"


def probe_wsl_engine(
    engine_name: str,
    distro: str | None = None,
    timeout_s: float = 5.0,
) -> WslEngineReport:
    """Probe whether an engine package is functional inside a WSL environment."""
    if not is_wsl_available():
        return WslEngineReport(
            engine_name=engine_name,
            available_in_wsl=False,
            distro=distro,
            version=None,
            diagnostic_message="WSL executable not available on this host.",
        )

    target_distro = distro or os.environ.get("UPSTREAM_WSL_DISTRO", DEFAULT_WSL_DISTRO)
    cmd = ["wsl"]
    if target_distro:
        cmd.extend(["-d", target_distro])
    cmd.extend(["python3", "-c", _format_probe_command(engine_name)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_s),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        logger.debug("WSL probe execution failed: %s", err)
        return WslEngineReport(
            engine_name=engine_name,
            available_in_wsl=False,
            distro=target_distro,
            version=None,
            diagnostic_message=f"WSL probe error: {err}",
        )

    if proc.returncode == 0:
        version_str = proc.stdout.strip() or "unknown"
        return WslEngineReport(
            engine_name=engine_name,
            available_in_wsl=True,
            distro=target_distro,
            version=version_str,
            diagnostic_message=(
                f"Available via WSL {target_distro}: {engine_name} {version_str}"
            ),
        )

    stderr_msg = proc.stderr.strip() or "import failed"
    return WslEngineReport(
        engine_name=engine_name,
        available_in_wsl=False,
        distro=target_distro,
        version=None,
        diagnostic_message=(
            f"Engine '{engine_name}' not installed in WSL ({target_distro}): {stderr_msg}"
        ),
    )


@functools.lru_cache(maxsize=32)
def get_wsl_engine_status(
    engine_name: str,
    distro: str | None = None,
) -> WslEngineReport:
    """Cached accessor for WSL engine availability."""
    return probe_wsl_engine(engine_name, distro=distro)
