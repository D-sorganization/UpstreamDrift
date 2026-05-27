"""Single source of truth for UpstreamDrift version and support metadata.

All version surfaces (server.py OpenAPI metadata, local_server.py, and the
root endpoint in routes/core.py) import __version__ from here so they stay
in sync with pyproject.toml.
"""

from __future__ import annotations

import platform
import sys
import warnings
from typing import TypeAlias

#: Canonical API version — must match pyproject.toml [project].version
__version__ = "2.1.1"

PlatformInfo: TypeAlias = dict[str, tuple[str, ...]]

SUPPORTED_PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13")
SUPPORTED_PYTHON_SYSTEMS = ("Linux", "Darwin", "Windows")
SUPPORTED_PLATFORMS: dict[str, PlatformInfo] = {
    "python_wheel": {
        "os": ("Linux x86_64", "macOS arm64", "Windows 10+ x86_64"),
        "python": SUPPORTED_PYTHON_VERSIONS,
        "tiers": ("core", "+extras"),
        "hardware": ("CPU",),
    },
    "docker_api": {
        "os": ("Linux x86_64",),
        "python": ("3.11",),
        "tiers": ("core", "extended"),
        "hardware": ("CPU", "optional CUDA 12"),
    },
    "tauri_desktop": {
        "os": ("Linux x86_64", "macOS arm64", "Windows 10+ x86_64"),
        "python": ("bundled",),
        "tiers": ("core", "extended"),
        "hardware": ("CPU",),
    },
    "rust_crate": {
        "os": ("Linux", "macOS", "Windows"),
        "python": ("n/a",),
        "tiers": ("n/a",),
        "hardware": ("CPU",),
    },
}


def _python_version_label(python_version: tuple[int, int]) -> str:
    """Return the support-matrix Python label for a major/minor tuple."""
    if not isinstance(python_version, tuple):
        raise TypeError("python_version must be a (major, minor) tuple")
    if len(python_version) != 2:
        raise ValueError("python_version must contain exactly two integers")
    major, minor = python_version
    if not isinstance(major, int) or not isinstance(minor, int):
        raise TypeError("python_version values must be integers")

    return f"{major}.{minor}"


def is_supported_python_platform(
    *,
    system_name: str,
    python_version: tuple[int, int],
) -> bool:
    """Return whether a Python runtime is in the release support matrix.

    Preconditions:
        ``system_name`` is a non-empty ``platform.system()`` value.
        ``python_version`` is a two-integer ``sys.version_info[:2]`` tuple.

    Postconditions:
        Returns ``True`` only for the canonical Python wheel matrix.
    """
    if not isinstance(system_name, str):
        raise TypeError("system_name must be a string")
    if not system_name.strip():
        raise ValueError("system_name must not be empty")

    python_label = _python_version_label(python_version)
    python_wheel = SUPPORTED_PLATFORMS["python_wheel"]

    return (
        system_name in SUPPORTED_PYTHON_SYSTEMS
        and python_label in python_wheel["python"]
    )


def warn_if_unsupported_platform(
    *,
    system_name: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> None:
    """Warn when the current Python runtime is outside the support matrix.

    Preconditions:
        Optional overrides must use the same shape as ``platform.system()`` and
        ``sys.version_info[:2]`` so tests can assert the contract directly.

    Postconditions:
        Emits ``UserWarning`` for out-of-matrix Python wheel combinations.
    """
    resolved_system = system_name if system_name is not None else platform.system()
    resolved_version = (
        python_version if python_version is not None else _current_python()
    )

    if is_supported_python_platform(
        system_name=resolved_system,
        python_version=resolved_version,
    ):
        return

    warnings.warn(
        f"UpstreamDrift {__version__} is not supported on "
        f"{resolved_system} / Python {_python_version_label(resolved_version)}. "
        "See docs/operations/production-readiness.md for the support matrix.",
        UserWarning,
        stacklevel=2,
    )


def _current_python() -> tuple[int, int]:
    """Return the active interpreter major/minor version."""
    return (sys.version_info.major, sys.version_info.minor)
